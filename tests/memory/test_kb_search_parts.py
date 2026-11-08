"""Keyword ranking, search filters and rank fusion (review F12, F14)."""

from __future__ import annotations

import contextlib
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from agentic_cli.memory.kb._db import Database
from agentic_cli.memory.kb._search import allowed_documents, fuse, keyword_ranked
from agentic_cli.memory._core.tokenize import tokenize
from agentic_cli.memory.kb.models import Document, DocumentChunk, SourceType
from tests.memory.kb_storage import A, B, C


def _db(tmp_path, *docs):
    db = Database.open(tmp_path / "kb.sqlite")
    with db.write():
        for doc_id, text, source, created in docs:
            doc = Document(id=doc_id, title=text[:10], content=text, source_type=source, created_at=created, updated_at=created)
            doc.chunks = [DocumentChunk(id=f"{doc_id}-0", document_id=doc_id, content=text, chunk_index=0)]
            db.insert_document(doc, [" ".join(tokenize(text))], None, None)
    return db


NOON = datetime(2026, 11, 7, 12)


@pytest.fixture
def db(tmp_path):
    db = _db(
        tmp_path,
        (A, "Otters nest in burrows; otters swim.", SourceType.USER, NOON),
        (B, "Herons wade, and otters watch them.", SourceType.ARXIV, NOON - timedelta(days=2)),
        (C, "水獭住在河边的洞穴里。", SourceType.WEB, NOON - timedelta(days=5)),
    )
    yield db
    db.close()


def test_fuse_adds_each_list_s_rank():
    assert [item for item, _ in fuse([["x", "y"], ["y", "z"]])] == ["y", "x", "z"]


def _fused_by_dict(rankings, owners, top_k, k=60):
    """Reciprocal rank fusion item by item, then each owner's best item: what
    the vectorized ``fuse`` must give exactly."""
    scores: dict = {}
    owner_of: dict = {}
    for ranking, owned in zip(rankings, owners):
        for rank, (item, owner) in enumerate(zip(ranking, owned)):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
            owner_of[item] = owner
    picked, seen = [], set()
    for item, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        if owner_of[item] not in seen:
            seen.add(owner_of[item])
            picked.append((item, score))
    return picked[:top_k]


@pytest.mark.parametrize("n", [200, 201, 260])
def test_fusion_keeps_every_tie_at_the_end_of_the_head_it_sorts(n):
    # Mirrored lists tie each chunk with its mirror; one document holds the
    # top chunks, so the second document's chunk sits where the sorted head
    # ends, tied with a chunk of a third document seen after it.
    semantic = [0, *range(1, n + 1)]
    keyword = [0, *range(n, 0, -1)]
    owner = {chunk: "big" for chunk in semantic}
    for position in range(1, n // 2 + 1):
        if 2 * position + 1 >= 64:
            owner[position], owner[n + 1 - position] = "second", "third"
            break
    owners = [[owner[c] for c in semantic], [owner[c] for c in keyword]]

    got = fuse([np.array(semantic), np.array(keyword)], owners=[np.array(o) for o in owners], top_k=2)

    assert got == _fused_by_dict([semantic, keyword], owners, 2)


@pytest.mark.parametrize("seed", range(40))
def test_fusion_is_the_item_by_item_fusion_ties_included(seed):
    rnd = random.Random(seed)
    chunks = list(range(1, rnd.randint(2, 300)))
    owner = {chunk: f"doc-{rnd.randrange(1 + len(chunks) // 3)}" for chunk in chunks}
    if seed % 4 == 1:  # one document holds most chunks: the head of the order grows to find others
        owner.update({chunk: "doc-big" for chunk in chunks if rnd.random() < 0.9})
    semantic = rnd.sample(chunks, rnd.randint(0, len(chunks)))
    keyword = rnd.sample(chunks, rnd.randint(0, len(chunks)))
    if seed % 4 == 0:  # mirrored ranks: every shared chunk ties with its mirror
        keyword = list(reversed(semantic))
    rankings = [semantic, keyword]
    owners = [[owner[c] for c in semantic], [owner[c] for c in keyword]]
    top_k = rnd.randint(1, 20)

    got = fuse([np.array(r, dtype=np.int64) for r in rankings], owners=[np.array(o) for o in owners], top_k=top_k)

    assert got == _fused_by_dict(rankings, owners, top_k)


@pytest.mark.parametrize("fts", [True, False])
def test_keyword_ranking_finds_every_chunk_holding_a_term(db, fts, monkeypatch):
    if not fts:
        monkeypatch.setattr(db, "_fts_module", False)
    with db.read():
        ranked = keyword_ranked(db.conn, tokenize("otters"), fts=db.fts)
    assert [doc for _, doc in ranked] == [A, B]  # A holds the term twice


def test_each_term_reaches_the_keyword_index_unchanged_and_a_query_is_split_alike(tmp_path):
    text = "Played on sixty-four squares: naïve_café, 水獭住在, मुझे किताब, Résumé 3.14"
    db = _db(tmp_path, (A, text, SourceType.USER, NOON))
    with db.read():
        found = [doc for _, doc in keyword_ranked(db.conn, tokenize("Sixty-Four SQUARES!"), fts=db.fts)]
    db.close()
    uri = (tmp_path / "kb.sqlite").absolute().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.execute("CREATE VIRTUAL TABLE temp.vocabulary USING fts5vocab('main', 'chunk_terms', 'row')")
        indexed = {term for (term,) in conn.execute("SELECT term FROM temp.vocabulary")}

    assert found == [A]
    assert indexed == set(tokenize(text))


def test_keyword_ranking_finds_a_chinese_word(db):
    with db.read():
        assert [doc for _, doc in keyword_ranked(db.conn, tokenize("水獭"), fts=db.fts)] == [C]


def test_date_to_keeps_the_whole_day_it_names(db):
    with db.read():
        assert allowed_documents(db.conn, {"date_to": "2026-11-07"}) == {A, B, C}
        assert allowed_documents(db.conn, {"date_to": "2026-11-06"}) == {B, C}


@pytest.fixture
def ahead_of_utc():
    """Local time 5 h 30 min ahead of UTC wherever the tests run: in UTC (as
    CI runs) a time with an offset reads the same converted or not."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX")
    saved = os.environ.get("TZ")
    os.environ["TZ"] = "IST-5:30"
    time.tzset()
    yield
    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


def test_a_date_with_an_offset_is_compared_as_local_time(db, ahead_of_utc):
    # NOON is local time; each bound is a minute from it, written in UTC.
    just_before = (NOON - timedelta(minutes=1)).astimezone(timezone.utc).isoformat()
    just_after = (NOON + timedelta(minutes=1)).astimezone(timezone.utc).isoformat()
    with db.read():
        assert allowed_documents(db.conn, {"date_from": just_before}) == {A}
        assert allowed_documents(db.conn, {"date_from": just_after}) == set()


def test_filters_combine_and_keywords_ignore_case(db):
    with db.read():
        assert allowed_documents(db.conn, {"source_type": "arxiv", "keywords": ["HERONS"]}) == {B}
        assert allowed_documents(db.conn, {"keywords": ["Выдры"]}) == set()


def test_an_unknown_source_type_is_an_error(db):
    with db.read(), pytest.raises(ValueError):
        allowed_documents(db.conn, {"source_type": "podcast"})
