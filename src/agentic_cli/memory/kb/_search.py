"""Keyword ranking, search filters and rank fusion for a knowledge base.

Both ranked lists are complete (every chunk with a vector, every chunk holding
a term), and the filters apply before ranking, so a search ranks once: no
re-run with more candidates when few documents pass (review F13).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from agentic_cli.memory.kb.models import SourceType

# Reciprocal rank fusion constant: the standard 60.
RRF_K = 60

_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")


def fuse(
    rankings: Sequence[Sequence[Any]],
    *,
    owners: Sequence[Sequence[Any]] | None = None,
    top_k: int | None = None,
    k: int = RRF_K,
) -> list[tuple[Any, float]]:
    """Merge ranked lists by reciprocal rank fusion, best first, as (item, score).

    Scores from different indexes are on different scales, so only ranks
    count: each list adds ``1 / (k + rank + 1)`` to an item's score. Ties keep
    the order in which items were first seen. With ``owners`` (each list's
    items' owners, in step with it: a chunk's document), only each owner's
    best item is kept; ``top_k`` cuts the result.

    Vectorized: a search fuses every chunk with a vector and every chunk
    holding a term, which can be a hundred thousand of each. With ``top_k``
    only the head of the fused order is sorted, a longer head each time it
    holds too few owners: each head is exact (every item tied with its last
    one included), so the result is the whole order's.
    """
    present = [i for i, ranking in enumerate(rankings) if len(ranking)]
    if not present:
        return []
    items = np.concatenate([np.asarray(rankings[i]) for i in present])
    weights = np.concatenate([1.0 / (k + np.arange(len(rankings[i]), dtype=np.float64) + 1) for i in present])
    unique, first, inverse = np.unique(items, return_index=True, return_inverse=True)
    scores = np.bincount(inverse.reshape(-1), weights=weights, minlength=len(unique))
    owner_of = None if owners is None else np.concatenate([np.asarray(owners[i]) for i in present])
    size = len(unique) if top_k is None else max(4 * top_k, 64)
    while True:
        order = _head(scores, first, size)
        if owner_of is None:
            picked = order if top_k is None else order[:top_k]
        else:
            picked = _first_of_each(order, owner_of[first[order]].tolist(), top_k)
        if top_k is None or len(picked) >= top_k or len(order) == len(unique):
            return list(zip(unique[picked].tolist(), scores[picked].tolist()))
        size *= 4


def _head(scores: np.ndarray, first: np.ndarray, size: int) -> np.ndarray:
    """The first ``size`` items of the order by (score descending, first
    seen), and every item tied with the last of them."""
    if size < len(scores):
        threshold = -np.partition(-scores, size - 1)[size - 1]
        chosen = np.flatnonzero(scores >= threshold)
    else:
        chosen = np.arange(len(scores))
    return chosen[np.lexsort((first[chosen], -scores[chosen]))]


def _first_of_each(order: np.ndarray, owners: list[Any], top_k: int | None) -> np.ndarray:
    """The items of ``order`` that come first for their owner, up to ``top_k``."""
    seen: set[Any] = set()
    kept: list[int] = []
    for position, owner in enumerate(owners):
        if owner not in seen:
            seen.add(owner)
            kept.append(position)
            if top_k is not None and len(kept) >= top_k:
                break
    return order[np.asarray(kept, dtype=np.int64)]


def keyword_ranked(
    conn: sqlite3.Connection, terms: Sequence[str], *, fts: bool, allowed: set[str] | None = None
) -> list[tuple[int, str]]:
    """The chunks holding any of ``terms`` as (rowid, document ID), best first.

    With the FTS5 index, by BM25; without it (a SQLite built without FTS5), by
    how many of the terms each chunk holds. ``allowed`` keeps only those
    documents' chunks.
    """
    unique = list(dict.fromkeys(terms))
    if not unique:
        return []
    if fts:
        query = " OR ".join('"' + term.replace('"', '""') + '"' for term in unique)
        rows = conn.execute(
            "SELECT chunks.rowid, chunks.document_id FROM chunk_terms JOIN chunks ON chunks.rowid = chunk_terms.rowid "
            "WHERE chunk_terms MATCH ? ORDER BY bm25(chunk_terms), chunks.rowid",
            (query,),
        )
    else:
        held = " + ".join(["(instr(' ' || terms || ' ', ?) > 0)"] * len(unique))
        rows = conn.execute(
            f"SELECT rowid, document_id FROM (SELECT rowid, document_id, {held} AS held FROM chunks) "
            "WHERE held > 0 ORDER BY held DESC, rowid",
            [f" {term} " for term in unique],
        )
    return [(int(rowid), str(doc_id)) for rowid, doc_id in rows if allowed is None or doc_id in allowed]


def allowed_documents(conn: sqlite3.Connection, filters: dict[str, Any]) -> set[str]:
    """The IDs of the documents ``filters`` keep.

    ``source_type``: a ``SourceType`` or its value. ``date_from`` / ``date_to``:
    ISO 8601 dates or times, compared with when the document was added; a
    date-only ``date_to`` keeps that whole day, and a time with an offset is
    compared as local time. ``keywords``: each must occur in the text, any
    case. Other keys are ignored.

    Raises:
        ValueError: an unknown source type, or a date that is not ISO 8601.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if "source_type" in filters:
        clauses.append("source_type = ?")
        params.append(SourceType(filters["source_type"]).value)
    if "date_from" in filters:
        clauses.append("created_at >= ?")
        params.append(_local(filters["date_from"]).isoformat())
    if "date_to" in filters:
        value = str(filters["date_to"])
        if _DATE_ONLY.fullmatch(value):
            clauses.append("created_at < ?")
            params.append((_local(value) + timedelta(days=1)).isoformat())
        else:
            clauses.append("created_at <= ?")
            params.append(_local(value).isoformat())
    for keyword in filters.get("keywords") or []:
        clauses.append("instr(kb_casefold(content), ?) > 0")
        params.append(str(keyword).casefold())
    sql = "SELECT id FROM documents" + (" WHERE " + " AND ".join(clauses) if clauses else "")
    return {str(row[0]) for row in conn.execute(sql, params)}


def _local(value: Any) -> datetime:
    moment = datetime.fromisoformat(str(value))
    return moment if moment.tzinfo is None else moment.astimezone().replace(tzinfo=None)
