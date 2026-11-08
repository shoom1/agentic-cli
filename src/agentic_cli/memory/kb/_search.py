"""Keyword ranking, search filters and rank fusion for a knowledge base.

Both ranked lists are complete (every chunk with a vector, every chunk holding
a term), and the filters apply before ranking, so a search ranks once: no
re-run with more candidates when few documents pass (review F13).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Hashable, Sequence
from datetime import datetime, timedelta
from typing import Any

from agentic_cli.memory.kb.models import SourceType

# Reciprocal rank fusion constant: the standard 60.
RRF_K = 60

_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")


def fuse(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[tuple[Hashable, float]]:
    """Merge ranked lists by reciprocal rank fusion, best first.

    Scores from different indexes are on different scales, so only ranks
    count: each list adds ``1 / (k + rank + 1)`` to an item's score. Ties keep
    the order in which items were first seen.
    """
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


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
