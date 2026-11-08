"""Helpers for the knowledge base storage tests: the database read raw, a 0.6
folder written by hand, and a recorder for what the package logs."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"
C = "33333333-3333-4333-8333-333333333333"
D = "44444444-4444-4444-8444-444444444444"
E = "55555555-5555-4555-8555-555555555555"


def read_rows(kb_dir: Path, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
    """``sql`` run read-only on ``kb_dir``'s database."""
    uri = (Path(kb_dir) / "kb.sqlite").absolute().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as conn:
        return conn.execute(sql, list(params)).fetchall()


def raw_write(kb_dir: Path, *statements: str | tuple[str, Iterable[Any]]) -> None:
    """Run ``statements`` on the database as a plain SQLite client would (no
    defensive mode): how a shipped database gets a trigger or a bad row."""
    with contextlib.closing(sqlite3.connect(Path(kb_dir) / "kb.sqlite", isolation_level=None)) as conn:
        for statement in statements:
            sql, params = (statement, ()) if isinstance(statement, str) else statement
            conn.execute(sql, list(params))


class LogRecorder:
    """Stands in for the modules' ``logger``: records (level, event, fields)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def _at(level: str):  # noqa: N805 - a method factory
        def record(self, event, **fields) -> None:
            self.records.append((level, event, fields))

        return record

    debug, info, warning, error = _at("debug"), _at("info"), _at("warning"), _at("error")

    def warnings(self, event: str) -> list[dict]:
        return [f for level, name, f in self.records if level == "warning" and name == event]

    def infos(self, event: str) -> list[dict]:
        return [f for level, name, f in self.records if level == "info" and name == event]

    def install(self, monkeypatch, *modules) -> "LogRecorder":
        for module in modules:
            monkeypatch.setattr(module, "logger", self, raising=False)
        return self


def header_06(doc_id: str, title: str, chunks: int = 1, **fields: Any) -> dict[str, Any]:
    """A 0.6 metadata.json header (v2/v3: no text; chunk texts in documents/<id>.json)."""
    header = {
        "id": doc_id,
        "title": title,
        "summary": f"About {title}.",
        "source_type": "user",
        "source_url": None,
        "file_path": None,
        "created_at": "2026-10-01T10:00:00",
        "updated_at": "2026-10-01T10:00:00",
        "metadata": {},
        "chunks": [
            {"id": f"{doc_id}-c{i}", "document_id": doc_id, "chunk_index": i, "metadata": {"title": title}}
            for i in range(chunks)
        ],
    }
    header.update(fields)
    return header


def write_06(
    folder: Path,
    documents: Iterable[tuple[str, str, str]],
    *,
    extra_headers: Iterable[Any] = (),
    position: int = 1,
) -> None:
    """A 0.6 knowledge base folder (format v3) holding ``documents``
    ((id, title, text) each, with a sidecar), ``extra_headers`` inserted at
    ``position`` of metadata.json, and an embeddings/ folder."""
    folder = Path(folder)
    for sub in ("documents", "files", "embeddings"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    headers: list[Any] = []
    for doc_id, title, text in documents:
        headers.append(header_06(doc_id, title))
        (folder / "documents" / f"{doc_id}.json").write_text(
            json.dumps({"content": text, "chunks": {f"{doc_id}-c0": text}})
        )
        (folder / "documents" / f"{doc_id}.md").write_text(f"---\nid: {doc_id}\n---\n# {title}\n")
    headers[position:position] = list(extra_headers)
    (folder / "metadata.json").write_text(
        json.dumps({"version": 3, "documents": headers, "updated_at": "2026-10-01T10:00:00"})
    )
    (folder / "embeddings" / "index.faiss").write_bytes(b"a 0.6 FAISS index")
    (folder / "embeddings" / "model.json").write_text('{"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384}')
