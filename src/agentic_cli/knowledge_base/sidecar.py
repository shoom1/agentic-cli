"""Pure formatting helpers for KB markdown artifacts.

Keeps the manager focused on persistence + lifecycle while exposing
deterministic, side-effect-free render/parse functions for sidecars
and the index.md file.
"""

from __future__ import annotations

import re
from typing import Any

from agentic_cli.knowledge_base.models import Document, SourceType


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def render_sidecar_markdown(doc: Document, payload: dict[str, Any]) -> str:
    """Render a per-document markdown sidecar.

    Args:
        doc: Document instance (must have id, title, source_type, created_at).
        payload: Dict with optional keys ``summary`` (str), ``claims``
            (list[str]), ``entities`` (dict[str, list[str]]). Missing or
            empty sections are omitted from the body.

    Returns:
        Markdown string with YAML frontmatter and body sections.
    """
    fm_lines = ["---"]
    fm_lines.append(f"id: {doc.id}")
    fm_lines.append(f"title: {doc.title}")
    fm_lines.append(f"source_type: {doc.source_type.value}")
    if doc.source_url:
        fm_lines.append(f"source_url: {doc.source_url}")
    authors = doc.metadata.get("authors") if doc.metadata else None
    if authors:
        fm_lines.append("authors: [" + ", ".join(authors) + "]")
    arxiv_id = doc.metadata.get("arxiv_id") if doc.metadata else None
    if arxiv_id:
        fm_lines.append(f"arxiv_id: {arxiv_id}")
    fm_lines.append(f"ingested_at: {doc.created_at.isoformat()}")
    fm_lines.append(f"chunks: {len(doc.chunks)}")
    if doc.file_path:
        fm_lines.append(f"file: files/{doc.file_path}")
    fm_lines.append("---")

    body: list[str] = [""]
    summary = payload.get("summary") or doc.summary or ""
    if summary:
        body.append("## Summary")
        body.append(summary)
        body.append("")

    claims = payload.get("claims") or []
    if claims:
        body.append("## Key Claims")
        for c in claims:
            body.append(f"- {c}")
        body.append("")

    entities = payload.get("entities") or {}
    if entities:
        body.append("## Key Entities")
        for kind, names in entities.items():
            if names:
                body.append(f"{kind}: " + ", ".join(names))
        body.append("")

    return "\n".join(fm_lines) + "\n" + "\n".join(body)


def parse_sidecar_frontmatter(md: str) -> dict[str, str]:
    """Parse the YAML frontmatter block out of a sidecar string.

    Returns a flat dict of string keys to string values. List/structured
    values are returned as their raw markdown representation. Used for
    audit/debug, not as a generic YAML parser.
    """
    m = _FRONTMATTER_RE.match(md)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def render_index_md(documents: list[Document], updated_at_iso: str) -> str:
    """Render the index.md file from a list of documents.

    Grouped by source_type, sorted by ingest date desc within each group.
    """
    by_type: dict[str, list[Document]] = {}
    for d in documents:
        by_type.setdefault(d.source_type.value, []).append(d)

    out: list[str] = ["# Knowledge Base Index", ""]
    out.append(f"Last updated: {updated_at_iso} · {len(documents)} documents")
    out.append("")

    for stype in sorted(by_type.keys()):
        docs = sorted(by_type[stype], key=lambda d: d.created_at, reverse=True)
        out.append(f"## {stype} ({len(docs)})")
        for d in docs:
            out.append(_index_line(d))
        out.append("")
    return "\n".join(out)


def _index_line(d: Document) -> str:
    """One-line entry in the index.md grouped section."""
    arxiv_id = d.metadata.get("arxiv_id") if d.metadata else None
    authors = d.metadata.get("authors") if d.metadata else None
    date = d.created_at.date().isoformat()
    if arxiv_id:
        prefix = f"[{arxiv_id}]"
    else:
        prefix = "-"
    parts = [prefix, d.title]
    if authors:
        parts.append(f"— {authors[0]} et al.")
    parts.append(f"— {date}")
    line = " ".join(parts)
    if not line.startswith("- "):
        line = "- " + line
    return line
