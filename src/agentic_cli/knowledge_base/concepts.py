"""Concept pages — agent-curated markdown notes layered on top of the KB.

Pure file operations; no vector/embedding coupling. Grep-based search
suited for the 100-500 concept ceiling.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any


_SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9\s-]")
_SLUG_WHITESPACE = re.compile(r"[\s_]+")
_SLUG_DASH_RUN = re.compile(r"-+")
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)", re.DOTALL)

MAX_SLUG_LENGTH = 80


def slug_from_title(title: str) -> str:
    """Convert a title into a kebab-case ASCII slug.

    Rules: lowercase ASCII only, punctuation stripped, whitespace runs
    collapsed to single dashes, max 80 chars (truncated at dash boundary
    when possible). Empty / all-punctuation input returns ``"untitled"``.
    """
    # Strip accents via NFKD decomposition then drop combining marks
    decomposed = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    lowered = ascii_only.lower()
    no_punct = _SLUG_INVALID_CHARS.sub(" ", lowered)
    dashed = _SLUG_WHITESPACE.sub("-", no_punct.strip())
    collapsed = _SLUG_DASH_RUN.sub("-", dashed).strip("-")
    if not collapsed:
        return "untitled"
    if len(collapsed) <= MAX_SLUG_LENGTH:
        return collapsed
    truncated = collapsed[:MAX_SLUG_LENGTH]
    # Prefer cutting at last dash so we don't split a word
    last_dash = truncated.rfind("-")
    if last_dash > 0:
        truncated = truncated[:last_dash]
    return truncated.rstrip("-")


def render_concept_markdown(
    slug: str,
    title: str,
    body: str,
    sources: list[str],
    created_at: datetime,
    updated_at: datetime,
) -> str:
    """Render a concept page with YAML frontmatter + body."""
    sources_str = "[" + ", ".join(sources) + "]"
    lines = [
        "---",
        f"slug: {slug}",
        f"title: {title}",
        f"created_at: {created_at.isoformat()}",
        f"updated_at: {updated_at.isoformat()}",
        f"sources: {sources_str}",
        "---",
        "",
    ]
    body_trimmed = body if body.endswith("\n") else body + "\n"
    return "\n".join(lines) + body_trimmed


def parse_concept_markdown(md: str) -> dict[str, Any] | None:
    """Parse a concept-page string into ``{slug, title, sources, body, ...}``.

    Returns ``None`` if no frontmatter fence is found.
    """
    m = _FRONTMATTER_RE.match(md)
    if not m:
        return None

    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()

    sources_raw = fm.get("sources", "[]")
    sources: list[str] = []
    if sources_raw.startswith("[") and sources_raw.endswith("]"):
        inner = sources_raw[1:-1].strip()
        if inner:
            sources = [s.strip() for s in inner.split(",") if s.strip()]

    return {
        "slug": fm.get("slug", ""),
        "title": fm.get("title", ""),
        "created_at": fm.get("created_at", ""),
        "updated_at": fm.get("updated_at", ""),
        "sources": sources,
        "body": m.group(2).lstrip("\n"),
    }
