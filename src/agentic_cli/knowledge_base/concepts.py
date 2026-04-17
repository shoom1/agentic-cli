"""Concept pages — agent-curated markdown notes layered on top of the KB.

Pure file operations; no vector/embedding coupling. Grep-based search
suited for the 100-500 concept ceiling.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_cli.file_utils import atomic_write_text


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


class ConceptStore:
    """File-backed store for agent-curated concept pages.

    All concepts live at ``<base_dir>/{slug}.md``. No database; each
    write is an atomic file replace. Search is grep-based.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def _concept_path(self, slug: str) -> Path:
        return self.base_dir / f"{slug}.md"

    def write(
        self,
        title: str,
        body: str,
        sources: list[str],
        slug: str = "",
        valid_ids_check: Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        """Create or overwrite a concept page.

        Args:
            title: Human-readable title.
            body: Markdown body (with or without its own ## sections).
            sources: Document IDs this concept cites. Must be non-empty
                after validation — an empty ``sources`` list or all-IDs-
                invalid returns ``{"success": False, ...}``.
            slug: Optional explicit slug. If empty, auto-generated from
                ``title`` and collision-suffixed on conflict. If explicit
                and already exists, the existing concept is overwritten
                (body replaced, sources merged as union).
            valid_ids_check: Optional ``Callable[[str], bool]``. When
                provided, source IDs that fail the check are dropped and
                returned in ``invalid_sources``. When all sources are
                invalid, the write fails.

        Returns:
            Dict with keys ``success``, ``slug``, ``path``, ``action``
            (``"created"`` or ``"updated"``), and ``invalid_sources``.
        """
        valid_sources, invalid = _partition_sources(sources, valid_ids_check)

        if not valid_sources:
            return {
                "success": False,
                "slug": "",
                "path": "",
                "action": "failed",
                "error": (
                    "concept page requires at least one valid source "
                    "document id; got none"
                ),
                "invalid_sources": invalid,
            }

        self.base_dir.mkdir(parents=True, exist_ok=True)

        if slug:
            # Explicit slug: overwrite if exists, merge sources
            existing = self.read(slug)
            action = "updated" if existing is not None else "created"
            if existing is not None:
                merged: list[str] = list(existing["sources"])
                for s in valid_sources:
                    if s not in merged:
                        merged.append(s)
                valid_sources = merged
                created_at = _parse_iso(existing["created_at"]) or datetime.now()
            else:
                created_at = datetime.now()
            final_slug = slug
        else:
            # Auto slug: collision-suffix if exists (never overwrites)
            base_slug = slug_from_title(title)
            final_slug = self._resolve_collision(base_slug)
            created_at = datetime.now()
            action = "created"

        now = datetime.now()
        path = self._concept_path(final_slug)
        md = render_concept_markdown(
            slug=final_slug,
            title=title,
            body=body,
            sources=valid_sources,
            created_at=created_at,
            updated_at=now,
        )
        atomic_write_text(path, md)

        return {
            "success": True,
            "slug": final_slug,
            "path": str(path),
            "action": action,
            "invalid_sources": invalid,
        }

    def read(self, slug: str) -> dict[str, Any] | None:
        """Load a concept page by slug. Returns None if missing."""
        path = self._concept_path(slug)
        if not path.exists():
            return None
        return parse_concept_markdown(path.read_text())

    def list(self) -> list[dict[str, Any]]:
        """Return all concepts as summary dicts, sorted by updated_at desc.

        Files missing frontmatter are silently skipped.
        """
        if not self.base_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in self.base_dir.glob("*.md"):
            parsed = parse_concept_markdown(path.read_text())
            if parsed is None:
                continue
            items.append({
                "slug": parsed["slug"] or path.stem,
                "title": parsed["title"],
                "updated_at": parsed["updated_at"],
                "sources": parsed["sources"],
            })
        items.sort(key=lambda it: it["updated_at"], reverse=True)
        return items

    def _resolve_collision(self, base_slug: str) -> str:
        """Return ``base_slug`` if free, else ``base_slug-2``, ``-3``, ..."""
        if not self._concept_path(base_slug).exists():
            return base_slug
        for suffix in range(2, 1000):
            candidate = f"{base_slug}-{suffix}"
            if not self._concept_path(candidate).exists():
                return candidate
        raise RuntimeError(
            f"exhausted collision suffixes 2..999 for slug {base_slug!r}"
        )


def _partition_sources(
    sources: list[str],
    valid_ids_check: Callable[[str], bool] | None,
) -> tuple[list[str], list[str]]:
    """Split a sources list into (valid, invalid) using the check callable.

    When ``valid_ids_check`` is None, all sources are considered valid.
    De-duplicates while preserving order.
    """
    seen: set[str] = set()
    valid: list[str] = []
    invalid: list[str] = []
    for s in sources:
        if not s or s in seen:
            continue
        seen.add(s)
        if valid_ids_check is None or valid_ids_check(s):
            valid.append(s)
        else:
            invalid.append(s)
    return valid, invalid


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
