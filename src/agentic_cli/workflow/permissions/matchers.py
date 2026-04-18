# src/agentic_cli/workflow/permissions/matchers.py
"""Matchers for capability targets.

Each capability namespace (``filesystem.*``, ``http.*``, ``shell.*``) has a
Matcher that knows how to canonicalize targets + patterns and compare them.
Unknown namespaces fall back to ``StringGlobMatcher``.
"""

from __future__ import annotations

import fnmatch
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from agentic_cli.workflow.permissions.store import PermissionContext


@runtime_checkable
class Matcher(Protocol):
    """Canonicalise and match targets for a capability namespace."""

    def canonicalize(self, s: str, ctx: PermissionContext) -> str: ...

    def matches(self, pattern: str, target: str) -> bool: ...


class StringGlobMatcher:
    """Default fallback: ``${...}`` substitution + ``fnmatch``."""

    def canonicalize(self, s: str, ctx: PermissionContext) -> str:
        return ctx.substitute(s).strip()

    def matches(self, pattern: str, target: str) -> bool:
        return fnmatch.fnmatchcase(target, pattern)


@lru_cache(maxsize=512)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a ``**``-aware glob into a compiled regex.

    * ``*`` matches one path segment (no ``/``).
    * ``**`` matches zero or more complete path segments (including their
      separating ``/`` characters).
    * ``?`` matches one non-``/`` character within a segment.
    """
    segments = pattern.split("/")
    # Build per-segment regex pieces.  Each piece is the regex for that
    # segment *without* surrounding slashes; we join with "/" afterwards,
    # but "**" segments need special junction treatment.
    pieces: list[str] = []
    for seg in segments:
        if seg == "**":
            pieces.append("**")  # placeholder; resolved during join below
            continue
        seg_re = ""
        for ch in seg:
            if ch == "*":
                seg_re += "[^/]*"
            elif ch == "?":
                seg_re += "[^/]"
            else:
                seg_re += re.escape(ch)
        pieces.append(seg_re)

    # Join pieces, collapsing "**" placeholders.
    # "**" as a segment means "zero or more <segment>/" sequences.
    # We absorb the "/" on the right side of "**" into its regex so that
    # a/z still matches a/**/z (zero repetitions absorbs the "/" between
    # "a" and "z" is just the normal "/" join when ** is absent).
    #
    # Strategy: iterate and build the body string, handling ** specially.
    body = ""
    for i, piece in enumerate(pieces):
        if piece == "**":
            # Replace the preceding "/" (already appended) with nothing,
            # then emit the zero-or-more pattern that *includes* a trailing "/".
            # We do this by emitting `(?:[^/]+/)*` and NOT prepending a "/".
            # But we already appended "/" before this piece if i > 0.
            # So strip the trailing "/" from body and emit the junction regex.
            if i > 0 and body.endswith("/"):
                body = body[:-1]
            # Zero or more <non-empty-segment>/ sequences
            body += "(?:/[^/]+)*"
        else:
            if i > 0:
                body += "/"
            body += piece

    return re.compile(f"^{body}$")


class PathMatcher:
    """Matcher for ``filesystem.*`` capabilities."""

    def canonicalize(self, s: str, ctx: PermissionContext) -> str:
        s = ctx.substitute(s)
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = ctx.workdir / p
        return str(Path(p).resolve(strict=False))

    def matches(self, pattern: str, target: str) -> bool:
        return bool(_glob_to_regex(pattern).match(target))
