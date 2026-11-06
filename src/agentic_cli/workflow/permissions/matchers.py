# src/agentic_cli/workflow/permissions/matchers.py
"""Matchers for capability targets.

Each capability namespace (``filesystem.*``, ``http.*``, ``shell.*``) has a
Matcher that canonicalizes rule patterns and call targets and compares them.
Unknown namespaces fall back to ``StringGlobMatcher``.

Patterns and targets are canonicalized by **different** methods:

* ``canonicalize_pattern`` is for rule targets (builtin rules, settings files).
  It expands ``${workdir}``, ``${home}`` and ``${app_name}``, and anchors a
  relative filesystem pattern to the project directory.
* ``canonicalize_target`` is for values taken from a tool call's arguments.
  It expands no placeholders, because the tool treats that text literally, and
  it resolves a filesystem path exactly as the tool will
  (:func:`agentic_cli.paths.resolve_path`). Using the pattern form here once
  let the engine judge one location while the tool acted on another.

A URL, pattern or target, is the one the HTTP client sends
(:func:`agentic_cli.urls.canonical_url`), so a rule judges the page and host
the request reaches, however the URL spells them.

``"*"`` is the wildcard only on the pattern side. As a *target* it is the
sentinel a targetless capability resolves to (and what a grant for it widens
to), so ``canonicalize_target`` never returns it for a real argument: a path
``*`` is a file named ``*``, a URL is canonicalized (``https://*/``), and a
bare ``*`` command or string is rejected (``ValueError``), which the engine
denies.

A deny rule is matched with ``matches_deny``: a filesystem deny ignores case
and Unicode normalization (see ``PathMatcher.matches_deny``), every other
matcher denies exactly as it allows.
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import SplitResult, urlsplit

from agentic_cli.paths import resolve_path
from agentic_cli.urls import canonical_url
from agentic_cli.workflow.permissions.store import PermissionContext


@runtime_checkable
class Matcher(Protocol):
    """Canonicalise and match targets for a capability namespace."""

    def canonicalize_pattern(self, s: str, ctx: PermissionContext) -> str: ...

    def canonicalize_target(self, s: str, ctx: PermissionContext) -> str: ...

    def matches(self, pattern: str, target: str) -> bool: ...

    def matches_deny(self, pattern: str, target: str) -> bool: ...


class StringGlobMatcher:
    """Default fallback: ``${...}`` substitution (patterns only) + ``fnmatch``."""

    def canonicalize_pattern(self, s: str, ctx: PermissionContext) -> str:
        if s == "*":
            return "*"
        return ctx.substitute(s).strip()

    def canonicalize_target(self, s: str, ctx: PermissionContext) -> str:
        return _literal_string_target(s)

    def matches(self, pattern: str, target: str) -> bool:
        if pattern == "*":
            return True
        return fnmatch.fnmatchcase(target, pattern)

    def matches_deny(self, pattern: str, target: str) -> bool:
        return self.matches(pattern, target)


def _literal_string_target(s: str) -> str:
    """A string argument as a target: stripped, never the wildcard sentinel."""
    target = s.strip()
    if target == "*":
        raise ValueError("'*' is not a valid target")
    return target


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

    def canonicalize_pattern(self, s: str, ctx: PermissionContext) -> str:
        if s == "*":
            return "*"
        s = ctx.substitute(s)
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = ctx.workdir / p
        return str(Path(p).resolve(strict=False))

    def canonicalize_target(self, s: str, ctx: PermissionContext) -> str:
        """The location the tool will act on, via the shared resolver.

        ``ctx`` is deliberately unused: a relative path is anchored where the
        tool anchors it (the process's current directory), not at
        ``ctx.workdir``. A ``*`` argument is a file named ``*``.
        """
        return str(resolve_path(s))

    def matches(self, pattern: str, target: str) -> bool:
        if pattern == "*":
            return True
        return bool(_glob_to_regex(pattern).match(target))

    def matches_deny(self, pattern: str, target: str) -> bool:
        """Whether a deny ``pattern`` covers ``target``, whatever its case.

        On a case-insensitive filesystem (APFS's default, NTFS) ``./.ENV`` and
        ``./.env`` are one file, and :func:`agentic_cli.paths.resolve_path`
        keeps the case it was given, so a deny compared case by case missed
        every other spelling. The two are compared as given and, failing
        that, casefolded after NFC (the glob characters are unaffected). That
        only denies more, so it holds on a case-sensitive filesystem too; an
        allow still compares case, and a case variant of an allowed path asks.
        """
        return self.matches(pattern, target) or self.matches(_fold(pattern), _fold(target))


def _fold(path: str) -> str:
    """A path's caseless form: NFC (a decomposed name equals a composed one),
    then casefolded."""
    return unicodedata.normalize("NFC", path).casefold()


class URLMatcher:
    """Matcher for ``http.*`` capabilities.

    A pattern and a target are both the URL the HTTP client sends
    (:func:`agentic_cli.urls.canonical_url`): ``/x/../admin``, ``/%61dmin``,
    ``host.`` and ``host。com`` are the page and the host they reach, so a
    deny written for one spelling holds for every other. A pattern keeps its
    wildcards (``*`` and ``**`` in the path, ``fnmatch`` in the query); the
    host and port are compared exactly. A URL written without a scheme is
    read as ``https://``. A target the client cannot send raises
    ``ValueError``, which the engine denies.
    """

    _DEFAULT_PORTS = {"http": 80, "https": 443}
    _SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")

    def canonicalize_pattern(self, s: str, ctx: PermissionContext) -> str:
        if s == "*":
            return "*"
        return self._canonical(ctx.substitute(s).strip())

    def canonicalize_target(self, s: str, ctx: PermissionContext) -> str:
        # Not stripped: the tool hands the client the argument as given.
        return self._canonical(s)

    def _canonical(self, s: str) -> str:
        return canonical_url(s if self._SCHEME.match(s) else f"https://{s}")

    def matches(self, pattern: str, target: str) -> bool:
        """Whether ``pattern`` covers ``target``; both canonical (see
        ``canonicalize_pattern`` and ``canonicalize_target``)."""
        if pattern == "*":
            return True
        pp = urlsplit(pattern if "://" in pattern else f"https://{pattern}")
        tt = urlsplit(target)
        if pp.scheme.lower() != tt.scheme.lower():
            return False
        if (pp.hostname or "").lower() != (tt.hostname or "").lower():
            return False
        # A URL without a port means its scheme's default one (normalising
        # drops it), never any port: ``https://host/**`` is not
        # ``https://host:8443/``, another service.
        if self._port(pp) != self._port(tt):
            return False
        path_re = _glob_to_regex(pp.path or "/")
        path = tt.path or "/"
        # ``/docs/**`` covers ``/docs/`` too (and ``/**`` the site root ``/``):
        # the trailing slash is the folder itself, not another segment.
        if not path_re.match(path) and not (
            pp.path.endswith("/**") and path.endswith("/") and path_re.match(path.rstrip("/"))
        ):
            return False
        if pp.query and not fnmatch.fnmatchcase(tt.query, pp.query):
            return False
        return True

    def matches_deny(self, pattern: str, target: str) -> bool:
        """A URL deny matches as an allow does: hosts are already lowercased,
        and a path is case-sensitive by definition."""
        return self.matches(pattern, target)

    def _port(self, parts: SplitResult) -> int | None:
        return parts.port or self._DEFAULT_PORTS.get(parts.scheme.lower())


class ShellMatcher:
    """Matcher for ``shell.*`` capabilities."""

    def canonicalize_pattern(self, s: str, ctx: PermissionContext) -> str:
        if s == "*":
            return "*"
        return ctx.substitute(s).strip()

    def canonicalize_target(self, s: str, ctx: PermissionContext) -> str:
        return _literal_string_target(s)

    def matches(self, pattern: str, target: str) -> bool:
        if pattern == "*":
            return True
        return fnmatch.fnmatchcase(target, pattern)

    def matches_deny(self, pattern: str, target: str) -> bool:
        return self.matches(pattern, target)


_MATCHERS: dict[str, Matcher] = {
    "filesystem": PathMatcher(),
    "http": URLMatcher(),
    "shell": ShellMatcher(),
}
_FALLBACK: Matcher = StringGlobMatcher()


def get_matcher(capability: str) -> Matcher:
    """Return the matcher for a capability name, or the string-glob fallback."""
    ns = capability.split(".", 1)[0]
    return _MATCHERS.get(ns, _FALLBACK)


def _cap_matches(rule_cap: str, target_cap: str) -> bool:
    """Capability-name glob: exact / ``ns.*`` / ``*``."""
    if rule_cap == "*" or rule_cap == target_cap:
        return True
    if rule_cap.endswith(".*"):
        return target_cap.startswith(rule_cap[:-1])
    return False
