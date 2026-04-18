# src/agentic_cli/workflow/permissions/matchers.py
"""Matchers for capability targets.

Each capability namespace (``filesystem.*``, ``http.*``, ``shell.*``) has a
Matcher that knows how to canonicalize targets + patterns and compare them.
Unknown namespaces fall back to ``StringGlobMatcher``.
"""

from __future__ import annotations

import fnmatch
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
