"""Capability declarations for tools.

A ``Capability`` describes one side effect a tool may perform, as a
``(namespace.action, target_source)`` pair. ``ResolvedCapability`` is the
concrete form emitted at call time, with the target extracted from the
tool's arguments and canonicalized by the namespace's matcher.

``EXEMPT`` is a sentinel used in ``@register_tool(capabilities=EXEMPT)``
to mark tools that explicitly require no permission check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Capability:
    """A capability a tool needs. Resolved against args at call time."""

    name: str                         # e.g. "filesystem.read"
    target_arg: str | None = None     # arg name holding the target; None → target "*"


@dataclass(frozen=True)
class ResolvedCapability:
    """Capability + concrete, canonicalized target."""

    name: str
    target: str


@dataclass(frozen=True)
class _CapabilityExempt:
    """Sentinel type; see ``EXEMPT`` for the singleton value."""

    def __bool__(self) -> bool:  # truthy so `not caps` ≠ "missing"
        return True


EXEMPT: Final[_CapabilityExempt] = _CapabilityExempt()

CapabilitiesSpec = list[Capability] | _CapabilityExempt
