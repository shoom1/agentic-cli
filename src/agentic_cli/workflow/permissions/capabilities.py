"""Capability declarations for tools.

A ``Capability`` describes one side effect a tool may perform, as a
``(namespace.action, target_source)`` pair. ``ResolvedCapability`` is the
concrete form emitted at call time, with the target extracted from the
tool's arguments (or the declared fixed target) and canonicalized by the
namespace's matcher.

Filesystem, network and shell capabilities act on a *resource*, so they must
name it: ``target_arg`` (the argument holding it) or ``target`` (a fixed value,
for a tool that always reaches the same endpoint). Without one the target is
the wildcard ``"*"``, and approving the tool would approve the capability for
every resource and every tool that declares it.

``EXEMPT`` is a sentinel used in ``@register_tool(capabilities=EXEMPT)``
to mark tools that explicitly require no permission check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


RESOURCE_NAMESPACES: Final[frozenset[str]] = frozenset({"filesystem", "http", "shell"})
"""Namespaces whose capabilities act on a resource named by a target."""


def is_resource_capability(name: str) -> bool:
    """True for a capability that acts on a resource (``filesystem.*``, ...)."""
    return name.split(".", 1)[0] in RESOURCE_NAMESPACES


@dataclass(frozen=True)
class Capability:
    """A capability a tool needs. Resolved against args at call time."""

    name: str                         # e.g. "filesystem.read"
    target_arg: str | None = None     # arg name holding the target
    optional: bool = False            # when the target arg is absent/empty, skip
    #                                   this capability (the side effect isn't
    #                                   performed) instead of resolving it to a
    #                                   spurious target. Only for genuinely
    #                                   optional args (e.g. an output path).
    target: str | None = None         # fixed target, for a tool that always
    #                                   reaches the same resource (e.g. one API
    #                                   endpoint). Neither set → target "*".

    def __post_init__(self) -> None:
        if self.target_arg is not None and self.target is not None:
            raise ValueError(
                f"Capability {self.name!r}: pass target_arg or target, not both"
            )


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
