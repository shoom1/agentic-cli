# src/agentic_cli/workflow/permissions/__init__.py
"""Centralised permission system.

See docs/superpowers/specs/2026-04-18-permissions-system-design.md for the
full design. Public API re-exported here; framework-bindings live under
``workflow/adk/permission_plugin.py`` and ``workflow/langgraph/permission_wrap.py``.
"""

from agentic_cli.workflow.permissions.capabilities import (
    Capability,
    CapabilitiesSpec,
    EXEMPT,
    ResolvedCapability,
)

try:
    from agentic_cli.workflow.permissions.engine import PermissionEngine
except ImportError:  # not yet implemented
    PermissionEngine = None  # type: ignore[assignment,misc]

try:
    from agentic_cli.workflow.permissions.rules import (
        AskScope,
        CheckResult,
        Effect,
        Rule,
        RuleSource,
    )
except ImportError:  # not yet implemented
    AskScope = CheckResult = Effect = Rule = RuleSource = None  # type: ignore[assignment,misc]

try:
    from agentic_cli.workflow.permissions.store import PermissionContext
except ImportError:  # not yet implemented
    PermissionContext = None  # type: ignore[assignment,misc]

__all__ = [
    "AskScope",
    "Capability",
    "CapabilitiesSpec",
    "CheckResult",
    "EXEMPT",
    "Effect",
    "PermissionContext",
    "PermissionEngine",
    "ResolvedCapability",
    "Rule",
    "RuleSource",
]
