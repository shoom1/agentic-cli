# src/agentic_cli/workflow/permissions/engine.py
"""Permission engine.

See docs/superpowers/specs/2026-04-18-permissions-system-design.md §4 for
the full decision flow. This file implements:

1. Rule loading from builtin + user + project JSON + in-memory session.
2. ``permissions_enabled=False`` short-circuit.
3. Full check() flow (Tasks 15–17).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agentic_cli.logging import Loggers
from agentic_cli.settings_persistence import (
    get_project_config_path,
    get_user_config_path,
)
from agentic_cli.workflow.permissions.capabilities import Capability, ResolvedCapability
from agentic_cli.workflow.permissions.matchers import get_matcher
from agentic_cli.workflow.permissions.rules import (
    AskScope,
    CheckResult,
    Effect,
    Rule,
    RuleSource,
)
from agentic_cli.workflow.permissions.store import (
    BUILTIN_RULES,
    PermissionContext,
    load_rules,
)

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

logger = Loggers.workflow()


class PermissionEngine:
    """Evaluate tool invocations against rules from four sources.

    Concurrency: one ``asyncio.Lock`` around the ask prompt only (rule
    matching is pure). See spec §4.4.
    """

    def __init__(
        self,
        settings: "BaseSettings",
        workflow: "BaseWorkflowManager",
        ctx: PermissionContext,
    ) -> None:
        self._settings = settings
        self._workflow = workflow
        self._ctx = ctx
        self._session_rules: list[Rule] = []
        self._ask_lock = asyncio.Lock()
        self._base_rules: list[Rule] = self._load_all_rules()

    # ------------------------------------------------------------------
    # Rule loading
    # ------------------------------------------------------------------

    def _load_all_rules(self) -> list[Rule]:
        rules: list[Rule] = []
        for r in BUILTIN_RULES:
            # Canonicalise each builtin rule's target (which may use ${workdir}/${home}).
            rules.append(
                Rule(
                    capability=r.capability,
                    target=get_matcher(r.capability).canonicalize(r.target, self._ctx),
                    effect=r.effect,
                    source=r.source,
                )
            )
        app = self._settings.app_name
        rules += load_rules(get_user_config_path(app), RuleSource.USER, self._ctx)
        rules += load_rules(get_project_config_path(app), RuleSource.PROJECT, self._ctx)
        return rules

    @property
    def rules(self) -> list[Rule]:
        """All currently-active rules, in source order (builtin→user→project→session)."""
        return list(self._base_rules) + list(self._session_rules)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check(
        self,
        tool_name: str,
        capabilities: list[Capability],
        args: dict,
    ) -> CheckResult:
        """Evaluate whether ``tool_name`` may run with ``args``."""
        if not self._settings.permissions_enabled:
            return CheckResult(True, "permissions disabled")
        # Full flow lands in Tasks 15–17.
        raise NotImplementedError
