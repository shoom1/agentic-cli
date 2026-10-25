"""Base CLI Application for agentic applications.

This module provides the base CLI application that:
1. Uses ThinkingPromptSession for all UI (thinking boxes, messages, etc.)
2. Connects to domain-specific workflow managers
3. Tracks message history for persistence
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NoReturn

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory

from thinking_prompt import AppInfo, ThinkingPromptSession, ThinkingPromptStyles

from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.cli.message_processor import MessageProcessor, _says_so_when_stopped
from agentic_cli.cli.session_args import SessionStart, add_session_arguments, parse_session_start
from agentic_cli.cli.session_resume import (
    PICK_LIMIT,
    picker,
    replay_lines,
    resumed_line,
    show_replay,
)
from agentic_cli.cli.usage_tracker import UsageTracker
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import MODEL_ROLE_KEY_PREFIX, BaseSettings
from agentic_cli.logging import Loggers, configure_logging
from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS, USER_PREFERENCE_KEYS
from agentic_cli.workflow.settings import OrchestratorType

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable, Sequence

    from thinking_prompt import Field

    from agentic_cli.models import CatalogView, Endpoint, ModelRef, ResolvedModel, RoleMap
    from agentic_cli.settings_persistence import SettingsSaveResult
    from agentic_cli.workflow import GoogleADKWorkflowManager, EventType, WorkflowEvent
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.context import CompactResult
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.cli()

# Synthetic dialog keys → the settings field their setter actually writes.
# "model" is not a field; update_setting() routes it via set_model() to
# default_model. Every "model_role:<role>" key (a role's model picker) writes
# model_roles, through set_model_role() (see _ui_key_target_field). A
# synthetic key without an entry here is treated as writing a field of the
# same name.
_UI_KEY_TARGET_FIELDS = {"model": "default_model"}


def _ui_key_target_field(key: str) -> str:
    """The settings field a ``/settings`` dialog key writes."""
    if key.startswith(MODEL_ROLE_KEY_PREFIX):
        return "model_roles"
    return _UI_KEY_TARGET_FIELDS.get(key, key)


def _is_model_picker(key: str) -> bool:
    return key == "model" or key.startswith(MODEL_ROLE_KEY_PREFIX)


def _setting_label(key: str, settings: BaseSettings) -> str:
    """How messages name a dialog key, as the dialog labels it: by the field's
    title (``UI Theme``, ``Max Retry Attempts``), ``Fast Model`` for a role's
    model picker, and otherwise by the key title-cased (``Model``)."""
    if key.startswith(MODEL_ROLE_KEY_PREFIX):
        return _role_picker_label(key)
    field = type(settings).model_fields.get(key)
    if field is not None and field.title:
        return field.title
    return key.replace("_", " ").title()


def _role_picker_label(key: str) -> str:
    """A role's model picker, ``model_role:fast``, is ``Fast Model``."""
    return f"{key.removeprefix(MODEL_ROLE_KEY_PREFIX).title()} Model"


@dataclass(frozen=True)
class _ModelChoices:
    """What the ``/settings`` model pickers offer and show, from one catalog.

    Attributes:
        options: The runnable references every model picker offers.
        current: Each role's model now, by role (``default`` included); None
            when nothing can fill it.
        thinking: What the thinking effort means for the default model, as
            ``high → provider:model: <label>``; None when it cannot be said.
    """

    options: list[str]
    current: Mapping[str, str | None]
    thinking: str | None


# === ADK notices ===

# ADK 2 announces each experimental feature it uses with a UserWarning, in one
# of two forms: a feature flag ("[EXPERIMENTAL] feature
# FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled."), or a class or function
# marked experimental ("[EXPERIMENTAL] GeminiContextCacheManager: This feature
# is experimental and may change ...", from a Gemini agent's first turn while
# prompt caching is on).
_ADK_EXPERIMENTAL_NOTICE = re.compile(
    r"\[EXPERIMENTAL\] (?:feature (?P<feature>\S+) is enabled|(?P<name>\w+): )"
)

# ADK 2's runner warns, through stdlib logging, that an app whose agents can
# transfer to each other has no context_cache_config: a native ADK YAML app
# with sub-agents (agents built from configs delegate and never transfer). The
# ADK manager sets one while prompt caching is on and a model caches, so the
# advice comes only when the user turned caching off or no model caches: it
# has nothing left to tell them.
_ADK_RUNNER_LOGGER = "google_adk.google.adk.runners"
_ADK_UNCACHED_TRANSFER = "can transfer between agents but has no context_cache_config"


class _AdkUncachedTransferFilter(logging.Filter):
    """Turns ADK's context-cache advice into a debug log entry."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _ADK_UNCACHED_TRANSFER not in message:
            return True
        logger.debug("adk_uncached_agent_transfer", message=message)
        return False


# While prompt caching is on, ADK's Gemini context cache warns through stdlib
# logging when it cannot create a cache or delete an old one (an expired cache
# is already gone on Google's side). The turn goes on uncached, so the warning
# has nothing for the user mid-conversation.
_ADK_CACHE_LOGGER = "google_adk.google.adk.models.gemini_context_cache_manager"
_ADK_CACHE_FAILURES = (
    ("Failed to create cache", "adk_cache_create_failed"),
    ("Failed to cleanup cache", "adk_cache_cleanup_failed"),
)


class _AdkCacheFailureFilter(logging.Filter):
    """Turns ADK's context-cache failure warnings into debug log entries."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for prefix, event in _ADK_CACHE_FAILURES:
            if message.startswith(prefix):
                logger.debug(event, message=message)
                return False
        return True


# ADK logs a failure's traceback through stdlib logging where the user is
# told otherwise. A summary that failed: ADK goes on with the whole history,
# and the summarizer itself warns the user once a turn (it never lets its own
# failure reach ADK, so this is a backstop). A model call that failed: the
# user reads its error as an ERROR event or as the turn's error. ADK logs one
# twice for the main agent, from the node and from the runner ("Root node %s
# failed.", matched exactly: "Root node %s was cancelled …" passes), and a
# failed overflow retry once more, from the plugin manager. ADK also warns on
# every request of a session holding a call without a result, which it drops:
# the next turn answers such a call (workflow/adk/interrupted_calls_plugin.py),
# so the warning has nothing for the user either. Each entry: the logger, what
# a record's message starts with or (``msg=``) its exact format, and the debug
# event it becomes.
_ADK_FAILURES = (
    ("google_adk.google.adk.apps.compaction", {"prefix": "Event compaction failed"},
     "adk_compaction_failed"),
    ("google_adk.google.adk.workflow._node_runner", {"prefix": "Node execution failed"},
     "adk_node_failed"),
    ("google_adk.google.adk.runners", {"msg": "Root node %s failed."}, "adk_root_node_failed"),
    ("google_adk.google.adk.plugins.plugin_manager",
     {"prefix": "Error in plugin 'context_overflow'"}, "adk_overflow_plugin_failed"),
    ("google_adk.google.adk.flows.llm_flows.tools._rearranger",
     {"prefix": "Dropping function calls with no matching function response"},
     "adk_dropped_open_calls"),
)


class _AdkFailureFilter(logging.Filter):
    """Turns ADK's failure tracebacks on one logger into a debug log entry.

    A record is matched by the start of its message (``prefix``) or by its
    format string itself (``msg``).
    """

    def __init__(self, event: str, *, prefix: str | None = None, msg: str | None = None) -> None:
        super().__init__()
        self._event = event
        self._prefix = prefix
        self._msg = msg

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._msg is not None:
            matched = record.msg == self._msg
        else:
            matched = self._prefix is not None and message.startswith(self._prefix)
        if not matched:
            return True
        error = record.exc_info[1] if record.exc_info else None
        logger.debug(
            self._event,
            message=message,
            error_type=type(error).__name__ if error is not None else None,
            error=str(error) if error is not None else None,
        )
        return False


def _install_adk_notice_hook() -> None:
    """Log ADK's notices instead of printing them.

    Python prints warnings, and the stdlib logging ``configure_logging`` sets
    up, to stderr, which in the CLI lands over the terminal UI. The hook logs
    ADK's experimental-feature notices at debug and passes every other warning
    to the hook it replaced; a filter on ADK's runner logger does the same for
    its context-cache advice, one on its context-cache logger for a cache it
    could not create or delete, one each on its compaction, node, runner
    and plugin-manager loggers for the traceback of a summary, a model call or
    an overflow retry that failed, and one on its rearranger for a call
    without a result it drops. Only
    the CLI installs them, so library users and the test suite keep Python's
    own handling. Installing twice changes nothing.
    """
    for name, kind in (
        (_ADK_RUNNER_LOGGER, _AdkUncachedTransferFilter),
        (_ADK_CACHE_LOGGER, _AdkCacheFailureFilter),
    ):
        adk_logger = logging.getLogger(name)
        if not any(isinstance(f, kind) for f in adk_logger.filters):
            adk_logger.addFilter(kind())
    for name, match, event in _ADK_FAILURES:
        adk_logger = logging.getLogger(name)
        if not any(isinstance(f, _AdkFailureFilter) for f in adk_logger.filters):
            adk_logger.addFilter(_AdkFailureFilter(event, **match))

    previous = warnings.showwarning
    if getattr(previous, "_logs_adk_notices", False):
        return

    def showwarning(message, category, filename, lineno, file=None, line=None):
        notice = _ADK_EXPERIMENTAL_NOTICE.match(str(message))
        if notice and issubclass(category, UserWarning):
            feature = (notice["feature"] or notice["name"]).removeprefix("FeatureName.")
            logger.debug("adk_experimental_feature", feature=feature)
            return
        previous(message, category, filename, lineno, file, line)

    showwarning._logs_adk_notices = True
    warnings.showwarning = showwarning


# === Deprecated orchestrators ===

# What the CLI tells a user of the LangGraph backend, deprecated in 0.7.0: once
# at startup, and again on a /settings switch to it. The manager's own
# DeprecationWarning is for library users; Python's default filters keep it off
# the terminal, so this is what a CLI user reads.
_LANGGRAPH_DEPRECATION = (
    "The LangGraph backend is deprecated and will be removed in 0.8.0. Set "
    "orchestrator to deepagents (pip install 'agentic-cli[deepagents]'); "
    "sessions saved by LangGraph do not carry over."
)


#: Why nothing is stored on the memory store, after a startup line.
_MEMORY_NOTE = " (session_store is memory: conversations end with the app)"


def _command_name() -> str:
    """The command this process was run as: ``python -m <package>``,
    ``python <script>.py``, or a console script's name."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return f"python -m {spec.name.removesuffix('.__main__')}"
    script = sys.argv[0] if sys.argv and sys.argv[0] else "app"
    if script.endswith(".py"):  # run as `python my_app.py`: name it so
        return f"python {script}"
    return os.path.basename(script)


def _runs_langgraph(settings: BaseSettings) -> bool:
    """Whether ``settings`` choose the LangGraph backend.

    Read as the factory reads it: an ``OrchestratorType``, or its plain value
    on a settings class that declares ``orchestrator: str``.
    """
    return getattr(settings, "orchestrator", None) == OrchestratorType.LANGGRAPH


def _warn_if_langgraph(settings: BaseSettings, session: ThinkingPromptSession) -> None:
    """Warn, through the session, when ``settings`` choose the LangGraph backend."""
    if _runs_langgraph(settings):
        session.add_warning(_LANGGRAPH_DEPRECATION)


# === Slash Command Completer ===


class SlashCommandCompleter(Completer):
    """Completer that only triggers for slash commands."""

    def __init__(self, commands: list[str]) -> None:
        """Initialize with a list of command names (without leading slash).

        Args:
            commands: List of command names, e.g., ["help", "settings", "quit"]
        """
        self.commands = sorted(commands)

    def get_completions(self, document: Document, complete_event):
        """Yield completions only when text starts with /."""
        text = document.text_before_cursor

        # Only complete if input starts with /
        if not text.startswith("/"):
            return

        # Get the partial command (without the leading /)
        partial = text[1:].lower()

        # Find matching commands
        for cmd in self.commands:
            if cmd.lower().startswith(partial):
                # Calculate how much to complete (replace everything after /)
                yield Completion(
                    text=f"/{cmd}",
                    start_position=-len(text),
                    display=f"/{cmd}",
                )


# === Theme ===

# agentic-cli's colors, over thinking-prompt's dark theme (get_theme("dark")).
# They are tuned for a dark background: every other theme is thinking-prompt's
# own.
_DARK_COLORS: dict[str, str] = {
    "thinking_box": "fg:#6c757d",
    "thinking_box_border": "fg:#495057",
    "thinking_box_hint": "fg:#6c757d italic",
    "user_message": "fg:#17a2b8",
    "assistant_message": "fg:#e9ecef",
    "system_message": "fg:#6c757d italic",
    "error_message": "fg:#dc3545 bold",
    "warning_message": "fg:#ffc107",
    "success_message": "fg:#28a745",
}

# What an app still overriding get_styles(), removed in 0.7.0, is told: its
# colors would otherwise be dropped without a word.
_GET_STYLES_REMOVED = (
    'get_styles() was replaced by get_theme(name) in agentic-cli 0.7.0: return '
    'your colors for "dark" from get_theme() (see the CHANGELOG)'
)


# === Base CLI Application ===


#: What ``/new`` shows while it saves the facts of the conversation it leaves.
SAVING_FACTS = "Saving facts from the conversation…"


class BaseCLIApp:
    """
    Base CLI Application for agentic applications.

    Domain-specific applications extend this class and provide via constructor:
    - app_info: The startup banner (name, version, welcome message)
    - agent_configs: List of agent configurations for the workflow
    - settings: Application settings instance

    Optional overrides:
    - register_commands(): Register domain-specific commands
    - get_theme(name): Customize a theme's colors
    - get_ui_setting_keys(): Customize settings dialog fields
    - apply_settings(): Handle additional custom settings
    """

    def __init__(
        self,
        app_info: AppInfo,
        agent_configs: list["AgentConfig"],
        settings: BaseSettings,
        session_id: str | None = None,
    ) -> None:
        """Initialize the CLI application.

        Args:
            app_info: The startup banner (name, version, welcome message)
            agent_configs: List of agent configurations for the workflow
            settings: Application settings instance
            session_id: The conversation to start in, new or stored (for
                library use and tests); without it a fresh id is generated.
                ``main()`` takes ``--continue`` and ``--resume [ID]``
                instead (``session_start``).

        Raises:
            TypeError: The app still defines ``get_styles()``, which
                ``get_theme(name)`` replaced.
        """
        if any("get_styles" in vars(cls) for cls in type(self).__mro__ if cls is not object):
            raise TypeError(_GET_STYLES_REMOVED)

        # Sessions are durable by default: a fresh id is generated when none is
        # given (resumable later with /resume or --resume); an explicit id
        # requests a resume.
        import uuid

        self._resume_requested = session_id is not None
        self._session_id = session_id or uuid.uuid4().hex[:12]
        # How run() starts (``session_start``), and what it opens once the prompt is up.
        self._session_start = SessionStart()
        self._pending_resume: str | None = None
        self._pending_pick = False
        # The command main() was run as, for the lines naming --resume.
        self._command: str | None = None
        #: The process's exit status after run(): 1 when the start was refused.
        self.exit_status = 0
        # === Configuration ===
        self._app_info = app_info
        self._settings = settings
        configure_logging(self._settings)
        _install_adk_notice_hook()

        logger.info("app_starting", app_name=self._settings.app_name)

        # === Command Registry ===
        self.command_registry = CommandRegistry()
        self._register_builtin_commands()
        self.register_commands()  # Domain-specific commands

        # === Components ===
        self._usage_tracker = UsageTracker()
        self._workflow_controller = WorkflowController(
            agent_configs=agent_configs,
            settings=settings,
        )
        self._workflow_controller.usage_tracker = self._usage_tracker
        self._message_processor = MessageProcessor()

        # Serializes turns: a user turn and a background-job resume turn must
        # never overlap (shared session + thinking boxes).
        self._turn_lock = asyncio.Lock()
        # Whether a turn ran in the current conversation since it was adopted:
        # its facts are saved on leaving it, or at exit, only then.
        self._turned = False

        # === UI: ThinkingPromptSession ===
        completer = SlashCommandCompleter(self.command_registry.get_completions())

        self.session = ThinkingPromptSession(
            message=">>> ",
            app_info=self._app_info,
            theme=self.get_theme(settings.ui_theme),
            history=InMemoryHistory(),
            completer=completer,
            complete_while_typing=True,
            completions_menu_height=8,
            enable_status_bar=True,
            status_text="Ctrl+C: cancel | Ctrl+D: exit | /help: commands",
            echo_input=False,  # We handle echoing manually based on command type
            # A finished box's text is not printed: the status lines a box
            # shows are not part of the conversation.
            echo_thinking=False,
            # Where a dialog that names no placement is drawn.
            dialog_placement=settings.dialog_placement,
        )

        # === State ===
        self.should_exit = False

        logger.debug("app_initialized_fast")

    def get_theme(self, name: str) -> str | ThinkingPromptStyles:
        """The session's theme for the theme ``name``.

        ``"dark"`` is thinking-prompt's dark theme in agentic-cli's colors.
        Any other name (``"light"``, ``"mono"``, ``"terminal"``, ``"auto"``)
        is returned as it is, for thinking-prompt's own theme: colors tuned
        for a dark background would be unreadable on a light one. ``"auto"``
        lets thinking-prompt choose from the terminal (mono under
        ``NO_COLOR``, light or dark from ``COLORFGBG``, else the terminal's
        own colors), so agentic-cli's colors do not apply there. The session
        is built in ``get_theme(settings.ui_theme)``.

        Override to recolor a theme, e.g. for ``"dark"`` return
        ``dataclasses.replace(ThinkingPromptStyles.dark(), thinking_box=...)``
        and ``super().get_theme(name)`` for the others.
        """
        if name == "dark":
            return dataclasses.replace(ThinkingPromptStyles.dark(), **_DARK_COLORS)
        return name

    def register_commands(self) -> None:
        """Register domain-specific commands.

        Override to register additional commands.
        """
        pass

    def get_ui_setting_keys(self) -> list[str]:
        """Get field names to display in the settings dialog.

        Override to customize which settings appear in the UI.
        Default: model, model_role:fast, model_role:strong, thinking_effort.
        ``model`` and each ``model_role:<role>`` are model pickers: they
        offer the model catalog's runnable models as ``provider:model``
        references, and write ``default_model`` and ``model_roles[<role>]``.
        An app adds a picker for a role of its own as ``model_role:<role>``.

        Only project-scoped settings and the user's preferences may appear.
        A key whose target field is in neither PROJECT_SETTABLE_KEYS nor
        USER_PREFERENCE_KEYS is excluded by _build_ui_items (with a
        warning), because a /settings edit to it would persist in the user
        ~/.{app}/settings.json and apply across all projects of the app. The
        preferences (``ui_theme``, ``dialog_placement``) are meant to apply
        everywhere, so they are saved there; the default list leaves them
        out, and an app shows them by listing them:
        ``[*super().get_ui_setting_keys(), "ui_theme", "dialog_placement"]``.

        Returns:
            List of field names that should appear in the settings UI
        """
        return ["model", "model_role:fast", "model_role:strong", "thinking_effort"]

    def _build_ui_items(self) -> list[Field]:
        """Build the settings dialog's fields using introspection.

        Uses get_ui_setting_keys() to determine which fields to show,
        then converts each field to an appropriate form field using
        the field's type annotation and metadata. The model pickers come
        from the model catalog (``_model_choices``), and the thinking
        effort's description says what it means for the default model.
        thinking_prompt's fields are frozen, so each is built complete.

        Returns:
            List of thinking_prompt form fields sorted by ui_order
        """
        from agentic_cli.cli.settings_introspection import field_to_ui_item, get_ui_order

        keys = self.get_ui_setting_keys()
        roles = [
            key.removeprefix(MODEL_ROLE_KEY_PREFIX)
            for key in keys
            if key.startswith(MODEL_ROLE_KEY_PREFIX)
        ]
        choices: _ModelChoices | None = None  # one catalog for every picker
        items: list[tuple[int, Field]] = []
        shown = PROJECT_SETTABLE_KEYS | USER_PREFERENCE_KEYS

        for key in keys:
            # Project-scope guard: the dialog may only expose settings the
            # project file can persist (PROJECT_SETTABLE_KEYS), and the user's
            # preferences (USER_PREFERENCE_KEYS), which are meant to apply in
            # every project. Any other user-scoped key would save to the user
            # ~/.{app}/settings.json and leak the change across all projects
            # of the app. model_endpoints is one: it holds the addresses a key
            # is sent to.
            if _ui_key_target_field(key) not in shown:
                logger.warning("user_scoped_setting_excluded_from_ui", key=key)
                continue

            if _is_model_picker(key):
                choices = choices or self._model_choices(roles)
                if not choices.options:
                    continue  # no provider to pick from
                # The default model first, then the roles in the app's order.
                items.append((10 if key == "model" else 11, _model_picker(key, choices)))
                continue

            # Handle regular fields from settings
            # Access model_fields from class to avoid Pydantic deprecation warning
            settings_cls = type(self._settings)
            if key in settings_cls.model_fields:
                field = settings_cls.model_fields[key]
                current = getattr(self._settings, key, None)
                description = None  # the field's own
                if key == "thinking_effort":
                    choices = choices or self._model_choices(roles)
                    description = choices.thinking or None
                item = field_to_ui_item(key, field, current, description=description)
                order = get_ui_order(field)
                items.append((order, item))

        # Sort by order and return items only
        return [item for _, item in sorted(items, key=lambda x: x[0])]

    def _model_choices(self, roles: "Sequence[str]" = ()) -> _ModelChoices:
        """The model pickers' options, each role's model, and the thinking description.

        Once the workflow is ready, from its catalog (listings included) and
        its default model; before that, from the shipped table's models of
        each provider with a key, and the default model the settings name (or
        compute). A role shows the model it maps, else the one it resolves
        to (an app's own role runs on the default model). Models a role
        names, and pinned ones, are offered too.

        Args:
            roles: The roles other than ``default`` whose model to show.
        """
        from agentic_cli.models import BUILTIN_ROLES, ModelCatalog
        from agentic_cli.workflow import model_services

        try:
            role_map = model_services.role_map_from_settings(self._settings)
        except ValueError:
            return _ModelChoices(options=[], current={}, thinking=None)
        endpoints = model_services.endpoints_from_settings(self._settings)
        controller = self._workflow_controller
        workflow = controller.workflow if controller.is_ready else None
        if workflow is not None and workflow.catalog is not None:
            catalog, default = workflow.catalog, workflow.model
        else:
            catalog = ModelCatalog(endpoints=endpoints)  # not loaded: the shipped table
            default = _resolved_role(role_map, "default", catalog, endpoints)

        current: dict[str, str | None] = {"default": default}
        for role in roles:
            mapped = role_map.spec_for(role)
            if mapped is not None:
                current[role] = str(mapped)
            elif role in BUILTIN_ROLES:
                current[role] = _resolved_role(role_map, role, catalog, endpoints)
            else:
                current[role] = default
        referenced = [*(ref for ref in current.values() if ref), *role_map.referenced()]
        return _ModelChoices(
            options=catalog.picker_refs(referenced=referenced, pinned=self._settings.pinned_models),
            current=current,
            thinking=self._thinking_meaning(default, catalog, endpoints),
        )

    def _thinking_meaning(
        self, model: str | None, catalog: "CatalogView", endpoints: "Mapping[str, Endpoint]"
    ) -> str | None:
        """``high → anthropic:claude-sonnet-4-6: adaptive, effort high``: the
        thinking effort, resolved on ``model`` as the workflow resolves it."""
        from agentic_cli.models import ModelResolutionError, RoleMap, resolve_model

        if model is None:
            return None
        effort = self._settings.thinking_effort
        try:
            resolved = resolve_model(
                agent=None,
                spec=model,
                generation=None,
                thinking_effort=effort,
                roles=RoleMap(default_model=None, model_roles={}),  # a reference needs none
                catalog=catalog,
                endpoints=endpoints,
            )
        except ModelResolutionError:
            return None
        meaning = resolved.thinking.label if resolved.thinking else "ignored (no thinking knob)"
        return f"{effort} → {resolved.ref}: {meaning}"

    async def save_settings(
        self, keys: "Iterable[str] | None" = None
    ) -> "SettingsSaveResult":
        """Save current settings, split by trust.

        ``keys`` restricts the save to those fields (``/settings`` passes the
        fields the user changed). Without it the whole live settings object is
        saved, including values that came from the environment.

        Allowlisted keys go to the project config
        (./.{app_name}/settings.json). User-scoped keys differing from their
        defaults go to the user config (~/.{app_name}/settings.json), where
        the loader trusts them — the project file may only carry allowlisted
        keys since P0-1. Secrets (API keys) are never saved.

        Returns:
            SettingsSaveResult with the written path(s)
        """
        from agentic_cli.settings_persistence import SettingsPersistence

        persistence = SettingsPersistence(self._settings.app_name)
        return persistence.save(self._settings, keys=keys)

    @property
    def workflow(self) -> "BaseWorkflowManager":
        """Get the workflow manager.

        Raises:
            RuntimeError: If workflow is not yet initialized
        """
        return self._workflow_controller.workflow

    @property
    def usage_tracker(self) -> UsageTracker:
        """Get the usage tracker."""
        return self._usage_tracker

    @property
    def settings(self) -> BaseSettings:
        """Get the application settings."""
        return self._settings

    @property
    def session_id(self) -> str | None:
        """Get the persistent session ID, if any."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        """Set the persistent session ID."""
        self._session_id = value

    @property
    def session_start(self) -> SessionStart:
        """How ``run()`` starts: a new conversation (the default), the newest
        stored one, one by id, or a picker (``cli/session_args.py``).

        ``main()`` sets it from the command line; set it before ``run()``.

        Raises:
            TypeError: On setting anything but a ``SessionStart``.
            ValueError: On setting anything but a new conversation for an
                app built with ``session_id=``, which already names the
                conversation it starts in.
        """
        return self._session_start

    @session_start.setter
    def session_start(self, value: SessionStart) -> None:
        if not isinstance(value, SessionStart):
            raise TypeError(f"session_start takes a SessionStart, not {type(value).__name__}")
        if self._resume_requested and value.mode != "new":
            raise ValueError(
                f"This app was built with session_id={self._session_id!r}; "
                f"it cannot also start with {value.mode!r}."
            )
        self._session_start = value

    @classmethod
    def main(cls, argv: "Sequence[str] | None" = None) -> NoReturn:
        """Run the app from the command line, then exit with its ``exit_status``.

        Point an app's console script at it (``myapp =
        "myapp.cli.app:MyApp.main"``). It parses ``argv`` (the process's
        arguments by default) for ``-c``/``--continue`` and ``-r``/``--resume
        [ID]`` (``cli/session_args.py``) and whatever ``add_arguments`` adds,
        builds the app with ``from_args``, sets its ``session_start``, and runs it.
        The status is 1 when ``--resume <id>`` cannot be opened (not stored,
        or the store failed) or the backend keeps no sessions, 130 after a
        Ctrl+C before the prompt, else 0.
        """
        import argparse

        command = _command_name()
        parser = argparse.ArgumentParser(prog=command)
        add_session_arguments(parser)
        cls.add_arguments(parser)
        args = parser.parse_args(None if argv is None else list(argv))
        app = cls.from_args(args)
        app.session_start = parse_session_start(args)
        app._command = command
        try:
            asyncio.run(app.run())
        except KeyboardInterrupt:  # Ctrl+C before the prompt: the terminal is not raw yet
            raise SystemExit(130) from None
        raise SystemExit(app.exit_status)

    @classmethod
    def add_arguments(cls, parser: "argparse.ArgumentParser") -> None:
        """Add the app's own command-line flags to ``parser`` (none by default).

        ``main()`` calls it after adding the session flags; ``from_args``
        reads what was given.
        """

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "BaseCLIApp":
        """Build the app ``main()`` runs: ``cls()`` by default.

        An app whose constructor takes arguments, or that reads flags of its
        own (``add_arguments``), overrides it. ``main()`` sets ``session_start`` on
        the app returned.
        """
        return cls()

    def stop(self) -> None:
        """Stop the application."""
        self.should_exit = True
        self.session.exit()

    async def apply_settings(self, changes: dict[str, Any]) -> bool:
        """Apply changed settings and reinitialize workflow if needed.

        All or nothing: if any value is invalid, or the workflow cannot be
        reinitialized with the new values, every field is restored and the
        workflow is brought back on the model it was running, so nothing
        unusable stays in memory to be saved by ``/settings``. A restart
        interrupted by Ctrl+C is undone the same way, without a word, and the
        cancel is re-raised.

        Settings are applied in two ways:
        1. Special settings (model, a role's model, thinking_effort) use
           dedicated setters and trigger workflow reinitialization
        2. All other settings are applied directly to the settings object

        A reinitialization keeps the sessions, as a model switch does. A
        switch to the deprecated LangGraph orchestrator ends with its
        deprecation warning. A change of ``ui_theme`` switches the session to
        ``get_theme(ui_theme)`` last, repainting the transcript; one rolled
        back leaves the theme as it was.

        Args:
            changes: Dictionary of changed settings (key -> new_value)

        Returns:
            True if the changes were applied, False if they were rolled back.
        """
        if not changes:
            self.session.add_message("system", "No changes made.")
            return True

        needs_reinit = False
        new_model = changes.get("model")
        snapshot = _snapshot_settings(self._settings)
        was_langgraph = _runs_langgraph(self._settings)
        theme_before = self._settings.ui_theme

        # `orchestrator` triggers a reinit too: the controller swaps the backend
        # when it changes (otherwise the change only took effect on restart).
        # So does a role's model ("model_role:<role>"): agents resolve their
        # roles when the workflow initializes. And so do the settings of when
        # to summarize and where a tool result is cut: each agent's
        # thresholds and cap are fixed when it is built.
        reinit_settings = {
            "model", "thinking_effort", "orchestrator",
            "context_summarize_at", "context_max_tokens", "tool_result_max_chars",
        }

        for key, value in changes.items():
            try:
                self._settings.update_setting(key, value)
                if key in reinit_settings or key.startswith(MODEL_ROLE_KEY_PREFIX):
                    needs_reinit = True
            except ValueError as e:
                _restore_settings(self._settings, snapshot)
                label = _setting_label(key, self._settings)
                self.session.add_error(f"Failed to set {label}: {e}")
                return False

        # Reinitialize workflow if needed
        controller = self._workflow_controller
        switched: list[str] = []
        if needs_reinit and controller.is_ready:
            previous = controller.workflow
            previous_model = previous.model
            previous_models = dict(getattr(previous, "agent_models", {}))
            try:
                await controller.reinitialize(model=new_model)
            except asyncio.CancelledError:
                # Ctrl+C stopped /settings mid-restart. Nothing of the Save
                # stays, as on a failure, and nothing is said: the session's
                # interruption is not a cancel of the dialog.
                _restore_settings(self._settings, snapshot)
                if not controller.is_ready:
                    try:
                        await controller.reinitialize(model=previous_model)
                    except Exception as restore_error:
                        # Down, on the old model: the next request retries,
                        # and reports a failure then.
                        logger.warning("settings_restore_failed", error=str(restore_error))
                raise
            except Exception as e:
                _restore_settings(self._settings, snapshot)
                self.session.add_error(
                    f"Failed to reinitialize workflow: {e}. Settings were not changed."
                )
                # A failed orchestrator swap leaves the old manager running; a
                # failed in-place reinitialize leaves it down, holding the new
                # model. Bring it back on the model it had.
                if not controller.is_ready:
                    try:
                        await controller.reinitialize(model=previous_model)
                    except Exception as restore_error:
                        self.session.add_error(
                            f"Could not restart the workflow on {previous_model}: "
                            f"{restore_error}"
                        )
                return False
            # An orchestrator swap starts another manager: nothing carries over.
            if controller.workflow is previous:
                switched = _provider_switches(
                    previous_models, getattr(previous, "agent_models", {})
                )
            _show_new_context(
                controller, self.session, swapped=controller.workflow is not previous
            )

        # Report changes
        for key, value in changes.items():
            self.session.add_success(f"{_setting_label(key, self._settings)}: {value}")
        for notice in switched:
            self.session.add_message("system", notice)
        if not was_langgraph:  # a switch to it, not a change made while on it
            _warn_if_langgraph(self._settings, self.session)
        if self._settings.ui_theme != theme_before:
            # Repaint: text printed before would keep the old theme's colors.
            # The transcript is printed again, the lines above included.
            theme = self.get_theme(self._settings.ui_theme)
            self.session.set_theme(theme, repaint=True)
        return True

    def _register_builtin_commands(self) -> None:
        """Register built-in commands."""
        from agentic_cli.cli.builtin_commands import (
            HelpCommand,
            ClearCommand,
            ExitCommand,
            StatusCommand,
            SandboxCommand,
            JobsCommand,
            ResumeSessionCommand,
            PapersCommand,
            SessionsCommand,
            CompactCommand,
            NewCommand,
            ModelsCommand,
        )
        from agentic_cli.cli.settings_command import SettingsCommand

        self.command_registry.register(HelpCommand())
        self.command_registry.register(ClearCommand())
        self.command_registry.register(ExitCommand())
        self.command_registry.register(StatusCommand())
        self.command_registry.register(SandboxCommand())
        self.command_registry.register(JobsCommand())
        self.command_registry.register(ResumeSessionCommand())
        self.command_registry.register(SettingsCommand())
        self.command_registry.register(PapersCommand())
        self.command_registry.register(SessionsCommand())
        self.command_registry.register(CompactCommand())
        self.command_registry.register(NewCommand())
        self.command_registry.register(ModelsCommand())

    async def process_input(self, user_input: str) -> None:
        """Process user input.

        Args:
            user_input: The raw user input string
        """
        user_input = user_input.strip()

        if not user_input:
            return

        # Route to appropriate handler
        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            await self._handle_message(user_input)

    async def _handle_command(self, user_input: str) -> None:
        """Handle slash command execution.

        Args:
            user_input: The command string starting with /
        """
        parts = user_input[1:].split(maxsplit=1)
        command_name = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        command = self.command_registry.get(command_name)
        if command:
            # Echo command if not silent
            if not command.silent:
                self.session.add_message("user", user_input)
            logger.debug("executing_command", command=command.name, args=args)
            try:
                await command.execute(args, self)
                logger.debug("command_completed", command=command.name)
            except Exception as e:
                self.session.add_error(f"Error executing command: {e}")
        else:
            # Echo unknown command before error
            self.session.add_message("user", user_input)
            self.session.add_error(f"Unknown command: /{command_name}")
            self.session.add_message("system", "Type /help to see available commands")

    async def _handle_message(self, message: str) -> None:
        """Route message through agentic workflow.

        Delegates to MessageProcessor for event stream handling.

        Args:
            message: User message to process
        """
        # Echo user input for regular messages
        self.session.add_message("user", message)

        # Delegate to message processor (one turn at a time).
        async with self._turn_lock:
            self._turned = True  # the conversation now holds this message
            await self._message_processor.process(
                message=message,
                workflow_controller=self._workflow_controller,
                ui=self.session,
                settings=self._settings,
                usage_tracker=self._usage_tracker,
                session_id=self._session_id,
            )

        # At the turn boundary, auto-resume any finished background jobs that
        # opted in (gated by the job_auto_resume setting).
        if getattr(self._settings, "job_auto_resume", False):
            await self.resume_finished_jobs()

    async def resume_finished_jobs(self) -> int:
        """Resume the agent for each finished, resume-flagged background job.

        Each resume is a serialized turn (via the turn lock) so it never
        overlaps a user turn or another resume. Used at turn boundaries (auto,
        gated) and by ``/jobs resume`` (explicit, ungated).

        Delivery follows the job's resume lifecycle: the job is *claimed*
        (pending → resuming) before the turn runs, so a crash cannot silently
        re-deliver it, and only recorded delivered once the turn actually
        completed — a failed or cancelled resume is recorded as such instead of
        being dropped.

        Returns:
            The number of jobs a resume turn was run for (unchanged meaning:
            "how many were picked up"). Whether each one was delivered is
            recorded on the job and reported by ``/jobs``; a failed delivery is
            surfaced to the user by the turn itself.
        """
        if not self._workflow_controller.is_ready:
            return 0
        jm = getattr(self._workflow_controller.workflow, "job_manager", None)
        if jm is None:
            return 0

        attempted = 0
        for record in jm.awaiting_resume():
            # Claim first: durable, so an interrupted delivery is recoverable
            # and two coordinators can't both deliver the same result.
            if not jm.begin_resume(record.job_id):
                continue
            attempted += 1
            await self._deliver_resume(jm, record)
        return attempted

    async def _deliver_resume(self, jm, record) -> bool:
        """Run one claimed job's resume turn and close out its transition.

        Every exit path closes the claim, so no record can be left ``RESUMING``
        in this process: a normal outcome records delivered/failed, an
        exception records failed with the reason, and a cancellation records
        failed before re-raising (cancellation still propagates).

        Args:
            jm: The JobManager holding the claim.
            record: The claimed job record.

        Returns:
            True if the result was delivered to the agent.
        """
        try:
            async with self._turn_lock:
                if getattr(record, "session_id", None) == self._session_id:
                    self._turned = True  # the resume adds to this conversation
                result = await self._message_processor.process_resume(
                    record=record,
                    workflow_controller=self._workflow_controller,
                    ui=self.session,
                    settings=self._settings,
                    usage_tracker=self._usage_tracker,
                    current_session_id=self._session_id,
                )
        except asyncio.CancelledError:
            jm.complete_resume(
                record.job_id, delivered=False, error="resume cancelled"
            )
            logger.info("job_resume_cancelled", job_id=record.job_id)
            raise
        except Exception as exc:  # noqa: BLE001 - the claim must always close
            jm.complete_resume(record.job_id, delivered=False, error=str(exc))
            logger.warning(
                "job_resume_raised", job_id=record.job_id, error=str(exc)
            )
            self.session.add_error(
                f"Background job '{record.name}' could not be resumed: {exc}"
            )
            return False

        jm.complete_resume(
            record.job_id, delivered=result.delivered, error=result.error
        )
        if not result.delivered:
            logger.info(
                "job_resume_not_delivered",
                job_id=record.job_id,
                status=result.status.value,
                error=result.error,
            )
        return result.delivered

    async def compact(self, focus: str | None = None) -> "CompactResult | None":
        """Summarize the conversation now (``/compact [focus]``), as a turn runs.

        The public twin of ``start_new_session()``: under the turn lock, the
        workflow's ``compact`` runs with the turn's events box, status line,
        token counts and Ctrl+C (``MessageProcessor.process_compact``), and
        what it did, or why nothing was, is said as ``/compact`` says it.

        Args:
            focus: What the summary should keep above all, if anything.

        Returns:
            What the summary did; None when nothing was summarized.
        """
        async with self._turn_lock:
            return await self._message_processor.process_compact(
                focus,
                workflow_controller=self._workflow_controller,
                ui=self.session,
                settings=self._settings,
                usage_tracker=self._usage_tracker,
                session_id=self._session_id,
            )

    async def start_new_session(self) -> str:
        """Start a new conversation in this process (``/new``), and return its id.

        Waits for a running turn or job resume (the turn lock), then:

        - with ``auto_extract_session_facts`` on, extracts the facts of the
          conversation being left when a turn ran in it since it was opened,
          as at exit, under a status line ("Saving facts from the
          conversation…"; a failure never stops it);
        - adopts a new session id once the workflow has taken it
          (``workflow.load_session``), and every later turn targets it;
        - ends the task box and forgets its tasks, zeroes the usage counters,
          and draws the status bar again: ``ctx`` waits for the new
          conversation's first call.

        It says ``"New conversation <new>. Previous: <old>, listed in
        /sessions."``, or ``"New conversation <new>."`` when the conversation
        left is in no store (no turn ran in it, or the workflow is not
        ready). Ctrl+C stops it like any request: "Operation cancelled...",
        once, and nothing has changed. What belongs to the process stays: the
        model and the settings, "Allow for this session" grants, sandbox
        kernels, and background jobs, each of which still reports to the
        conversation that started it (a job's resume runs in its record's
        session, and leaves the current conversation's ``ctx`` and tasks
        alone).

        Returns:
            The new session id (12 hex characters).
        """
        import uuid

        with _says_so_when_stopped(self.session):
            async with self._turn_lock:
                old = self._session_id
                controller = self._workflow_controller
                ready = controller.is_ready
                if self._turned:  # facts once per conversation used
                    await self._saving_facts(old)
                stored = ready and await self._is_stored(controller.workflow, old)
                new = uuid.uuid4().hex[:12]
                if ready:
                    await controller.workflow.load_session(new)
                self._session_id = new
                self._turned = False
                self._message_processor.reset()
                self._usage_tracker.reset()
                # The trigger as a /settings rebuild shows it; no prompt yet.
                _show_new_context(controller, self.session, swapped=False)
                logger.info("new_session_started", session_id=new, previous=old)
                line = f"New conversation {new}."
                if stored:
                    line += f" Previous: {old}, listed in /sessions."
                self.session.add_message("system", line)
                return new

    async def resume_session(self, session_id: str) -> bool:
        """Switch to the stored conversation ``session_id`` (``/resume``), and show where it left off.

        Waits for a running turn or job resume (the turn lock), then reads the
        conversation first, its row (``workflow.session_info``) and its
        messages, and makes what it will show: one that is not stored, or
        cannot be read, changes nothing. Then, as ``start_new_session`` does,
        it saves the facts of the conversation being left (when
        ``auto_extract_session_facts`` is on and a turn ran in it since it was
        opened), adopts the id once the workflow has taken it
        (``load_session``; one that fails changes nothing either), ends the
        task box and zeroes the usage counters. It says "Resumed:
        <title> (<id>, 12 turns, last active 2h ago)." and replays the last
        messages (``cli/session_resume.py``). Ctrl+C stops it like any
        request, "Operation cancelled...", once, and nothing has changed.

        Returns:
            True once ``session_id`` is the current conversation. False, with
            nothing changed, when it is already ("Already in this
            conversation."), is not stored ("No conversation '<id>'
            found."), or could not be read ("Could not open '<id>': ...").
        """
        with _says_so_when_stopped(self.session):
            return await self._resume(session_id)

    async def _resume(self, session_id: str) -> bool:
        """``resume_session`` without its "Operation cancelled..." line: at
        startup (``_open_on_startup``) a Ctrl+C ends the app, not the switch."""
        async with self._turn_lock:
            if session_id == self._session_id:
                self.session.add_message("system", "Already in this conversation.")
                return False
            controller = self._workflow_controller
            if not controller.is_ready:
                self.session.add_error(
                    f"Could not open '{session_id}': the workflow is not ready."
                )
                return False
            workflow = controller.workflow
            try:
                info = await workflow.session_info(session_id)
                messages = (
                    await workflow.recent_messages(session_id, sys.maxsize) if info else []
                )
            except Exception as error:  # noqa: BLE001 - a store error changes nothing
                logger.debug("session_open_failed", session_id=session_id, exc_info=True)
                reason = str(error) or type(error).__name__
                self.session.add_error(f"Could not open '{session_id}': {reason}")
                return False
            if info is None:
                self.session.add_error(f"No conversation '{session_id}' found.")
                return False
            # Made before anything changes: a display error changes nothing.
            line = resumed_line(info, datetime.now(timezone.utc))
            replay = replay_lines(messages)
            old = self._session_id
            if self._turned:  # facts once per conversation used
                await self._saving_facts(old)
            try:
                await workflow.load_session(session_id)
            except Exception as error:  # noqa: BLE001 - the id is not adopted
                logger.debug("session_open_failed", session_id=session_id, exc_info=True)
                reason = str(error) or type(error).__name__
                self.session.add_error(f"Could not open '{session_id}': {reason}")
                return False
            self._session_id = session_id
            self._turned = False
            self._message_processor.reset()
            self._usage_tracker.reset()
            _show_new_context(controller, self.session, swapped=False)
            logger.info("session_resumed", session_id=session_id, previous=old)
            self.session.add_message("system", line)
            show_replay(self.session, replay)
            return True

    async def pick_session(self) -> str | None:
        """Ask which stored conversation to open: the newest ``PICK_LIMIT`` but this one.

        Not under the turn lock: it waits on a dialog (``resume_session``
        takes the lock).

        Returns:
            The id picked, or None: Esc, or no other conversation is stored
            ("No other conversations to resume.").

        Raises:
            RuntimeError: The workflow is not ready, or the session is not
                running (a dialog needs it); ``/resume`` checks both.
        """
        workflow = self._workflow_controller.workflow
        rows = [
            info for info in await workflow.list_sessions(limit=PICK_LIMIT + 1)
            if info.session_id != self._session_id
        ][:PICK_LIMIT]
        if not rows:
            note = _MEMORY_NOTE if self._settings.session_store == "memory" else ""
            self.session.add_message("system", f"No other conversations to resume.{note}")
            return None
        dialog, ids = picker(rows, datetime.now(timezone.utc))
        picked = await self.session.show_dialog(dialog)
        return None if picked is None else ids.get(picked)

    async def _saving_facts(self, session_id: str) -> None:
        """Extract ``session_id``'s facts (if enabled) under a status line saying so."""
        if not (
            getattr(self._settings, "auto_extract_session_facts", False)
            and self._workflow_controller.is_ready
        ):
            return
        box = self.session.start_thinking(lambda: SAVING_FACTS, content_format="ansi")
        try:
            await self._extract_session_facts(session_id)
        finally:
            # Out of the transcript, like a turn's events box, a cancel included.
            box.finish(add_to_history=False, echo_to_console=False)

    @staticmethod
    async def _is_stored(workflow: Any, session_id: str) -> bool:
        """Whether ``session_id``'s conversation is in the workflow's store (``/sessions``)."""
        if not getattr(workflow, "supports_sessions", False):
            return False
        try:
            return bool(await workflow.session_exists(session_id))
        except Exception:
            logger.debug("session_lookup_failed", session_id=session_id, exc_info=True)
            return False

    async def _extract_session_facts(self, session_id: str | None = None) -> None:
        """Extract the facts of ``session_id``'s conversation into memory (if enabled).

        ``session_id`` defaults to the current conversation (at exit).
        Gated by ``auto_extract_session_facts``; a no-op when the workflow is
        not ready. Never raises: a conversation must be left whatever its
        facts. Runs outside any turn, so its model call reaches no usage
        counter.
        """
        if not getattr(self._settings, "auto_extract_session_facts", False):
            return
        if not self._workflow_controller.is_ready:
            return
        workflow = self._workflow_controller.workflow
        session_id = session_id or self._session_id
        try:
            facts = await workflow.on_session_end(
                session=workflow.session_ref(session_id, self._settings.default_user)
            )
        except Exception:
            logger.debug("session_fact_extraction_failed", exc_info=True)
            return
        if facts:
            logger.info("session_facts_extracted", count=len(facts), session_id=session_id)

    async def _adopt_session_on_startup(self) -> bool:
        """Adopt this run's conversation before the prompt; False when the app must not start.

        Runs for every startup: a fresh auto-generated id must be adopted too,
        otherwise turns fall back to the manager's ``default_session`` and
        unnamed runs silently share one conversation. By ``session_start``:

        - ``"continue"`` finds the newest stored conversation, which
          ``_open_on_startup`` resumes once the prompt is up, or says there is
          none ("No conversation to continue; starting a new one.");
        - ``"resume"`` checks its conversation is stored first: one that is
          not is refused ("No conversation '<id>' found. Run <app> --resume
          to pick one."), ``exit_status`` is 1, and False is returned;
        - ``"pick"`` starts in the new conversation, and the picker opens
          once the prompt is up; ``"new"`` starts in the new conversation.

        A workflow that cannot start keeps the prompt, so ``/settings`` can
        fix it, and says how to resume then; a backend without sessions
        refuses every start but ``"new"`` (False, status 1). A store that
        fails refuses ``"resume"`` too, and starts the others in a new
        conversation, saying why. On the memory store a line says why
        nothing is stored.
        """
        mode, target = self._session_start.mode, self._session_start.session_id
        if not await self._workflow_controller.ensure_initialized(self.session):
            self.session.add_warning({
                "resume": f"Could not open '{target}': the workflow did not start. "
                          f"Fix it with /settings, then /resume {target}.",
                "continue": "Could not continue: the workflow did not start. "
                            "Fix it with /settings, then /resume.",
                "pick": "Could not open the picker: the workflow did not start. "
                        "Fix it with /settings, then /resume.",
            }.get(mode, "Cannot adopt session — workflow not initialized."))
            return True

        workflow = self._workflow_controller.workflow
        if mode != "new" and not getattr(workflow, "supports_sessions", False):
            self.session.add_error("This app's backend keeps no sessions.")
            self.exit_status = 1
            return False
        note = _MEMORY_NOTE if self._settings.session_store == "memory" else ""
        try:
            if mode == "resume":
                if await workflow.session_info(target) is None:
                    run = f" Run {self._command} --resume to pick one." if self._command else ""
                    self.session.add_error(f"No conversation '{target}' found.{run}{note}")
                    self.exit_status = 1
                    return False
                self._pending_resume = target
            elif mode == "continue":
                newest = await workflow.list_sessions(limit=1)
                if newest:
                    self._pending_resume = newest[0].session_id
                else:
                    self.session.add_message(
                        "system", f"No conversation to continue; starting a new one.{note}"
                    )
            elif mode == "pick":
                self._pending_pick = True
        except Exception as error:  # noqa: BLE001 - the store failed, not the app
            logger.debug("startup_session_lookup_failed", mode=mode, exc_info=True)
            reason = str(error) or type(error).__name__
            if mode == "resume":
                self.session.add_error(f"Could not open '{target}': {reason}")
                self.exit_status = 1
                return False
            self.session.add_warning(
                f"Could not list the saved conversations: {reason}. Starting a new one."
            )

        try:
            resumed = await workflow.load_session(self._session_id)
        except Exception as error:  # noqa: BLE001 - turns name their conversation anyway
            logger.debug("session_adopt_failed", exc_info=True)
            self.session.add_warning(
                f"Cannot adopt session — {str(error) or type(error).__name__}"
            )
            return True
        if resumed:
            self.session.add_success(f"Session '{self._session_id}' resumed.")
        elif self._resume_requested:
            # An explicit session_id that didn't exist yet: tell the user it's new.
            self.session.add_message("system", f"New session '{self._session_id}'.")
        return True

    async def _open_on_startup(self) -> None:
        """Once the prompt is up: resume what ``--continue``/``--resume <id>``
        chose, or open the picker for ``--resume`` alone.

        A dialog needs the running session, and a replay reads best under the
        welcome banner, so this waits until the session runs. It runs outside
        the input handler, so a Ctrl+C here ends the app, as at an idle prompt
        (its resume, cancelled on the way out, says nothing of a cancel). A
        failure is said and logged: the new conversation goes on.
        """
        if self._pending_resume is None and not self._pending_pick:
            return
        while not self.session.app.is_running:
            await asyncio.sleep(0.02)
        target = self._pending_resume
        try:
            if target is None:
                target = await self.pick_session()
            if target is not None:
                await self._resume(target)
        except Exception as error:  # noqa: BLE001 - the new conversation goes on
            logger.debug("startup_open_failed", session_id=target, exc_info=True)
            reason = str(error) or type(error).__name__
            if target is None:
                self.session.add_error(f"Could not list the saved conversations: {reason}")
            else:
                self.session.add_error(f"Could not open '{target}': {reason}")

    def _resume_hint(self) -> str:
        """The exit line that names how to come back to the current conversation."""
        if self._command:
            return f"Resume this conversation with: {self._command} --resume {self._session_id}"
        return f"Resume this conversation with /resume {self._session_id}"

    async def _extract_session_facts_on_exit(self) -> None:
        """Extract key facts from the session into memory on exit (if enabled).

        Gated by the ``auto_extract_session_facts`` setting; a no-op when the
        workflow isn't ready or has no memory store. Never raises — fact
        extraction must not block a clean shutdown. The conversation is the
        current one, which the manager also holds (``load_session``), and only
        when a turn ran in it since it was opened: a resumed conversation left
        as it was has nothing new.
        """
        if self._turned:
            await self._extract_session_facts()

    async def run(self) -> None:
        """Run the main application loop."""
        logger.info("repl_starting")

        from agentic_cli.cli.job_monitor import JobMonitor

        job_monitor = JobMonitor(
            self.session, self._workflow_controller, settings=self._settings
        )

        # Once, before anything waits for the workflow: a backend that cannot
        # start (its extra missing, or a key) is still deprecated.
        _warn_if_langgraph(self._settings, self.session)

        async with self._workflow_controller.background_init(self.session):
            # Adopt this run's session id (generated, or session_id=) so every
            # turn targets it; a fresh id starts a new durable session rather
            # than falling back to the shared 'default_session'. --resume <id>
            # of a conversation that is not stored ends here, before the prompt.
            if not await self._adopt_session_on_startup():
                logger.info("app_start_refused", exit_status=self.exit_status)
                return

            # Register input handler
            @self.session.on_input
            async def handle_input(text: str) -> None:
                if self.should_exit:
                    return
                await self.process_input(text)

            # Keep long-running jobs visible in the status bar while the
            # session runs, independent of the agent loop.
            async with job_monitor.running():
                # What --continue / --resume open, once the prompt is up.
                opening = asyncio.create_task(self._open_on_startup())
                try:
                    # Run the session - user sees prompt immediately!
                    await self.session.run_async()
                finally:
                    opening.cancel()
                    await asyncio.gather(opening, return_exceptions=True)

            # Extract session facts into memory (if enabled) while the workflow
            # is still alive: leaving this context closes the manager, and fact
            # extraction needs the live session store and an LLM call.
            await self._extract_session_facts_on_exit()
            # Named at exit when a later run can resume it: stored, not in memory.
            controller = self._workflow_controller
            resumable = (
                self._settings.session_store != "memory"
                and controller.is_ready
                and await self._is_stored(controller.workflow, self._session_id)
            )

        # No save-on-exit: durable session stores persist continuously per turn.

        logger.info("app_ending")
        if resumable:
            self.session.add_message("system", self._resume_hint())
        self.session.add_message("system", "Goodbye!")


def _model_picker(key: str, choices: _ModelChoices) -> Field:
    """The dialog field of one model picker: ``model`` or ``model_role:<role>``."""
    from pydantic.fields import FieldInfo

    from agentic_cli.cli.settings_introspection import field_to_ui_item

    if key == "model":
        role, title = "default", "Default Model"
        description = "Agents without a model of their own run on it"
    else:
        role = key.removeprefix(MODEL_ROLE_KEY_PREFIX)
        title = _role_picker_label(key)
        description = f"Agents whose model is {role!r} run on it"
    field = FieldInfo(default=None, title=title, description=description)
    return field_to_ui_item(
        key=key,
        field=field,
        current_value=choices.current.get(role),
        dynamic_options=choices.options,
        max_width=_MODEL_PICKER_MAX_WIDTH,
    )


# The widest a model picker's dropdown grows: a reference and its padding. A
# dropdown's default width cuts a reference to its first 14 characters
# ("google:gemini-"), so the models could not be told apart. SettingsCommand's
# dialog leaves room for it beside the longest label.
_MODEL_PICKER_MAX_WIDTH = 48


def _resolved_role(
    roles: "RoleMap", role: str, catalog: "CatalogView", endpoints: "Mapping[str, Endpoint]"
) -> str | None:
    """The reference ``role`` resolves to, or None when nothing can fill it."""
    from agentic_cli.models import ModelResolutionError

    try:
        return str(roles.resolve(role, catalog=catalog, endpoints=endpoints))
    except ModelResolutionError:
        return None


def _show_new_context(
    controller: WorkflowController, session: Any, *, swapped: bool
) -> None:
    """Show ``ctx`` against the rebuilt agents' trigger now, not after the next call.

    A reinitialize may move the main agent's trigger (a model of another
    window, ``context_summarize_at``, ``context_max_tokens``), and the last
    prompt may already be past the new one. The usage tracker takes the
    trigger and the window flag from ``workflow.context_budget()``: none when
    the manager manages no context (``KeyError``), has no such method, or
    is not ready (it failed to start; the status bar then says so). An
    orchestrator swap starts another manager, whose conversation is another
    one: the tracker forgets the context (the session's totals stay). The
    status bar is then drawn again.

    Args:
        controller: The app's workflow controller, after the reinitialize.
        session: The UI session whose status bar is drawn.
        swapped: The reinitialize replaced the manager.
    """
    tracker = controller.usage_tracker
    if tracker is not None and swapped:
        tracker.reset_context()
    elif tracker is not None:
        budget = None
        # A workflow that is not ready (it failed to start) has no budget,
        # and the controller hands out none: the status bar says why.
        if controller.is_ready:
            context_budget = getattr(controller.workflow, "context_budget", None)
            try:
                budget = context_budget() if context_budget is not None else None
            except (KeyError, RuntimeError):
                budget = None
        tracker.context_trigger = budget.trigger if budget is not None else None
        tracker.window_known = budget.window_known if budget is not None else None
    controller.update_status_bar(session)


def _provider_switches(
    before: Mapping[str, ResolvedModel], after: Mapping[str, ResolvedModel]
) -> list[str]:
    """What each switch to another provider's model costs, one line per switch.

    The conversation is kept, but no model is sent another provider's
    reasoning (both backends leave it out), and the new provider has none of
    the old one's prompt cache. A switch within one provider loses neither.
    """
    agents: dict[tuple[ModelRef, ModelRef], list[str]] = {}
    for name, model in after.items():
        old = before.get(name)
        if old is not None and old.record.kind is not model.record.kind:
            agents.setdefault((old.ref, model.ref), []).append(name)
    return [
        f"{', '.join(names)}: {old} → {new}. The conversation is kept, but earlier "
        f"reasoning from {old.provider} is not sent to the new model, and the prompt "
        "cache starts over."
        for (old, new), names in agents.items()
    ]


def _snapshot_settings(settings: BaseSettings) -> tuple[dict[str, Any], set[str]]:
    """Capture every field value (and which were set) so a change can be undone."""
    return dict(settings.__dict__), set(settings.__pydantic_fields_set__)


def _restore_settings(
    settings: BaseSettings, snapshot: tuple[dict[str, Any], set[str]]
) -> None:
    """Put back the values captured by :func:`_snapshot_settings`.

    Written directly, like the dedicated setters do: they were valid when
    captured, and revalidating could itself fail.
    """
    values, fields_set = snapshot
    for key, value in values.items():
        if settings.__dict__.get(key) is not value:
            object.__setattr__(settings, key, value)
    object.__setattr__(settings, "__pydantic_fields_set__", fields_set)
