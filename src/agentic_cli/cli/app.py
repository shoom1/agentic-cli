"""Base CLI Application for agentic applications.

This module provides the base CLI application that:
1. Uses ThinkingPromptSession for all UI (thinking boxes, messages, etc.)
2. Connects to domain-specific workflow managers
3. Opens, switches and resumes the conversation its turns run in (the
   workflow's session store keeps each one)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import re
import shlex
import shutil
import sys
import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NoReturn, Self

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory

from thinking_prompt import AppInfo, ThinkingPromptSession, ThinkingPromptStyles

from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.cli.message_processor import (
    CANCELLED_LINE,
    MessageProcessor,
    TurnStatus,
    _says_so_when_stopped,
)
from agentic_cli.cli.conversation import ConversationCore, Hold, Place, Switch, SwitchParts
from agentic_cli.cli.session_args import (
    SessionArgumentParser,
    SessionStart,
    add_session_arguments,
    parse_session_start,
)
from agentic_cli.cli.session_resume import (
    PICK_LIMIT,
    picker,
    replay_lines,
    resumed_line,
    show_replay,
)
from agentic_cli.cli.usage_tracker import UsageTracker
from agentic_cli.cli.waiting_jobs import Here, told_here, waiting_here
from agentic_cli.cli.workflow_controller import WorkflowController, WorkflowState
from agentic_cli.config import MODEL_ROLE_KEY_PREFIX, BaseSettings
from agentic_cli.logging import LOG_STREAM, Loggers, configure_logging, log_lines_to
from agentic_cli.settings_persistence import (
    HAND_EDITED_KEYS,
    NEW_CONVERSATION_KEYS,
    PROJECT_SETTABLE_KEYS,
    REBUILD_KEYS,
    RESTART_KEYS,
    USER_PREFERENCE_KEYS,
)
from agentic_cli.workflow.factory import check_backend, orchestrator_of
from agentic_cli.workflow.settings import OrchestratorType

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable, Sequence

    from thinking_prompt import Field

    from agentic_cli.cli.job_monitor import JobMonitor
    from agentic_cli.models import CatalogView, ModelRef, Provider, ResolvedModel, RoleMap
    from agentic_cli.settings_persistence import SettingsSaveResult
    from agentic_cli.workflow import GoogleADKWorkflowManager, EventType, WorkflowEvent
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.context import CompactResult
    from agentic_cli.workflow.sessions import SessionInfo
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


def _stderr_is_terminal() -> bool:
    """Whether stderr is a terminal (the one the prompt is drawn on): not
    redirected to a file, as ``app 2>run.log`` does."""
    try:
        return sys.stderr is not None and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


@contextmanager
def _log_lines_above_prompt(session: ThinkingPromptSession) -> Iterator[None]:
    """While the prompt runs, print each log line above it, as the session's
    own output.

    Written to stderr, a line landed wherever the cursor was (on the prompt
    line), and the next redraw did not know it was there. A line logged on
    the event loop's thread is shown at once, so it keeps its place among
    the session's output (a turn's warning before the error that follows
    it); one from another thread (a tool's worker thread, an executor's) is
    handed to the loop. A line logged while one is being shown goes to
    stderr, so showing never recurses. With stderr redirected (``2>file``),
    the lines stay on it. Python's warnings go the same way
    (``_install_adk_notice_hook``).
    """
    if not _stderr_is_terminal():
        yield
        return

    from rich.text import Text

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    busy = threading.local()

    def show(line: str) -> None:
        busy.now = True
        try:
            session.add_rich(Text.from_ansi(line))
        except Exception:
            # Not logged: the log line would come back here.
            sys.stderr.write(line + "\n")
        finally:
            busy.now = False

    def sink(line: str) -> None:
        if getattr(busy, "now", False):
            sys.stderr.write(line + "\n")
        elif threading.get_ident() == loop_thread:
            show(line)
        else:
            busy.now = True
            try:
                loop.call_soon_threadsafe(show, line)
            finally:
                busy.now = False

    with log_lines_to(sink):
        yield


def _install_adk_notice_hook() -> None:
    """Log ADK's notices instead of printing them.

    While the prompt runs, the log lines ``configure_logging`` sets up print
    above it (``_log_lines_above_prompt``), and so do Python's warnings: this
    hook has Python's own hook write them to the same stream (another hook,
    such as logging's ``captureWarnings``, gets them as given). The hook logs ADK's
    experimental-feature notices at debug and passes every other warning to
    the hook it replaced; a filter on ADK's runner logger does the same for
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
        if file is None and previous is warnings._showwarning_orig:
            file = LOG_STREAM
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


#: What setting ``BaseCLIApp.session_id`` warns, deprecated in 0.7.0: a setter
#: cannot run the switch, which awaits the workflow.
_SESSION_ID_SETTER_DEPRECATED = (
    "Setting BaseCLIApp.session_id is deprecated: use `await app.resume_session(id)` "
    "or `await app.start_new_session()`. The setter holds the conversation and starts "
    "its task list and usage counters over, but the workflow does not load it until "
    "the next turn names it, and the memories of the conversation left are not proposed."
)

#: How ``/settings`` reports a change of a key read only as the app starts
#: (``settings_persistence.RESTART_KEYS``).
_AT_NEXT_START = "(takes effect at the next start)"

#: How ``/settings`` reports a change of the backend of new conversations
#: (``settings_persistence.NEW_CONVERSATION_KEYS``) that the open one keeps.
_AT_NEXT_NEW = "(applies to the next /new)"

#: The keys besides the rebuild keys whose change waits for a start in
#: progress, as a rebuild does: whether an orchestrator change starts a new
#: conversation, or moves the run's, needs that start's outcome. The session
#: store is fixed for the run as its first start begins
#: (``WorkflowController.store``), and ``log_level`` and ``log_format`` are
#: read by the app's constructor only: a change of them does not wait.
_READ_BY_THE_START = frozenset({"orchestrator"})

#: What ``_move_here`` returns when it finds, under the turn lock, that the
#: move no longer applies, decided as for a running workflow: something of
#: the conversation is stored by now, or the next start no longer builds
#: afresh (it stays; the change applies to the next ``/new``) ...
_STAYS = "stays"
#: ... or the workflow came up meanwhile, nothing of it stored (a new
#: conversation on the new backend, ``start_new_session``).
_STARTS_ANEW = "starts anew"

#: Why nothing is stored on the memory store, after a startup line.
_MEMORY_NOTE = " (session_store is memory: conversations end with the app)"


def _passed_over_line(passed_over: list[tuple[str, int]], *, continuing: str | None) -> str:
    """What ``--continue`` says of the newer conversations other processes have
    open: each by its id and holder, past three a count."""
    if len(passed_over) == 1:
        (session_id, holder), = passed_over
        held = (
            f"The newest conversation, '{session_id}', is open in another process ({holder})"
        )
    else:
        names = [f"'{session_id}' ({holder})" for session_id, holder in passed_over[:3]]
        if len(passed_over) > 3:
            names.append(f"{len(passed_over) - 3} more")
        held = (
            f"The newest conversations, {', '.join(names[:-1])} and {names[-1]}, "
            "are open elsewhere"
        )
    then = f"continuing '{continuing}'" if continuing else "starting a new one"
    return f"{held}; {then}."


def _command_name() -> str:
    """The command this process was run as, to type again.

    ``<python> -m <package>`` (the interpreter as it was run, ``python3``
    say); ``<python> <script>`` for a script or a zipapp; a console script's
    name when PATH finds it, else its path. On Windows pip's launcher runs a
    zip whose module is ``__main__``, which is no ``-m`` run.
    """
    python = os.path.basename(sys.executable) or "python"
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name and spec.name != "__main__":
        return f"{python} -m {spec.name.removesuffix('.__main__')}"
    script = sys.argv[0] if sys.argv and sys.argv[0] else "app"
    if script.endswith((".py", ".pyz")) or os.path.isdir(script):
        return f"{python} {shlex.quote(script)}"
    name = re.split(r"[\\/]", script)[-1].removesuffix(".exe")
    found = shutil.which(name)
    if found is not None and _same_program(found, script):
        return name
    return shlex.quote(script)


def _same_program(found: str, script: str) -> bool:
    """Whether PATH's ``found`` is ``script`` (an .exe found for a launcher too)."""
    def key(path: str) -> str:
        return os.path.splitext(os.path.normcase(path).replace("\\", "/"))[0]

    return key(found) == key(script) or key(os.path.realpath(found)) == key(
        os.path.realpath(script)
    )


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


# === A finished job's delivery ===

#: Set while the CLI tells finished jobs on its own (a turn's end, a
#: conversation opened), not because the user asked (``/jobs resume``).
_TELLING_ON_ITS_OWN: ContextVar[bool] = ContextVar("telling_on_its_own", default=False)


async def _claim_delivery(jm: Any, job_id: str) -> bool:
    """Claim the job's delivery (``jm.begin_resume``) on a worker thread.

    The claim takes the jobs directory's lock, which another CLI of the app
    may hold, so it runs off the event loop. A Ctrl+C while it is taken stops
    the wait, not the claim, which its thread may still take: a claim taken
    then is handed back (``JobManager.release_resume``), never left
    ``RESUMING``, and the job waits for a later delivery, nothing having
    reached the agent.

    Returns:
        True if this process now delivers the job.
    """
    claim = asyncio.ensure_future(asyncio.to_thread(jm.begin_resume, job_id))
    try:
        return await asyncio.shield(claim)
    except asyncio.CancelledError:
        claim.add_done_callback(lambda done: _release_abandoned_claim(jm, job_id, done))
        raise


def _release_abandoned_claim(jm: Any, job_id: str, claim: "asyncio.Future[bool]") -> None:
    """Hand back, on a worker thread, a claim taken for a delivery Ctrl+C stopped."""
    if claim.cancelled() or claim.exception() is not None or not claim.result():
        return
    logger.info("job_resume_claim_released", job_id=job_id)

    def release() -> None:
        try:
            jm.release_resume(job_id)
        except Exception:  # noqa: BLE001 - startup recovery fails it once we are gone
            logger.warning("job_resume_release_failed", job_id=job_id, exc_info=True)

    asyncio.get_running_loop().run_in_executor(None, release)


# === Slash Command Completer ===


class SlashCommandCompleter(Completer):
    """Completer that only triggers for slash commands."""

    def __init__(self, commands: "CommandRegistry | Iterable[str]") -> None:
        """Initialize with the commands to complete.

        Args:
            commands: A ``CommandRegistry``, read each time a completion is
                asked for (a command registered later completes too, as the
                app's session does), or command names without the leading
                slash, e.g. ``["help", "settings", "quit"]``, fixed here.
        """
        if isinstance(commands, CommandRegistry):
            self._registry: CommandRegistry | None = commands
            self._names: list[str] = []
        else:
            self._registry = None
            self._names = sorted(commands)

    @property
    def commands(self) -> list[str]:
        """The command names and aliases it completes, sorted."""
        if self._registry is not None:
            return self._registry.get_completions()
        return self._names

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


#: What ``/new`` and ``/resume`` show while the memories to propose are looked
#: for in the conversation they leave (the name is from when it saved facts).
SAVING_FACTS = "Looking for memories to propose…"


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
    - apply_settings(changes): Handle additional custom settings
    - save_settings(keys): Persist the settings ``/settings`` changed
    - add_arguments(parser), from_args(args): The app's own command-line
      flags, and how ``main()`` builds the app from them (classmethods)

    ``register_commands()`` and ``get_theme()`` run during construction:
    ``register_commands()`` last, ``get_theme()`` as the session is built
    in its theme, everything else in place. An attribute of the app's own
    that one reads is set before ``super().__init__()``.
    """

    #: (job, conversation) pairs whose "could not be resumed now" was said:
    #: the deliveries the CLI makes on its own say it once per conversation.
    #: An immutable class default, replaced on each write, for the same reason.
    _said_not_told: frozenset[tuple[str, str]] = frozenset()

    #: The status bar's job monitor while ``run()`` runs (None before, and
    #: for an app built with ``__new__``).
    _job_monitor: "JobMonitor | None" = None

    #: The lock on the conversation ``--continue``/``--resume`` chose, taken
    #: before the prompt and kept by the switch to it once the prompt is up
    #: (``_open_on_startup``); None otherwise.
    _pending_hold: Hold | None = None

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
                ``main()`` takes ``--continue`` and ``--resume <id>``
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
        # The conversation the turns run in: its id, whether a turn ran in it
        # since it was opened, and the lock held on it against another
        # process opening it, at the place the settings keep it in (read
        # again as run() starts the workflow: cli/conversation.py).
        self._conversation_core = ConversationCore(
            session_id or uuid.uuid4().hex[:12], Place.of(settings),
            opened_empty=session_id is None,  # one given may be stored
            deferred=session_id is None,  # held from its first write
        )
        # How run() starts (``session_start``), and what it opens once the prompt is up.
        self._session_start = SessionStart()
        self._pending_resume: str | None = None
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

        # === Components ===
        self._usage_tracker = UsageTracker()
        self._workflow_controller = WorkflowController(
            agent_configs=agent_configs,
            settings=settings,
        )
        self._workflow_controller.usage_tracker = self._usage_tracker
        self._message_processor = MessageProcessor()

        # Serializes what writes to the conversation: a turn, a job's resume,
        # /compact, /new and /resume (not /settings, nor the /resume picker,
        # which waits on a dialog). thinking-prompt runs one input handler at
        # a time, so it orders them against the startup resume
        # (_open_on_startup, a task of its own) and an app's own tasks.
        self._turn_lock = asyncio.Lock()
        self.should_exit = False

        # === Commands: registered once the session exists (below) ===
        self.command_registry = CommandRegistry()

        # === UI: ThinkingPromptSession ===
        # get_theme() runs here, with every attribute above in place: its
        # theme is the one the session is built with.
        self.session = ThinkingPromptSession(
            message=">>> ",
            app_info=self._app_info,
            theme=self.get_theme(settings.ui_theme),
            history=InMemoryHistory(),
            # Reads the registry as it completes: a command registered later
            # completes too.
            completer=SlashCommandCompleter(self.command_registry),
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

        # === Hooks: last, on the built app ===
        self._register_builtin_commands()
        self.register_commands()  # Domain-specific commands

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
        is built in ``get_theme(settings.ui_theme)``: it is called while the
        constructor builds the session, with every other attribute of the
        app in place, and again when ``/settings`` changes ``ui_theme``. An
        attribute of the app's own that it reads is set before
        ``super().__init__()``.

        Override to recolor a theme, e.g. for ``"dark"`` return
        ``dataclasses.replace(ThinkingPromptStyles.dark(), thinking_box=...)``
        and ``super().get_theme(name)`` for the others.
        """
        if name == "dark":
            return dataclasses.replace(ThinkingPromptStyles.dark(), **_DARK_COLORS)
        return name

    def register_commands(self) -> None:
        """Register domain-specific commands.

        Override to register additional commands
        (``self.command_registry.register(...)``). Called last in the
        constructor, after the built-ins (``_register_builtin_commands``),
        once the settings, the usage tracker, the workflow controller and
        the session exist. An attribute of the app's own that it reads is
        set before ``super().__init__()``. A command registered later works
        too: the slash-command completer and ``/help`` read the registry each
        time.
        """
        pass

    def get_ui_setting_keys(self) -> list[str]:
        """Get field names to display in the settings dialog.

        Override to customize which settings appear in the UI.
        Default: model, model_role:fast, model_role:strong, thinking_effort.
        ``model`` and each ``model_role:<role>`` are model pickers: they
        offer the list (``ModelCatalog.offered``: the ``models`` entries,
        then the models the settings and the agents name) as
        ``provider:model`` references, and write ``default_model`` and
        ``model_roles[<role>]``.
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

        When a Save takes effect is declared for each of those keys beside
        them in ``settings_persistence``: one in ``REBUILD_KEYS`` (the model
        pickers, ``thinking_effort``, the summary thresholds,
        ``tool_result_max_chars``, the retries and
        ``anthropic_request_timeout``) rebuilds the workflow; one in
        ``NEW_CONVERSATION_KEYS`` (``orchestrator``) starts a new conversation
        on that backend when the open one holds nothing yet, else applies to
        the next ``/new``; one in ``RESTART_KEYS`` (``session_store``,
        ``log_level``, ``log_format``) is saved, and its line says it takes
        effect at the next start; one in ``LIVE_KEYS`` (the rest) applies at
        once.

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
        shown = (PROJECT_SETTABLE_KEYS | USER_PREFERENCE_KEYS) - HAND_EDITED_KEYS

        for key in keys:
            # Project-scope guard: the dialog may only expose settings the
            # project file can persist (PROJECT_SETTABLE_KEYS), and the user's
            # preferences (USER_PREFERENCE_KEYS), which are meant to apply in
            # every project. Any other user-scoped key would save to the user
            # ~/.{app}/settings.json and leak the change across all projects
            # of the app. Nor does it expose a hand-edited key
            # (HAND_EDITED_KEYS: models, and model_providers, which holds the
            # addresses a key is sent to): no save writes one, so an edit
            # would be gone at the next start.
            target = _ui_key_target_field(key)
            if target not in shown:
                if target in HAND_EDITED_KEYS:
                    logger.warning("hand_edited_setting_excluded_from_ui", key=key)
                else:
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

        The options are the list (``ModelCatalog.offered``), with each role's
        model (``model_services.role_models``, as ``/models`` has it). Once
        the workflow is ready, from its catalog (listings
        included) and its default model; before that, from the catalog it
        would load, unloaded (the entries and the shipped table), and the
        default model the settings name (or compute). A role shows the model
        it maps, else the one it resolves to (an app's own role runs on the
        default model).

        Args:
            roles: The roles other than ``default`` whose model to show.
        """
        from agentic_cli.models import BUILTIN_ROLES
        from agentic_cli.workflow import model_services

        try:
            role_map = model_services.role_map_from_settings(self._settings)
        except ValueError:
            return _ModelChoices(options=[], current={}, thinking=None)
        providers = model_services.providers_from_settings(self._settings)
        controller = self._workflow_controller
        workflow = controller.workflow if controller.is_ready else None
        if workflow is not None and workflow.catalog is not None:
            catalog, default = workflow.catalog, workflow.model
        else:
            # Not loaded: the entries, what the settings and agents name, and
            # the shipped table.
            catalog = model_services.build_catalog(
                self._settings, providers, agent_configs=self.agent_configs
            )
            default = _resolved_role(role_map, "default", catalog, providers)

        current: dict[str, str | None] = {"default": default}
        for role in roles:
            mapped = role_map.spec_for(role)
            if mapped is not None:
                current[role] = str(mapped)
            elif role in BUILTIN_ROLES:
                current[role] = _resolved_role(role_map, role, catalog, providers)
            else:
                current[role] = default
        # Every role's model, whichever pickers the app shows: /models offers
        # the same list.
        role_models = model_services.role_models(self._settings, catalog, providers, default)
        return _ModelChoices(
            options=catalog.offered(extra=list(role_models)),
            current=current,
            thinking=self._thinking_meaning(default, catalog, providers),
        )

    def _thinking_meaning(
        self, model: str | None, catalog: "CatalogView", providers: "Mapping[str, Provider]"
    ) -> str | None:
        """``high → anthropic:claude-sonnet-5-5: adaptive, effort high``: the
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
                providers=providers,
            )
        except ModelResolutionError:
            return None
        meaning = resolved.thinking.label if resolved.thinking else "ignored (no thinking knob)"
        return f"{effort} → {resolved.ref}: {meaning}"

    async def save_settings(self, keys: "Iterable[str]") -> "SettingsSaveResult":
        """Save the settings ``keys`` name, split by trust.

        ``/settings`` passes the fields the user changed: only those are
        written, as picked (see ``SettingsPersistence.save``).

        Allowlisted keys go to the project config
        (./.{app_name}/settings.json). User-scoped keys go to the user config
        (~/.{app_name}/settings.json), where the loader trusts them — the
        project file may only carry allowlisted keys since P0-1. Secrets
        (API keys) are never saved. The files are read and written on a
        worker thread, off the event loop.

        Returns:
            SettingsSaveResult with the written path(s)

        Raises:
            TypeError: ``keys`` is None (an override that passes its own
                ``keys=None`` default on has no keys to write).
        """
        from agentic_cli.settings_persistence import SettingsPersistence

        persistence = SettingsPersistence(self._settings.app_name)
        return await asyncio.to_thread(persistence.save, self._settings, keys=keys)

    @property
    def workflow(self) -> "BaseWorkflowManager":
        """Get the workflow manager.

        Raises:
            RuntimeError: If workflow is not yet initialized
        """
        return self._workflow_controller.workflow

    @property
    def workflow_error(self) -> Exception | None:
        """Why the workflow is not running: the error its start, or its last
        rebuild, failed with (``/status`` shows it). None while it runs, while
        it is still starting, and before it was ever started."""
        return self._workflow_controller.init_error

    @property
    def agent_configs(self) -> tuple["AgentConfig", ...]:
        """The agents the workflow is built from, ready or not (``/models``
        and the model pickers offer their models)."""
        return self._workflow_controller.agent_configs

    @property
    def usage_tracker(self) -> UsageTracker:
        """Get the usage tracker."""
        return self._usage_tracker

    @property
    def settings(self) -> BaseSettings:
        """Get the application settings."""
        return self._settings

    @property
    def _conversation(self) -> ConversationCore:
        """The conversation the app is in (``cli/conversation.py``). An app
        built with ``__new__`` (tests, an application's own construction) gets
        one, with no id, on first use."""
        try:
            return self.__dict__["_conversation_core"]
        except KeyError:
            core = self.__dict__["_conversation_core"] = ConversationCore()
            return core

    @property
    def _session_id(self) -> str | None:
        """The conversation's id (``_conversation.session_id``)."""
        return self._conversation.session_id

    @_session_id.setter
    def _session_id(self, value: str | None) -> None:
        self._conversation.session_id = value

    @property
    def _turned(self) -> bool:
        """A turn ran in the conversation since it was opened
        (``_conversation.turned``): its memories are proposed on leaving it."""
        return self._conversation.turned

    @_turned.setter
    def _turned(self, value: bool) -> None:
        self._conversation.turned = value

    @property
    def session_id(self) -> str | None:
        """The id of the conversation the turns run in.

        Setting it is deprecated (0.7.0): switch with ``await
        resume_session(id)`` or ``await start_new_session()``, which the
        workflow loads and which propose the memories of the conversation
        left. The setter still switches, as far as a setter can (the switch
        awaits the workflow): it holds the conversation, letting go of the
        one before, and starts the task list and the usage counters over,
        with a ``DeprecationWarning``. The workflow is not told until the next
        turn names the id, and the memories of the conversation left are not
        proposed. Set it between turns. None names no conversation: the turns
        run in the one the workflow loaded last, which a ready workflow's
        ``session_id`` names and which is then held (with none ready, the
        conversation held stays held).

        Raises:
            RuntimeError: On setting an id another process has open; nothing
                changes.
        """
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        warnings.warn(_SESSION_ID_SETTER_DEPRECATED, DeprecationWarning, stacklevel=2)
        parts = self._switch_parts()
        holder = self._conversation.switch_now(value, parts)
        if holder is not None:
            name = value if value is not None else getattr(parts.workflow, "session_id", value)
            raise RuntimeError(f"Conversation '{name}' is open in another process ({holder}).")

    @property
    def session_start(self) -> SessionStart:
        """How ``run()`` starts: a new conversation (the default), the newest
        stored one no other process has open, or one by id
        (``cli/session_args.py``).

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
        <id>`` (``cli/session_args.py``) and whatever ``add_arguments`` adds,
        builds the app with ``from_args``, sets its ``session_start``, and runs it.
        The status is 1 when ``--resume <id>`` cannot be opened (not stored,
        or the store failed) or the backend keeps no sessions, 130 after a
        Ctrl+C before the prompt, else 0.
        """
        command = _command_name()
        parser = SessionArgumentParser(prog=command)
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
    def from_args(cls, args: "argparse.Namespace") -> Self:
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
        cancel is re-raised. A new conversation on another backend
        interrupted once that backend runs (while the old one is released,
        which goes on) is too late to undo: the change stands and is reported
        as usual before the cancel is re-raised (``/settings`` saves it), so
        the settings name the backend that runs.

        Each change is set on the settings object (``update_setting``; the
        model pickers and the thinking effort through their own setters) and
        takes effect as ``settings_persistence`` declares for its field:

        1. ``REBUILD_KEYS`` (read when the workflow builds its agents and
           their models: the models, the summary thresholds, the result cap,
           the retries and request timeout) reinitialize the workflow in
           place, on the backend it runs;
        2. ``NEW_CONVERSATION_KEYS`` (``orchestrator``: the backend of new
           conversations; a conversation never changes backend) start a new
           conversation on the new backend at once when the open one holds
           nothing yet (``ConversationCore.fresh``: it began in this run and
           no turn ran in it), as ``/new <backend>`` does, and the line of
           the new conversation names its backend; else they are saved, and
           their line says "(applies to the next /new)": nothing is rebuilt;
        3. ``RESTART_KEYS`` (read only as the app starts: ``session_store``,
           ``log_level``, ``log_format``) are applied and saved, and their
           line says "(takes effect at the next start)";
        4. any other key (``LIVE_KEYS``, or an app's own) applies at once:
           what reads it reads the settings each time.

        A reinitialization keeps the sessions, as a model switch does. One
        asked for while the workflow is still starting waits for that start
        first, as a turn does ("Waiting for initialization...": the start
        resolved its models from the settings before the change), and so
        does a change of ``session_store`` or ``orchestrator``, which that
        start reads; Ctrl+C
        stops the wait, re-raised, and changes nothing. When that start
        fails, the change is applied as for any workflow that is not
        running: nothing is rebuilt, and the next start uses it. A change of
        the orchestrator or ``session_store`` made while the workflow is down
        (whose next start builds on the new values) holds the conversation
        where that start will run it (``_hold_it_at``), and is refused, every
        field restored, when another process has it open there. A
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

        # When each change takes effect (settings_persistence): a key the
        # workflow reads as it builds its agents and models rebuilds it; the
        # backend of new conversations starts one on it, or waits for /new;
        # one read only as the app starts is said to take effect at the next
        # start; the rest apply at once, read where they are used.
        needs_reinit = any(_ui_key_target_field(key) in REBUILD_KEYS for key in changes)
        at_next_start = {key for key in changes if _ui_key_target_field(key) in RESTART_KEYS}

        read_by_the_start = needs_reinit or any(
            _ui_key_target_field(key) in _READ_BY_THE_START for key in changes
        )

        controller = self._workflow_controller
        if read_by_the_start and controller.state is WorkflowState.INITIALIZING:
            # The start in progress read the settings as they are (its models
            # first, its session store too): wait for it, as a turn does, so
            # the rebuild below runs on the new values and the session store
            # is not taken half way. Ctrl+C stops the wait before anything
            # changed. A start that fails leaves the workflow down, and the
            # change is applied for the next start, as for any failed workflow.
            async with _says_so_when_stopped(self.session):
                await controller.ensure_initialized(self.session)

        new_model = changes.get("model")
        snapshot = _snapshot_settings(self._settings)
        # The backend the conversation runs on, before the change.
        backend_before = controller.backend if "orchestrator" in changes else None
        was_langgraph = _runs_langgraph(self._settings)
        theme_before = self._settings.ui_theme

        for key, value in changes.items():
            try:
                self._settings.update_setting(key, value)
            except ValueError as e:
                _restore_settings(self._settings, snapshot)
                label = _setting_label(key, self._settings)
                self.session.add_error(f"Failed to set {label}: {e}")
                return False

        # The backend new conversations run on, and whether it is another than
        # the open conversation's. A conversation never changes backend once
        # anything of it is stored (ConversationCore.fresh), read under the
        # turn lock: after a startup resume or an app's task that holds it.
        # With nothing stored, a running workflow leaves it for a new
        # conversation there at once (no rebuild first: the new manager is
        # built on the new values), and a workflow that is down, whose next
        # start builds afresh, has that start run it there (below); else the
        # change waits for /new, and the conversation stays where it is.
        target = orchestrator_of(self._settings).value
        if "orchestrator" in changes:
            try:  # no backend that cannot run is taken, for now or the next /new
                check_backend(target)
            except ImportError as error:
                _restore_settings(self._settings, snapshot)
                label = _setting_label("orchestrator", self._settings)
                self.session.add_error(f"Failed to set {label}: {_first_line(error)}")
                return False
        moves_backend = "orchestrator" in changes and target != backend_before
        movable = False
        if moves_backend:
            try:
                async with self._turn_lock:
                    movable = self._conversation.fresh
            except asyncio.CancelledError:  # waiting for it: nothing changes
                _restore_settings(self._settings, snapshot)
                raise
        starts_anew = movable and controller.is_ready
        moves_here = movable and not controller.is_ready and controller.next_start_builds
        waits_for_new = {
            key for key in changes if _ui_key_target_field(key) in NEW_CONVERSATION_KEYS
        }
        at_next_new = waits_for_new if moves_backend and not (starts_anew or moves_here) else set()

        # The workflow is down, its next start builds afresh, and nothing of
        # the conversation is stored: that start runs it on the new backend,
        # where it is held (from its first write, when nothing holds it yet).
        # Decided again under the turn lock, which the move waits for: what
        # it finds then is decided as for a running workflow.
        if moves_here:
            place = dataclasses.replace(self._place(), backend=target)
            try:
                moved = await self._move_here(place)
            except asyncio.CancelledError:  # it stays where it was
                _restore_settings(self._settings, snapshot)
                raise
            except Exception as error:
                _restore_settings(self._settings, snapshot)
                self.session.add_error(
                    f"Could not set Orchestrator: {str(error) or type(error).__name__}. "
                    "Settings were not changed."
                )
                return False
            if moved == _STAYS:  # a turn wrote to it meanwhile, say: it stays
                at_next_new = waits_for_new
            elif moved == _STARTS_ANEW:  # the workflow came up meanwhile
                starts_anew = True
            elif not moved.made:  # open there in another process: nothing changes
                _restore_settings(self._settings, snapshot)
                self.session.add_error(
                    f"Could not set Orchestrator: conversation '{self._session_id}' is open "
                    f"in another process ({moved.holder}) on {place.backend} ({place.store}). "
                    "Settings were not changed."
                )
                return False

        # Reinitialize workflow if needed
        switched: list[str] = []
        interrupted: asyncio.CancelledError | None = None
        if needs_reinit and controller.is_ready and not starts_anew:
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
                # A failed reinitialize leaves the manager down, holding the
                # new model. Bring it back on the model it had.
                if not controller.is_ready:
                    try:
                        await controller.reinitialize(model=previous_model)
                    except Exception as restore_error:
                        self.session.add_error(
                            f"Could not restart the workflow on {previous_model}: "
                            f"{restore_error}"
                        )
                return False
            switched = _provider_switches(previous_models, getattr(previous, "agent_models", {}))
            _show_new_context(controller, self.session)

        if starts_anew:
            try:
                # The app's own /new (an override applies), on the settings'
                # orchestrator, which names the new backend now.
                await self.start_new_session()
            except asyncio.CancelledError as cancel:
                if controller.backend != target:
                    # Stopped before the new backend ran: nothing of the Save
                    # stays ("Operation cancelled...", as /new says it).
                    _restore_settings(self._settings, snapshot)
                    raise
                # It runs, and its conversation is open: too late to undo. The
                # change is reported (and saved), then the cancel goes on.
                interrupted = cancel
            except Exception as error:
                _restore_settings(self._settings, snapshot)
                reason = _first_line(error)
                self.session.add_error(
                    f"Could not start a new conversation on {target}: {reason}. "
                    "Settings were not changed."
                )
                return False

        # Report changes
        for key, value in changes.items():
            line = f"{_setting_label(key, self._settings)}: {value}"
            if key in at_next_start:
                line += f" {_AT_NEXT_START}"
            elif key in at_next_new:
                line += f" {_AT_NEXT_NEW}"
            self.session.add_success(line)
        for notice in switched:
            self.session.add_message("system", notice)
        if not was_langgraph and not starts_anew:  # a switch to it (the new
            _warn_if_langgraph(self._settings, self.session)  # conversation said it)
        if self._settings.ui_theme != theme_before:
            # Repaint: text printed before would keep the old theme's colors.
            # The transcript is printed again, the lines above included.
            theme = self.get_theme(self._settings.ui_theme)
            self.session.set_theme(theme, repaint=True)
        if interrupted is not None:
            raise interrupted
        return True

    async def _move_here(self, place: Place) -> Switch | str:
        """Have the next start run the conversation on ``place``'s backend.

        Decided again under the turn lock, which it waits for (an app's task
        may hold it, and run a turn or start the workflow meanwhile): it
        moves only while the workflow is down, its next start builds a
        manager afresh, and nothing of the conversation is stored
        (``ConversationCore.fresh``): one with anything stored never changes
        backend. Else it returns what it found, decided as for a running
        workflow: ``_STAYS`` (something of it is stored, or the next start
        revives a manager kept on its backend: the change waits for
        ``/new``) or ``_STARTS_ANEW`` (the workflow came up, nothing of it
        stored: the caller starts a new conversation there, as on a running
        workflow; this one stays where it ran).

        One this run made is held from its first write, so mostly there is
        no hold to move; when one holds it (a line typed while the workflow
        was down), its lock is taken there. The controller is told the
        backend its next start builds (``switch_backend``), then the switch
        to the same id there is made at once (no workflow runs: no await),
        letting go of where it was held. A cancel while it waits (for the
        turn lock, or for the controller) leaves it where it was, on the
        backend it had.

        Returns:
            How the switch ended: made, or refused (another process has it
            open there, and nothing changed); else ``_STAYS`` or
            ``_STARTS_ANEW`` (nothing changed).
        """
        core, controller = self._conversation, self._workflow_controller
        async with self._turn_lock:
            if not core.fresh:
                return _STAYS
            if controller.is_ready:
                return _STARTS_ANEW
            if not controller.next_start_builds:
                return _STAYS
            if core.deferred:  # nothing holds it yet: none to move
                hold: Hold | int = Hold(self._session_id, place, None)
            else:
                hold = core.acquire(self._session_id, place)
            if isinstance(hold, int):
                return Switch(made=False, holder=hold)
            try:
                await controller.switch_backend(place.backend)
            except BaseException:
                core.abandon(hold)
                raise
            return await core.switch(
                self._session_id, self._switch_parts(), place=place, propose=False, hold=hold
            )

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
        from agentic_cli.cli.memory_command import MemoryCommand
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
        self.command_registry.register(MemoryCommand())

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
                # A command that raises is a bug: the line says what, the log
                # which command and its type, and its traceback at debug. At
                # info: a warning would print above the prompt (the default
                # level), beside the line the user already sees.
                logger.info(
                    "command_failed", command=command.name,
                    error_type=type(e).__name__, error=str(e),
                )
                logger.debug("command_failed", command=command.name, exc_info=e)
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
        # Delegate to message processor (one turn at a time). Echoed as its
        # turn starts: a line typed while the conversation --continue/--resume
        # asked for opens shows under its replay; one Ctrl+C stops while it
        # waits is shown and said cancelled (it never ran), not dropped unseen.
        try:
            await self._turn_lock.acquire()
        except asyncio.CancelledError:
            self.session.add_message("user", message)
            self.session.add_message("system", CANCELLED_LINE)
            raise
        try:
            self.session.add_message("user", message)
            if not self._held_to_write():
                return
            self._turned = True  # the conversation now holds this message
            reached = False  # the turn reached a ready workflow, which stores it
            try:
                result = await self._message_processor.process(
                    message=message,
                    workflow_controller=self._workflow_controller,
                    ui=self.session,
                    settings=self._settings,
                    usage_tracker=self._usage_tracker,
                    session_id=self._session_id,
                )
                reached = getattr(result, "status", None) is not TurnStatus.UNAVAILABLE
            except BaseException:
                reached = self._workflow_controller.is_ready  # it may have begun
                raise
            finally:
                if reached:
                    self._conversation.written = True
        finally:
            self._turn_lock.release()

        # At the turn boundary, auto-resume any finished background jobs that
        # opted in (gated by the job_auto_resume setting).
        if getattr(self._settings, "job_auto_resume", False):
            await self._resume_on_its_own()

    async def _resume_on_its_own(self) -> None:
        """``resume_finished_jobs``, as the CLI runs it unasked (``_TELLING_ON_ITS_OWN``)."""
        token = _TELLING_ON_ITS_OWN.set(True)
        try:
            await self.resume_finished_jobs()
        finally:
            _TELLING_ON_ITS_OWN.reset(token)

    async def resume_finished_jobs(self) -> int:
        """Resume the agent for each finished, resume-flagged background job.

        Only the jobs that wait for the current conversation
        (``cli/waiting_jobs.py``: its own, and those with no conversation or
        whose conversation is gone, which are told it is unavailable), as the
        status bar's ``↻N`` and the line an opened conversation says count
        them: another conversation's job waits until that conversation is
        opened (``resume_session``, or at startup), instead of answering in
        this one's transcript. Each resume is a
        serialized turn (via the turn lock) so it never overlaps a user turn
        or another resume. Used at turn boundaries (auto, gated), when a
        conversation is opened (gated) and by ``/jobs resume`` (explicit,
        ungated).

        Delivery follows the job's resume lifecycle: the job is *claimed*
        (pending → resuming) once the turn lock is held, the workflow found
        ready and the conversation looked up, and before the turn runs, so a
        crash cannot silently re-deliver it, and only recorded
        delivered once the turn actually completed — a failed or cancelled
        resume is recorded as such instead of being dropped. The check above
        is made again under the lock (``_deliver_resume``): the conversation
        may have changed while the delivery waited for it.

        Returns:
            The number of jobs picked up: a resume turn run for it, or, for
            one that could not be told now (its conversation's lookup failed,
            before any claim), the failure said. Whether each one was
            delivered is recorded on the job and reported by ``/jobs``; a
            failed delivery is surfaced to the user by the turn itself.
        """
        if not self._workflow_controller.is_ready:
            return 0
        jm = getattr(self._workflow_controller.workflow, "job_manager", None)
        if jm is None:
            return 0

        attempted = 0
        # Ctrl+C stops these waits with no box open: they say so themselves.
        async with _says_so_when_stopped(self.session):
            records = await asyncio.to_thread(jm.awaiting_resume)
        for record in records:
            if not await self._told_here(record):
                continue  # told when its conversation is opened
            if await self._deliver_resume(jm, record) is not None:
                attempted += 1
        return attempted

    def _here(self) -> Here:
        """The current conversation, as ``cli/waiting_jobs.py`` names it: (user, session)."""
        return getattr(self._settings, "default_user", None), self._session_id

    async def _told_here(self, record: Any) -> bool:
        """Whether ``record``, a job waiting for delivery, is told in this conversation.

        A Ctrl+C during the conversation's lookup says the request stopped.
        """
        workflow = self._workflow_controller.workflow
        async with _says_so_when_stopped(self.session):
            return await told_here(
                record, self._here(), getattr(workflow, "can_resume", None)
            )

    async def _deliver_resume(self, jm, record) -> bool | None:
        """Claim one job and run its resume turn, and close out its transition.

        Under the turn lock the job is looked at again: another conversation
        may have become current while this waited for it (an app's own task
        that called ``resume_session`` or ``start_new_session`` took the lock
        first). A job that now waits for its conversation is not claimed, nor
        told here, where ``ctx`` and the task list are another conversation's:
        it stays pending until its conversation is opened. The claim is durable, so an interrupted
        delivery is recoverable and two coordinators cannot both deliver the
        same result.

        The job is claimed only once its result can reach the agent:
        ``process_resume`` calls ``claim`` after it found the workflow ready
        and looked the conversation up, before it says anything. A delivery
        stopped before then (the workflow not started, Ctrl+C or a failure
        during the lookup) claims nothing, and the job stays pending for a
        later one. A failure is said: each time the user asks (``/jobs
        resume``), and once per job and conversation when the CLI tells
        jobs on its own (``_TELLING_ON_ITS_OWN``: a turn's end, a
        conversation opened), not at every turn.

        Every exit path closes a claim, so no record can be left ``RESUMING``
        in this process: a normal outcome records delivered/failed, an
        exception records failed with the reason, and a cancellation records
        failed before re-raising (cancellation still propagates): the turn
        may have written the job's notice to the conversation already.

        The job manager is called on a worker thread (``_claim_delivery``,
        ``complete_resume``): it takes the jobs directory's lock, which
        another CLI of the app may hold. Each call is awaited before the next
        step, and one a second Ctrl+C stops waiting for still finishes.

        Args:
            jm: The JobManager.
            record: The job record.

        Returns:
            True if the result was delivered to the agent; False if it was
            not (claimed and not delivered, or a failure before the claim,
            said); None if it was not claimed and nothing was said (it
            waits for its conversation, the workflow was not ready, or
            another coordinator has it).
        """
        claimed = False

        async def claim() -> bool:
            nonlocal claimed
            claimed = await _claim_delivery(jm, record.job_id)
            if claimed and getattr(record, "session_id", None) == self._session_id:
                self._turned = True  # the resume adds to this conversation
                # Defensive: a job of this conversation was submitted by a
                # turn, which stored it already (fresh was False before).
                self._conversation.written = True
            return claimed

        try:
            async with self._turn_lock:
                if not await self._told_here(record):
                    return None
                if not self._held_to_write():  # its notice is written here
                    return None
                result = await self._message_processor.process_resume(
                    record=record,
                    workflow_controller=self._workflow_controller,
                    ui=self.session,
                    settings=self._settings,
                    usage_tracker=self._usage_tracker,
                    claim=claim,
                )
        except asyncio.CancelledError:
            if claimed:
                logger.info("job_resume_cancelled", job_id=record.job_id)
                await asyncio.to_thread(
                    jm.complete_resume, record.job_id, delivered=False,
                    error="resume cancelled",
                )
            raise
        except Exception as exc:  # noqa: BLE001 - the claim must always close
            if not claimed:
                # Before the claim (the conversation's lookup, say): nothing
                # reached the agent, and a later delivery tells it.
                logger.warning(
                    "job_resume_unclaimed_failure", job_id=record.job_id, error=str(exc)
                )
                said = (record.job_id, self._session_id)
                if not (_TELLING_ON_ITS_OWN.get() and said in self._said_not_told):
                    self.session.add_error(
                        f"Background job '{record.name}' could not be resumed now: {exc}"
                    )
                    self._said_not_told = self._said_not_told | {said}
                return False
            await asyncio.to_thread(
                jm.complete_resume, record.job_id, delivered=False, error=str(exc)
            )
            logger.warning(
                "job_resume_raised", job_id=record.job_id, error=str(exc)
            )
            self.session.add_error(
                f"Background job '{record.name}' could not be resumed: {exc}"
            )
            return False

        if not claimed:
            return None  # stopped before its result could reach the agent
        await asyncio.to_thread(
            jm.complete_resume, record.job_id, delivered=result.delivered,
            error=result.error,
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
            if not self._held_to_write():
                return None
            return await self._message_processor.process_compact(
                focus,
                workflow_controller=self._workflow_controller,
                ui=self.session,
                settings=self._settings,
                usage_tracker=self._usage_tracker,
                session_id=self._session_id,
            )

    async def start_new_session(self, backend: str | None = None) -> str:
        """Start a new conversation in this process (``/new``), and return its id.

        Waits for a running turn or job resume (the turn lock), then switches
        as every change of conversation does (``ConversationCore.switch``):

        - with ``auto_extract_session_facts`` on, proposes memories from the
          conversation being left when a turn ran in it since it was opened,
          as at exit, under a status line ("Looking for memories to
          propose…"; a failure never stops it);
        - adopts a new session id once the workflow has taken it
          (``workflow.load_session``), and every later turn targets it; this
          process holds it, and lets the one left go;
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
        conversation that started it (a finished job waits until that
        conversation is opened).

        A conversation runs on one backend for its whole life (each keeps
        its history in its own store), so the new one runs on ``backend``,
        by default the settings' ``orchestrator``, the backend of new
        conversations. On another backend than the one running, the
        memories of the conversation left are proposed first (its backend
        holds it), then a manager of ``backend`` is started
        (``WorkflowController.switch_backend``: a failure there changes
        nothing), and the line names it: ``"New conversation <new> on
        deepagents."``, with ``" Previous: <old>, kept by adk."`` when the
        conversation left is stored; one on LangGraph is told it is
        deprecated. A Ctrl+C once the new backend runs (the old one is being
        released) is too late to undo: the new conversation is opened there
        first, then the cancel goes on. The backend chosen is not saved to
        any settings file.

        Args:
            backend: The backend the new conversation runs on: ``adk``,
                ``deepagents`` or ``langgraph``.

        Returns:
            The new session id (12 hex characters).

        Raises:
            ValueError: ``backend`` is no orchestrator.
            ImportError: Its extra is not installed (the message says how to
                install it); nothing changed.
        """
        target = check_backend(backend).value if backend is not None else None
        async with _says_so_when_stopped(self.session):
            async with self._turn_lock:
                return await self._new_conversation(target)

    async def _new_conversation(self, backend: str | None) -> str:
        """``start_new_session``'s switch, under the turn lock: on ``backend``
        (the settings' orchestrator when None)."""
        import uuid

        controller = self._workflow_controller
        old, old_backend = self._session_id, controller.backend
        target = backend or orchestrator_of(self._settings).value
        moving = target != old_backend
        if moving and backend is None:
            check_backend(target)  # before anything changes
        parts = self._switch_parts()
        stored = parts.workflow is not None and await self._is_stored(parts.workflow, old)
        place, interrupted = None, None
        if moving:
            if self._turned:  # proposed while the backend that holds them runs
                await self._saving_facts(old)
                self._turned = False
            try:
                await controller.switch_backend(target)
            except asyncio.CancelledError as cancel:
                if controller.backend != target:
                    raise  # it never ran: nothing changed
                interrupted = cancel  # it runs: its conversation is opened, then the cancel
            parts = self._switch_parts()
            place = dataclasses.replace(self._place(), backend=target)  # the run's store
        new = uuid.uuid4().hex[:12]

        def opened() -> None:
            """Say where the app is now."""
            if moving:
                line = f"New conversation {new} on {target}."
                if stored:
                    line += f" Previous: {old}, kept by {old_backend}."
            else:
                line = f"New conversation {new}."
                if stored:
                    line += f" Previous: {old}, listed in /sessions."
            self.session.add_message("system", line)
            if moving and target == OrchestratorType.LANGGRAPH.value:
                self.session.add_warning(_LANGGRAPH_DEPRECATION)

        try:
            switched = await self._conversation.switch(
                new, parts, place=place, propose=not moving, generated=True
            )
        except asyncio.CancelledError:
            if self._session_id == new:
                # On another backend the switch is made before the load, which
                # the cancel stopped: the app is there, and says so.
                opened()
                _show_new_context(controller, self.session)
            raise
        if not switched.made:  # a new id: never open elsewhere in practice
            raise RuntimeError(f"'{new}' is open in another process ({switched.holder})")
        logger.info("new_session_started", session_id=new, previous=old, backend=target)
        opened()
        if interrupted is not None:
            raise interrupted
        return new

    async def resume_session(self, session_id: str) -> bool:
        """Switch to the stored conversation ``session_id`` (``/resume``), and show where it left off.

        Waits for a running turn or job resume (the turn lock), then reads the
        conversation first, its row (``workflow.session_info``) and its
        messages, and makes what it will show: one that is not stored, or
        cannot be read, changes nothing. Then, as ``start_new_session`` does,
        it proposes memories from the conversation being left (when
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
            found.", also when it is deleted while it opens), or could not
            be read ("Could not open '<id>': ...").
        """
        async with _says_so_when_stopped(self.session):
            opened = await self._open_conversation(session_id)
        if opened:
            await self._tell_its_finished_jobs()
        return opened

    async def _resume(self, session_id: str) -> bool:
        """``resume_session`` without its "Operation cancelled..." line: at
        startup (``_open_on_startup``, which holds the turn lock) a Ctrl+C
        ends the app, not the switch. The lock the start took on
        ``session_id`` before the prompt is the one the switch keeps.

        Its finished jobs are named, not run: this runs outside the input
        handler, where Ctrl+C cannot stop a turn. They are told at the first
        turn boundary (``job_auto_resume``) or by ``/jobs resume``.
        """
        hold, self._pending_hold = self._pending_hold, None
        if hold is not None and hold.session_id != session_id:
            self._conversation.abandon(hold)
            hold = None
        opened = await self._open(session_id, hold)
        if opened:
            await self._name_its_waiting_jobs()
        return opened

    async def _name_its_waiting_jobs(self) -> None:
        """Say how many finished jobs wait for this conversation, and how to tell them.

        Counted as the status bar's ``↻N`` and ``/jobs resume`` count them
        (``cli/waiting_jobs.py``). The job manager is asked on a worker
        thread: it takes the jobs directory's lock, which another CLI of the
        app may hold.
        """
        workflow = self._workflow_controller.workflow
        jm = getattr(workflow, "job_manager", None)
        if jm is None:
            return
        waiting = await waiting_here(
            await asyncio.to_thread(jm.awaiting_resume), self._here(),
            getattr(workflow, "can_resume", None),
        )
        if len(waiting) == 1:
            self.session.add_message(
                "system", "A finished job waits for this conversation: /jobs resume tells it now."
            )
        elif waiting:
            self.session.add_message(
                "system",
                f"{len(waiting)} finished jobs wait for this conversation: "
                "/jobs resume tells them now.",
            )

    async def _tell_its_finished_jobs(self) -> None:
        """Once a conversation is open, its finished jobs resume (``job_auto_resume``).

        After the switch, outside the turn lock: each resume is a turn of its own.
        """
        if getattr(self._settings, "job_auto_resume", False):
            await self._resume_on_its_own()

    async def _open_conversation(self, session_id: str) -> bool:
        """The switch to a stored conversation, under the turn lock (``resume_session``)."""
        async with self._turn_lock:
            return await self._open(session_id)

    async def _open(self, session_id: str, hold: Hold | None = None) -> bool:
        """Open the stored conversation ``session_id``: its row and messages
        are read first, then the switch (``ConversationCore.switch``).

        The caller holds the turn lock. ``hold`` is ``session_id``'s lock,
        taken already (at startup, before the prompt): it is kept once the
        switch is made, and let go on every other way out.
        """
        switching = False
        try:
            if session_id == self._session_id:
                self.session.add_message("system", "Already in this conversation.")
                return False
            parts = self._switch_parts()
            workflow = parts.workflow
            if workflow is None:
                self.session.add_error(
                    f"Could not open '{session_id}': the workflow is not ready."
                )
                return False
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
            switching = True  # the switch keeps the hold or lets it go
            try:
                switched = await self._conversation.switch(
                    session_id, parts, stored=True, hold=hold
                )
            except Exception as error:  # noqa: BLE001 - the id is not adopted
                if self._session_id == session_id:
                    raise  # opened: what failed after it is the caller's to say
                logger.debug("session_open_failed", session_id=session_id, exc_info=True)
                reason = str(error) or type(error).__name__
                self.session.add_error(f"Could not open '{session_id}': {reason}")
                return False
        finally:
            if hold is not None and not switching:
                self._conversation.abandon(hold)
        if switched.holder is not None:
            self.session.add_error(
                f"Could not open '{session_id}': it is open in another process "
                f"({switched.holder})."
            )
            return False
        if switched.gone:  # deleted since it was read (another process's /sessions --delete)
            self.session.add_error(f"No conversation '{session_id}' found.")
            return False
        logger.info("session_resumed", session_id=session_id, previous=old)
        self.session.add_message("system", line)
        show_replay(self.session, replay)
        return True

    def _held_to_write(self) -> bool:
        """Hold the conversation before it is written, when its id was made by
        this run (``ConversationCore.hold_to_write``); the caller holds the
        turn lock. False, said, when another process has it open (a collision
        of random ids): the write must not happen."""
        holder = self._conversation.hold_to_write()
        if holder is None:
            return True
        self.session.add_error(
            f"Conversation '{self._session_id}' is open in another process ({holder})."
        )
        return False

    def _switch_parts(self, *, drawn: bool = True) -> SwitchParts:
        """What a switch acts on besides the conversation, as the app has it now.

        Args:
            drawn: Draw the status line after the switch (not before the
                prompt: the start draws it).
        """
        controller = self._workflow_controller
        self._place()  # known before anything is held (an app built with __new__)

        def status() -> None:
            if drawn:  # the trigger as a /settings rebuild shows it; no prompt yet
                _show_new_context(controller, self.session)

        return SwitchParts(
            workflow=controller.workflow if controller.is_ready else None,
            processor=self._message_processor,
            tracker=self._usage_tracker,
            propose=self._saving_facts,
            status=status,
        )

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
            note = "" if self._place().locked else _MEMORY_NOTE
            self.session.add_message("system", f"No other conversations to resume.{note}")
            return None
        # Drawn where the settings say now: /settings may have changed the
        # placement since the session started with its own.
        dialog, ids = picker(
            rows, datetime.now(timezone.utc), placement=self._settings.dialog_placement
        )
        picked = await self.session.show_dialog(dialog)
        return None if picked is None else ids.get(picked)

    async def _saving_facts(self, session_id: str) -> None:
        """Propose memories from ``session_id``'s conversation (if enabled) under a
        status line saying so, then name the proposals waiting (``/new`` and
        ``/resume``)."""
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
        await self._name_waiting_proposals()

    async def _name_waiting_proposals(self) -> None:
        """Say how many memory proposals wait for review, if any (``/memory inbox``)."""
        from agentic_cli.cli.memory_command import proposals_line

        if not self._workflow_controller.is_ready:
            return
        store = getattr(self._workflow_controller.workflow, "memory_store", None)
        if store is None:
            return
        try:
            count = len(await asyncio.to_thread(store.proposals))
        except Exception:  # a line, never a failure
            logger.debug("memory_proposals_unreadable", exc_info=True)
            return
        line = proposals_line(count)
        if line:
            self.session.add_message("system", line)

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
        """Propose memories from ``session_id``'s conversation (if enabled).

        ``session_id`` defaults to the current conversation (at exit).
        Gated by ``auto_extract_session_facts``; a no-op when the workflow is
        not ready. Never raises: a conversation must be left whatever its
        memories. Runs outside any turn, so its model call reaches no usage
        counter.
        """
        if not getattr(self._settings, "auto_extract_session_facts", False):
            return
        if not self._workflow_controller.is_ready:
            return
        workflow = self._workflow_controller.workflow
        session_id = session_id or self._session_id
        try:
            proposals = await workflow.on_session_end(
                session=workflow.session_ref(session_id, self._settings.default_user)
            )
        except Exception:
            logger.debug("session_fact_extraction_failed", exc_info=True)
            return
        if proposals:
            logger.info("memory_proposals_filed", count=len(proposals), session_id=session_id)

    async def _adopt_session_on_startup(self) -> bool:
        """Adopt this run's conversation before the prompt; False when the app must not start.

        Runs for every startup: a fresh auto-generated id must be adopted too,
        otherwise turns fall back to the manager's ``default_session`` and
        unnamed runs silently share one conversation. The run's own
        conversation is held and loaded through the switch every change of
        conversation makes (``ConversationCore.switch``), even when the
        workflow did not start: the turns write to it once ``/settings``
        fixes the start, so no other process may open it meanwhile. By
        ``session_start``:

        - ``"continue"`` finds the newest stored conversation no other
          process has open (``_newest_free``, among the newest ``PICK_LIMIT``
          when the newest is held), which ``_open_on_startup``
          resumes once the prompt is up, or says there is none ("No
          conversation to continue; starting a new one.");
        - ``"resume"`` checks its conversation is stored first: one that is
          not is refused ("No conversation '<id>' found: <app> --continue
          opens the newest, /resume in the app picks one."), ``exit_status``
          is 1, and False is returned;
        - ``"new"`` starts in the new conversation.

        The conversation ``"continue"`` or ``"resume"`` chose is held from
        here (another process asking for it is refused), beside the run's
        own, until the switch to it once the prompt is up keeps it or lets
        it go.

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
            }.get(
                mode,
                "The workflow did not start. Fix it with /settings; "
                "your next message starts it again.",
            ))
            return await self._adopt_its_own()

        workflow = self._workflow_controller.workflow
        if mode != "new" and not getattr(workflow, "supports_sessions", False):
            self.session.add_error("This app's backend keeps no sessions.")
            self.exit_status = 1
            return False
        note = "" if self._place().locked else _MEMORY_NOTE
        try:
            if mode == "resume":
                if await workflow.session_info(target) is None:
                    # On the memory store there is nothing to open instead.
                    run = (
                        f": {self._command} --continue opens the newest, "
                        "/resume in the app picks one."
                        if self._command and not note else "."
                    )
                    self.session.add_error(f"No conversation '{target}' found{run}{note}")
                    self.exit_status = 1
                    return False
                taken = self._conversation.acquire(target, self._place())
                if isinstance(taken, int):
                    self.session.add_error(
                        f"Conversation '{target}' is open in another process ({taken})."
                    )
                    self.exit_status = 1
                    return False
                self._pending_resume, self._pending_hold = target, taken
            elif mode == "continue":
                # A listing can read whole sessions (ADK's from before the
                # title keys): the newest alone, unless it is held.
                self._pending_hold, passed_over = self._newest_free(
                    await workflow.list_sessions(limit=1)
                )
                if passed_over:
                    self._pending_hold, passed_over = self._newest_free(
                        await workflow.list_sessions(limit=PICK_LIMIT)
                    )
                if self._pending_hold is not None:
                    self._pending_resume = self._pending_hold.session_id
                if passed_over:
                    self.session.add_warning(_passed_over_line(
                        passed_over, continuing=self._pending_resume
                    ))
                elif self._pending_resume is None:
                    self.session.add_message(
                        "system", f"No conversation to continue; starting a new one.{note}"
                    )
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

        return await self._adopt_its_own()

    async def _adopt_its_own(self) -> bool:
        """Load the conversation the run starts in (the switch to it, before
        the prompt: no memories to propose, no status line yet, which the
        start draws), holding an id given to the constructor; one this run
        made is in no store, so it is held from its first write
        (``ConversationCore.deferred``) and a run that writes nothing makes
        no lock file. False, status 1, when another process has a given id
        open."""
        session_id = self._session_id
        try:
            switched = await self._conversation.switch(
                session_id, self._switch_parts(drawn=False), propose=False
            )
        except Exception as error:  # noqa: BLE001 - turns name their conversation anyway
            logger.debug("session_adopt_failed", exc_info=True)
            self.session.add_warning(
                f"Cannot adopt session — {str(error) or type(error).__name__}"
            )
            return True  # made all the same: the switch is made before the load
        if not switched.made:
            if self._pending_hold is not None:
                self._conversation.abandon(self._pending_hold)
                self._pending_resume = self._pending_hold = None
            self.session.add_error(
                f"Conversation '{session_id}' is open in another process ({switched.holder})."
            )
            self.exit_status = 1
            return False
        if switched.loaded:
            self.session.add_success(f"Session '{session_id}' resumed.")
        elif switched.loaded is False and self._resume_requested:
            # An explicit session_id that didn't exist yet: tell the user it's new.
            self.session.add_message("system", f"New session '{session_id}'.")
        return True

    def _newest_free(self, rows: "list[SessionInfo]") -> tuple[Hold | None, list[tuple[str, int]]]:
        """The newest of ``rows`` no other process has open, now held here
        (beside the conversation held; the switch to it keeps it or lets it go).

        Returns:
            Its hold (None when every one is open elsewhere, or there is none),
            and the newer ones passed over, each with the process holding it.
        """
        passed_over: list[tuple[str, int]] = []
        for info in rows:
            taken = self._conversation.acquire(info.session_id, self._place())
            if not isinstance(taken, int):
                return taken, passed_over
            passed_over.append((info.session_id, taken))
        return None, passed_over

    async def _open_on_startup(self) -> None:
        """Once the prompt is up: name the memory proposals waiting, then resume
        what ``--continue``/``--resume <id>`` chose.

        A replay reads best under the welcome banner, so this waits until the
        session runs. It runs outside
        the input handler, so a Ctrl+C here ends the app, as at an idle prompt
        (its resume, cancelled on the way out, says nothing of a cancel). A
        failure is said and logged: the new conversation goes on.

        With a conversation to resume it holds the turn lock from its start,
        which comes before the prompt can take a line (``run()`` starts it
        first): a turn typed or pasted ahead waits for the resume and runs in
        the conversation asked for, not in the run's new one.
        """
        target = self._pending_resume
        async with self._turn_lock if target is not None else contextlib.nullcontext():
            while not self.session.app.is_running:
                await asyncio.sleep(0.02)
            await self._name_waiting_proposals()
            if target is None:
                return
            try:
                await self._resume(target)  # one not opened stays in the run's own
            except Exception as error:  # noqa: BLE001 - the new conversation goes on
                logger.debug("startup_open_failed", session_id=target, exc_info=True)
                reason = str(error) or type(error).__name__
                if self._session_id == target:  # opened: the next turn goes there
                    self.session.add_error(
                        f"Opened '{target}', but could not show where it left off: {reason}"
                    )
                else:
                    self.session.add_error(f"Could not open '{target}': {reason}")

    def _place(self) -> Place:
        """Where the conversation is kept: its backend and store, as the
        workflow runs them.

        Recorded when the workflow starts (``run()``), for a new conversation
        on another backend, and when ``/settings`` changes the orchestrator
        while the workflow is down and nothing of the conversation is stored
        (``_move_here``); never read from the settings in between. The store
        is the run's for its whole life (a change of ``session_store`` takes
        effect at the next app start). An app built with ``__new__`` reads
        its settings on first use.
        """
        core = self._conversation
        if core.place is None:
            core.place = Place.of(self._settings)
        return core.place

    def _conversation_key(self, session_id: str) -> tuple:
        """What names a stored conversation for its lock: the same id in
        another backend or store is another conversation."""
        return self._place().key(session_id)

    @contextmanager
    def conversation_held(self, session_id: str) -> Iterator[int | None]:
        """Hold ``session_id``'s conversation for the block, beside the one
        this process is in, so no other process opens it meanwhile
        (``/sessions --delete``). When the block ends, the status bar's
        job monitor forgets what it knew of jobs' conversations: one may be
        gone, and its jobs told here.

        Yields:
            None while it is held (on the memory store nothing is held, nor
            needed); else the process id that has it open (0 when unknown),
            and nothing is held.
        """
        taken = self._conversation.acquire(session_id, self._place())
        if isinstance(taken, int):
            yield taken
            return
        try:
            yield None
        finally:
            self._conversation.abandon(taken)
            if self._job_monitor is not None:
                self._job_monitor.forget_conversations()

    def _hold(self, session_id: str) -> int | None:
        """Hold ``session_id``'s conversation for this process, letting go of the one before.

        Two CLIs on one conversation lose a turn (Deep Agents keeps one of two
        runs from the same checkpoint; ADK fails the second writer), so one
        another process holds is refused. Nothing is held on the memory store,
        whose conversations are this process's alone.

        Returns:
            None once it is held; else the process id of the one holding it
            (0 when unknown), and the conversation held before stays held.
        """
        return self._conversation.hold(session_id, self._place())

    def _let_go(self) -> None:
        """At exit: let go of the conversation held, and of one the start held
        for ``--continue``/``--resume`` and did not open (an exit before it)."""
        if self._pending_hold is not None:
            self._conversation.abandon(self._pending_hold)
            self._pending_hold = None
        self._conversation.release()

    def _resume_hint(self) -> str:
        """The exit line that names how to come back to the current conversation.

        ``--resume`` (and ``/resume``) open a conversation on the settings'
        orchestrator, the backend of new conversations: one that runs on
        another (``/new <backend>``) is found once the orchestrator names
        its backend, and the line says so.
        """
        if self._command:
            hint = f"Resume this conversation with: {self._command} --resume {self._session_id}"
        else:
            hint = f"Resume this conversation with /resume {self._session_id}"
        backend = self._place().backend
        if backend != orchestrator_of(self._settings).value:
            hint += f" (set orchestrator to {backend} first: it runs there)"
        return hint

    async def _extract_session_facts_on_exit(self) -> None:
        """Propose memories from the session on exit (if enabled).

        Gated by the ``auto_extract_session_facts`` setting; a no-op when the
        workflow isn't ready or has no memory store. Never raises — memory
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
            self.session, self._workflow_controller, settings=self._settings,
            conversation=lambda: (self._settings.default_user, self._session_id),
        )
        self._job_monitor = job_monitor  # told when a conversation is deleted

        # Once, before anything waits for the workflow: a backend that cannot
        # start (its extra missing, or a key) is still deprecated.
        _warn_if_langgraph(self._settings, self.session)
        # The place the workflow starts in keys the conversation's lock.
        self._conversation.place = Place.of(self._settings)

        async with self._workflow_controller.background_init(self.session):
            # Adopt this run's session id (generated, or session_id=) so every
            # turn targets it; a fresh id starts a new durable session rather
            # than falling back to the shared 'default_session'. --resume <id>
            # of a conversation that is not stored ends here, before the prompt.
            if not await self._adopt_session_on_startup():
                logger.info("app_start_refused", exit_status=self.exit_status)
                self._let_go()
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
                    # Run the session: the prompt shows now, after the
                    # start above (which waited for the workflow).
                    with _log_lines_above_prompt(self.session):
                        await self.session.run_async()
                finally:
                    opening.cancel()
                    await asyncio.gather(opening, return_exceptions=True)

            # Propose memories from the session (if enabled) while the workflow
            # is still alive: leaving this context closes the manager, and the
            # extraction needs the live session store and an LLM call.
            await self._extract_session_facts_on_exit()
            # Named at exit when a later run can resume it: stored, not in memory.
            controller = self._workflow_controller
            resumable = (
                self._place().locked  # stored, not on the memory store
                and controller.is_ready
                and await self._is_stored(controller.workflow, self._session_id)
            )

        # No save-on-exit: durable session stores persist continuously per turn.

        logger.info("app_ending")
        self._let_go()
        if resumable:
            self.session.add_message("system", self._resume_hint())
        self.session.add_message("system", "Goodbye!")


def _first_line(error: BaseException) -> str:
    """What a line says of ``error``: its first line (a backend's missing extra
    adds the import's own error after it), or its type when it has no words."""
    return (str(error) or type(error).__name__).split("\n")[0]


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
    roles: "RoleMap", role: str, catalog: "CatalogView", providers: "Mapping[str, Provider]"
) -> str | None:
    """The reference ``role`` resolves to, or None when nothing can fill it."""
    from agentic_cli.models import ModelResolutionError

    try:
        return str(roles.resolve(role, catalog=catalog, providers=providers))
    except ModelResolutionError:
        return None


def _show_new_context(controller: WorkflowController, session: Any) -> None:
    """Show ``ctx`` against the rebuilt agents' trigger now, not after the next call.

    A reinitialize may move the main agent's trigger (a model of another
    window, ``context_summarize_at``, ``context_max_tokens``), and the last
    prompt may already be past the new one; a switch of conversation starts
    one whose trigger the workflow knows. The usage tracker takes the
    trigger and the window flag from ``workflow.context_budget()``: none when
    the manager manages no context (``KeyError``), has no such method, or
    is not ready (it failed to start; the status bar then says so). The
    status bar is then drawn again.

    Args:
        controller: The app's workflow controller, after the reinitialize.
        session: The UI session whose status bar is drawn.
    """
    tracker = controller.usage_tracker
    if tracker is not None:
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
