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
import os
import re
import shlex
import shutil
import sys
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NoReturn, Self

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory

from thinking_prompt import AppInfo, ThinkingPromptSession, ThinkingPromptStyles

from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.cli.message_processor import MessageProcessor, _says_so_when_stopped
from agentic_cli.cli.conversation import (
    NOT_KEPT,
    ConversationCore,
    Hold,
    Place,
    SwitchParts,
    TurnParts,
    telling_on_its_own,
)
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
from agentic_cli.cli.settings_ui import (
    _LANGGRAPH_DEPRECATION,
    SettingsUI,
    _first_line,
    _show_new_context,
    _warn_if_langgraph,
)
from agentic_cli.cli.stored_conversations import StoredConversations
from agentic_cli.cli.usage_tracker import UsageTracker
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings
from agentic_cli.logging import Loggers, configure_logging, log_lines_to
from agentic_cli.workflow.adk.log_filters import install_log_filters
from agentic_cli.workflow.factory import check_backend, orchestrator_of
from agentic_cli.workflow.session_readers import reader_backends
from agentic_cli.workflow.settings import OrchestratorType

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterable, Sequence

    from thinking_prompt import Field

    from agentic_cli.cli.job_monitor import JobMonitor
    from agentic_cli.settings_persistence import SettingsSaveResult
    from agentic_cli.workflow import GoogleADKWorkflowManager, EventType, WorkflowEvent
    from agentic_cli.workflow.base_manager import BaseWorkflowManager
    from agentic_cli.workflow.context import CompactResult
    from agentic_cli.workflow.sessions import SessionInfo
    from agentic_cli.workflow.config import AgentConfig

logger = Loggers.cli()


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
    (``workflow/adk/log_filters.py::install_log_filters``).
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


#: What setting ``BaseCLIApp.session_id`` warns, deprecated in 0.7.0: a setter
#: cannot run the switch, which awaits the workflow.
_SESSION_ID_SETTER_DEPRECATED = (
    "Setting BaseCLIApp.session_id is deprecated: use `await app.resume_session(id)` "
    "or `await app.start_new_session()`. The setter holds the conversation and starts "
    "its task list and usage counters over, but the workflow does not load it until "
    "the next turn names it, and the memories of the conversation left are not proposed."
)

#: Where ``/new``'s line says the conversation left is on the memory store.
_NOT_KEPT_LEFT = "not kept (session_store is memory)"


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

    #: The status bar's job monitor while ``run()`` runs (None before, and
    #: for an app built with ``__new__``).
    _job_monitor: "JobMonitor | None" = None

    #: The lock on the conversation ``--continue``/``--resume`` chose, taken
    #: before the prompt and kept by the switch to it once the prompt is up
    #: (``_open_on_startup``); None otherwise.
    _pending_hold: Hold | None = None

    #: What ``--continue`` chose before the workflow was built
    #: (``_start_on_its_backend``): the hold on the conversation it opens,
    #: and the newer ones passed over; None when it chose nothing then.
    _chosen_at_start: "tuple[Hold | None, list[tuple[str, int]]] | None" = None

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

        logger.info("app_starting", app_name=self._settings.app_name)

        # === Components ===
        self._usage_tracker = UsageTracker()
        self._workflow_controller = WorkflowController(
            agent_configs=agent_configs,
            settings=settings,
        )
        self._workflow_controller.usage_tracker = self._usage_tracker
        self._message_processor = MessageProcessor()
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
        """Build the settings dialog's fields (``cli/settings_ui.py``:
        ``SettingsUI.fields``).

        ``get_ui_setting_keys()`` says which fields to show; each is
        converted to a form field from its settings field's type annotation
        and metadata. The model pickers come from the model catalog, and the
        thinking effort's description says what it means for the default
        model. ``/settings`` builds its dialog from this method, so an
        override applies.

        Returns:
            List of thinking_prompt form fields sorted by ui_order
        """
        return SettingsUI(self).fields()

    async def save_settings(self, keys: "Iterable[str]") -> "SettingsSaveResult":
        """Save the settings ``keys`` name, split by trust.

        ``/settings`` passes the fields the user changed: only those are
        written, as picked (see ``SettingsPersistence.save``).

        Allowlisted keys go to the project config
        (./.{app_name}/settings.json). User-scoped keys go to the user config
        (~/.{app_name}/settings.json), where the loader trusts them — the
        project file may only carry allowlisted keys since P0-1. Secrets
        (API keys) are never saved. The files are read and written on a
        worker thread, off the event loop (``SettingsUI.save``).

        Returns:
            SettingsSaveResult with the written path(s)

        Raises:
            TypeError: ``keys`` is None (an override that passes its own
                ``keys=None`` default on has no keys to write).
        """
        return await SettingsUI(self).save(keys)

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
    def _turn_lock(self) -> asyncio.Lock:
        """The conversation's turn lock (``_conversation.turn_lock``): a turn,
        a job's notice, ``/compact``, ``/new`` and ``/resume`` take it."""
        return self._conversation.turn_lock

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
    def session_backend(self) -> str:
        """The backend that keeps the conversation the turns run in (``adk``,
        ``deepagents``, ...): the one the workflow runs, which a conversation
        never changes (``/new <backend>`` and ``/resume`` of another
        backend's conversation start that backend)."""
        return self._place().backend

    @property
    def session_store(self) -> str:
        """The session store the conversations of this run are kept in
        (``sqlite``, ``postgres``, ``memory``): the settings' as the run
        started, for its whole life (a change of ``session_store`` takes
        effect at the next start). On ``memory`` the app keeps one
        conversation at a time: ``/new`` drops the one it leaves, and
        ``/resume``, ``/sessions``, ``--continue`` and ``--resume`` are
        refused (``cli/conversation.py::NOT_KEPT``)."""
        return self._place().store

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
        the store failed, or the session store is memory, which keeps none)
        or the backend keeps no sessions, 130 after a Ctrl+C before the
        prompt, else 0.
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
        """Apply changed settings and reinitialize workflow if needed
        (``cli/settings_ui.py``: ``SettingsUI.apply``). ``/settings`` applies
        a Save through this method, so an override applies.

        All or nothing: if any value is invalid, or the workflow cannot be
        reinitialized with the new values, every field is restored and the
        workflow is brought back on the model it was running, so nothing
        unusable stays in memory to be saved by ``/settings``. Ctrl+C stops
        it wherever it waits (for a start in progress, the turn lock, the
        rebuild, the restart on the old model after a failed one, a new
        conversation's backend) as it stops a turn: "Operation cancelled..."
        is said once, the Save is undone the same way (one stopped in that
        restart leaves the workflow down on the old model, for the next
        request to start), and the cancel is re-raised. A new conversation on another
        backend interrupted once that backend runs (while the old one is
        released, which goes on) is too late to undo: the change stands and
        is reported as usual before the cancel is re-raised (``/settings``
        saves it), so the settings name the backend that runs.

        Each change is set on the settings object (``update_setting``; the
        model pickers and the thinking effort through their own setters) and
        takes effect as ``settings_persistence`` declares for its field:

        1. ``REBUILD_KEYS`` (read when the workflow builds its agents and
           their models: the models, the summary thresholds, the result cap,
           the retries and request timeout) reinitialize the workflow in
           place, on the backend it runs;
        2. ``NEW_CONVERSATION_KEYS`` (``orchestrator``: the backend of new
           conversations; a conversation never changes backend) start a new
           conversation on the new backend at once when nothing of the open
           one is stored (``ConversationCore.fresh``: it began in this run as
           a new one, or a load found it empty, and no turn of it ran on a
           ready workflow, even one cancelled or failed, which may have
           stored its input; read under the turn lock), as ``/new <backend>``
           does, and the line of the new conversation names its backend;
           else they are saved, and their line says "(applies to the next
           /new)": nothing is rebuilt;
        3. ``RESTART_KEYS`` (read only as the app starts: ``session_store``,
           ``log_level``, ``log_format``) are applied and saved, and their
           line says "(takes effect at the next start)";
        4. any other key (``LIVE_KEYS``, or an app's own) applies at once:
           what reads it reads the settings each time.

        A reinitialization keeps the sessions, as a model switch does. One
        asked for while the workflow is still starting waits for that start
        first, as a turn does ("Waiting for initialization...": the start
        resolved its models from the settings before the change), and so
        does a change of ``orchestrator``, which needs that start's outcome
        (one of ``session_store`` does not wait: the run keeps the store its
        first start took); Ctrl+C stops the wait, and changes nothing. When
        that start fails, the change is applied as for any workflow that is
        not running: nothing is rebuilt, and the next start uses it. A change
        of the orchestrator made while the workflow is down, its next start
        building afresh, has that start run a conversation with nothing
        stored on the new backend (``SettingsUI._move_here``, decided again
        under the turn lock: one stored by then stays, and one whose workflow
        came up meanwhile is left for a new conversation, as on a running
        workflow). Held from its first write, it has no lock to
        move; one a line typed while the workflow was down holds is refused,
        every field restored, when another process has it open there. A
        change of ``session_store`` moves nothing and no lock: it takes
        effect at the next start. A
        switch to the deprecated LangGraph orchestrator ends with its
        deprecation warning. A change of ``ui_theme`` switches the session to
        ``get_theme(ui_theme)`` last, repainting the transcript; one rolled
        back leaves the theme as it was.

        Args:
            changes: Dictionary of changed settings (key -> new_value)

        Returns:
            True if the changes were applied, False if they were rolled back.
        """
        return await SettingsUI(self).apply(changes)

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
        """Run the user's ``message`` as a turn of the conversation
        (``ConversationCore.run_turn``: one at a time, under the turn lock),
        then, at the turn boundary, tell the finished jobs that wait for it
        (``job_auto_resume``) through the app's ``resume_finished_jobs``.

        Args:
            message: User message to process
        """
        if not await self._conversation.run_turn(message, self._turn_parts()):
            return  # open in another process: nothing ran
        # At the turn boundary, auto-resume any finished background jobs that
        # opted in (gated by the job_auto_resume setting).
        if getattr(self._settings, "job_auto_resume", False):
            await self._resume_on_its_own()

    def _turn_parts(self) -> TurnParts:
        """What a write to the conversation runs on besides it, as the app has it now."""
        return TurnParts(
            controller=self._workflow_controller,
            processor=self._message_processor,
            ui=self.session,
            settings=self._settings,
            tracker=self._usage_tracker,
            kept_elsewhere=self._kept_elsewhere,
            drop=self._drop,
        )

    async def _resume_on_its_own(self) -> None:
        """``resume_finished_jobs``, as the CLI runs it unasked (``telling_on_its_own``)."""
        with telling_on_its_own():
            await self.resume_finished_jobs()

    async def resume_finished_jobs(self) -> int:
        """Resume the agent for each finished, resume-flagged background job.

        Only the jobs that wait for the current conversation
        (``cli/waiting_jobs.py``: its own, and those with no conversation or
        whose conversation is gone, which are told it is unavailable), as the
        status bar's ``↻N`` and the line an opened conversation says count
        them: another conversation's job waits until that conversation is
        opened (``resume_session``, or at startup), instead of answering in
        this one's transcript. Each resume is a serialized turn (via the turn
        lock) so it never overlaps a user turn or another resume. Used at
        turn boundaries (auto, gated), when a conversation is opened (gated)
        and by ``/jobs resume`` (explicit, ungated).

        A job is claimed only once its result can reach the agent, and its
        delivery recorded as it ended: delivered, failed, or cancelled
        (``ConversationCore.deliver_finished_jobs``, which runs it).

        Returns:
            The number of jobs picked up: a resume turn run for it, or, for
            one that could not be told now (its conversation's lookup failed,
            before any claim), the failure said. Whether each one was
            delivered is recorded on the job and reported by ``/jobs``; a
            failed delivery is surfaced to the user by the turn itself.
        """
        return await self._conversation.deliver_finished_jobs(self._turn_parts())

    async def _kept_elsewhere(self, record: Any) -> bool:
        """Whether another installed backend than the running one keeps
        ``record``'s conversation: a job of it waits for it there, to be told
        once ``/resume`` opens it on its own backend. A store that cannot be
        read counts as keeping it (unknown: the job waits). Never raises."""
        session_id = getattr(record, "session_id", None)
        if session_id is None or not self._place().locked:
            return False  # the memory store's conversations are its manager's alone
        try:
            # A look the app makes on its own, again and again (the status
            # bar's recheck, each turn's end, /jobs resume): a store that
            # cannot be read is warned of once a run, then at debug.
            async with self._stored(quiet=True) as stored:
                return await stored.kept_elsewhere(
                    session_id, getattr(record, "user_id", None)
                )
        except Exception:  # noqa: BLE001 - unknown: it waits for its conversation
            logger.debug("job_conversation_lookup_failed", session_id=session_id, exc_info=True)
            return True

    async def compact(self, focus: str | None = None) -> "CompactResult | None":
        """Summarize the conversation now (``/compact [focus]``), as a turn runs.

        The public twin of ``start_new_session()``: under the turn lock, the
        workflow's ``compact`` runs with the turn's events box, status line,
        token counts and Ctrl+C (``MessageProcessor.process_compact``), and
        what it did, or why nothing was, is said as ``/compact`` says it
        (``ConversationCore.compact``).

        Args:
            focus: What the summary should keep above all, if anything.

        Returns:
            What the summary did; None when nothing was summarized.
        """
        return await self._conversation.compact(focus, self._turn_parts())

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

        On the memory store (``session_store=memory``, which writes no
        conversation to disk) the app keeps one conversation at a time: the
        one left is dropped from the workflow's store once the switch is
        made, after its memories are proposed (its ``delete_session``; a
        Ctrl+C then stops the wait, not the drop), and the line says
        ``"Previous: <old>, not kept (session_store is memory)."``. A finished
        job of it is told as unavailable, as a deleted conversation's is.
        One it cannot drop then (the workflow is down, the manager a failed
        reinitialization kept holding it still, or its delete raises) is
        remembered, and dropped once a workflow is ready, before the next
        turn, ``/compact``, delivery of finished jobs or ``/new`` uses it
        (``ConversationCore.drop_left``); its job is then told as
        unavailable.

        A conversation runs on one backend for its whole life (each keeps
        its history in its own store), so the new one runs on ``backend``,
        by default the settings' ``orchestrator``, the backend of new
        conversations. On another backend than the one running, the
        memories of the conversation left are proposed first (its backend
        holds it), then a manager of ``backend`` is started
        (``WorkflowController.switch_backend``: a failure there changes
        nothing), and the line names it: ``"New conversation <new> on
        deepagents."``, with ``" Previous: <old>, kept by adk."`` when the
        conversation left is stored (on the memory store it goes with the
        manager released, store and all: "not kept"); one on LangGraph is
        told it is deprecated. A Ctrl+C once the new backend runs (the old
        one is being released) is too late to undo: the new conversation is
        opened there first, then the cancel goes on. The backend chosen is
        not saved to any settings file.

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
        # Those an earlier /new could not drop go before the workflow is used.
        await self._conversation.drop_left(self._turn_parts())
        parts = self._switch_parts()
        stored = parts.workflow is not None and await self._is_stored(parts.workflow, old)
        kept = self._place().kept
        # On the memory store the conversation left is dropped once the switch
        # is made: on another backend its manager goes, and its store with it.
        # While the workflow is down (the manager a failed reinitialization
        # kept, its store with it) it is dropped once a workflow is ready.
        leaving = not kept and not moving
        dropping = parts.workflow if stored and leaving else None
        later = leaving and parts.workflow is None and old is not None
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
                    where = f"kept by {old_backend}" if kept else _NOT_KEPT_LEFT
                    line += f" Previous: {old}, {where}."
            else:
                line = f"New conversation {new}."
                if stored:
                    where = "listed in /sessions" if kept else _NOT_KEPT_LEFT
                    line += f" Previous: {old}, {where}."
            self.session.add_message("system", line)
            if moving and target == OrchestratorType.LANGGRAPH.value:
                self.session.add_warning(_LANGGRAPH_DEPRECATION)

        try:
            switched = await self._conversation.switch(
                new, parts, place=place, propose=not moving, generated=True
            )
            if switched.made and (dropping is not None or later):
                # Kept until the store has it no more: a delete that raises,
                # even after a Ctrl+C stopped the wait, leaves it to the next
                # point (ConversationCore.drop_left).
                self._conversation.to_drop.append(old)
                if dropping is not None and await self._drop(dropping, old):
                    self._conversation.to_drop.remove(old)
        except asyncio.CancelledError:
            if self._session_id == new:
                # On another backend the switch is made before the load, which
                # the cancel stopped, and a drop goes on to its end: the app
                # is there, and says so.
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

        Waits for a running turn or job resume (the turn lock), then finds
        the conversation on the backend that keeps it, the running one's or
        another installed backend's (``list_conversations``'s stores; a
        conversation belongs to one backend for its whole life), and reads
        it first, its row and its messages: one that is not stored, or
        cannot be read, changes nothing. Then, as ``start_new_session`` does,
        it proposes memories from the conversation being left (when
        ``auto_extract_session_facts`` is on and a turn ran in it since it was
        opened), adopts the id once the workflow has taken it
        (``load_session``; one that fails changes nothing either), ends the
        task box and zeroes the usage counters. It says "Resumed:
        <title> (<id>, 12 turns, last active 2h ago)." and replays the last
        messages (``cli/session_resume.py``). Ctrl+C stops it like any
        request, "Operation cancelled...", once, and nothing has changed.

        A conversation another backend keeps opens there: the app follows
        the conversation. It is held there first (one another process has
        open is refused), the memories of the one left are proposed while
        its backend runs, then that backend starts on the run's services
        (``WorkflowController.switch_backend``: the session grants, jobs,
        sandbox kernels, knowledge bases and embedding model stay the same),
        and the line says so: "Resumed on deepagents: …". A backend that
        fails to start changes nothing ("Could not open '<id>': deepagents
        did not start: …"); a Ctrl+C once it runs (the old one being
        released) is too late to undo: the conversation is opened there,
        then the cancel goes on.

        An id two backends keep (left by the live switch of backend an
        earlier version made) opens on the running backend, and the app
        says the other has one too ("'<id>' is also a conversation on
        deepagents: /resume lists it."); the picker's
        (``pick_session``) names the backend it was picked on, and opens
        there.

        On the memory store, which keeps one conversation at a time, it is
        refused at once with ``NOT_KEPT`` ("session_store is memory:
        conversations are not kept. ..."): nothing is looked up or held.

        Returns:
            True once ``session_id`` is the current conversation. False, with
            nothing changed, when it is already ("Already in this
            conversation."), is not stored ("No conversation '<id>'
            found.", also when it is deleted while it opens, with the
            backends that could not be read), could not be read ("Could not
            open '<id>': ..."), its backend did not start, or the run's
            store is the memory store.
        """
        if not self._place().kept:  # one conversation at a time: none to go back to
            self.session.add_error(NOT_KEPT)
            return False
        backend = getattr(session_id, "backend", None)  # the picker's
        async with _says_so_when_stopped(self.session):
            opened = await self._open_conversation(str(session_id), backend)
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
        (``cli/waiting_jobs.py``, through ``ConversationCore.waiting_jobs``).
        """
        waiting = await self._conversation.waiting_jobs(self._turn_parts())
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

    async def _open_conversation(self, session_id: str, backend: str | None = None) -> bool:
        """The switch to a stored conversation, under the turn lock (``resume_session``)."""
        async with self._turn_lock:
            return await self._open(session_id, backend=backend)

    async def _open(
        self, session_id: str, hold: Hold | None = None, *, backend: str | None = None
    ) -> bool:
        """Open the stored conversation ``session_id``, on the backend that
        keeps it: its row and messages are read first, then the switch
        (``ConversationCore.switch``). One another backend keeps opens there
        (``_open_elsewhere``).

        The caller holds the turn lock. ``hold`` is ``session_id``'s lock,
        taken already (at startup, before the prompt) where it is kept: it
        names the backend, it is kept once the switch is made, and let go on
        every other way out. ``backend`` names the backend otherwise (the
        picker's row); without either, the running one's is preferred.
        """
        if hold is not None:
            backend = hold.place.backend
        switching = False
        try:
            if session_id == self._session_id and backend in (None, self.session_backend):
                self.session.add_message("system", "Already in this conversation.")
                await self._name_its_twins(session_id)
                return False
            parts = self._switch_parts()
            workflow = parts.workflow
            if workflow is None:
                self.session.add_error(
                    f"Could not open '{session_id}': the workflow is not ready."
                )
                return False
            try:
                async with self._stored() as stored:
                    rows = await stored.where(session_id)
                    notes = stored.notes()
                found = [row for row in rows if backend in (None, row.backend)]
                if not found:
                    self.session.add_error(
                        " ".join([f"No conversation '{session_id}' found.", *notes])
                    )
                    return False
                info = found[0]
                twins = [row.backend for row in rows if row is not info]
                elsewhere = info.backend != self._workflow_controller.backend
                if not elsewhere:  # another backend's are read once it runs
                    messages = await workflow.recent_messages(session_id, sys.maxsize)
            except Exception as error:  # noqa: BLE001 - a store error changes nothing
                logger.debug("session_open_failed", session_id=session_id, exc_info=True)
                reason = str(error) or type(error).__name__
                self.session.add_error(f"Could not open '{session_id}': {reason}")
                return False
            if elsewhere:
                switching = True  # it keeps the hold or lets it go
                return await self._open_elsewhere(info, hold, twins)
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
        self._say_twins(session_id, twins)
        return True

    async def _open_elsewhere(
        self, info: "SessionInfo", hold: Hold | None, twins: "list[str | None]"
    ) -> bool:
        """Open ``info``'s conversation, which another backend than the
        running one keeps: that backend runs from now on (``_open``'s).

        The caller holds the turn lock. In order: the conversation is held
        where it is kept (``hold``, when the caller took it; one another
        process has open is refused, and nothing starts), and looked at
        again there (one deleted since the look is refused, nothing
        started: a CLI's delete takes the hold first); the memories of
        the conversation left are proposed while its backend still runs; the
        controller starts the conversation's backend on the run's services
        (``switch_backend``); and the switch to it is made there, at once
        (``ConversationCore.switch``, at its place: made before the load,
        since the conversation left is not on the new backend). Its messages
        are read from the new backend then, for the replay.

        A backend that fails to start, or a Ctrl+C before it runs, changes
        nothing (memories proposed stay proposed); a Ctrl+C once it runs
        (the old one being released) opens the conversation there, then the
        cancel goes on. A Ctrl+C during the load there, the switch made, says
        where the app is ("Resumed on <backend>: …", no replay) and draws the
        status line before it goes on. One deleted after all (by a deleter
        that holds no lock) leaves a new conversation open there, said.
        """
        controller = self._workflow_controller
        session_id, backend = info.session_id, info.backend
        place = dataclasses.replace(self._place(), backend=backend)
        if hold is None:
            taken = self._conversation.acquire(session_id, place)
            if isinstance(taken, int):
                self.session.add_error(
                    f"Could not open '{session_id}': it is open in another process ({taken})."
                )
                return False
            hold = taken
        # Looked at again once held: another process's /sessions --delete
        # takes the hold first, so one gone by now is gone for good.
        try:
            async with self._stored(only=backend) as stored:
                still = await stored.kept_elsewhere(session_id)
        except BaseException:
            self._conversation.abandon(hold)
            raise
        if not still:
            self._conversation.abandon(hold)
            self.session.add_error(f"No conversation '{session_id}' found.")
            return False
        old, interrupted = self._session_id, None
        try:
            if self._turned:  # proposed while the backend that holds them runs
                await self._saving_facts(old)
                self._turned = False
            try:
                await controller.switch_backend(backend)
            except asyncio.CancelledError as cancel:
                if controller.backend != backend:
                    raise  # it never ran: nothing changed
                interrupted = cancel  # it runs: its conversation is opened, then the cancel
        except Exception as error:
            self._conversation.abandon(hold)
            logger.debug("session_open_failed", session_id=session_id, exc_info=True)
            self.session.add_error(
                f"Could not open '{session_id}': {backend} did not start: {_first_line(error)}"
            )
            return False
        except BaseException:
            self._conversation.abandon(hold)
            raise
        # The backend runs: the switch is made there, whatever comes next.
        core = self._conversation
        try:
            switched = await core.switch(
                session_id, self._switch_parts(), place=place, stored=True, hold=hold,
                propose=False,
            )
        except asyncio.CancelledError:
            if (core.session_id, core.place) == (session_id, place):
                # Made before the load, which the cancel stopped: the app is
                # there, and says so (no replay), as /new <backend> does.
                self.session.add_message(
                    "system", resumed_line(info, datetime.now(timezone.utc), backend=backend)
                )
                _show_new_context(controller, self.session)
            raise
        logger.info("session_resumed", session_id=session_id, previous=old, backend=backend)
        if switched.gone:
            # Deleted as it opened, by a deleter that holds no lock: the
            # backend runs, and the conversation left is not on it, so a new
            # conversation opens there, never the id deleted.
            import uuid

            new = uuid.uuid4().hex[:12]
            await core.switch(new, self._switch_parts(), generated=True, propose=False)
            self.session.add_error(
                f"No conversation '{session_id}' found on {backend}: it was deleted as it "
                f"opened. New conversation {new} on {backend}."
            )
        else:
            self.session.add_message(
                "system", resumed_line(info, datetime.now(timezone.utc), backend=backend)
            )
            try:
                messages = await controller.workflow.recent_messages(session_id, sys.maxsize)
            except Exception as error:  # noqa: BLE001 - opened: the replay is a view
                logger.debug("session_replay_failed", session_id=session_id, exc_info=True)
                self.session.add_error(
                    f"Opened '{session_id}', but could not show where it left off: "
                    f"{_first_line(error)}"
                )
            else:
                show_replay(self.session, replay_lines(messages))
        self._say_twins(session_id, twins)
        if interrupted is not None:
            raise interrupted
        return True

    async def _name_its_twins(self, session_id: str) -> None:
        """Say which other backends keep a conversation of the app's id (it
        is in it already): their stores alone are read. Never raises."""
        try:
            async with self._stored() as stored:
                rows = await stored.where(session_id, running=False)
        except Exception:  # noqa: BLE001 - a line, never a failure
            logger.debug("session_twins_unread", session_id=session_id, exc_info=True)
            return
        self._say_twins(session_id, [row.backend for row in rows])

    def _say_twins(self, session_id: str, twins: "list[str | None]") -> None:
        """Say that other backends keep a conversation of the id just opened too."""
        names = sorted({twin for twin in twins if twin and twin != self.session_backend})
        if names:
            self.session.add_message(
                "system",
                f"'{session_id}' is also a conversation on {' and '.join(names)}: "
                f"/resume lists {'it' if len(names) == 1 else 'them'}.",
            )

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

        Every installed backend's (``list_conversations``): once a row is on
        another backend than the running one, each row names its backend,
        and an id two backends keep is two rows (the conversation the app is
        in, its id on its backend, is left out). Not under the turn lock: it
        waits on a dialog (``resume_session`` takes the lock).

        Returns:
            The id picked, or None: Esc, no other conversation is stored
            ("No other conversations to resume."), or the run's store is
            the memory store, which keeps none (``NOT_KEPT`` said, no
            dialog). The id is a
            ``PickedConversation`` (``cli/session_resume.py``): a ``str``
            whose ``backend`` names the backend that keeps it, which
            ``resume_session`` opens it on.

        Raises:
            RuntimeError: The workflow is not ready, or the session is not
                running (a dialog needs it); ``/resume`` checks both.
        """
        if not self._place().kept:  # one conversation at a time: none to pick
            self.session.add_error(NOT_KEPT)
            return None
        if not self._workflow_controller.is_ready:
            raise RuntimeError("Workflow not initialized yet")
        running = self._workflow_controller.backend
        here = (self._session_id, self.session_backend)
        rows = [
            info for info in await self.list_conversations(limit=PICK_LIMIT + 1)
            if (info.session_id, info.backend) != here
        ][:PICK_LIMIT]
        if not rows:
            self.session.add_message("system", "No other conversations to resume.")
            return None
        # Drawn where the settings say now: /settings may have changed the
        # placement since the session started with its own.
        dialog, ids = picker(
            rows, datetime.now(timezone.utc), placement=self._settings.dialog_placement,
            backends=any(info.backend != running for info in rows),
        )
        picked = await self.session.show_dialog(dialog)
        return None if picked is None else ids.get(picked)

    async def list_conversations(self, limit: int | None = None) -> "list[SessionInfo]":
        """The stored conversations of every installed backend, newest first.

        A conversation belongs to one backend for its whole life, and each
        keeps its own store: the running workflow lists its own backend's
        (all of them while it does not run: each installed backend's reader
        does), and each other installed backend's ``SessionReader``
        (``workflow/session_readers.py``) its own, read in the run's session
        store (``cli/stored_conversations.py``). Each row names its backend
        (``SessionInfo.backend``): an id two backends keep is two rows. A
        backend whose store cannot be read is left out (logged), and so is
        one whose extra is not installed.

        Args:
            limit: Only the newest this many (None: all of them).

        Raises:
            Exception: What the running workflow's listing raises.
        """
        async with self._stored() as stored:
            return await stored.rows(limit)

    async def delete_conversation(self, session_id: str, backend: str | None = None) -> bool:
        """Delete the stored conversation ``session_id`` on ``backend``.

        On the running workflow's backend (the default) its
        ``delete_session`` deletes it; on another, that backend's reader
        does. Nothing is held here: hold it first (``conversation_held``),
        so no other process has it open (``/sessions --delete`` does).

        Args:
            session_id: The conversation.
            backend: The backend that keeps it: the running one by default.

        Returns:
            Whether it was stored there.

        Raises:
            ValueError: ``backend``'s conversations cannot be read (its extra
                is not installed, or no reader reads them).
            Exception: What the backend's delete raises.
        """
        backend = backend or self._workflow_controller.backend
        async with self._stored(only=backend) as stored:
            return await stored.delete(session_id, backend)

    def _stored(self, *, only: str | None = None, quiet: bool = False) -> StoredConversations:
        """One look at the stored conversations of every installed backend
        (``only``'s alone, when named), in the run's store: the running
        workflow answers for its own backend while it is ready. A ``quiet``
        look (one the app makes on its own) warns of a store that cannot be
        read once a run; the others, which the user asked for, each time."""
        controller = self._workflow_controller
        warned = None
        if quiet:  # made on first use: an app built with __new__ has one too
            warned = self.__dict__.setdefault("_unreadable_warned", set())
        return StoredConversations(
            self._settings, self._place().store,
            workflow=controller.workflow if controller.is_ready else None,
            backend=controller.backend, only=only, warned=warned,
        )

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

    async def _drop(self, workflow: Any, session_id: str) -> bool:
        """Delete ``session_id``'s conversation from ``workflow``'s store: on
        the memory store the CLI keeps one conversation at a time, and
        ``/new`` drops the one it left, once the switch is made (its memories
        were proposed by then). One it left while the workflow was down is
        dropped through this too, once a workflow is ready, before it is used
        (``ConversationCore.drop_left``).

        Shielded: a Ctrl+C stops the wait, not the delete, so no conversation
        is left half deleted. Never raises but that cancel: a failed delete
        is logged. The status bar's job monitor then forgets what it knew of
        jobs' conversations: one of this conversation's jobs is told as
        unavailable now.

        Returns:
            Whether the store has it no more: dropped, or not there (the
            manager that kept it was released since); False when the delete
            raised.
        """
        async def drop() -> bool:
            try:
                held = await workflow.delete_session(session_id)
            except Exception:  # noqa: BLE001 - the new conversation goes on
                logger.debug("conversation_drop_failed", session_id=session_id, exc_info=True)
                gone = False
            else:
                logger.info("conversation_dropped", session_id=session_id, held=bool(held))
                gone = True
            if self._job_monitor is not None:
                self._job_monitor.forget_conversations()
            return gone

        return await asyncio.shield(asyncio.ensure_future(drop()))

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

        - ``"continue"`` finds the newest stored conversation of every
          installed backend that no other process has open (``_newest_free``,
          among the newest ``PICK_LIMIT`` when the newest is held; chosen
          before the workflow was built, on the backend that keeps it, by
          ``_start_on_its_backend``), which ``_open_on_startup`` resumes once
          the prompt is up, or says there is none ("No conversation to
          continue; starting a new one.");
        - ``"resume"`` checks its conversation is stored first, on any
          installed backend (the workflow runs the one that keeps it, found
          before it was built): one that is not is refused ("No
          conversation '<id>' found: <app> --continue opens the newest,
          /resume in the app picks one."), ``exit_status`` is 1, and False
          is returned;
        - ``"new"`` starts in the new conversation.

        The conversation ``"continue"`` or ``"resume"`` chose is held from
        here (another process asking for it is refused), beside the run's
        own, where it is kept, until the switch to it once the prompt is up
        keeps it or lets it go. One the start found on another backend than
        the one it runs (a look before the build that failed) opens there
        then (``resume_session``'s switch of backend).

        A workflow that cannot start keeps the prompt, so ``/settings`` can
        fix it, and says how to resume then (the conversation ``--continue``
        chose named, and let go); a backend without sessions
        refuses every start but ``"new"`` (False, status 1). A store that
        fails refuses ``"resume"`` too, and starts the others in a new
        conversation, saying why.

        The memory store keeps no conversation to go back to: ``"resume"``
        is refused with ``NOT_KEPT`` (False, status 1, as for an id that is
        not stored: the user asked for that one, and a script sees that it
        did not open), and ``"continue"`` starts a new conversation, saying
        why; neither waits for the workflow, nor looks anything up.
        """
        mode, target = self._session_start.mode, self._session_start.session_id
        chosen, self._chosen_at_start = self._chosen_at_start, None
        if chosen is not None:  # --continue chose before the build (on its backend)
            self._pending_hold = chosen[0]
        if mode != "new" and not self._place().kept:  # nothing kept to go back to
            if mode == "resume":
                self.session.add_error(NOT_KEPT)
                self.exit_status = 1
                return False
            self.session.add_warning(f"{NOT_KEPT} Starting a new conversation.")
            mode = "new"
        if not await self._workflow_controller.ensure_initialized(self.session):
            # The backend the start moved to for the conversation asked for
            # (not the settings') is named: the run stays on it.
            backend = self._workflow_controller.backend
            failed = (
                "the workflow did not start" if backend == orchestrator_of(self._settings).value
                else f"its backend, {backend}, did not start"
            )
            if self._pending_hold is not None:  # the one --continue chose
                continuing = self._pending_hold.session_id
                self._conversation.abandon(self._pending_hold)
                self._pending_hold = None
                self.session.add_warning(
                    f"Could not continue '{continuing}': {failed}. "
                    f"Fix it with /settings, then /resume {continuing}."
                )
                return await self._adopt_its_own()
            self.session.add_warning({
                "resume": f"Could not open '{target}': {failed}. "
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
        try:
            if mode == "resume":
                # On every installed backend: the running one's first.
                async with self._stored() as stored:
                    rows = await stored.where(target)
                    unread = stored.notes()
                if not rows:
                    run = (
                        f": {self._command} --continue opens the newest, "
                        "/resume in the app picks one."
                        if self._command else "."
                    )
                    self.session.add_error(" ".join(
                        [f"No conversation '{target}' found{run}", *unread]
                    ))
                    self.exit_status = 1
                    return False
                place = dataclasses.replace(self._place(), backend=rows[0].backend)
                taken = self._conversation.acquire(target, place)
                if isinstance(taken, int):
                    self.session.add_error(
                        f"Conversation '{target}' is open in another process ({taken})."
                    )
                    self.exit_status = 1
                    return False
                self._pending_resume, self._pending_hold = target, taken
            elif mode == "continue":
                if chosen is not None:  # chosen before the build, on its backend
                    passed_over = chosen[1]
                else:
                    self._pending_hold, passed_over = await self._newest_stored()
                if self._pending_hold is not None:
                    self._pending_resume = self._pending_hold.session_id
                if passed_over:
                    self.session.add_warning(_passed_over_line(
                        passed_over, continuing=self._pending_resume
                    ))
                elif self._pending_resume is None:
                    self.session.add_message(
                        "system", "No conversation to continue; starting a new one."
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
        (``ConversationCore.deferred``) and a run in which no turn runs makes
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

    async def _newest_stored(
        self, stored: StoredConversations | None = None
    ) -> tuple[Hold | None, list[tuple[str, int]]]:
        """What ``--continue`` opens: the newest stored conversation of every
        installed backend that no other process has open (``_newest_free``),
        read through ``stored`` (``list_conversations`` by default).

        A listing can read whole sessions (ADK's from before the title
        keys): the newest alone, unless it is held.
        """
        async def newest(limit: int) -> "list[SessionInfo]":
            if stored is None:
                return await self.list_conversations(limit=limit)
            return await stored.rows(limit)

        hold, passed_over = self._newest_free(await newest(1))
        if passed_over:
            hold, passed_over = self._newest_free(await newest(PICK_LIMIT))
        return hold, passed_over

    async def _start_on_its_backend(self) -> None:
        """Before the workflow is built: find the backend that keeps the
        conversation ``--resume <id>`` or ``--continue`` opens, so the run
        builds that one (``WorkflowController.switch_backend`` while none
        runs), not the settings', then the other.

        Every installed backend's store is read through its reader (none
        runs yet). ``--resume <id>`` moves to the backend that keeps it, the
        settings' preferred when both do; ``--continue`` chooses the newest
        no other process has open, and holds it (``_chosen_at_start``: the
        start opens it once the workflow runs, without another look). A
        target nobody has, a store that cannot be read, the memory store
        (nothing to read before the workflow runs), or a settings' backend
        whose own conversations no reader reads (LangGraph): nothing moves,
        and the start finds the target as it always did.
        """
        mode, target = self._session_start.mode, self._session_start.session_id
        controller = self._workflow_controller
        if mode == "new" or not self._place().locked:
            return
        if controller.backend not in reader_backends():
            return
        try:
            async with self._stored() as stored:  # nothing runs: every backend's reader
                if mode == "resume":
                    rows = await stored.where(target)
                    if not rows or any(row.backend == controller.backend for row in rows):
                        return
                    backend = rows[0].backend
                else:
                    hold, passed_over = await self._newest_stored(stored)
                    if hold is None and not passed_over:
                        return  # nothing anywhere: the start looks again
                    self._chosen_at_start = (hold, passed_over)
                    backend = hold.place.backend if hold is not None else None
        except Exception:  # noqa: BLE001 - the start looks again, as it always did
            logger.debug("startup_backend_lookup_failed", mode=mode, exc_info=True)
            return
        if backend is not None and backend != controller.backend:
            logger.info("startup_backend_chosen", backend=backend, mode=mode)
            await controller.switch_backend(backend)

    def _newest_free(self, rows: "list[SessionInfo]") -> tuple[Hold | None, list[tuple[str, int]]]:
        """The newest of ``rows`` no other process has open, now held here,
        on the backend that keeps it (beside the conversation held; the
        switch to it keeps it or lets it go).

        Returns:
            Its hold (None when every one is open elsewhere, or there is none),
            and the newer ones passed over, each with the process holding it.
        """
        passed_over: list[tuple[str, int]] = []
        for info in rows:
            place = self._place()
            if info.backend is not None:
                place = dataclasses.replace(place, backend=info.backend)
            taken = self._conversation.acquire(info.session_id, place)
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
        (``SettingsUI._move_here``); never read from the settings in
        between. The store is the run's for its whole life (a change of
        ``session_store`` takes effect at the next app start). An app built
        with ``__new__`` reads its settings on first use.
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
    def conversation_held(
        self, session_id: str, backend: str | None = None
    ) -> Iterator[int | None]:
        """Hold ``session_id``'s conversation for the block, beside the one
        this process is in, so no other process opens it meanwhile
        (``/sessions --delete``). When the block ends, the status bar's
        job monitor forgets what it knew of jobs' conversations: one may be
        gone, and its jobs told here.

        Args:
            session_id: The conversation.
            backend: The backend that keeps it (the same id on another
                backend is another conversation): the running one by
                default.

        Yields:
            None while it is held (on the memory store nothing is held, nor
            needed); else the process id that has it open (0 when unknown),
            and nothing is held.
        """
        place = self._place()
        if backend is not None:
            place = dataclasses.replace(place, backend=backend)
        taken = self._conversation.acquire(session_id, place)
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

        ``--resume`` (and ``/resume``) open a conversation on the backend
        that keeps it, whatever the settings' orchestrator, but in the
        settings' session store: one this run keeps in its store after a
        change of ``session_store`` (which takes effect at the next start)
        is found once the settings name that store, and the line says so.
        A backend no reader reads (LangGraph) is found only while it runs:
        one of its conversations needs the orchestrator set first, and the
        line says so too.
        """
        if self._command:
            hint = f"Resume this conversation with: {self._command} --resume {self._session_id}"
        else:
            hint = f"Resume this conversation with /resume {self._session_id}"
        place = self._place()
        unread = (
            place.backend not in reader_backends()
            and place.backend != orchestrator_of(self._settings).value
        )
        settings_first = [f"orchestrator to {place.backend}"] if unread else []
        if place.store != self._settings.session_store:
            settings_first.append(f"session_store to {place.store}")
        if settings_first:
            why = "it runs there" if unread else "it is stored there"
            hint += f" (set {' and '.join(settings_first)} first: {why})"
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

    def _new_job_monitor(self) -> "JobMonitor":
        """The status bar's job monitor ``run()`` runs: ``↻N`` counts the
        jobs that wait for this conversation, as ``cli/waiting_jobs.py`` says
        (a conversation another installed backend keeps is not gone)."""
        from agentic_cli.cli.job_monitor import JobMonitor

        return JobMonitor(
            self.session, self._workflow_controller, settings=self._settings,
            conversation=lambda: (self._settings.default_user, self._session_id),
            kept_elsewhere=self._kept_elsewhere,
        )

    async def run(self) -> None:
        """Run the main application loop.

        It installs the filters that log what ADK would print over the
        prompt (``workflow/adk/log_filters.py``) first: an app that is built
        but never run, as a library user's, keeps Python's own handling.
        """
        logger.info("repl_starting")
        install_log_filters()

        job_monitor = self._new_job_monitor()
        self._job_monitor = job_monitor  # told when a conversation is deleted

        # Once, before anything waits for the workflow: a backend that cannot
        # start (its extra missing, or a key) is still deprecated.
        _warn_if_langgraph(self._settings, self.session)
        # The place the workflow starts in keys the conversation's lock: the
        # backend of the conversation --resume/--continue opens, found first.
        self._conversation.place = Place.of(self._settings)
        await self._start_on_its_backend()
        self._conversation.place = Place.of(
            self._settings, backend=self._workflow_controller.backend
        )

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
