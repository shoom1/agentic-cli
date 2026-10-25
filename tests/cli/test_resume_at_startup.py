"""Starting an app on a stored conversation: ``BaseCLIApp.main()`` and ``start``.

``main()`` parses the session flags (and the app's own, ``add_arguments``),
builds the app (``from_args``), sets its ``session_start``, runs it and exits with its
``exit_status``. How ``run()`` starts:

- ``--continue`` resumes the newest stored conversation once the prompt is
  up, or says there is none ("No conversation to continue; starting a new
  one.");
- ``--resume <id>`` resumes that one once the prompt is up; one that is not
  stored is refused before the prompt: "No conversation '<id>' found. Run
  <app> --resume to pick one.", and the status is 1;
- ``--resume`` alone opens the picker once the prompt is up;
- a workflow that cannot start keeps the prompt (``/settings`` can fix it)
  and says how to resume after; a backend without sessions refuses
  ``--continue`` and ``--resume`` (status 1); ``session_store=memory`` says
  why nothing is stored;
- a store that fails says so: ``--resume <id>`` is refused (status 1), the
  others start a new conversation; a resume that fails once the prompt is
  up says so too, and an exit while it runs says nothing of a cancel.

At exit, a stored conversation (not on the memory store) is named: "Resume
this conversation with: <app> --resume <id>", or "… with /resume <id>" for
an app not run through ``main()``.

The app runs for real over ``tests/cli/console_app.py``'s scripted manager,
its stored conversations seeded; the session is a fake that keeps the prompt
up until the startup work is done.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli import AppInfo  # noqa: E402
from agentic_cli.cli import app as app_module  # noqa: E402
from agentic_cli.cli.app import BaseCLIApp  # noqa: E402
from agentic_cli.cli.session_args import SessionStart  # noqa: E402
from agentic_cli.config import BaseSettings  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from tests.cli.console_app import ScriptedManager  # noqa: E402

NOW = datetime.now(timezone.utc)


class Session:
    """The terminal: records what the app shows; its prompt stays up until
    ``until`` says the startup work is done, then plays ``inputs``."""

    def __init__(self, *, inputs=(), picks=(), until=None) -> None:
        self.shown: list[tuple[str, str]] = []
        self.inputs = list(inputs)
        self.picks = list(picks)
        self.dialogs: list = []
        self.until = until
        self.prompted = False
        self.app = SimpleNamespace(is_running=False)  # until the prompt is up
        self._handler = None

    def add_warning(self, text): self.shown.append(("warning", text))
    def add_error(self, text): self.shown.append(("error", text))
    def add_success(self, text): self.shown.append(("success", text))
    def add_message(self, role, text): self.shown.append((role, text))
    def add_response(self, text, markdown=False): self.shown.append(("response", text))
    def add_rich(self, renderable): self.shown.append(("rich", str(renderable)))
    def set_status(self, text): pass
    def exit(self): pass

    def start_thinking(self, *args, **kwargs):
        return SimpleNamespace(finish=lambda **kw: None, is_finished=False,
                               append=lambda *a, **k: None)

    def on_input(self, handler):
        self._handler = handler
        return handler

    async def show_dialog(self, dialog):
        self.dialogs.append(dialog)
        (field,) = dialog.fields
        pick = self.picks.pop(0)
        return None if pick is None else field.options[pick]

    async def run_async(self) -> None:
        self.prompted = True
        self.app.is_running = True
        self.shown.append(("prompt", ""))
        for _ in range(500):
            if self.until is None or self.until(self):
                break
            await asyncio.sleep(0)
        for text in self.inputs:
            await self._handler(text)

    def said(self, kind: str) -> list[str]:
        return [text for k, text in self.shown if k == kind]


@pytest.fixture(autouse=True)
def quiet_app(monkeypatch):
    """No global logging configuration or warning hook left behind."""
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    """Every log call of ``cli/app.py``, with its level.

    Recorded on the module's logger: ``structlog.testing.capture_logs()``
    sees nothing once an earlier test has cached a configured logger.
    """
    events: list[dict] = []

    class Recorder:
        def __getattr__(self, level: str):
            def log(event: str, **fields) -> None:
                events.append({"event": event, "log_level": level, **fields})

            return log

    monkeypatch.setattr(app_module, "logger", Recorder())
    return events


@pytest.fixture
def manager(monkeypatch, tmp_path) -> ScriptedManager:
    """The scripted manager every app here builds, with no conversation stored.

    HOME is the test's: the app's conversation locks live in its workspace.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = ScriptedManager()
    monkeypatch.setattr(
        "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings",
        lambda **kwargs: manager,
    )
    return manager


def store(manager: ScriptedManager, session_id: str, *texts: str, hours: float) -> None:
    """A stored conversation: ``texts`` alternate the user's and the replies."""
    for i, text in enumerate(texts):
        manager._said(session_id, "user" if i % 2 == 0 else "assistant", text)
    manager._sessions.add(session_id)
    manager._updated[session_id] = NOW - timedelta(hours=hours)


class App(BaseCLIApp):
    def __init__(self, store: str = "sqlite") -> None:
        super().__init__(
            app_info=AppInfo(name="Startup test", version="0"),
            agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
            settings=BaseSettings(session_store=store),
        )


def app_with(session: Session, start: SessionStart | None = None, **kwargs) -> App:
    app = App(**kwargs)
    app.session = session
    if start is not None:
        app.session_start = start
    return app


def resumed(session: Session) -> bool:
    return any(text.startswith("Resumed:") for text in session.said("system"))


class TestItsFinishedJobs:
    async def test_at_startup_they_are_named_not_run(self, manager, monkeypatch):
        """Outside the input handler Ctrl+C cannot stop a turn: the jobs wait
        for the first turn boundary, or for /jobs resume."""
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        record = SimpleNamespace(job_id="j1", name="build", session_id="abc",
                                 user_id=None, resume_on_complete=True)
        manager.job_manager = SimpleNamespace(awaiting_resume=lambda: [record])
        session = Session(until=resumed)
        app = App()
        app.session = session
        app.session_start = SessionStart("resume", "abc")
        app._settings.job_auto_resume = True
        ran: list = []

        async def process_resume(**kwargs):
            ran.append(kwargs["record"].job_id)

        monkeypatch.setattr(app._message_processor, "process_resume", process_resume)

        await app.run()

        assert ran == []
        assert "A finished job waits for this conversation: /jobs resume tells it now." in (
            session.said("system")
        )


class TestContinue:
    async def test_it_resumes_the_newest_once_the_prompt_is_up(self, manager):
        store(manager, "old", "Plan the trip", "Sure.", hours=5)
        store(manager, "newer", "Fix the build", "Done.", hours=1)
        session = Session(until=resumed)
        app = app_with(session, SessionStart("continue"))

        await app.run()

        assert app.session_id == "newer" and app.exit_status == 0
        assert session.said("system")[0] == (
            "Resumed: Fix the build (newer, 1 turn, last active 1h ago)."
        )
        # Once the prompt is up, under the banner.
        assert session.shown.index(("prompt", "")) < session.shown.index(
            ("system", "Resumed: Fix the build (newer, 1 turn, last active 1h ago).")
        )
        assert ("user", "Fix the build") in session.shown
        assert ("response", "Done.") in session.shown

    async def test_with_nothing_stored_it_starts_a_new_one(self, manager):
        session = Session()
        app = app_with(session, SessionStart("continue"))
        fresh = app.session_id

        await app.run()

        assert app.session_id == fresh and session.prompted
        assert "No conversation to continue; starting a new one." in session.said("system")

    async def test_on_the_memory_store_it_says_why(self, manager):
        session = Session()
        app = app_with(session, SessionStart("continue"), store="memory")

        await app.run()

        assert (
            "No conversation to continue; starting a new one. "
            "(session_store is memory: conversations end with the app)"
        ) in session.said("system")


class TestResumeById:
    async def test_it_resumes_that_one(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = Session(until=resumed)
        app = app_with(session, SessionStart("resume", "abc"))

        await app.run()

        assert app.session_id == "abc" and resumed(session)

    async def test_one_that_is_not_stored_is_refused_before_the_prompt(self, manager):
        session = Session()
        app = app_with(session, SessionStart("resume", "zzz"))

        await app.run()

        assert app.exit_status == 1 and not session.prompted
        assert session.said("error") == ["No conversation 'zzz' found."]
        assert "Goodbye!" not in session.said("system")

    async def test_the_refusal_names_the_command_main_ran(self, manager):
        session = Session()
        app = app_with(session, SessionStart("resume", "zzz"))
        app._command = "research-demo"

        await app.run()

        assert session.said("error") == [
            "No conversation 'zzz' found. Run research-demo --resume to pick one."
        ]

    async def test_on_the_memory_store_the_refusal_says_why(self, manager):
        session = Session()
        app = app_with(session, SessionStart("resume", "zzz"), store="memory")

        await app.run()

        assert session.said("error") == [
            "No conversation 'zzz' found. "
            "(session_store is memory: conversations end with the app)"
        ]


class TestPick:
    async def test_the_picker_opens_once_the_prompt_is_up(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = Session(picks=[0], until=resumed)
        app = app_with(session, SessionStart("pick"))

        await app.run()

        (dialog,) = session.dialogs
        assert dialog.title == "Resume a conversation"
        assert app.session_id == "abc"

    async def test_on_the_memory_store_with_nothing_stored_it_says_why(self, manager):
        session = Session(until=lambda s: any("No other" in t for t in s.said("system")))
        app = app_with(session, SessionStart("pick"), store="memory")

        await app.run()

        assert session.dialogs == []
        assert (
            "No other conversations to resume. "
            "(session_store is memory: conversations end with the app)"
        ) in session.said("system")

    async def test_escape_keeps_the_new_conversation(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = Session(picks=[None], until=lambda s: bool(s.dialogs))
        app = app_with(session, SessionStart("pick"))
        fresh = app.session_id

        await app.run()

        assert app.session_id == fresh and not resumed(session)


class TestWhenItCannot:
    async def test_a_workflow_that_cannot_start_keeps_the_prompt(
        self, manager, tmp_path, monkeypatch
    ):
        (tmp_path / "init-fails").touch()
        (tmp_path / "ready").touch()  # a later retry would start
        monkeypatch.setattr(ScriptedManager, "initializations", 0)
        session = Session()
        app = app_with(session, SessionStart("resume", "abc"))

        await app.run()

        assert session.prompted and app.exit_status == 0
        assert (
            "Could not open 'abc': the workflow did not start. "
            "Fix it with /settings, then /resume abc."
        ) in session.said("warning")

    async def test_with_the_workflow_down_the_picker_is_not_opened(
        self, manager, tmp_path, monkeypatch
    ):
        (tmp_path / "init-fails").touch()
        (tmp_path / "ready").touch()
        monkeypatch.setattr(ScriptedManager, "initializations", 0)
        session = Session()
        app = app_with(session, SessionStart("pick"))

        await app.run()

        assert session.prompted and session.dialogs == [] and app.exit_status == 0
        assert (
            "Could not open the picker: the workflow did not start. "
            "Fix it with /settings, then /resume."
        ) in session.said("warning")

    @pytest.mark.parametrize("start", [
        SessionStart("continue"), SessionStart("pick"), SessionStart("resume", "abc"),
    ])
    async def test_a_backend_without_sessions_refuses_every_resume(
        self, manager, monkeypatch, start
    ):
        monkeypatch.setattr(ScriptedManager, "supports_sessions", False)
        session = Session()
        app = app_with(session, start)

        await app.run()

        assert app.exit_status == 1 and not session.prompted
        assert session.said("error") == ["This app's backend keeps no sessions."]

    async def test_a_store_that_fails_to_look_a_conversation_up_refuses_it(
        self, manager, monkeypatch
    ):
        async def failing(session_id, *, user_id=None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(manager, "session_info", failing)
        session = Session()
        app = app_with(session, SessionStart("resume", "abc"))

        await app.run()

        assert app.exit_status == 1 and not session.prompted
        assert session.said("error") == ["Could not open 'abc': database is locked"]

    async def test_a_store_that_fails_to_list_starts_a_new_conversation(
        self, manager, monkeypatch
    ):
        async def failing(*, user_id=None, limit=None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(manager, "list_sessions", failing)
        session = Session()
        app = app_with(session, SessionStart("continue"))

        await app.run()

        assert session.prompted and app.exit_status == 0
        assert session.said("warning") == [
            "Could not list the saved conversations: database is locked. "
            "Starting a new one."
        ]

    async def test_a_store_that_fails_to_adopt_this_runs_conversation_keeps_the_prompt(
        self, manager, monkeypatch
    ):
        async def failing(session_id, user_id=None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(manager, "load_session", failing)
        session = Session()
        app = app_with(session)

        await app.run()

        assert session.prompted and app.exit_status == 0
        assert session.said("warning") == ["Cannot adopt session — database is locked"]

    async def test_a_resume_that_fails_once_the_prompt_is_up_says_so(
        self, manager, monkeypatch
    ):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)

        def boom(*args, **kwargs):
            raise RuntimeError("the status bar broke")

        monkeypatch.setattr(app_module, "_show_new_context", boom)
        session = Session(until=lambda s: bool(s.said("error")))
        app = app_with(session, SessionStart("resume", "abc"))

        await app.run()

        assert session.said("error") == ["Could not open 'abc': the status bar broke"]

    @pytest.mark.parametrize("start, broken", [
        (SessionStart("resume", "abc"), "session_info"),
        (SessionStart("continue"), "list_sessions"),
        (SessionStart(), "load_session"),
        (SessionStart("resume", "abc"), "show"),  # once the prompt is up
    ])
    async def test_a_failure_is_logged_at_debug_never_drawn_over_the_ui(
        self, manager, monkeypatch, logged, start, broken
    ):
        """At the default log level a warning prints its traceback on the
        terminal, over the prompt; the user is told in one line instead."""
        def boom(*args, **kwargs):
            raise RuntimeError("database is locked")

        async def failing(*args, **kwargs):
            boom()

        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        if broken == "show":
            monkeypatch.setattr(app_module, "_show_new_context", boom)
        else:
            monkeypatch.setattr(manager, broken, failing)
        session = Session(until=lambda s: bool(s.said("error") or s.said("warning")))
        app = app_with(session, start)

        await app.run()

        assert session.said("error") or session.said("warning")
        assert [e["log_level"] for e in logged if e.get("exc_info")] == ["debug"]

    async def test_an_exit_while_it_resumes_says_nothing_of_a_cancel(self, manager, monkeypatch):
        """At startup a Ctrl+C (or /exit) ends the app: its resume is
        cancelled on the way out, and that is no "Operation cancelled..."."""
        from agentic_cli.cli.message_processor import CANCELLED_LINE

        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        reading = asyncio.Event()

        async def slow(session_id, limit=20, *, user_id=None):
            reading.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(manager, "recent_messages", slow)
        session = Session(until=lambda s: reading.is_set())
        app = app_with(session, SessionStart("resume", "abc"))

        await app.run()

        assert CANCELLED_LINE not in session.said("system")
        assert session.said("system")[-1] == "Goodbye!"

    def test_a_start_with_a_session_id_given_is_refused(self, manager):
        app = BaseCLIApp(
            app_info=AppInfo(name="Startup test", version="0"),
            agent_configs=[AgentConfig(name="assistant", prompt="Help.")],
            settings=BaseSettings(session_store="memory"),
            session_id="mine",
        )

        with pytest.raises(ValueError, match="session_id"):
            app.session_start = SessionStart("continue")
        app.session_start = SessionStart()  # a new conversation: the id given

    def test_a_start_must_be_a_session_start(self, manager):
        with pytest.raises(TypeError, match="SessionStart"):
            App().session_start = "continue"  # type: ignore[assignment]


class TestAConversationOpenElsewhere:
    """Two CLIs on one conversation lose a turn: another process's is refused."""

    @staticmethod
    def held_elsewhere(app: App, session_id: str) -> None:
        from agentic_cli.cli.conversation_lock import ConversationLock

        other = ConversationLock(app.settings.sessions_dir / "locks")
        assert other.take(app._conversation_key(session_id)) is None
        TestAConversationOpenElsewhere.others.append(other)

    others: list = []

    async def test_resume_of_one_open_elsewhere_is_refused_before_the_prompt(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = Session()
        app = app_with(session, SessionStart("resume", "abc"))
        self.held_elsewhere(app, "abc")

        await app.run()

        assert app.exit_status == 1 and not session.prompted
        assert session.said("error") == [
            f"Conversation 'abc' is open in another process ({os.getpid()})."
        ]

    async def test_continue_starts_a_new_one_when_the_newest_is_open_elsewhere(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = Session()
        app = app_with(session, SessionStart("continue"))
        self.held_elsewhere(app, "abc")

        await app.run()

        assert session.prompted and app.exit_status == 0 and app.session_id != "abc"
        assert session.said("warning") == [
            f"The newest conversation, 'abc', is open in another process ({os.getpid()}); "
            "starting a new one."
        ]

    async def test_the_app_holds_the_conversation_it_opened(self, manager):
        from agentic_cli.cli.conversation_lock import ConversationLock

        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        held: list = []

        def looked(session) -> bool:
            if resumed(session) and not held:  # while the app has it open
                other = ConversationLock(app.settings.sessions_dir / "locks")
                held.append(other.take(app._conversation_key("abc")))
            return bool(held)

        session = Session(until=looked)
        app = app_with(session, SessionStart("resume", "abc"))
        await app.run()

        assert held == [os.getpid()]


class TestTheExitHint:
    async def test_a_stored_conversation_is_named_at_exit(self, manager):
        session = Session(inputs=["Hello."])
        app = app_with(session)

        await app.run()

        assert session.said("system")[-2:] == [
            f"Resume this conversation with /resume {app.session_id}", "Goodbye!",
        ]

    async def test_it_names_the_command_main_ran(self, manager):
        session = Session(inputs=["Hello."])
        app = app_with(session)
        app._command = "python -m research_demo"

        await app.run()

        assert session.said("system")[-2] == (
            f"Resume this conversation with: python -m research_demo --resume {app.session_id}"
        )

    async def test_a_conversation_with_no_turn_is_not_named(self, manager):
        session = Session()
        app = app_with(session)

        await app.run()

        assert session.said("system")[-1] == "Goodbye!"
        assert not any(t.startswith("Resume this") for t in session.said("system"))

    async def test_nor_one_on_the_memory_store(self, manager):
        session = Session(inputs=["Hello."])
        app = app_with(session, store="memory")

        await app.run()

        assert not any(t.startswith("Resume this") for t in session.said("system"))


class TestMain:
    class FlagApp(App):
        """An app with a flag of its own, built from the command line."""

        last: "TestMain.FlagApp | None" = None
        session_fake: Session | None = None

        @classmethod
        def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--shout", action="store_true")

        @classmethod
        def from_args(cls, args: argparse.Namespace) -> "TestMain.FlagApp":
            app = cls()
            app.shout = args.shout
            app.session = cls.session_fake
            cls.last = app
            return app

    def run_main(self, *argv: str, session: Session) -> int:
        self.FlagApp.last, self.FlagApp.session_fake = None, session
        with pytest.raises(SystemExit) as exited:
            self.FlagApp.main(list(argv))
        return exited.value.code

    def test_it_exits_with_the_apps_status(self, manager):
        code = self.run_main("--resume", "zzz", session=Session())

        assert code == 1
        assert self.FlagApp.last.session_start == SessionStart("resume", "zzz")

    def test_it_continues_and_reads_the_apps_own_flags(self, manager):
        store(manager, "abc", "Plan the trip", "Sure.", hours=2)
        session = Session(until=resumed)

        code = self.run_main("--continue", "--shout", session=session)

        assert code == 0 and self.FlagApp.last.shout is True
        assert self.FlagApp.last.session_id == "abc"

    def test_a_ctrl_c_before_the_prompt_ends_it_with_130_not_a_traceback(
        self, manager, monkeypatch
    ):
        """Before the prompt the terminal is not raw yet: Ctrl+C is a
        KeyboardInterrupt out of ``asyncio.run``, as during a slow start."""
        async def interrupted(app):
            raise KeyboardInterrupt

        monkeypatch.setattr(self.FlagApp, "run", interrupted)
        self.FlagApp.session_fake = Session()
        try:
            self.FlagApp.main(["--resume", "abc"])
        except BaseException as ended:  # noqa: BLE001 - one escaping would stop pytest
            outcome = ended
        assert isinstance(outcome, SystemExit) and outcome.code == 130

    def test_its_command_is_the_script_or_the_module_run(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["/usr/local/bin/research-demo"])
        monkeypatch.setitem(sys.modules, "__main__", ModuleType("__main__"))
        assert app_module._command_name() == "research-demo"

        main = ModuleType("__main__")
        main.__spec__ = SimpleNamespace(name="research_demo.__main__")
        monkeypatch.setitem(sys.modules, "__main__", main)
        assert app_module._command_name() == "python -m research_demo"

    def test_a_script_run_by_python_is_named_as_it_was_run(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "__main__", ModuleType("__main__"))
        monkeypatch.setattr(sys, "argv", ["my_app.py"])
        assert app_module._command_name() == "python my_app.py"
        monkeypatch.setattr(sys, "argv", ["/home/me/apps/my_app.py"])
        assert app_module._command_name() == "python /home/me/apps/my_app.py"
