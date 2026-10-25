"""``/compact [focus]``, ``/new`` and ``BaseCLIApp.start_new_session()``.

``/compact`` runs the workflow's ``compact`` as a turn runs
(``MessageProcessor.process_compact``), under the app's turn lock: the events
box and its status line ("Summarizing the earlier conversation…" while the
summary is written), the token counts and Ctrl+C behave as in a turn, and
``CONTEXT_TRIMMED`` says what it did. Nothing to summarize is "Nothing to
summarize yet."; a backend without it says "/compact is not supported on this
backend"; a summary that fails is "Could not summarize: <error>", once. The
summary's usage adds to the totals but never moves ``ctx``: the next turn's
call shows the smaller context.

``/new`` calls ``start_new_session()``: under the turn lock, the facts of the
conversation being left are extracted first (when that is on, under "Saving
facts from the conversation…"), a new session id is adopted once the
workflow has taken it, the task box and the usage counters are reset, and
the line names the new conversation, and the old one when it is stored.
Ctrl+C stops it with "Operation cancelled...", once, and nothing changed.
The model, grants, sandbox kernels and jobs stay, and a job still reports to
the conversation that started it, without moving ``ctx`` or the task list of
the current one.

The app is a real ``BaseCLIApp`` whose session is a fake and whose controller
holds the manager: a fake one, or the real ADK manager on ``FakeAPI`` (Claude,
the 16,000-token trigger of ``tests/workflow/test_adk_compaction.py``).
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

pytest.importorskip("google.adk")

from agentic_cli.cli import AppInfo  # noqa: E402
from agentic_cli.cli import app as app_module  # noqa: E402
from agentic_cli.cli.app import BaseCLIApp  # noqa: E402
from agentic_cli.cli.message_processor import CANCELLED_LINE  # noqa: E402
from agentic_cli.tools.jobs import JobState  # noqa: E402
from agentic_cli.tools.registry import ToolCategory, register_tool  # noqa: E402
from agentic_cli.workflow.adk.compaction import SUMMARY_FAILED  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.context import CompactResult, overflow_message  # noqa: E402
from agentic_cli.workflow.events import EventType, WorkflowEvent  # noqa: E402
from agentic_cli.workflow.permissions import EXEMPT  # noqa: E402
from agentic_cli.workflow.permissions.rules import Effect, Rule, RuleSource  # noqa: E402
from agentic_cli.workflow.service_registry import (  # noqa: E402
    JOB_MANAGER,
    PERMISSION_ENGINE,
    SANDBOX_MANAGER,
)
from agentic_cli.workflow.sessions import SessionRef  # noqa: E402
from tests.cli.dialog_fakes import autospec_box, autospec_session, opening_boxes  # noqa: E402
from tests.workflow.test_adk_compaction import (  # noqa: E402,F401
    PROMPTS,
    THREE,
    Claude,
    calls,
    project,
    says,
    serve,
    settings,
    solo,
    summaries_in,
    three_turns,
)

NOTHING = "Nothing to summarize yet."
NOT_SUPPORTED = "/compact is not supported on this backend"
SUMMARIZING = "Summarizing the earlier conversation…"


@pytest.fixture(autouse=True)
def quiet_app(monkeypatch):
    """An app that leaves no global logging configuration or warning hook behind."""
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)


def app_on(manager, app_settings) -> BaseCLIApp:
    """A real app whose controller holds ``manager`` (initialized) and whose
    session is a fake, each ``start_thinking`` opening a box of its own."""
    app = BaseCLIApp(
        app_info=AppInfo(name="Compact test", version="0"),
        agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"])],
        settings=app_settings,
    )
    app.session = opening_boxes(autospec_session())
    app._workflow_controller._workflow = manager
    return app


async def events_of(app: BaseCLIApp, manager, session_id: str | None = None) -> list:
    """The stored events of the app's conversation (or ``session_id``'s)."""
    session = await manager._session_service.get_session(
        app_name=manager.app_name, user_id=app.settings.default_user,
        session_id=session_id or app.session_id,
    )
    return session.events


def said(session, method: str) -> list[str]:
    """What the fake session was told through ``method``, in order."""
    calls_made = getattr(session, method).call_args_list
    if method == "add_message":
        return [c.args[1] for c in calls_made]
    return [c.args[0] for c in calls_made]


class Workflow:
    """What ``/compact`` and ``/new`` touch on a workflow manager."""

    is_initialized = True
    model = "fake:model"
    supports_compact = True
    supports_sessions = True

    def __init__(self, result=None, error: Exception | None = None, reports: list | None = None):
        self.result, self.error, self.reports = result, error, reports or []
        self.compacted: list[tuple[SessionRef, str | None]] = []
        self.turns: list[str | None] = []
        self.calls: list[str] = []
        self.ended: list[SessionRef | None] = []
        self.end_error: Exception | None = None
        #: Set to hold load_session until it is set.
        self.load_gate: asyncio.Event | None = None
        self.loading = asyncio.Event()
        self.reinitialize = AsyncMock()
        self.cleanup = AsyncMock()

    def session_ref(self, session_id=None, user_id=None) -> SessionRef:
        return SessionRef("fake-app", user_id or "default_user", session_id or "default_session")

    def set_input_callback(self, callback) -> None:
        pass

    def clear_input_callback(self) -> None:
        pass

    async def compact(self, session, focus=None):
        from agentic_cli.workflow import live_events

        self.compacted.append((session, focus))
        for event in self.reports:
            live_events.report(event)
        if self.error is not None:
            raise self.error
        return self.result

    async def process(self, message, user_id, session_id=None):
        self.turns.append(session_id)
        yield WorkflowEvent.text(f"Echo: {message}")
        usage = WorkflowEvent.llm_usage(model=self.model, prompt_tokens=4_000,
                                        completion_tokens=5)
        usage.metadata.update(agent="solo", context_trigger=16_000, window_known=True)
        yield usage

    async def load_session(self, session_id, user_id=None) -> bool:
        self.calls.append(f"load_session {session_id}")
        self.loading.set()
        if self.load_gate is not None:
            await self.load_gate.wait()
        return False

    async def session_exists(self, session_id, *, user_id=None) -> bool:
        return session_id in self.turns

    async def on_session_end(self, messages=None, *, session=None) -> list[str]:
        self.calls.append("on_session_end")
        self.ended.append(session)
        if self.end_error is not None:
            raise self.end_error
        return ["The user likes tea."]


def fake_settings(**values):
    from agentic_cli.config import BaseSettings

    return BaseSettings(session_store="memory", **values)


# --- /compact -------------------------------------------------------------------------------


class TestCompactCommand:
    async def test_nothing_to_summarize_says_so(self, project):
        workflow = Workflow(result=None)
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert workflow.compacted == [
            (workflow.session_ref(app.session_id, app.settings.default_user), None)
        ]
        assert said(app.session, "add_message") == ["/compact", NOTHING]
        assert said(app.session, "add_error") == []

    async def test_the_focus_is_the_rest_of_the_line(self, project):
        workflow = Workflow(result=None)
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact keep the API decisions")

        assert [focus for _, focus in workflow.compacted] == ["keep the API decisions"]

    async def test_a_backend_without_it_says_so(self, project):
        workflow = Workflow()
        workflow.supports_compact = False
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert workflow.compacted == []
        assert said(app.session, "add_warning") == [NOT_SUPPORTED]
        assert app.session.start_thinking.call_count == 0

    async def test_a_backend_that_raises_not_implemented_says_so_once(self, project):
        workflow = Workflow(error=NotImplementedError("a native ADK YAML app"))
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert said(app.session, "add_warning") == [NOT_SUPPORTED]
        assert said(app.session, "add_error") == []
        assert NOTHING not in said(app.session, "add_message")

    async def test_a_summary_that_fails_is_said_once(self, project):
        workflow = Workflow(error=RuntimeError("the model refused"))
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert said(app.session, "add_error") == ["Could not summarize: the model refused"]
        assert NOTHING not in said(app.session, "add_message")

    async def test_an_error_without_words_is_named_by_its_type(self, project):
        workflow = Workflow(error=TimeoutError())
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert said(app.session, "add_error") == ["Could not summarize: TimeoutError"]

    async def test_a_native_adk_app_says_so_and_opens_no_box(self, project, monkeypatch, tmp_path):
        from tests.workflow.test_adk_context_cache import _native_yaml

        serve(monkeypatch, Claude())
        manager = GoogleADKWorkflowManager(
            agent_configs=[], settings=settings(),
            adk_config_path=_native_yaml(tmp_path, "claude-sonnet-4-6"),
        )
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            await app.process_input("/compact")
        finally:
            await manager.cleanup()

        assert said(app.session, "add_warning") == [NOT_SUPPORTED]
        assert app.session.start_thinking.call_count == 0

    async def test_what_it_did_is_its_report(self, project):
        trimmed = WorkflowEvent.context_trimmed(
            6, 3, "compact", "solo", messages_summarized=4, tokens_before=9_000,
            tokens_after=2_000,
        )
        workflow = Workflow(result=CompactResult(4, 9_000, 2_000), reports=[trimmed])
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert said(app.session, "add_warning") == ["Context summarized: 4 messages, 9k → 2k tokens"]
        assert NOTHING not in said(app.session, "add_message")

    async def test_a_result_it_did_not_report_is_said_anyway(self, project):
        workflow = Workflow(result=CompactResult(4, 9_000, 2_000))
        app = app_on(workflow, fake_settings())

        await app.process_input("/compact")

        assert said(app.session, "add_warning") == ["Context summarized: 4 messages, 9k → 2k tokens"]

    async def test_it_waits_for_a_running_turn(self, project):
        workflow = Workflow(result=None)
        app = app_on(workflow, fake_settings())

        async with app._turn_lock:
            compacting = asyncio.create_task(app.process_input("/compact"))
            await asyncio.sleep(0.05)
            waited = workflow.compacted == [] and not compacting.done()
        await asyncio.wait_for(compacting, 5)

        assert waited
        assert len(workflow.compacted) == 1


class TestTheAppsCompact:
    """``BaseCLIApp.compact()``: what ``/compact`` runs, for an app to call too."""

    async def test_it_summarizes_as_a_turn_runs_and_returns_what_it_did(self, project):
        result = CompactResult(4, 9_000, 2_000)
        workflow = Workflow(result=result)
        app = app_on(workflow, fake_settings())

        returned = await app.compact("keep the API decisions")

        assert returned is result
        assert workflow.compacted == [
            (workflow.session_ref(app.session_id, app.settings.default_user),
             "keep the API decisions")
        ]
        assert app.session.start_thinking.call_count == 1  # the turn's events box

    async def test_it_waits_for_a_running_turn(self, project):
        workflow = Workflow(result=None)
        app = app_on(workflow, fake_settings())

        async with app._turn_lock:
            compacting = asyncio.create_task(app.compact())
            await asyncio.sleep(0.05)
            waited = workflow.compacted == [] and not compacting.done()
        assert await asyncio.wait_for(compacting, 5) is None

        assert waited

    async def test_the_command_runs_it_with_the_rest_of_the_line(self, project, monkeypatch):
        app = app_on(Workflow(), fake_settings())
        asked: list = []

        async def compact(focus=None):
            asked.append(focus)

        monkeypatch.setattr(app, "compact", compact)
        await app.process_input("/compact keep the API decisions")
        await app.process_input("/compact")

        assert asked == ["keep the API decisions", None]


class TestCompactOnADK:
    async def test_the_line_shows_while_it_summarizes_then_what_it_did(
        self, project, monkeypatch
    ):
        answers = serve(monkeypatch, Claude(
            solo=[says("A1."), says("A2."), says("A3."), says("A4.", 2_400)],
        ))
        manager = solo()
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        tracker = app.usage_tracker
        try:
            for text in THREE:
                await app.process_input(text)
            before = (tracker.format_context(), tracker.prompt_tokens)
            answers.hold = True
            compacting = asyncio.create_task(app.process_input("/compact"))
            await asyncio.wait_for(answers.held.wait(), 5)
            await asyncio.sleep(0.05)
            status = app.session.start_thinking.call_args_list[-1].args[0]
            while_summarizing = status()
            answers.release.set()
            await asyncio.wait_for(compacting, 5)
            after = (tracker.format_context(), tracker.prompt_tokens)
            session = await events_of(app, manager)
            await app.process_input("Four?")
            next_turn = tracker.format_context()
        finally:
            answers.release.set()
            await manager.cleanup()

        assert while_summarizing == SUMMARIZING
        (warning,) = said(app.session, "add_warning")
        assert re.fullmatch(r"Context summarized: 2 messages, \S+ → \S+ tokens", warning)
        assert len(summaries_in(session)) == 1
        # The summary's usage adds to the totals; ctx stays the main agent's
        # until its next call, which sends the smaller context.
        assert before == ("ctx 4k/16k (25%)", 12_000)
        assert after == ("ctx 4k/16k (25%)", 19_000)
        assert next_turn == "ctx 2.4k/16k (15%)"
        assert app.session.boxes[-2].finish.call_args == call(
            add_to_history=False, echo_to_console=False
        )

    async def test_before_the_first_turn_there_is_nothing_to_summarize(
        self, project, monkeypatch
    ):
        serve(monkeypatch, three_turns())
        manager = solo()
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            await app.process_input("/compact")
        finally:
            await manager.cleanup()

        assert said(app.session, "add_message")[-1] == NOTHING

    async def test_ctrl_c_says_so_once_and_writes_nothing(self, project, monkeypatch):
        answers = serve(monkeypatch, three_turns())
        manager = solo()
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            for text in THREE:
                await app.process_input(text)
            answers.hold = True
            compacting = asyncio.create_task(app.process_input("/compact"))
            await asyncio.wait_for(answers.held.wait(), 5)
            compacting.cancel()
            with pytest.raises(asyncio.CancelledError):  # not a TimeoutError: no hang
                await asyncio.wait_for(compacting, 5)
            after_cancel = summaries_in(await events_of(app, manager))
            answers.release.set()
            # The turn lock is free again: the next /compact runs at once.
            await asyncio.wait_for(app.process_input("/compact"), 5)
            after_retry = summaries_in(await events_of(app, manager))
        finally:
            answers.release.set()
            await manager.cleanup()

        messages = said(app.session, "add_message")
        assert messages.count(CANCELLED_LINE) == 1
        assert NOTHING not in messages
        assert said(app.session, "add_error") == []
        assert after_cancel == []
        assert len(after_retry) == 1

    async def test_a_summary_that_fails_is_said_once(self, project, monkeypatch):
        import httpx

        refused = httpx.Response(400, json={"type": "error", "error": {
            "type": "invalid_request_error", "message": "max_tokens: 8000 > 4096",
        }})
        serve(monkeypatch, three_turns(summaries=[refused]))
        manager = solo()
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            for text in THREE:
                await app.process_input(text)
            await app.process_input("/compact")
        finally:
            await manager.cleanup()

        (error,) = said(app.session, "add_error")
        assert error.startswith("Could not summarize: ") and "max_tokens: 8000 > 4096" in error
        assert SUMMARY_FAILED not in " ".join(said(app.session, "add_warning"))
        assert NOTHING not in said(app.session, "add_message")


# --- /new and start_new_session() -----------------------------------------------------------


class TestStartNewSession:
    async def test_it_adopts_a_new_id_and_says_so(self, project):
        """Before any turn the conversation being left is in no store: the
        line names the new one alone."""
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        old = app.session_id

        new = await app.start_new_session()

        assert re.fullmatch(r"[0-9a-f]{12}", new) and new != old
        assert app.session_id == new
        assert workflow.calls == [f"load_session {new}"]
        assert said(app.session, "add_message") == [f"New conversation {new}."]

    async def test_after_a_turn_the_line_names_the_old_conversation(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        old = app.session_id
        await app.process_input("Hello.")

        new = await app.start_new_session()

        assert said(app.session, "add_message")[-1] == (
            f"New conversation {new}. Previous: {old}, listed in /sessions."
        )

    async def test_with_the_workflow_failed_to_start_it_still_starts_one(self, project):
        """The workflow is down (no key, a failed rebuild): the new id is
        adopted and said, the status bar says why, and the next turn's retry
        of initialization runs in the new conversation."""
        app = app_on(Workflow(), fake_settings())
        controller = app._workflow_controller
        controller._workflow, controller._init_error = None, RuntimeError("no API key")
        old = app.session_id

        await app.process_input("/new")

        assert said(app.session, "add_error") == []
        assert app.session_id != old
        assert said(app.session, "add_message")[-1] == f"New conversation {app.session_id}."
        assert app.session.set_status.call_args.args[0] == "Init failed - check API keys"

    async def test_the_next_turn_runs_in_the_new_conversation(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        old = app.session_id
        await app.process_input("Hello.")

        await app.process_input("/new")
        await app.process_input("Again.")

        assert workflow.turns == [old, app.session_id] and app.session_id != old

    async def test_usage_and_the_task_box_start_over(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        await app.process_input("Hello.")
        processor = app._message_processor
        box = processor._task_box = autospec_box()
        processor._last_task_progress = processor._last_task_content = "[ ] Write the report"
        assert app.usage_tracker.format_status_bar() != ""

        await app.start_new_session()
        status = app.session.set_status.call_args.args[0]
        await app.process_input("Again.")

        assert box.finish.call_args == call(add_to_history=False, echo_to_console=False)
        assert (processor._task_box, processor._last_task_progress,
                processor._last_task_content) == (None, None, None)
        assert "Tokens:" not in status and "ctx" not in status
        # The next turn opens no task box from the old conversation's list.
        titles = [c.kwargs.get("title") for c in app.session.start_thinking.call_args_list]
        assert "Tasks" not in titles
        assert app.usage_tracker.invocation_count == 1  # the new turn's call alone

    async def test_the_facts_of_the_old_conversation_are_extracted_first(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        old = app.session_id
        await app.process_input("Hello.")

        new = await app.start_new_session()

        assert workflow.calls == ["on_session_end", f"load_session {new}"]
        assert workflow.ended == [workflow.session_ref(old, app.settings.default_user)]

    async def test_facts_are_saved_under_a_status_line(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        lines: list[str] = []

        async def on_session_end(messages=None, *, session=None):
            status = app.session.start_thinking.call_args.args[0]
            lines.append(status())
            return []

        workflow.on_session_end = on_session_end
        await app.process_input("Hello.")
        await app.start_new_session()

        assert lines == [app_module.SAVING_FACTS]
        box = app.session.boxes[-1]  # after the turn's own
        assert box.finish.call_args == call(add_to_history=False, echo_to_console=False)

    async def test_ctrl_c_while_facts_are_saved_says_so_once_and_changes_nothing(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        await app.process_input("Hello.")
        old, counted = app.session_id, app.usage_tracker.invocation_count
        started = asyncio.Event()

        async def on_session_end(messages=None, *, session=None):
            started.set()
            await asyncio.Event().wait()

        workflow.on_session_end = on_session_end
        starting = asyncio.create_task(app.process_input("/new"))
        await asyncio.wait_for(started.wait(), 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 5)

        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1
        assert app.session_id == old
        assert app.usage_tracker.invocation_count == counted
        assert app.session.boxes[-1].finish.call_args == call(
            add_to_history=False, echo_to_console=False
        )
        assert not any(m.startswith("New conversation") for m in said(app.session, "add_message"))

    async def test_ctrl_c_while_the_workflow_takes_the_new_id_changes_nothing(self, project):
        """The id is adopted only once the workflow has taken it."""
        workflow = Workflow()
        workflow.load_gate = asyncio.Event()
        app = app_on(workflow, fake_settings())
        await app.process_input("Hello.")
        old, counted = app.session_id, app.usage_tracker.invocation_count

        starting = asyncio.create_task(app.start_new_session())
        await asyncio.wait_for(workflow.loading.wait(), 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 5)

        assert app.session_id == old
        assert app.usage_tracker.invocation_count == counted
        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1

    async def test_the_exit_extracts_the_facts_of_the_current_conversation(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        await app.process_input("Hello.")
        opened = app.session.start_thinking.call_count

        await app._extract_session_facts_on_exit()

        assert workflow.ended == [workflow.session_ref(app.session_id, app.settings.default_user)]
        assert app.session.start_thinking.call_count == opened  # no box once the session ended

    async def test_a_conversation_no_turn_ran_in_has_no_facts_to_save(self, project):
        """Facts are saved once per conversation used, not on every leaving."""
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))

        await app.start_new_session()
        await app._extract_session_facts_on_exit()

        assert "on_session_end" not in workflow.calls
        assert app.session.start_thinking.call_count == 0

    async def test_a_new_conversation_that_fails_after_saving_them_saves_none_again(
        self, project, monkeypatch
    ):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        await app.process_input("Hello.")

        async def unavailable(session_id, user_id=None) -> bool:
            raise RuntimeError("database is locked")

        monkeypatch.setattr(workflow, "load_session", unavailable)
        with pytest.raises(RuntimeError):
            await app.start_new_session()
        await app._extract_session_facts_on_exit()

        assert workflow.calls.count("on_session_end") == 1

    async def test_without_fact_extraction_none_are(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=False))

        await app.start_new_session()

        assert "on_session_end" not in workflow.calls

    async def test_a_failed_extraction_never_stops_it(self, project):
        workflow = Workflow()
        workflow.end_error = RuntimeError("the store is gone")
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        await app.process_input("Hello.")

        new = await app.start_new_session()

        assert app.session_id == new
        assert said(app.session, "add_message")[-1].startswith(f"New conversation {new}.")
        assert said(app.session, "add_error") == []

    async def test_it_waits_for_a_running_turn(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        old = app.session_id

        async with app._turn_lock:
            starting = asyncio.create_task(app.start_new_session())
            await asyncio.sleep(0.05)
            waited = app.session_id == old and not starting.done()
        new = await asyncio.wait_for(starting, 5)

        assert waited and app.session_id == new

    async def test_the_workflow_is_kept(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())

        await app.start_new_session()

        assert app._workflow_controller.workflow is workflow
        assert workflow.reinitialize.await_count == workflow.cleanup.await_count == 0

    async def test_clear_still_only_clears_the_screen(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        old = app.session_id

        await app.process_input("/clear")

        assert app.session.clear.call_count == 1
        assert app.session_id == old and workflow.calls == []


#: The call id each start of the probe job was given.
STARTED: list[str] = []


@register_tool(category=ToolCategory.EXECUTION, capabilities=EXEMPT, long_running=True,
               requires=(JOB_MANAGER, SANDBOX_MANAGER),
               description="Start a background job (an exempt test tool).")
def new_session_probe_start_job(command: str = "", tool_context=None) -> dict:
    """Start a job and return its id at once, as a job starter does."""
    STARTED.append(getattr(tool_context, "function_call_id", None))
    return {"success": True, "job_id": "j1", "state": "running"}


class TestANewSessionOnADK:
    async def test_before_any_turn_the_line_names_no_old_conversation(
        self, project, monkeypatch
    ):
        serve(monkeypatch, Claude())
        manager = solo()
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            old = app.session_id
            new = await app.start_new_session()
            listed = [s.session_id for s in await manager.list_sessions()]
        finally:
            await manager.cleanup()

        assert said(app.session, "add_message")[-1] == f"New conversation {new}."
        assert old not in listed

    async def test_a_resume_of_another_conversation_leaves_ctx_and_the_tasks_alone(
        self, project, monkeypatch
    ):
        """After /new, a job started before it resumes its own conversation:
        its usage counts in the totals, but ``ctx`` and the task list stay
        the current conversation's, and its line names its conversation."""
        serve(monkeypatch, Claude(solo=[
            calls("new_session_probe_start_job", "toolu_j1", 12_000, command="make"),
            says("Started.", 12_000), says("Hello back.", 2_000),
            says("The build finished.", 12_500),
        ]))
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                       tools=[new_session_probe_start_job],
                                       include_state_tools=False)],
            settings=settings(),
        )
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        tracker, processor = app.usage_tracker, app._message_processor
        completed: list = []
        jobs = SimpleNamespace(
            complete_resume=lambda job_id, delivered, error=None: completed.append(
                (job_id, delivered)
            )
        )
        try:
            old = app.session_id
            await app.process_input("Start the job.")
            await app.start_new_session()
            await app.process_input("Hello.")
            processor._last_task_progress = processor._last_task_content = "[ ] Write it"
            before = (tracker.format_context(), tracker.prompt_tokens)
            opened = app.session.start_thinking.call_count
            record = SimpleNamespace(
                job_id="j1", name="build", tool="new_session_probe_start_job",
                call_id=STARTED[-1], call_name="new_session_probe_start_job",
                session_id=old, user_id=app.settings.default_user,
                state=JobState.SUCCEEDED, exit_code=0, error=None,
            )
            delivered = await app._deliver_resume(jobs, record)
            after = (tracker.format_context(), tracker.prompt_tokens)
            titles = [c.kwargs.get("title")
                      for c in app.session.start_thinking.call_args_list[opened:]]
            status = app.session.set_status.call_args.args[0]
        finally:
            await manager.cleanup()

        assert delivered and completed == [("j1", True)]
        assert before[0] == after[0] == "ctx 2k/16k (12%)"
        assert after[1] == before[1] + 12_500  # the totals still count it
        assert "ctx 2k/16k (12%)" in status
        assert (processor._last_task_progress, processor._last_task_content) == (
            "[ ] Write it", "[ ] Write it"
        )
        assert "Tasks" not in titles  # the current list is not drawn for it
        assert said(app.session, "add_message")[-1] == (
            f"↻ Background job 'build' finished (✓ succeeded) — resuming its "
            f"conversation {old}."
        )

    async def test_grants_jobs_and_the_model_stay_and_a_job_reports_to_its_conversation(
        self, project, monkeypatch
    ):
        answers = serve(monkeypatch, Claude(solo=[
            calls("new_session_probe_start_job", "toolu_j1", command="make"),
            says("Started."), says("Hello back."), says("The build finished."),
        ]))
        manager = GoogleADKWorkflowManager(
            agent_configs=[AgentConfig(name="solo", prompt=PROMPTS["solo"],
                                       tools=[new_session_probe_start_job],
                                       include_state_tools=False)],
            settings=settings(),
        )
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            old = app.session_id
            await app.process_input("Start the job.")
            engine = manager._services[PERMISSION_ENGINE]
            grant = Rule("http.read", "https://example.com/**", Effect.ALLOW, RuleSource.SESSION)
            engine._session_rules.append(grant)
            services, model = dict(manager._services), manager.model

            new = await app.start_new_session()
            await app.process_input("Hello.")
            record = SimpleNamespace(
                job_id="j1", name="build", tool="new_session_probe_start_job",
                call_id=STARTED[-1], call_name="new_session_probe_start_job",
                session_id=old, user_id=app.settings.default_user,
                state=JobState.SUCCEEDED, exit_code=0, error=None,
            )
            resumed = await app._message_processor.process_resume(
                record, app._workflow_controller, app.session, app.settings,
                usage_tracker=app.usage_tracker,
            )
            old_events = await events_of(app, manager, old)
            new_events = await events_of(app, manager, new)
            services_after, model_after = dict(manager._services), manager.model
        finally:
            await manager.cleanup()

        assert resumed.delivered
        # The job's result went to the conversation that started it.
        resume_request = json.dumps(answers.bodies("solo")[-1])
        assert "Start the job." in resume_request and "Hello." not in resume_request
        assert any(e.author == "solo" and e.content and e.content.parts
                   and e.content.parts[0].text == "The build finished." for e in old_events)
        assert not any("The build finished." in json.dumps(e.model_dump(mode="json"))
                       for e in new_events)
        # The process's own: the same services (grants, jobs and sandbox
        # kernels among them), the model.
        assert {JOB_MANAGER, SANDBOX_MANAGER, PERMISSION_ENGINE} <= set(services)
        assert {k: id(v) for k, v in services_after.items()} == {
            k: id(v) for k, v in services.items()
        }
        assert grant in engine._session_rules and model_after == model


# --- the commands the overflow message names ------------------------------------------------


def test_every_command_the_overflow_message_names_is_registered(project):
    app = app_on(Workflow(), fake_settings())

    named = re.findall(r"(?<!\w)/(\w+)", overflow_message("anthropic:claude-sonnet-4-6"))

    assert named == ["compact", "new"]
    assert [name for name in named if app.command_registry.get(name) is None] == []
