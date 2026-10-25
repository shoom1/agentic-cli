"""``BaseCLIApp.resume_session()`` and ``pick_session()``: reopening a stored conversation.

``resume_session(id)`` waits for a running turn (the turn lock) and reads the
conversation first, its row and its messages: one that is not stored, or that
cannot be read, changes nothing. Then, as ``/new`` does, it saves the facts
of the conversation being left (when it is stored: a fresh one has none),
adopts the id once the workflow has taken it,
ends the task box and zeroes the usage counters; it says "Resumed: <title>
(<id>, 12 turns, last active 2h ago)." and replays the last 10 messages (the
user's as typed, the main agent's as a turn shows them), each cut at 2,000
characters, after "… N earlier messages" when there are more. Ctrl+C stops it
with "Operation cancelled...", once, and nothing has changed.

Facts are saved once per conversation actually used: on leaving it (``/new``,
``/resume``) or at exit, only when a turn ran in it since it was opened.

``pick_session()`` lists the newest 50 other conversations in a dialog whose
only field is a choice list, newest first, each row how long ago, its turns,
its title and its id (the parts that tell rows apart first, since a narrow
terminal clips a row's end), and returns the id picked, or None: Esc, or
nothing else stored ("No other conversations to resume.").

The last class runs a resume on the real ADK and Deep Agents managers.

The app is a real ``BaseCLIApp`` whose session is a fake and whose
controller holds a fake workflow (``tests/cli/test_compact_and_new.py``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import call

import pytest

pytest.importorskip("google.adk")

from thinking_prompt import ChoiceField  # noqa: E402

from agentic_cli.cli.message_processor import CANCELLED_LINE  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.deepagents import manager as deepagents_manager  # noqa: E402
from agentic_cli.workflow.deepagents.manager import DeepAgentsWorkflowManager  # noqa: E402
from agentic_cli.cli.session_resume import (  # noqa: E402
    PICK_LIMIT,
    ago,
    resumed_line,
)
from agentic_cli.workflow.sessions import SessionInfo  # noqa: E402
from tests.cli.dialog_fakes import autospec_box  # noqa: E402
from tests.cli.test_compact_and_new import (  # noqa: E402,F401
    Claude,
    Workflow as CompactWorkflow,
    app_on,
    events_of,
    fake_settings,
    project,
    quiet_app,
    said,
    says,
    serve,
    solo,
)
from tests.conftest import MockContext  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402

NOW = datetime.now(timezone.utc)


def conversation(session_id: str, title: str, messages: int, *, hours: float = 2.0):
    """A stored conversation: its row and its ``messages`` texts, alternating.

    Its time is ``hours`` before the call, not before ``NOW``: the picker
    reads the clock when it runs, maybe minutes after this module loaded.
    """
    texts = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{session_id} message {i}"}
        for i in range(messages)
    ]
    info = SessionInfo(
        session_id, title, (messages + 1) // 2, datetime.now(timezone.utc) - timedelta(hours=hours)
    )
    return info, texts


class Workflow(CompactWorkflow):
    """The fake workflow, with stored conversations to resume."""

    def __init__(self, *stored, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stored = {info.session_id: (info, texts) for info, texts in stored}
        self.info_error: Exception | None = None
        self.load_error: Exception | None = None

    async def load_session(self, session_id, user_id=None) -> bool:
        if self.load_error is not None:
            self.calls.append(f"load_session {session_id}")
            raise self.load_error
        return await super().load_session(session_id, user_id)

    async def session_info(self, session_id, *, user_id=None):
        self.calls.append(f"session_info {session_id}")
        if self.info_error is not None:
            raise self.info_error
        entry = self.stored.get(session_id)
        return entry[0] if entry else None

    async def recent_messages(self, session_id, limit=20, *, user_id=None):
        return self.stored[session_id][1][-limit:]

    async def list_sessions(self, *, user_id=None, limit=None):
        rows = sorted((info for info, _ in self.stored.values()),
                      key=lambda info: info.updated, reverse=True)
        return rows if limit is None else rows[:limit]


def replayed(app) -> list[tuple[str, str]]:
    """What the session was shown, in order: (kind, text)."""
    shown = []
    for name, args, kwargs in app.session.mock_calls:
        if name == "add_message":
            shown.append((args[0], args[1]))
        elif name == "add_response":
            shown.append(("response", args[0]))
    return shown


class TestAgo:
    @pytest.mark.parametrize("delta, text", [
        (timedelta(seconds=20), "just now"),
        (timedelta(minutes=5), "5m ago"),
        (timedelta(hours=2, minutes=59), "2h ago"),
        (timedelta(days=3, hours=1), "3d ago"),
    ])
    def test_how_long_ago(self, delta, text):
        assert ago(NOW - delta, NOW) == text

    def test_an_unknown_time_is_nothing(self):
        assert ago(None, NOW) == ""

    def test_a_time_without_a_zone_is_taken_as_utc(self):
        naive = (NOW - timedelta(hours=3)).replace(tzinfo=None)
        assert ago(naive, NOW) == "3h ago"


class TestResumedLine:
    def test_the_title_the_id_the_turns_and_how_long_ago(self):
        info = SessionInfo("abc", "Plan the trip", 12, NOW - timedelta(hours=2))
        assert resumed_line(info, NOW) == "Resumed: Plan the trip (abc, 12 turns, last active 2h ago)."

    def test_one_turn_no_title_and_unknowns(self):
        assert resumed_line(SessionInfo("abc", "", 1, None), NOW) == (
            "Resumed: (no prompt yet) (abc, 1 turn)."
        )
        assert resumed_line(SessionInfo("abc"), NOW) == "Resumed: (no prompt yet) (abc)."


class TestResumeSession:
    async def test_it_adopts_the_conversation_and_shows_where_it_left_off(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 4))
        app = app_on(workflow, fake_settings())

        resumed = await app.resume_session("abc")

        assert resumed is True and app.session_id == "abc"
        assert workflow.calls == ["session_info abc", "load_session abc"]
        assert replayed(app) == [
            ("system", "Resumed: Plan the trip (abc, 2 turns, last active 2h ago)."),
            ("user", "abc message 0"), ("response", "abc message 1"),
            ("user", "abc message 2"), ("response", "abc message 3"),
        ]
        assert app.session.add_response.call_args == call("abc message 3", markdown=True)

    async def test_only_the_last_ten_are_replayed_after_how_many_came_before(self, project):
        app = app_on(Workflow(conversation("abc", "Long", 13)), fake_settings())

        await app.resume_session("abc")

        shown = replayed(app)
        assert shown[1] == ("system", "… 3 earlier messages")
        assert [text for _, text in shown[2:]] == [f"abc message {i}" for i in range(3, 13)]

    async def test_one_earlier_message_is_said_so(self, project):
        app = app_on(Workflow(conversation("abc", "Long", 11)), fake_settings())

        await app.resume_session("abc")

        assert replayed(app)[1] == ("system", "… 1 earlier message")

    async def test_a_reply_of_content_blocks_shows_its_text(self, project):
        """A backend may hand content blocks; tool calls are never replayed."""
        info = SessionInfo("abc", "Find it", 1, NOW)
        workflow = Workflow((info, [
            {"role": "user", "content": "Find it."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
            ]},
            {"role": "assistant", "content": None},
        ]))
        app = app_on(workflow, fake_settings())

        await app.resume_session("abc")

        assert replayed(app)[1:] == [("user", "Find it."), ("response", "Let me search.")]

    async def test_a_long_message_is_cut(self, project):
        info = SessionInfo("abc", "Long", 1, NOW)
        workflow = Workflow((info, [{"role": "assistant", "content": "x" * 2_500}]))
        app = app_on(workflow, fake_settings())

        await app.resume_session("abc")

        assert replayed(app)[-1] == ("response", "x" * 2_000 + "… (cut)")

    async def test_the_next_turn_runs_in_it(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings())

        await app.resume_session("abc")
        await app.process_input("Again.")

        assert workflow.turns == ["abc"]

    async def test_usage_and_the_task_box_start_over(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings())
        await app.process_input("Hello.")
        processor = app._message_processor
        box = processor._task_box = autospec_box()
        processor._last_task_progress = processor._last_task_content = "[ ] Write it"

        await app.resume_session("abc")
        status = app.session.set_status.call_args.args[0]

        assert box.finish.call_args == call(add_to_history=False, echo_to_console=False)
        assert (processor._task_box, processor._last_task_progress) == (None, None)
        assert "Tokens:" not in status and "ctx" not in status
        assert app.usage_tracker.invocation_count == 0

    async def test_the_facts_of_the_conversation_left_are_saved_after_it_is_read(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        old = app.session_id
        await app.process_input("Hello.")

        await app.resume_session("abc")

        assert workflow.calls == ["session_info abc", "on_session_end", "load_session abc"]
        assert workflow.ended == [workflow.session_ref(old, app.settings.default_user)]

    async def test_a_conversation_left_with_no_turn_has_no_facts_to_save(self, project):
        """A fresh conversation (at startup, or after /new) is in no store."""
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))

        await app.resume_session("abc")

        assert workflow.calls == ["session_info abc", "load_session abc"]
        assert app.session.start_thinking.call_count == 0

    async def test_a_conversation_that_is_not_stored_changes_nothing(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))
        old = app.session_id

        resumed = await app.resume_session("zzz")

        assert resumed is False and app.session_id == old
        assert workflow.calls == ["session_info zzz"]
        assert said(app.session, "add_error") == ["No conversation 'zzz' found."]

    async def test_the_current_conversation_is_said_so(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())

        resumed = await app.resume_session(app.session_id)

        assert resumed is False and workflow.calls == []
        assert said(app.session, "add_message") == ["Already in this conversation."]

    async def test_a_store_that_fails_changes_nothing(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        workflow.info_error = RuntimeError("database is locked")
        app = app_on(workflow, fake_settings())
        old = app.session_id

        resumed = await app.resume_session("abc")

        assert resumed is False and app.session_id == old
        assert said(app.session, "add_error") == ["Could not open 'abc': database is locked"]
        assert "load_session abc" not in workflow.calls

    async def test_a_store_that_fails_to_open_it_changes_nothing(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        workflow.load_error = RuntimeError("database is locked")
        app = app_on(workflow, fake_settings())
        old = app.session_id

        resumed = await app.resume_session("abc")

        assert resumed is False and app.session_id == old
        assert said(app.session, "add_error") == ["Could not open 'abc': database is locked"]
        assert not any(m.startswith("Resumed:") for m in said(app.session, "add_message"))

    async def test_a_workflow_that_did_not_start_changes_nothing(self, project):
        app = app_on(Workflow(), fake_settings())
        controller = app._workflow_controller
        controller._workflow, controller._init_error = None, RuntimeError("no API key")
        old = app.session_id

        resumed = await app.resume_session("abc")

        assert resumed is False and app.session_id == old
        assert said(app.session, "add_error") == [
            "Could not open 'abc': the workflow is not ready."
        ]

    async def test_ctrl_c_while_the_workflow_takes_it_changes_nothing(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        workflow.load_gate = asyncio.Event()
        app = app_on(workflow, fake_settings())
        old = app.session_id

        resuming = asyncio.create_task(app.resume_session("abc"))
        await asyncio.wait_for(workflow.loading.wait(), 5)
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(resuming, 5)

        assert app.session_id == old
        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1
        assert not any(m.startswith("Resumed:") for m in said(app.session, "add_message"))

    async def test_it_waits_for_a_running_turn(self, project):
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings())

        async with app._turn_lock:
            resuming = asyncio.create_task(app.resume_session("abc"))
            await asyncio.sleep(0.05)
            waited = workflow.calls == [] and not resuming.done()
        await asyncio.wait_for(resuming, 5)

        assert waited and app.session_id == "abc"


class _Jobs:
    """A job manager holding finished jobs that wait to be told."""

    def __init__(self, *records) -> None:
        self.records = list(records)
        self.claimed: list[str] = []

    def awaiting_resume(self):
        return [r for r in self.records if r.job_id not in self.claimed]

    def begin_resume(self, job_id: str) -> bool:
        self.claimed.append(job_id)
        return True

    def complete_resume(self, job_id: str, *, delivered: bool, error=None) -> None:
        pass


class TestAConversationOpenElsewhere:
    async def test_it_is_refused_and_nothing_changes(self, project, monkeypatch):
        import os

        from agentic_cli.cli.conversation_lock import ConversationLock

        monkeypatch.setenv("HOME", str(project))
        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        from agentic_cli.config import BaseSettings

        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        before = app.session_id
        other = ConversationLock(app.settings.sessions_dir / "locks")
        assert other.take(app._conversation_key("abc")) is None

        assert await app.resume_session("abc") is False

        assert app.session_id == before
        assert not [c for c in workflow.calls if c.startswith("load_session")]
        assert said(app.session, "add_error") == [
            f"Could not open 'abc': it is open in another process ({os.getpid()})."
        ]


class TestTheLockOnASwitchThatDoesNotHappen:
    """The conversation the app is in stays held, and the one it did not
    open is not, however the switch ends."""

    @staticmethod
    def app_holding(project, monkeypatch):
        from agentic_cli.config import BaseSettings

        monkeypatch.setenv("HOME", str(project))
        workflow = Workflow(conversation("a", "A", 2), conversation("b", "B", 2))
        app = app_on(workflow, BaseSettings(session_store="sqlite"))
        assert app._hold(app.session_id) is None
        return app, workflow

    @staticmethod
    def free(app, session_id: str) -> bool:
        from agentic_cli.cli.conversation_lock import ConversationLock

        other = ConversationLock(app.settings.sessions_dir / "locks")
        taken = other.take(app._conversation_key(session_id)) is None
        other.release()
        return taken

    async def test_a_ctrl_c_while_it_loads(self, project, monkeypatch):
        app, workflow = self.app_holding(project, monkeypatch)
        old, loading = app.session_id, asyncio.Event()

        async def load_session(session_id, user_id=None):
            loading.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(workflow, "load_session", load_session)
        switch = asyncio.create_task(app.resume_session("b"))
        await asyncio.wait_for(loading.wait(), 5)
        switch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await switch

        assert app.session_id == old
        assert not self.free(app, old) and self.free(app, "b")

    async def test_a_load_that_fails(self, project, monkeypatch):
        app, workflow = self.app_holding(project, monkeypatch)
        old = app.session_id
        workflow.load_error = RuntimeError("database is locked")

        assert await app.resume_session("b") is False

        assert not self.free(app, old) and self.free(app, "b")


class TestItsFinishedJobs:
    @pytest.mark.parametrize("auto, told", [(True, ["j1"]), (False, [])])
    async def test_they_are_told_once_it_is_open_when_auto_resume_is_on(
        self, project, auto, told
    ):
        from agentic_cli.cli.message_processor import TurnResult, TurnStatus

        workflow = Workflow(conversation("abc", "Plan the trip", 2))
        app = app_on(workflow, fake_settings(job_auto_resume=auto))
        me = app.settings.default_user
        workflow.job_manager = _Jobs(
            SimpleNamespace(job_id="j1", name="build", session_id="abc", user_id=me),
            SimpleNamespace(job_id="j2", name="other", session_id="xyz", user_id=me),
        )
        seen: list[tuple[str, str]] = []

        async def process_resume(*, record, current_session_id=None, **_):
            seen.append((record.job_id, current_session_id))
            return TurnResult(TurnStatus.COMPLETED)

        app._message_processor.process_resume = process_resume

        assert await app.resume_session("abc")
        assert seen == [(job, "abc") for job in told]


class TestPickSession:
    async def test_it_lists_the_others_newest_first_and_returns_the_pick(self, project):
        workflow = Workflow(
            conversation("old", "Plan the trip", 4, hours=26),
            conversation("new", "Fix the build", 2, hours=0.1),
        )
        app = app_on(workflow, fake_settings())
        workflow.stored[app.session_id] = conversation(app.session_id, "This one", 2, hours=0)

        def pick_the_second(dialog):
            (field,) = dialog.fields
            pick_the_second.dialog = dialog
            return field.options[1]

        app.session.show_dialog.side_effect = pick_the_second
        picked = await app.pick_session()

        dialog = pick_the_second.dialog
        (field,) = dialog.fields
        assert dialog.title == "Resume a conversation"
        assert isinstance(field, ChoiceField)
        assert field.options == [
            "6m ago   · 1 turn    · Fix the build · new",
            "1d ago   · 2 turns   · Plan the trip · old",
        ]
        assert picked == "old"

    async def test_escape_picks_nothing(self, project):
        app = app_on(Workflow(conversation("abc", "Plan the trip", 2)), fake_settings())
        app.session.show_dialog.return_value = None

        assert await app.pick_session() is None

    async def test_with_nothing_else_stored_no_picker_opens(self, project):
        workflow = Workflow()
        app = app_on(workflow, fake_settings())
        workflow.stored[app.session_id] = conversation(app.session_id, "This one", 2)

        assert await app.pick_session() is None
        assert app.session.show_dialog.call_count == 0
        assert said(app.session, "add_message") == [
            "No other conversations to resume. "
            "(session_store is memory: conversations end with the app)"
        ]

    async def test_it_holds_the_newest_fifty(self, project):
        workflow = Workflow(*(
            conversation(f"s{i:02d}", f"Topic {i}", 2, hours=i) for i in range(60)
        ))
        app = app_on(workflow, fake_settings())
        app.session.show_dialog.return_value = None

        await app.pick_session()

        (field,) = app.session.show_dialog.call_args.args[0].fields
        assert len(field.options) == PICK_LIMIT == 50
        assert field.options[0].endswith(" · s00") and field.options[-1].endswith(" · s49")


class TestFactsAreSavedOncePerConversationUsed:
    async def test_switching_back_and_forth_without_a_turn_saves_none(self, project):
        workflow = Workflow(conversation("a", "A", 2), conversation("b", "B", 2))
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))

        for session_id in ("a", "b", "a"):
            await app.resume_session(session_id)
        await app._extract_session_facts_on_exit()

        assert workflow.ended == []
        assert app.session.start_thinking.call_count == 0

    async def test_a_turn_in_it_saves_its_facts_once(self, project):
        workflow = Workflow(conversation("a", "A", 2), conversation("b", "B", 2))
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))

        await app.resume_session("a")
        await app.process_input("Hello.")
        await app.resume_session("b")
        await app.resume_session("a")
        await app._extract_session_facts_on_exit()

        assert workflow.ended == [workflow.session_ref("a", app.settings.default_user)]

    async def test_at_exit_after_a_turn(self, project):
        workflow = Workflow(conversation("a", "A", 2))
        app = app_on(workflow, fake_settings(auto_extract_session_facts=True))

        await app.resume_session("a")
        await app.process_input("Hello.")
        await app._extract_session_facts_on_exit()

        assert workflow.ended == [workflow.session_ref("a", app.settings.default_user)]


class TestOnRealBackends:
    """A resume on the real managers: the next turn lands in the conversation
    resumed, the manager holds its id, and the replay is its text."""

    async def _resume_after_another(self, app, manager):
        first = app.session_id
        await app.process_input("One in A.")
        await app.process_input("Two in A.")
        await app.start_new_session()
        await app.process_input("One in B.")
        resumed = await app.resume_session(first)
        shown = replayed(app)[-5:]
        await app.process_input("Three in A.")
        return first, resumed, shown

    def _expected(self, first: str) -> list[tuple[str, str]]:
        return [
            ("system", f"Resumed: One in A. ({first}, 2 turns, last active just now)."),
            ("user", "One in A."), ("response", "A1."),
            ("user", "Two in A."), ("response", "A2."),
        ]

    async def test_on_adk(self, project, monkeypatch):
        serve(monkeypatch, Claude(solo=[says("A1."), says("A2."), says("B1."), says("A3.")]))
        manager = solo()
        await manager.initialize_services()
        app = app_on(manager, manager._settings)
        try:
            first, resumed, shown = await self._resume_after_another(app, manager)
            stored = await events_of(app, manager, first)
            manager_id = manager.session_id
        finally:
            await manager.cleanup()

        assert resumed and app.session_id == first == manager_id
        assert shown == self._expected(first)
        typed = [e.content.parts[0].text for e in stored if e.author == "user" and e.content]
        assert typed == ["One in A.", "Two in A.", "Three in A."]

    async def test_on_deep_agents(self, project, monkeypatch):
        pytest.importorskip("deepagents")
        monkeypatch.setattr(deepagents_manager, "build_chat_model",
                            ScriptedModels(solo=[ai("A1."), ai("A2."), ai("B1."), ai("A3.")]))
        with MockContext(google_api_key="test-key", session_store="memory") as context:
            manager = DeepAgentsWorkflowManager(
                agent_configs=[AgentConfig(name="solo", prompt="You help.")],
                settings=context.settings, model="gemini-2.5-flash",
            )
            await manager.initialize_services()
            app = app_on(manager, context.settings)
            try:
                first, resumed, shown = await self._resume_after_another(app, manager)
                recent = await manager.recent_messages(first, 10)
                manager_id = manager.session_id
            finally:
                await manager.cleanup()

        assert resumed and app.session_id == first == manager_id
        assert shown == self._expected(first)
        assert [m["content"] for m in recent if m["role"] == "user"] == [
            "One in A.", "Two in A.", "Three in A.",
        ]
