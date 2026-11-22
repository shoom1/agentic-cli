"""``/agents``: the open conversation's sub-agent runs (``cli/agents_command.py``).

``/agents`` is a picker of the instances (id, type, status, title, each
followed by the ones it started), and Enter opens its view, as ``/agent <id>``
does (``tests/cli/test_agent_view.py``). ``/agents <id>`` shows one instance
or run: its type, parent and title, then each run's
status and time, its brief's other fields (a follow-up's message), the first
lines of its result, its error and its tokens by model. An instance that was
followed up shows every run in its panel.
``/agent stop <id>`` and ``/agent close <id>`` are the user's stop and close,
each saying what it did or why not. The runs are a real manager's records,
filled as its event processors fill them (``AgentRuns.observe`` under the
conversation's turn), and the tokens its usage ledger's; nothing reaches a
provider. A manager that did not run them (a restart) loads them from the
conversation's store as ``/agents`` asks, each run's tokens its record's.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from rich.console import Console

from agentic_cli.cli import agents_command
from agentic_cli.cli.agents_command import (
    AGENT_USAGE,
    NEW_USAGE,
    NO_RUNS,
    AgentCommand,
    AgentsCommand,
)
from agentic_cli.cli.builtin_commands import HelpCommand
from agentic_cli.cli.commands import CommandCategory, CommandRegistry
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.sessions import RunRef, SessionRef, reset_active_turn, set_active_turn
from tests.cli.dialog_fakes import autospec_session

SONNET = "anthropic:claude-sonnet-4-5"
HAIKU = "anthropic:claude-haiku-4-5"
START = datetime(2026, 11, 13, 9, 0, 0)
#: When the command is run: a minute after the first run started.
NOW = START + timedelta(seconds=60)


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    """A temp home and project, a fake Claude key, and a fixed clock."""
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for variable in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test-key")
    monkeypatch.setattr(agents_command, "_now", lambda: NOW)


def built() -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["researcher"],
                        description="Leads."),
            AgentConfig(name="researcher", prompt="Research.", description="Finds facts."),
        ],
        settings=BaseSettings(session_store="memory", default_model=SONNET),
    )


@pytest.fixture
async def manager():
    """A real manager (built, not started: its runs and ledgers need no services)."""
    made = built()
    yield made
    await made.cleanup()


class Runs:
    """Records sub-agent runs in ``manager`` as its event processors do: each
    ``AGENT_RUN`` observed under the conversation's turn, each model call in
    its usage ledger. Times are seconds after ``START``."""

    def __init__(self, manager: GoogleADKWorkflowManager, conversation: SessionRef) -> None:
        self.manager = manager
        self.conversation = conversation

    def start(self, agent: str, call_id: str, request: str, *, at: int,
              parent: RunRef | None = None, brief: dict[str, str] | None = None,
              tools: list[str] | None = None, model: str | None = None) -> RunRef:
        ref = self.manager.agent_runs.run_for(self.conversation, parent, call_id, agent)
        title = request.splitlines()[0]
        self._observe(WorkflowEvent.agent_run(
            ref, "started", title=title, call_id=call_id, brief=brief, tools=tools,
            model=model,
        ), at)
        return ref

    def follow(self, instance: str, call_id: str, message: str, *, at: int) -> RunRef:
        """The instance's next run, on ``message`` (a follow-up)."""
        ref = self.manager.agent_runs.next_run(self.conversation, instance, parent_run=None)
        self._observe(WorkflowEvent.agent_run(
            ref, "started", title=message.splitlines()[0], call_id=call_id, input=message,
        ), at)
        return ref

    def now(self, ref: RunRef, status: str, *, at: int) -> None:
        """``ref``'s run says how it works now: ``queued``, ``running`` or ``input_required``."""
        self._observe(WorkflowEvent.agent_run(ref, status), at)

    def end(self, ref: RunRef, status: str, *, at: int, result: str = "",
            error: str | None = None, partial: bool = False, plan: str | None = None,
            tasks: list[dict] | None = None, usage: dict | None = None) -> None:
        self._observe(WorkflowEvent.agent_run(
            ref, status, result=result, error=error, partial=partial, plan=plan, tasks=tasks,
            usage=usage,
        ), at)

    def used(self, ref: RunRef, model: str, prompt: int, completion: int) -> None:
        event = WorkflowEvent.llm_usage(model, prompt_tokens=prompt, completion_tokens=completion)
        event.run = ref
        event.metadata.update(agent=ref.agent, purpose="subagent")
        self.manager.usage_of(self.conversation).record(event)

    def _observe(self, event: WorkflowEvent, at: int) -> None:
        event.timestamp = START + timedelta(seconds=at)
        token = set_active_turn(self.conversation)
        try:
            self.manager.agent_runs.observe(event)
        finally:
            reset_active_turn(token)


def crew_ran(manager: GoogleADKWorkflowManager) -> None:
    """Five runs in conversation ``s1``: two finished, one of them with a run
    of its own under it, one still running, one refused, one failed."""
    runs = Runs(manager, manager.session_ref("s1"))
    first = runs.start("researcher", "c1", "Find X.\nIn the docs.", at=0)
    runs.used(first, SONNET, 1_200, 300)
    runs.used(first, HAIKU, 100, 10)
    writer = runs.start("writer", "w1", "Draft it.", at=2, parent=first)
    runs.used(writer, SONNET, 800, 2_000)
    runs.end(writer, "completed", at=42, result="Draft of the summary", partial=True)
    runs.end(first, "completed", at=12,
             result="X is 42.\nSource: a.\n\nSource: b.\nSource: c.")
    runs.start("researcher", "c2", "Find Y.", at=50)
    checker = runs.start("checker", "c3", "Check it.", at=51)
    runs.end(checker, "refused", at=52,
             result="The subagent stopped because a model refused the request.")
    fetcher = runs.start("fetcher", "c4", "Fetch Z.", at=53)
    runs.end(fetcher, "failed", at=55, error="overloaded")


async def agents(manager, args: str = "", *, session_id: str = "s1", picked=None):
    """Run ``/agents <args>`` in an app whose conversation is ``session_id``;
    its picker answers ``picked(options)`` (None: Esc)."""
    app = SimpleNamespace(workflow=manager, session_id=session_id, session=autospec_session())

    async def show_dialog(dialog):
        return picked(dialog.fields[0].options) if picked is not None else None

    app.session.show_dialog.side_effect = show_dialog
    await AgentsCommand().execute(args, app)
    return app.session


async def agent(manager, args: str, *, session_id: str = "s1"):
    """Run ``/agent <args>`` in an app whose conversation is ``session_id``
    (the workflow's own is another: a workflow that started after the app
    opened its conversation never loaded it)."""
    app = SimpleNamespace(workflow=manager, session_id=session_id, session=autospec_session())
    await AgentCommand().execute(args, app)
    return app.session


def rows_of(table) -> list[tuple[str, ...]]:
    """A rich table's rows, as text."""
    columns = [[str(cell) for cell in column.cells] for column in table.columns]
    return list(zip(*columns))


def options_of(session) -> list[str]:
    """The options of the picker ``/agents`` showed."""
    (dialog,) = session.show_dialog.call_args.args
    (field,) = dialog.fields
    return list(field.options)


class TestThePicker:
    async def test_each_instance_its_type_status_and_title_its_sub_agents_under_it(
        self, manager
    ):
        crew_ran(manager)

        session = await agents(manager)

        (dialog,) = session.show_dialog.call_args.args
        assert dialog.title == "Sub-agents"
        assert options_of(session) == [
            "researcher-1 · researcher · idle · Find X.",
            "  writer-1 · writer · idle · Draft it.",
            "researcher-2 · researcher · running · Find Y.",
            "checker-1 · checker · idle · Check it.",
            "fetcher-1 · fetcher · idle · Fetch Z.",
        ]
        session.add_rich.assert_not_called()

    async def test_escape_shows_nothing(self, manager):
        crew_ran(manager)

        session = await agents(manager)

        session.add_rich.assert_not_called()
        session.add_warning.assert_not_called()

    async def test_a_conversation_without_runs_says_so(self, manager):
        session = await agents(manager)

        session.add_message.assert_called_once_with("system", NO_RUNS)
        session.show_dialog.assert_not_called()

    async def test_another_conversations_runs_are_not_listed(self, manager):
        crew_ran(manager)

        session = await agents(manager, session_id="s2")

        session.add_message.assert_called_once_with("system", NO_RUNS)

    async def test_an_instance_with_follow_ups_is_one_option(self, manager):
        runs = Runs(manager, manager.session_ref("s1"))
        first = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(first, "completed", at=5, result="X is 42.")
        second = runs.follow("researcher-1", "c2", "Now find Y.", at=10)
        runs.end(second, "completed", at=15, result="Y is 7.")

        session = await agents(manager)

        assert options_of(session) == ["researcher-1 · researcher · idle · Find X."]


class TestOne:
    async def test_an_instance_its_parent_title_run_result_and_tokens_by_model(self, manager):
        crew_ran(manager)

        session = await agents(manager, "researcher-1")

        (panel,) = session.add_rich.call_args.args
        assert str(panel.title) == "[bold]researcher-1[/bold]"
        assert rows_of(panel.renderable) == [
            ("Type", "researcher"),
            ("Parent", "root"),
            ("Title", "Find X."),
            ("Run #1", "completed · 12s"),
            ("  Result", "X is 42.\nSource: a.\nSource: b.\n…"),
            ("  Tokens", f"{SONNET}: 1.2k in / 300 out · 1 call"),
            ("", f"{HAIKU}: 100 in / 10 out · 1 call"),
        ]

    @pytest.mark.parametrize("tools, model, shown", [
        (["lookup", "fetch_page"], "fast",
         [("Model", "fast"), ("Tools", "lookup, fetch_page")]),
        ([], None, [("Tools", "none of its own")]),
        (None, HAIKU, [("Model", HAIKU)]),
    ])
    async def test_an_instances_model_choice_and_narrowed_tools_after_its_title(
        self, manager, tools, model, shown
    ):
        runs = Runs(manager, manager.session_ref("s1"))
        ref = runs.start("researcher", "c1", "Find X.", at=0, tools=tools, model=model)
        runs.end(ref, "completed", at=5, result="X is 42.")

        session = await agents(manager, "researcher-1")

        (panel,) = session.add_rich.call_args.args
        assert rows_of(panel.renderable)[:3 + len(shown)] == [
            ("Type", "researcher"), ("Parent", "root"), ("Title", "Find X."), *shown,
        ]

    async def test_a_runs_own_plan_and_tasks_after_its_result(self, manager):
        runs = Runs(manager, manager.session_ref("s1"))
        ref = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(ref, "completed", at=5, result="X is 42.", plan="- [ ] find X\n- [ ] check X",
                 tasks=[{"id": "1", "description": "find X", "status": "completed"},
                        {"id": "2", "description": "check X", "status": "pending"}])

        session = await agents(manager, "researcher-1")

        (panel,) = session.add_rich.call_args.args
        assert rows_of(panel.renderable) == [
            ("Type", "researcher"),
            ("Parent", "root"),
            ("Title", "Find X."),
            ("Run #1", "completed · 5s"),
            ("  Result", "X is 42."),
            ("  Plan", "- [ ] find X\n- [ ] check X"),
            ("  Tasks", "1/2\n[ ] check X\n[✓] find X"),
        ]

    async def test_a_run_without_a_plan_or_tasks_shows_neither(self, manager):
        runs = Runs(manager, manager.session_ref("s1"))
        ref = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(ref, "completed", at=5, result="X is 42.")

        session = await agents(manager, "researcher-1")

        (panel,) = session.add_rich.call_args.args
        assert [row[0] for row in rows_of(panel.renderable)] == [
            "Type", "Parent", "Title", "Run #1", "  Result",
        ]

    async def test_a_run_by_its_id_under_the_run_that_started_it(self, manager):
        crew_ran(manager)

        session = await agents(manager, "writer-1#1")

        (panel,) = session.add_rich.call_args.args
        assert rows_of(panel.renderable) == [
            ("Type", "writer"),
            ("Parent", "researcher-1#1"),
            ("Title", "Draft it."),
            ("Run #1", "completed (partial) · 40s"),
            ("  Result", "Draft of the summary"),
            ("  Tokens", f"{SONNET}: 800 in / 2k out · 1 call"),
        ]

    async def test_a_runs_brief_each_field_it_filled(self, manager):
        runs = Runs(manager, manager.session_ref("s1"))
        ref = runs.start("researcher", "c1", "Find X", at=0, brief={
            "title": "Find X", "objective": "Find X in the docs.",
            "deliverable": "One number.", "done_when": "X is [named].",
        })
        runs.end(ref, "completed", at=5, result="X is 42.")

        session = await agents(manager, "researcher-1")

        (panel,) = session.add_rich.call_args.args
        assert rows_of(panel.renderable) == [
            ("Type", "researcher"),
            ("Parent", "root"),
            ("Title", "Find X"),
            ("Run #1", "completed · 5s"),
            ("  Objective", "Find X in the docs."),
            ("  Deliverable", "One number."),
            ("  Done when", "X is [named]."),  # data, never markup
            ("  Result", "X is 42."),
        ]

    async def test_a_failed_run_says_why(self, manager):
        crew_ran(manager)

        session = await agents(manager, "fetcher-1")

        (panel,) = session.add_rich.call_args.args
        assert ("Run #1", "failed · 2s") in rows_of(panel.renderable)
        assert ("  Error", "overloaded") in rows_of(panel.renderable)

    async def test_a_running_run_has_no_result_yet(self, manager):
        crew_ran(manager)

        session = await agents(manager, "researcher-2")

        (panel,) = session.add_rich.call_args.args
        assert rows_of(panel.renderable) == [
            ("Type", "researcher"),
            ("Parent", "root"),
            ("Title", "Find Y."),
            ("Run #1", "running · 10s"),
        ]

    async def test_an_instances_follow_up_shows_its_message(self, manager):
        runs = Runs(manager, manager.session_ref("s1"))
        first = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(first, "completed", at=5, result="X is 42.")
        second = runs.follow("researcher-1", "c2", "Now find Y.\nThe same way.", at=10)
        runs.end(second, "completed", at=15, result="Y is 7.")

        shown = await agents(manager, "researcher-1")

        (panel,) = shown.add_rich.call_args.args
        assert rows_of(panel.renderable) == [
            ("Type", "researcher"),
            ("Parent", "root"),
            ("Title", "Find X."),
            ("Run #1", "completed · 5s"),
            ("  Result", "X is 42."),
            ("Run #2", "completed · 5s"),
            ("  Message", "Now find Y.\nThe same way."),
            ("  Result", "Y is 7."),
        ]

    async def test_a_working_run_says_whether_it_works_waits_for_a_slot_or_needs_the_user(
        self, manager
    ):
        runs = Runs(manager, manager.session_ref("s1"))
        runs.start("researcher", "c1", "Find X.", at=0)
        queued = runs.start("researcher", "c2", "Find Y.", at=1)
        runs.now(queued, "queued", at=1)
        asking = runs.start("researcher", "c3", "Find Z.", at=2)
        runs.now(asking, "input_required", at=30)

        shown = [await agents(manager, name)
                 for name in ("researcher-1", "researcher-2", "researcher-3")]

        panels = [session.add_rich.call_args.args[0] for session in shown]
        assert [rows_of(panel.renderable)[3] for panel in panels] == [
            ("Run #1", "running · 1m 00s"), ("Run #1", "queued · 59s"),
            ("Run #1", "needs you · 58s"),
        ]

    async def test_a_title_is_shown_as_written(self, manager):
        """Titles are the model's words: markup in one is text, never style."""
        runs = Runs(manager, manager.session_ref("s1"))
        runs.start("researcher", "c1", "Fix [/x] and [bold]this[/bold]", at=0)

        listed = await agents(manager)
        shown = await agents(manager, "researcher-1")

        assert options_of(listed) == [
            "researcher-1 · researcher · running · Fix [/x] and [bold]this[/bold]",
        ]
        (panel,) = shown.add_rich.call_args.args
        out = io.StringIO()
        Console(file=out, width=200, color_system=None).print(panel)
        assert "Fix [/x] and [bold]this[/bold]" in out.getvalue()

    @pytest.mark.parametrize(
        "seconds, shown",
        [(0, "0s"), (59, "59s"), (60, "1m 00s"), (185, "3m 05s"), (3_725, "1h 02m")],
    )
    async def test_the_time_a_run_took(self, manager, seconds, shown):
        runs = Runs(manager, manager.session_ref("s1"))
        ref = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(ref, "completed", at=seconds, result="X.")

        session = await agents(manager, "researcher-1")

        (panel,) = session.add_rich.call_args.args
        assert rows_of(panel.renderable)[3] == ("Run #1", f"completed · {shown}")

    async def test_an_unknown_id_says_so(self, manager):
        crew_ran(manager)

        session = await agents(manager, "nobody-1")

        session.add_warning.assert_called_once_with(
            "No sub-agent nobody-1 in this conversation. Type /agents for the list."
        )
        session.add_rich.assert_not_called()


class TestAfterARestart:
    """The runs are the conversation's records, kept beside it in its store: a
    manager that did not run them loads them as ``/agents`` asks, and shows
    each run's tokens from its record (its usage ledger holds none of them)."""

    async def test_the_stored_runs_are_listed_with_their_tokens(self, manager):
        earlier = built()
        runs = Runs(earlier, earlier.session_ref("s1"))
        first = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(first, "completed", at=12, result="X is 42.", usage={
            SONNET: {"input": 1_200, "output": 300}, HAIKU: {"input": 100, "output": 10},
        })
        runs.start("researcher", "c2", "Find Y.", at=50)  # working when the app was killed
        await manager.initialize_services()
        await manager._save_agent_records(
            manager.session_ref("s1"), earlier.agent_runs.export(earlier.session_ref("s1"))
        )

        listed = await agents(manager)
        interrupted = await agents(manager, "researcher-2")
        shown = await agents(manager, "researcher-1")

        assert options_of(listed) == [
            "researcher-1 · researcher · idle · Find X.",
            "researcher-2 · researcher · idle · Find Y.",
        ]
        (label, value) = rows_of(interrupted.add_rich.call_args.args[0].renderable)[3]
        assert (label, value.split(" · ")[0]) == ("Run #1", "interrupted")
        (panel,) = shown.add_rich.call_args.args
        assert rows_of(panel.renderable)[-2:] == [
            ("  Tokens", f"{SONNET}: 1.2k in / 300 out"),
            ("", f"{HAIKU}: 100 in / 10 out"),
        ]

    async def test_records_that_cannot_be_read_are_said_and_the_runs_held_shown(
        self, manager, monkeypatch
    ):
        crew_ran(manager)

        async def unreadable(conversation: SessionRef) -> None:
            raise OSError("disk unreadable")

        monkeypatch.setattr(manager, "load_agents", unreadable)
        session = await agents(manager)

        session.add_warning.assert_called_once_with(
            "Sub-agent records not read from the store: disk unreadable"
        )
        assert len(options_of(session)) == 5


    async def test_an_error_without_a_message_is_named_by_its_type(self, manager, monkeypatch):
        async def unreadable(conversation: SessionRef) -> None:
            raise TimeoutError()

        monkeypatch.setattr(manager, "load_agents", unreadable)
        session = await agents(manager)

        session.add_warning.assert_called_once_with(
            "Sub-agent records not read from the store: TimeoutError"
        )


class TestTheAgentCommand:
    """``/agent stop <id>`` and ``/agent close <id>`` stop and close one
    sub-agent, each saying what it did or why not (``/agent <id>`` opens its
    view: ``tests/cli/test_agent_view.py``)."""

    async def test_stop_stops_an_idle_one_and_says_only_the_user_continues_it(self, manager):
        crew_ran(manager)

        session = await agent(manager, "stop researcher-1")

        session.add_success.assert_called_once_with(
            "Stopped researcher-1: only you can start its next run."
        )
        conversation = manager.session_ref("s1")
        assert manager.agent_runs.instance(conversation, "researcher-1").status == "stopped"
        assert options_of(await agents(manager))[0] == (
            "researcher-1 · researcher · stopped · Find X."
        )

    async def test_a_stop_that_leaves_it_closed_says_so(self, manager, monkeypatch):
        """A type that is not continuable closes as its stopped run ends."""
        crew_ran(manager)
        conversation = manager.session_ref("s1")

        async def stops_and_closes(instance, **ids):
            last = manager.agent_runs.record(conversation, instance)
            token = set_active_turn(conversation)
            try:
                manager.agent_runs.observe(
                    WorkflowEvent.agent_instance(last.ref, "closed", reason="ended")
                )
            finally:
                reset_active_turn(token)
            return True

        monkeypatch.setattr(manager, "stop_instance", stops_and_closes)

        session = await agent(manager, "stop researcher-1")

        session.add_success.assert_called_once_with(
            "Stopped researcher-1: it is closed, as it takes no more runs."
        )

    async def test_close_closes_it_and_its_sub_agents_and_says_its_work_stays(self, manager):
        crew_ran(manager)

        session = await agent(manager, "close researcher-1")

        session.add_success.assert_called_once_with(
            "Closed researcher-1, writer-1; /agents researcher-1 still shows their work."
        )
        assert options_of(await agents(manager))[:2] == [
            "researcher-1 · researcher · closed · Find X.",
            "  writer-1 · writer · closed · Draft it.",
        ]

    async def test_each_says_why_it_did_nothing(self, manager):
        crew_ran(manager)
        await agent(manager, "stop researcher-1")
        await agent(manager, "close checker-1")

        said = [
            await agent(manager, args)
            for args in ("stop researcher-1", "stop checker-1", "close checker-1",
                         "stop nobody-1", "close nobody-1")
        ]

        assert [s.add_message.call_args.args for s in said[:3]] == [
            ("system", "researcher-1 is stopped already."),
            ("system", "checker-1 is closed already."),
            ("system", "checker-1 is closed already."),
        ]
        assert [s.add_warning.call_args.args for s in said[3:]] == [
            ("No sub-agent nobody-1 in this conversation. Type /agents for the list.",),
        ] * 2
        assert not any(s.add_success.called for s in said)

    async def test_without_an_id_it_says_how(self, manager):
        assert AGENT_USAGE == (
            "Usage: /agent <id> | /agent new <type> <objective> | /agent stop <id> "
            "| /agent close <id>"
        )
        said = [await agent(manager, args) for args in ("", "stop", "close")]

        assert [s.add_warning.call_args.args for s in said] == [
            (AGENT_USAGE,), ("Usage: /agent stop <id>",), ("Usage: /agent close <id>",),
        ]

    async def test_records_that_cannot_be_read_stop_nothing(self, manager, monkeypatch):
        crew_ran(manager)

        async def unreadable(conversation: SessionRef) -> None:
            raise OSError("disk unreadable")

        monkeypatch.setattr(manager, "load_agents", unreadable)
        session = await agent(manager, "stop researcher-1")

        session.add_warning.assert_called_once_with(
            "Sub-agent records not read from the store: disk unreadable"
        )
        session.add_success.assert_not_called()
        assert manager.agent_runs.instance(manager.session_ref("s1"), "researcher-1").status == (
            "idle"
        )


class TestNew:
    """``/agent new <type> <objective>``: the type, then the rest of the line
    as typed, handed to the app's ``start_agent`` (``tests/cli/test_agent_view.py``
    runs it)."""

    @staticmethod
    async def new(manager, args: str):
        """``/agent <args>`` in an app that records what it was asked to start."""
        started: list[tuple[str, str]] = []

        async def start_agent(agent_type: str, objective: str) -> bool:
            started.append((agent_type, objective))
            return True

        app = SimpleNamespace(workflow=manager, session_id="s1", session=autospec_session(),
                              start_agent=start_agent)
        await AgentCommand().execute(args, app)
        return app.session, started

    async def test_the_objective_is_the_rest_of_the_line_as_typed(self, manager):
        _, started = await self.new(manager, 'new  researcher  Find --all of "X",\tnow. ')

        assert started == [("researcher", 'Find --all of "X",\tnow.')]

    @pytest.mark.parametrize("args", ["new", "new researcher", "new researcher   "])
    async def test_without_a_type_or_an_objective_it_says_how(self, manager, args):
        session, started = await self.new(manager, args)

        assert NEW_USAGE == "Usage: /agent new <type> <objective>"
        session.add_warning.assert_called_once_with(NEW_USAGE)
        assert started == []

    async def test_help_shows_it_with_an_example(self):
        registry = CommandRegistry()
        registry.register(HelpCommand())
        registry.register(AgentCommand())
        session = autospec_session()

        await HelpCommand().execute("agent", SimpleNamespace(command_registry=registry,
                                                             session=session))

        console = Console(file=io.StringIO(), record=True, width=120)
        console.print(*session.add_rich.call_args.args)
        text = console.export_text()
        assert "/agent <id> | new <type> <objective> | stop <id> | close <id>" in text
        assert "/agent new researcher Find what changed in the 2.0 release." in text
        assert "start a new one" in text


class _Down:
    """An app whose workflow is not running."""

    def __init__(self, error: Exception | None) -> None:
        self.session = autospec_session()
        self.session_id = "s1"
        self.workflow_error = error

    @property
    def workflow(self):
        raise RuntimeError("Workflow not initialized")


class TestWithoutAWorkflow:
    async def test_while_it_starts(self):
        app = _Down(None)

        await AgentsCommand().execute("", app)

        app.session.add_warning.assert_called_once_with(
            "Sub-agent runs not available (initializing...)."
        )

    async def test_when_it_failed_to_start(self):
        app = _Down(RuntimeError("no key"))

        await AgentsCommand().execute("researcher-1", app)

        app.session.add_warning.assert_called_once_with(
            "Sub-agent runs not available (failed to start)."
        )

    @pytest.mark.parametrize("args", ["stop researcher-1", "new researcher Find X."])
    async def test_agent_says_the_same(self, args):
        app = _Down(None)

        await AgentCommand().execute(args, app)

        app.session.add_warning.assert_called_once_with(
            "Sub-agent runs not available (initializing...)."
        )


def test_it_is_a_built_in_workflow_command():
    from agentic_cli.cli.app import BaseCLIApp

    registry = CommandRegistry()
    BaseCLIApp._register_builtin_commands(SimpleNamespace(command_registry=registry))

    command = registry.get("agents")
    assert isinstance(command, AgentsCommand)
    assert command.category is CommandCategory.WORKFLOW
    assert command.usage == "/agents [<id>]"
    one = registry.get("agent")
    assert isinstance(one, AgentCommand)
    assert one.category is CommandCategory.WORKFLOW
    assert one.usage == "/agent <id> | new <type> <objective> | stop <id> | close <id>"
    assert one.description == (
        "Open one of this conversation's sub-agents, start a new one, or stop or close one"
    )
    assert one.examples == [
        "/agent researcher-2",
        "/agent new researcher Find what changed in the 2.0 release.",
        "/agent stop researcher-2",
        "/agent close researcher-2",
    ]
