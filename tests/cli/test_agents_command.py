"""``/agents``: the open conversation's sub-agent runs (``cli/agents_command.py``).

``/agents`` is a table of the runs, each followed by the runs it started, a
level deeper: instance, type, status, run, title, tokens in/out, time.
``/agents <id>`` shows one instance or run: its type, parent and title, then
each run's status and time, the first lines of its result, its error and its
tokens by model. The runs are a real manager's records, filled as its event
processors fill them (``AgentRuns.observe`` under the conversation's turn),
and the tokens its usage ledger's; nothing reaches a provider.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from rich.console import Console

from agentic_cli.cli import agents_command
from agentic_cli.cli.agents_command import AgentsCommand
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


@pytest.fixture
async def manager():
    """A real manager (built, not started: its runs and ledgers need no services)."""
    built = GoogleADKWorkflowManager(
        agent_configs=[
            AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["researcher"],
                        description="Leads."),
            AgentConfig(name="researcher", prompt="Research.", description="Finds facts."),
        ],
        settings=BaseSettings(session_store="memory", default_model=SONNET),
    )
    yield built
    await built.cleanup()


class Runs:
    """Records sub-agent runs in ``manager`` as its event processors do: each
    ``AGENT_RUN`` observed under the conversation's turn, each model call in
    its usage ledger. Times are seconds after ``START``."""

    def __init__(self, manager: GoogleADKWorkflowManager, conversation: SessionRef) -> None:
        self.manager = manager
        self.conversation = conversation

    def start(self, agent: str, call_id: str, request: str, *, at: int,
              parent: RunRef | None = None) -> RunRef:
        ref = self.manager.agent_runs.run_for(self.conversation, parent, call_id, agent)
        self._observe(WorkflowEvent.agent_run(
            ref, "started", title=request.splitlines()[0], call_id=call_id
        ), at)
        return ref

    def end(self, ref: RunRef, status: str, *, at: int, result: str = "",
            error: str | None = None, partial: bool = False) -> None:
        self._observe(WorkflowEvent.agent_run(
            ref, status, result=result, error=error, partial=partial
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


async def agents(manager, args: str = "", *, session_id: str = "s1"):
    """Run ``/agents <args>`` in an app whose conversation is ``session_id``."""
    app = SimpleNamespace(workflow=manager, session_id=session_id, session=autospec_session())
    await AgentsCommand().execute(args, app)
    return app.session


def rows_of(table) -> list[tuple[str, ...]]:
    """A rich table's rows, as text."""
    columns = [[str(cell) for cell in column.cells] for column in table.columns]
    return list(zip(*columns))


class TestTheList:
    async def test_every_run_each_followed_by_the_runs_it_started(self, manager):
        crew_ran(manager)

        session = await agents(manager)

        (table,) = session.add_rich.call_args.args
        assert table.title == "Sub-agents"
        assert [column.header for column in table.columns] == [
            "Agent", "Type", "Status", "Run", "Title", "Tokens in/out", "Time",
        ]
        assert rows_of(table) == [
            ("researcher-1", "researcher", "completed", "#1", "Find X.", "1.3k / 310", "12s"),
            ("  writer-1", "writer", "completed (partial)", "#1", "Draft it.", "800 / 2k", "40s"),
            ("researcher-2", "researcher", "running", "#1", "Find Y.", "—", "10s"),
            ("checker-1", "checker", "refused", "#1", "Check it.", "—", "1s"),
            ("fetcher-1", "fetcher", "failed", "#1", "Fetch Z.", "—", "2s"),
        ]

    async def test_a_conversation_without_runs_says_so(self, manager):
        session = await agents(manager)

        session.add_message.assert_called_once_with(
            "system", "No sub-agent runs in this conversation yet."
        )
        session.add_rich.assert_not_called()

    async def test_another_conversations_runs_are_not_listed(self, manager):
        crew_ran(manager)

        session = await agents(manager, session_id="s2")

        session.add_message.assert_called_once_with(
            "system", "No sub-agent runs in this conversation yet."
        )

    async def test_a_title_is_shown_as_written(self, manager):
        """Titles are the model's words: markup in one is text, never style."""
        runs = Runs(manager, manager.session_ref("s1"))
        runs.start("researcher", "c1", "Fix [/x] and [bold]this[/bold]", at=0)

        session = await agents(manager)

        (table,) = session.add_rich.call_args.args
        out = io.StringIO()
        Console(file=out, width=200, color_system=None).print(table)
        assert "Fix [/x] and [bold]this[/bold]" in out.getvalue()

    @pytest.mark.parametrize(
        "seconds, shown",
        [(0, "0s"), (59, "59s"), (60, "1m 00s"), (185, "3m 05s"), (3_725, "1h 02m")],
    )
    async def test_the_time_a_run_took(self, manager, seconds, shown):
        runs = Runs(manager, manager.session_ref("s1"))
        ref = runs.start("researcher", "c1", "Find X.", at=0)
        runs.end(ref, "completed", at=seconds, result="X.")

        session = await agents(manager)

        (table,) = session.add_rich.call_args.args
        assert rows_of(table)[0][-1] == shown


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

    async def test_an_unknown_id_says_so(self, manager):
        crew_ran(manager)

        session = await agents(manager, "nobody-1")

        session.add_warning.assert_called_once_with(
            "No sub-agent nobody-1 in this conversation. Type /agents for the list."
        )
        session.add_rich.assert_not_called()


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


def test_it_is_a_built_in_workflow_command():
    from agentic_cli.cli.app import BaseCLIApp

    registry = CommandRegistry()
    BaseCLIApp._register_builtin_commands(SimpleNamespace(command_registry=registry))

    command = registry.get("agents")
    assert isinstance(command, AgentsCommand)
    assert command.category is CommandCategory.WORKFLOW
    assert command.usage == "/agents [<id>]"
