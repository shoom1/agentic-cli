"""A sub-agent's view (``cli/agent_view.py``): ``/agent <id>`` and ``/back``.

The app is a real ``BaseCLIApp`` around a real manager, the shared
contract's crew (``tests/workflow/test_subagents_contract.py``: ADK on
``FakeAPI``, Deep Agents on ``ScriptedChat``), its session a fake. Entering a
view prints its rule and a recap of the instance, the prompt becomes
``<id>> `` and the status bar names it, its context and its tokens; a line
typed there is the instance's next run (``run_instance``), drawn as a turn
and run in a task of its own, or, while its run works, given to that run
(``steer_instance``); Ctrl+C stops the run in view (``AgentView.stop``: the
user's stop). ``/back`` waits for a run still working, prints its rule and,
once a run the user started there ended, tells the main agent in a notice
turn (``tell_root``); the main agent's notices (finished jobs) wait while a
view is open. ``/compact`` is refused there, ``/new`` and ``/resume`` leave
it first, ``/status`` shows the instance's context, and a closed instance's
view is read-only.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.agent_view import (
    BACK_RULE,
    COMPACT_IN_VIEW,
    JOBS_WAIT,
    NOT_IN_VIEW,
    READ_ONLY,
    STEERED,
    TELLING,
)
from agentic_cli.cli.agents_command import NO_SUCH_AGENT
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.message_processor import CANCELLED_LINE
from agentic_cli.cli.usage_tracker import UsageTracker, format_tokens
from agentic_cli.tools.jobs.backends import TERMINAL_STATES
from agentic_cli.workflow.agents import (
    INSTANCE_IDLE,
    INSTANCE_STOPPED,
    RUN_CANCELLED,
    RUN_COMPLETED,
)
from agentic_cli.workflow.agents.envelope import view_notice
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.loop_guard import no_blocking_calls
from tests.workflow.test_subagents_contract import (  # noqa: F401 - fixtures
    BLOCKING,
    HELD,
    SESSION,
    USER,
    Crew,
    agent,
    backend,
    calls,
    contract_start_job,
    delegate,
    follow_team,
    make_crew,
    parent_team,
    project,
    says,
    until,
    use,
)

RULE = "── researcher-1 · researcher · idle ──"


@pytest.fixture(autouse=True)
def quiet_app(monkeypatch):
    """No global logging configuration or warning hook left behind."""
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "install_log_filters", lambda: None)


async def app_on(crew: Crew) -> BaseCLIApp:
    """A real app whose controller holds ``crew``'s manager (started), in the
    crew's conversation; its session a fake, each box a box of its own."""
    await crew.manager.initialize_services()
    app = BaseCLIApp(
        app_info=AppInfo(name="View test", version="0"),
        agent_configs=[agent("coordinator")],
        settings=crew.manager.settings,
        session_id=SESSION,
    )
    app.session = opening_boxes(autospec_session())
    app.session.message = ">>> "
    app._workflow_controller._workflow = crew.manager
    return app


async def typed(app: BaseCLIApp, *lines: str) -> None:
    """Each line as the user types it, the view's run waited for."""
    for line in lines:
        await app.process_input(line)
        if app.agent_view is not None:
            await asyncio.wait_for(app.agent_view.settled(), 10)


async def in_a_view(
    make_crew, *, researcher=(), coordinator=(), team=None, **crew_args
) -> tuple[Crew, BaseCLIApp]:
    """A crew whose coordinator delegated once (``researcher-1``: "X is 42.",
    1,200 tokens in, 40 out), and an app in ``researcher-1``'s view."""
    crew = make_crew(
        team or follow_team(),
        coordinator=[calls(delegate("researcher", "Find X.\nIn the docs.", "c1")),
                     says("Done."), *coordinator],
        researcher=[says("X is 42.", usage=(1_200, 40)), *researcher],
        **crew_args,
    )
    app = await app_on(crew)
    await typed(app, "Find X.", "/agent researcher-1")
    return crew, app


def said(session, method: str) -> list[str]:
    """What the fake session was told through ``method``, in order."""
    made = getattr(session, method).call_args_list
    return [c.args[1] if method == "add_message" else c.args[0] for c in made]


def rows_of(table) -> list[tuple[str, ...]]:
    """A rich table's rows, as text."""
    columns = [[str(cell) for cell in column.cells] for column in table.columns]
    return list(zip(*columns))


def status(app: BaseCLIApp) -> str:
    """The status bar as last drawn."""
    return app.session.set_status.call_args.args[0]


def ctx(crew: Crew, prompt: int) -> str:
    """``ctx``, for the researcher's last prompt against its own trigger."""
    budget = crew.manager.context_budget("researcher")
    return UsageTracker(
        last_prompt_tokens=prompt, context_trigger=budget.trigger,
        window_known=budget.window_known,
    ).format_context()


class TestEntering:
    """``/agent <id>`` (or Enter in ``/agents``' picker) opens the view: its
    rule and recap, its prompt, its status bar."""

    async def test_its_rule_and_recap_then_its_prompt_and_status_bar(self, make_crew):
        crew, app = await in_a_view(make_crew)

        assert said(app.session, "add_message")[-2:] == ["/agent researcher-1", RULE]
        (recap,) = app.session.add_rich.call_args.args
        assert rows_of(recap) == [
            ("Title", "Find X."), ("Objective", "Find X.\nIn the docs."),
            ("Last run", "#1 · completed"), ("Result", "X is 42."),
            ("Tokens", "1.2k in / 40 out"),
        ]
        assert app.session.message == "researcher-1> "
        assert app.agent_view.instance == "researcher-1"
        assert status(app) == (
            f"researcher-1 · idle | {ctx(crew, 1_200)} | Tokens: 1.2k in / 40 out"
            " | Ctrl+C: cancel | /back: main"
        )

    async def test_enter_in_the_agents_picker_opens_it(self, make_crew):
        crew = make_crew(
            follow_team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[says("X is 42.")],
        )
        app = await app_on(crew)

        async def first(dialog):
            return dialog.fields[0].options[0]

        app.session.show_dialog.side_effect = first
        await typed(app, "Find X.", "/agents")

        assert said(app.session, "add_message")[-1] == RULE
        assert app.session.message == "researcher-1> "

    async def test_an_unknown_id_opens_nothing(self, make_crew):
        _, app = await in_a_view(make_crew)
        await typed(app, "/back", "/agent researcher-9")

        app.session.add_warning.assert_called_once_with(
            NO_SUCH_AGENT.format(id="researcher-9")
        )
        assert app.agent_view is None
        assert app.session.message == ">>> "

    async def test_a_closed_instances_view_has_no_prompt_and_takes_no_line(self, make_crew):
        crew, app = await in_a_view(make_crew, team=follow_team(continuable=False))
        await typed(app, "Go on.")

        assert said(app.session, "add_message")[-1] == "── researcher-1 · researcher · closed ──"
        assert app.session.message == ""
        app.session.add_warning.assert_called_once_with(READ_ONLY.format(instance="researcher-1"))
        assert crew.requests("researcher") == 1
        await typed(app, "/back")
        assert app.session.message == ">>> "


class TestALineTyped:
    """A line typed in the view is the instance's next run, drawn as a turn,
    or, while that run works, given to it; Ctrl+C stops the run in view."""

    async def test_an_idle_ones_line_is_its_next_run_drawn_as_a_turn(self, make_crew):
        crew, app = await in_a_view(make_crew, researcher=[says("Y is 7.", usage=(1_500, 20))])

        await typed(app, "Now find Y.")

        assert said(app.session, "add_response") == ["Done.", "Y is 7."]
        assert said(app.session, "add_message")[-1] == "Now find Y."
        assert [(r.ref.run, r.status, r.started_by) for r in crew.records()] == [
            ("researcher-1#1", RUN_COMPLETED, "agent:root"),
            ("researcher-1#2", RUN_COMPLETED, "user"),
        ]
        assert crew.requests("coordinator") == 2  # the main agent did not run
        assert crew.last_message("researcher", 1) == "Now find Y."
        assert "X is 42." in crew.bodies("researcher")[1]
        assert app.session.message == "researcher-1> "
        assert status(app).startswith(
            f"researcher-1 · idle | {ctx(crew, 1_500)} | Tokens: 2.7k in / 60 out"
        )

    async def test_a_line_typed_while_its_run_works_is_given_to_that_run(self, make_crew):
        crew, app = await in_a_view(
            make_crew, team=parent_team(),
            researcher=[calls(use("contract_hold", "r1")), says("X is 42; Z is 9.")],
        )

        await app.process_input("Hold on.")
        await asyncio.wait_for(HELD["started"].wait(), 10)
        await app.process_input("Also find Z.")
        HELD["go"].set()
        await asyncio.wait_for(app.agent_view.settled(), 10)

        assert said(app.session, "add_message")[-2:] == [
            "Also find Z.", STEERED.format(instance="researcher-1"),
        ]
        assert crew.last_message("researcher", -1) == "Also find Z."
        assert [r.ref.run for r in crew.records()] == ["researcher-1#1", "researcher-1#2"]
        assert said(app.session, "add_response")[-1] == "X is 42; Z is 9."

    async def test_ctrl_c_stops_the_run_in_view_the_users_stop(self, make_crew):
        crew, app = await in_a_view(
            make_crew, researcher=[calls(use("contract_block", "r1")), says("Went on.")],
        )

        await app.process_input("Wait here.")
        await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        assert app.agent_view.stop() is True
        await asyncio.wait_for(app.agent_view.settled(), 10)

        second = crew.records()[-1]
        assert (second.ref.run, second.status, second.cancelled_by) == (
            "researcher-1#2", RUN_CANCELLED, "user",
        )
        assert crew.instances() == [("researcher-1", INSTANCE_STOPPED)]
        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1
        assert app.session.message == "researcher-1> "
        assert status(app).startswith("researcher-1 · stopped")
        await typed(app, "Go on.")
        assert crew.instances() == [("researcher-1", INSTANCE_IDLE)]
        assert said(app.session, "add_response")[-1] == "Went on."

    async def test_after_a_model_switch_in_a_view_a_line_runs_on_the_new_model(self, make_crew):
        crew, app = await in_a_view(make_crew, researcher=[says("Y is 7.")])

        assert await app.apply_settings({"model": "claude-haiku-4-5"})
        await typed(app, "Now find Y.")

        assert crew.models("researcher") == ["claude-sonnet-4-6", "claude-haiku-4-5"]
        assert "X is 42." in crew.bodies("researcher")[1]
        assert [(r.ref.run, r.status) for r in crew.records()] == [
            ("researcher-1#1", RUN_COMPLETED), ("researcher-1#2", RUN_COMPLETED),
        ]
        assert app.agent_view.instance == "researcher-1"
        assert app.session.message == "researcher-1> "
        assert said(app.session, "add_response")[-1] == "Y is 7."
        assert status(app).startswith("researcher-1 · idle")


class TestLeavingAndSwitching:
    """``/back`` puts the prompt and the status bar back; ``/agent`` with
    another id leaves this view for that one; an instance closed in its own
    view leaves it read-only."""

    async def test_back_prints_its_rule_and_puts_the_prompt_and_bar_back(self, make_crew):
        _, app = await in_a_view(make_crew)

        await typed(app, "/back")

        assert said(app.session, "add_message")[-1] == BACK_RULE
        assert app.agent_view is None
        assert app.session.message == ">>> "
        assert status(app).endswith("Ctrl+C: cancel | /help: commands")

    async def test_outside_a_view_it_says_so(self, make_crew):
        crew = make_crew(follow_team())
        app = await app_on(crew)

        await typed(app, "/back")

        assert said(app.session, "add_message")[-1] == NOT_IN_VIEW

    async def test_agent_with_another_id_leaves_this_view_for_that_one(self, make_crew):
        crew = make_crew(
            follow_team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1"),
                               delegate("researcher", "Find Y.", "c2")), says("Done.")],
            researcher=[says("X is 42."), says("Y is 7.")],
        )
        app = await app_on(crew)

        await typed(app, "Find X and Y.", "/agent researcher-1", "/agent researcher-2")

        assert said(app.session, "add_message")[-3:] == [
            "/agent researcher-2", BACK_RULE, "── researcher-2 · researcher · idle ──",
        ]
        assert app.session.message == "researcher-2> "

    async def test_closing_the_instance_in_view_leaves_its_view_read_only(self, make_crew):
        crew, app = await in_a_view(make_crew)

        await typed(app, "/agent close researcher-1")

        assert app.session.message == ""
        assert status(app).startswith("researcher-1 · closed")


# -- /back's notice; the main conversation's commands in a view ------------------------


class TestTheMainAgentIsTold:
    """``/back`` tells the main agent once, in a notice turn, when a run the
    user started there ended; its own notices wait while a view is open."""

    async def test_after_a_run_of_yours_the_main_agent_is_told_once(self, make_crew):
        crew, app = await in_a_view(
            make_crew, coordinator=[says("Noted: Y is 7.")], researcher=[says("Y is 7.")],
        )

        await typed(app, "Now find Y.", "/back")

        latest = crew.manager.instance_result("researcher-1", session_id=SESSION)
        assert said(app.session, "add_message")[-3:] == [
            "/back", BACK_RULE, TELLING.format(instance="researcher-1"),
        ]
        assert crew.last_message("coordinator") == view_notice("researcher-1", latest)
        assert said(app.session, "add_response")[-1] == "Noted: Y is 7."
        assert app.agent_view is None
        assert app.session.message == ">>> "
        assert not status(app).startswith("researcher-1")
        recent = await crew.manager.recent_messages(SESSION, user_id=USER)
        assert [m["content"] for m in recent] == ["Find X.", "Done.", "Noted: Y is 7."]
        # Back in the view and out again: nothing new to tell.
        await typed(app, "/agent researcher-1", "/back")
        assert crew.requests("coordinator") == 3

    async def test_without_a_run_of_yours_nothing_is_told(self, make_crew):
        crew, app = await in_a_view(make_crew)

        await typed(app, "/back")

        assert said(app.session, "add_message")[-1] == BACK_RULE
        assert crew.requests("coordinator") == 2

    async def test_it_waits_for_the_run_in_view_and_tells_its_result(self, make_crew):
        crew, app = await in_a_view(
            make_crew, team=parent_team(), coordinator=[says("Noted.")],
            researcher=[calls(use("contract_hold", "r1")), says("Z is 9.")],
        )

        await app.process_input("Hold on.")
        await asyncio.wait_for(HELD["started"].wait(), 10)
        back = asyncio.create_task(app.process_input("/back"))
        await asyncio.sleep(0.05)
        assert not back.done()
        HELD["go"].set()
        await asyncio.wait_for(back, 10)

        assert '"result": "Z is 9."' in crew.last_message("coordinator")
        assert app.agent_view is None

    async def test_the_main_agents_notices_wait_until_back(self, make_crew):
        crew, app = await in_a_view(
            make_crew,
            team=[agent("coordinator", sub_agents=("researcher",)),
                  agent("researcher", tools=(contract_start_job,))],
            coordinator=[says("Noted."), says("The build finished.")],
            researcher=[calls(use("contract_start_job", "r1")), says("Started the build.")],
        )
        app.settings.update_setting("job_auto_resume", True)
        jobs = crew.manager.job_manager

        await typed(app, "Start the build.")
        (record,) = jobs.list()
        await until(lambda: jobs.get(record.job_id).state in TERMINAL_STATES)
        await typed(app, "/jobs resume")

        assert said(app.session, "add_message")[-1] == JOBS_WAIT
        assert crew.requests("coordinator") == 2
        await typed(app, "/back")
        assert said(app.session, "add_response")[-2:] == ["Noted.", "The build finished."]
        assert jobs.awaiting_resume() == []


class TestTheMainConversationsCommands:
    """In a view ``/compact`` is refused, ``/new`` and ``/resume`` leave it
    first, and ``/status`` shows the instance's context."""

    async def test_compact_is_refused(self, make_crew):
        crew, app = await in_a_view(make_crew)

        await typed(app, "/compact")

        app.session.add_warning.assert_called_once_with(COMPACT_IN_VIEW)
        assert crew.requests("coordinator") == 2
        assert app.agent_view is not None

    async def test_new_leaves_the_view_first(self, make_crew):
        crew, app = await in_a_view(make_crew)

        await typed(app, "/new")

        messages = said(app.session, "add_message")
        assert messages[-2] == BACK_RULE
        assert messages[-1].startswith("New conversation ")
        assert app.agent_view is None
        assert app.session.message == ">>> "

    async def test_resume_leaves_the_view_first(self, make_crew):
        crew, app = await in_a_view(make_crew, store="sqlite")

        await typed(app, "/resume zzz")

        assert BACK_RULE in said(app.session, "add_message")
        assert said(app.session, "add_error")[-1].startswith("No conversation 'zzz' found")
        assert app.agent_view is None

    async def test_status_shows_the_instances_context(self, make_crew):
        crew, app = await in_a_view(make_crew)
        budget = crew.manager.context_budget("researcher")

        await typed(app, "/status")

        (panel,) = app.session.add_rich.call_args.args
        rows = dict(rows_of(panel.renderable))
        assert rows["Sub-agent"] == "researcher-1 (researcher, idle)"
        assert rows["Summarize At"] == format_tokens(budget.trigger)
        assert rows["Last Prompt"] == "1.2k"
        assert rows["Tokens"] == "1.2k in / 40 out"
        assert "Main Agent" not in rows


class TestOffTheLoop:
    async def test_it_blocks_no_event_loop(self, make_crew):
        crew = make_crew(
            follow_team(), store="sqlite",
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done."),
                         says("Noted.")],
            researcher=[says("X is 42."), says("Y is 7.")],
        )
        app = await app_on(crew)
        await typed(app, "Find X.")

        with no_blocking_calls():
            await typed(app, "/agent researcher-1", "Now find Y.", "/back")

        assert said(app.session, "add_response") == ["Done.", "Y is 7.", "Noted."]
