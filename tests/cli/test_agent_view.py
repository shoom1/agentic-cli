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
import contextlib
import json
from types import MethodType, SimpleNamespace

import pytest

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli import workflow_controller
from agentic_cli.cli.agent_view import (
    BACK_RULE,
    COMPACT_IN_VIEW,
    JOBS_WAIT,
    NOT_IN_VIEW,
    READ_ONLY,
    STEERED,
    TELLING,
    bind_ctrl_c,
)
from agentic_cli.cli import agents_command
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
    RUN_INPUT_REQUIRED,
    RUN_INTERRUPTED,
)
from agentic_cli.workflow.agents.envelope import view_notice
from tests.cli.dialog_fakes import autospec_session, opening_boxes
from tests.loop_guard import no_blocking_calls
from tests.workflow.test_subagents_contract import (  # noqa: F401 - fixtures
    BLOCKING,
    FOLLOWING,
    HELD,
    SESSION,
    SMALL_ENTRY,
    SMALL_MODEL,
    STARTING,
    USER,
    Crew,
    agent,
    backend,
    calls,
    contract_ask,
    contract_hold,
    contract_start_job,
    delegate,
    envelope_read,
    follow_team,
    make_crew,
    message,
    parent_team,
    project,
    says,
    until,
    use,
    user_team,
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


async def settled_or_fail(app: BaseCLIApp) -> None:
    """Wait for the view's run to end; one that does not is the test's failure
    (its tasks are cancelled so that teardown does not wait for them)."""
    try:
        await asyncio.wait_for(app.agent_view.settled(), 10)
    except asyncio.TimeoutError:
        for run in list(app.agent_view._runs):
            run.cancel()
        raise AssertionError("the run in view did not end within 10 s") from None


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
        await settled_or_fail(app)

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

    async def test_ctrl_c_with_a_second_line_waiting_says_so_once(self, make_crew):
        crew, app = await in_a_view(
            make_crew, researcher=[calls(use("contract_block", "r1")), says("Went on.")],
        )

        await app.process_input("Wait here.")
        await app.process_input("And then this.")  # waits for the first run
        await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        assert app.agent_view.stop() is True
        await settled_or_fail(app)

        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1
        assert [r.ref.run for r in crew.records()] == ["researcher-1#1", "researcher-1#2"]

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

    async def test_a_result_over_the_cap_is_told_cut_with_its_note(self, make_crew):
        """The envelope the notice carries passes the tool result cap, as a
        ``task`` result does: cut to ``tool_result_max_chars``, with the note
        that names its full output, saved for ``read_file``."""
        crew, app = await in_a_view(
            make_crew, coordinator=[says("Noted.")], researcher=[says("Y is 7. " * 1_000)],
        )
        crew.manager.settings.tool_result_max_chars = 2_000

        await typed(app, "Now find Y.", "/back")

        notice = crew.last_message("coordinator")
        frame = view_notice("researcher-1", {})[: -len("{}")]
        assert notice.startswith(frame), notice[:200]
        sent = notice[len(frame):]
        assert len(sent) <= 2_000, len(sent)
        read = json.loads(sent)
        assert (read["run"], read["status"], read["success"]) == (
            "researcher-1#2", RUN_COMPLETED, True,
        )
        assert read["result"].startswith("Y is 7.") and "characters cut]" in read["result"]
        assert read["truncated"].startswith("Cut: ") and "tool-output/researcher-1-" in (
            read["truncated"]
        )
        assert crew.manager.instance_result("researcher-1", session_id=SESSION) == read

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

    async def test_status_has_one_model_row_the_instances(self, make_crew):
        crew, app = await in_a_view(make_crew)

        await typed(app, "/status")

        (panel,) = app.session.add_rich.call_args.args
        models = [value for label, value in rows_of(panel.renderable) if label == "Model"]
        assert models == [str(crew.manager.agent_models["researcher"].ref)]


class TestAnInstanceOnAModelChoice:
    """An instance on a model choice with narrowed tools, continued after a
    restart (its records restored in a new manager): ``/agents <id>`` and its
    view say its model and tools, its view's context is its choice's, and
    its next run is on that model with those tools."""

    async def test_after_a_restart_its_view_and_panel_say_so(self, make_crew):
        first = make_crew(
            follow_team(model_choices=(SMALL_MODEL,)), store="sqlite",
            models=SMALL_ENTRY,
            coordinator=[calls(delegate("researcher", "Find X.", "c1",
                                        model=SMALL_MODEL,
                                        tools=["contract_lookup"])), says("Done.")],
            researcher=[says("X is 42.", usage=(1_200, 40))],
        )
        await first.turn()
        await first.manager.cleanup()
        second = make_crew(
            follow_team(model_choices=(SMALL_MODEL,)), store="sqlite",
            models=SMALL_ENTRY,
            researcher=[says("Y is 7.", usage=(900, 30))],
        )
        app = await app_on(second)

        await typed(app, "/agents researcher-1")
        (panel,) = app.session.add_rich.call_args.args
        rows = dict(rows_of(panel.renderable))
        await typed(app, "/agent researcher-1", "Now find Y.")
        bar = status(app)
        await typed(app, "/status")
        (shown,) = app.session.add_rich.call_args.args

        assert (rows["Model"], rows["Tools"]) == (SMALL_MODEL, "contract_lookup")
        assert second.sent_models("researcher") == ["claude-haiku-4-5"]
        assert second.tool_names("researcher") == ["contract_lookup"]
        budget = second.manager.context_budget("researcher", model=SMALL_MODEL)
        assert budget != second.manager.context_budget("researcher")
        assert UsageTracker(
            last_prompt_tokens=900, context_trigger=budget.trigger,
            window_known=budget.window_known,
        ).format_context() in bar
        context = dict(rows_of(shown.renderable))
        assert (context["Model"], context["Summarize At"]) == (
            SMALL_MODEL, format_tokens(budget.trigger)
        )


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


# -- Ctrl+C on the run in view ---------------------------------------------------------


class _Viewed:
    """What the Ctrl+C binding reads of a view: whether its run works, and its stop."""

    def __init__(self) -> None:
        self.working = False
        self.stopped = 0

    def stop(self) -> bool:
        self.stopped += 1
        return True


class TestCtrlCInAView:
    """The run in view works outside the input handler (the prompt stays open
    to steer it), so thinking-prompt's Ctrl+C, which cancels the handler, does
    not reach it: a binding of the session's own stops it, active only while
    that run works (``tests/cli/test_console_turns.py`` presses it)."""

    def test_it_stops_the_run_in_view_only_while_one_works(self):
        from prompt_toolkit.keys import Keys
        from thinking_prompt import ThinkingPromptSession

        session = ThinkingPromptSession(message=">>> ")
        view = _Viewed()

        def runs_on_ctrl_c():
            """The handler prompt_toolkit runs: the last binding whose filter holds."""
            bound = session.app.key_bindings.get_bindings_for_keys((Keys.ControlC,))
            return [binding for binding in bound if binding.filter()][-1].handler

        assert bind_ctrl_c(session, lambda: view) is True
        idle = runs_on_ctrl_c()
        view.working = True
        runs_on_ctrl_c()(None)

        assert idle.__name__ == "cancel"  # thinking-prompt's own
        assert view.stopped == 1

    def test_a_fake_session_gets_none(self):
        assert bind_ctrl_c(autospec_session(), lambda: None) is False


class TestExitInAView:
    async def test_a_run_still_working_at_exit_is_interrupted(self, make_crew, monkeypatch):
        """``/exit`` while the run in view works: the process exit interrupts
        the run, as it would any run (only Ctrl+C in a view is the user's
        stop), before the workflow goes (whose cleanup waits for it). The run
        ends ``interrupted`` and the sub-agent stays ``idle``, so after a
        restart its parent continues it."""
        crew = make_crew(
            follow_team(), store="sqlite",
            coordinator=[calls(delegate("researcher", "Find X.", "c1")), says("Done.")],
            researcher=[says("X is 42."), calls(use("contract_block", "r1"))],
        )
        monkeypatch.setattr(
            workflow_controller, "create_workflow_manager_from_settings",
            lambda **kwargs: crew.manager,
        )
        app = BaseCLIApp(
            app_info=AppInfo(name="View test", version="0"),
            agent_configs=[agent("coordinator")],
            settings=crew.manager.settings,
            session_id=SESSION,
        )
        app.session = opening_boxes(autospec_session())
        app.session.app = SimpleNamespace(is_running=True)

        async def then_exit() -> None:
            """The user's lines, then ``/exit`` once the run in view works."""
            await app.process_input("Find X.")
            await app.process_input("/agent researcher-1")
            await app.process_input("Wait here.")
            await asyncio.wait_for(BLOCKING["started"].wait(), 10)

        app.session.run_async.side_effect = then_exit
        running = asyncio.create_task(app.run())
        exited, _ = await asyncio.wait({running}, timeout=20)
        if not exited:
            # Its workflow's cleanup waits for the run in view: unwind it
            # without the view's own stop (which is what failed), and bound
            # the wait, so that this fails the test instead of stalling it.
            for run in list(app.agent_view._runs):
                run.cancel()
            await asyncio.wait({running}, timeout=10)
            if not running.done():
                running.cancel()
                await asyncio.wait({running}, timeout=5)

        assert exited, "the app did not exit: the run in view was still working"
        last = crew.records()[-1]
        assert (last.ref.run, last.status, last.cancelled_by) == (
            "researcher-1#2", RUN_INTERRUPTED, None,
        )
        assert crew.instances() == [("researcher-1", INSTANCE_IDLE)]

        restarted = make_crew(
            follow_team(), store="sqlite", subagents=FOLLOWING,
            coordinator=[calls(message("researcher-1", "Go on.", "c2")), says("Done.")],
            researcher=[says("Went on.")],
        )
        await asyncio.wait_for(restarted.turn("Go on."), 20)

        read = envelope_read(restarted.results_read("coordinator")["c2"])
        assert (read["run"], read["status"], read["result"]) == (
            "researcher-1#3", RUN_COMPLETED, "Went on.",
        )


class TestCtrlCNotBound:
    async def test_a_session_that_cannot_take_the_binding_is_logged(self, make_crew, monkeypatch):
        """Without the binding Ctrl+C in a view is thinking-prompt's own (it
        ends the boxes, and the run goes on): the log says so at info."""
        crew = make_crew(follow_team())
        monkeypatch.setattr(
            workflow_controller, "create_workflow_manager_from_settings",
            lambda **kwargs: crew.manager,
        )
        events: list[dict] = []

        class Recorder:
            def __getattr__(self, level: str):
                def log(event: str, **fields) -> None:
                    events.append({"event": event, "log_level": level, **fields})

                return log

        monkeypatch.setattr(app_module, "logger", Recorder())
        app = BaseCLIApp(
            app_info=AppInfo(name="View test", version="0"),
            agent_configs=[agent("coordinator")],
            settings=crew.manager.settings,
            session_id=SESSION,
        )
        app.session = opening_boxes(autospec_session())
        app.session.app = SimpleNamespace(is_running=True)  # no prompt_toolkit key bindings

        await asyncio.wait_for(app.run(), 20)

        (logged,) = [e for e in events if e["event"] == "agent_view_ctrl_c_not_bound"]
        assert logged["log_level"] == "info"


# -- one dialog at a time: a run's question and a command's dialog ------------------------


class Dialogs:
    """``show_dialog`` as thinking-prompt's: one dialog at a time, a second
    one raises. Each dialog stays open until its title's gate is set."""

    def __init__(self) -> None:
        self.open: str | None = None
        self.shown: list[str] = []
        self.refused: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.released = False

    def gate(self, title: str) -> asyncio.Event:
        gate = self.gates.setdefault(title, asyncio.Event())
        if self.released:
            gate.set()
        return gate

    def release_all(self) -> None:
        """Let every dialog, now or later, close: a failing test must not leave
        a run waiting for the user."""
        self.released = True
        HELD["go"].set()
        for gate in self.gates.values():
            gate.set()

    async def show(self, dialog):
        if self.open is not None:
            self.refused.append(dialog.title)
            raise RuntimeError("A dialog is already being shown")
        self.open = dialog.title
        self.shown.append(dialog.title)
        try:
            await self.gate(dialog.title).wait()
            return None if dialog.title == "Sub-agents" else True
        finally:
            self.open = None


async def asking_view(make_crew) -> tuple[Crew, BaseCLIApp, Dialogs]:
    """A view of ``researcher-1`` whose next run holds, then asks the user."""
    team = [
        agent("coordinator", sub_agents=("researcher",)),
        agent("researcher", tools=(contract_hold, contract_ask)),
    ]
    crew, app = await in_a_view(
        make_crew, team=team,
        researcher=[calls(use("contract_hold", "r1")), calls(use("contract_ask", "r2")),
                    says("Asked.")],
    )
    dialogs = Dialogs()

    async def show_dialog(session, dialog):
        return await dialogs.show(dialog)

    # A real method, as a real session's is: the app serializes those.
    app.session.show_dialog = MethodType(show_dialog, app.session)
    app.session = app.session
    return crew, app, dialogs


class TestOneDialogAtATime:
    """The view's run works outside the input handler, so a command can open
    a dialog while the run asks its question. Whichever comes second waits
    for the first: a question is never answered unseen (a second
    ``show_dialog`` raises, and a permission prompt that raised is a denial),
    and a command's dialog never fails because a question is open."""

    async def test_a_question_waits_for_an_open_picker(self, make_crew):
        crew, app, dialogs = await asking_view(make_crew)
        try:
            await app.process_input("Ask me.")
            await asyncio.wait_for(HELD["started"].wait(), 10)
            picker = asyncio.create_task(app.process_input("/agents"))
            await until(lambda: dialogs.shown == ["Sub-agents"])
            HELD["go"].set()
            await until(lambda: crew.records()[-1].status == RUN_INPUT_REQUIRED)

            assert dialogs.refused == []
            assert dialogs.shown == ["Sub-agents"]
            dialogs.gate("Sub-agents").set()
            await asyncio.wait_for(picker, 10)
            await until(lambda: "researcher-1 asks" in dialogs.shown)
            dialogs.gate("researcher-1 asks").set()
            await asyncio.wait_for(app.agent_view.settled(), 10)

            assert dialogs.refused == []
            assert crew.records()[-1].status == RUN_COMPLETED
            app.session.add_error.assert_not_called()
        finally:
            dialogs.release_all()

    async def test_a_command_waits_for_an_open_question(self, make_crew, monkeypatch):
        crew, app, dialogs = await asking_view(make_crew)
        reached = asyncio.Event()
        built = agents_command.picker

        def noting(*args, **kwargs):
            reached.set()
            return built(*args, **kwargs)

        monkeypatch.setattr(agents_command, "picker", noting)
        try:
            await app.process_input("Ask me.")
            await asyncio.wait_for(HELD["started"].wait(), 10)
            HELD["go"].set()
            await until(lambda: dialogs.shown == ["researcher-1 asks"])
            picker = asyncio.create_task(app.process_input("/agents"))
            await until(reached.is_set)  # the command has its picker and shows it

            assert dialogs.refused == []
            assert dialogs.shown == ["researcher-1 asks"]
            dialogs.gate("researcher-1 asks").set()
            await until(lambda: "Sub-agents" in dialogs.shown)
            dialogs.gate("Sub-agents").set()
            await asyncio.wait_for(picker, 10)
            await asyncio.wait_for(app.agent_view.settled(), 10)

            assert dialogs.refused == []
            assert crew.records()[-1].status == RUN_COMPLETED
            app.session.add_error.assert_not_called()
        finally:
            dialogs.release_all()

    async def test_an_apps_own_command_waits_for_an_open_question(self, make_crew):
        """A command of an app that calls ``app.session.show_dialog`` itself
        is serialized too: the session's dialogs all take the one lock."""
        from thinking_prompt import Dialog

        from agentic_cli.cli.commands import Command, CommandCategory

        reached = asyncio.Event()

        class Mine(Command):
            def __init__(self) -> None:
                super().__init__(
                    name="mine", description="An app's own dialog", usage="/mine",
                    category=CommandCategory.WORKFLOW,
                )

            async def execute(self, args, app) -> None:
                reached.set()
                await app.session.show_dialog(Dialog("Mine", fields=[]))

        crew, app, dialogs = await asking_view(make_crew)
        app.command_registry.register(Mine())
        try:
            await app.process_input("Ask me.")
            await asyncio.wait_for(HELD["started"].wait(), 10)
            HELD["go"].set()
            await until(lambda: dialogs.shown == ["researcher-1 asks"])
            mine = asyncio.create_task(app.process_input("/mine"))
            await until(reached.is_set)

            assert dialogs.refused == []
            assert dialogs.shown == ["researcher-1 asks"]
            dialogs.gate("researcher-1 asks").set()
            await until(lambda: "Mine" in dialogs.shown)
            dialogs.gate("Mine").set()
            await asyncio.wait_for(mine, 10)
            await settled_or_fail(app)

            assert dialogs.refused == []
            app.session.add_error.assert_not_called()
        finally:
            dialogs.release_all()


# -- a workflow that is down ----------------------------------------------------------------


class Starts:
    """The manager's starts, failing while ``failing`` is set: a model switch
    whose rebuild fails (and whose way back fails too) leaves the workflow
    down until a start succeeds."""

    def __init__(self, crew: Crew) -> None:
        self.failing = True
        self.tried = 0
        original = crew.manager._load_models

        async def load() -> None:
            self.tried += 1
            if self.failing:
                raise RuntimeError("scripted start failure")
            await original()

        crew.manager._load_models = load


class TestAWorkflowThatIsDown:
    """After a model switch whose rebuild failed, the main conversation's line
    retries the start; a line in a view does too, and ``/back`` still tells the
    main agent about a run the user started before."""

    async def test_a_line_retries_the_start_as_the_main_conversations_does(self, make_crew):
        crew, app = await in_a_view(make_crew, researcher=[says("Y is 7.")])
        starts = Starts(crew)
        assert not await app.apply_settings({"model": "claude-haiku-4-5"})
        assert not app._workflow_controller.is_ready
        starts.failing = False

        await typed(app, "Now find Y.")

        app.session.add_warning.assert_not_called()
        assert [(r.ref.run, r.status) for r in crew.records()] == [
            ("researcher-1#1", RUN_COMPLETED), ("researcher-1#2", RUN_COMPLETED),
        ]
        assert said(app.session, "add_response")[-1] == "Y is 7."

    async def test_back_retries_the_start_and_tells_the_main_agent(self, make_crew):
        crew, app = await in_a_view(
            make_crew, coordinator=[says("Noted.")], researcher=[says("Y is 7.")],
        )
        await typed(app, "Now find Y.")
        starts = Starts(crew)
        assert not await app.apply_settings({"model": "claude-haiku-4-5"})
        starts.failing = False

        await typed(app, "/back")

        assert app.agent_view is None
        assert said(app.session, "add_response")[-1] == "Noted."
        assert '"result": "Y is 7."' in crew.last_message("coordinator")


# -- what the main conversation keeps while a view is open ---------------------------------


class TestTheMainConversationInAView:
    async def test_the_main_agents_context_stays_put_after_a_view_run(self, make_crew):
        crew = make_crew(
            follow_team(),
            coordinator=[calls(delegate("researcher", "Find X.", "c1"), usage=(900, 9)),
                         says("Done.", usage=(1_000, 10))],
            researcher=[says("X is 42.", usage=(1_200, 40)), says("Y is 7.", usage=(5_000, 20))],
        )
        app = await app_on(crew)
        await typed(app, "Find X.")
        before = app.usage_tracker.last_prompt_tokens
        assert before  # the main agent's own call moved it

        await typed(app, "/agent researcher-1", "Now find Y.")

        # The instance's call is a sub-agent's: it never moves the main agent's ctx.
        assert app.usage_tracker.last_prompt_tokens == before

    async def test_finished_jobs_wait_in_a_view_and_the_bar_counts_them(self, make_crew):
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

        await app._new_job_monitor().poll_once()

        assert "↻1" in status(app)
        assert status(app).startswith("researcher-1 · ")  # still the view's bar
        assert await app.resume_finished_jobs() == 0
        assert crew.requests("coordinator") == 2
        assert [r.job_id for r in jobs.awaiting_resume()] == [record.job_id]


# -- /agent new: a new instance's view, its first run streamed there ---------------------

NEW_RULE = "── helper-1 · helper · running ──"


async def new_view(make_crew, *, team=None, **crew_args) -> tuple[Crew, BaseCLIApp]:
    """A crew with a helper only the user may start (``user_team``), and an app
    in its main conversation."""
    crew = make_crew(team or user_team(), subagents=STARTING, **crew_args)
    return crew, await app_on(crew)


class TestANewInstancesView:
    """``/agent new <type> <objective>`` starts an instance for the user
    (``start_instance``) and opens its view once its run's start names it:
    the run streams there, a line typed steers it, Ctrl+C is the user's stop
    and ``/back`` tells the main agent once, as for a follow-up run."""

    async def test_its_view_opens_and_its_first_run_streams_there(self, make_crew):
        crew, app = await new_view(make_crew, helper=[says("Helped with X.", usage=(900, 30))])

        await typed(app, "/agent new helper Help with X.")

        assert said(app.session, "add_message")[-2:] == ["/agent new helper Help with X.", NEW_RULE]
        (recap,) = app.session.add_rich.call_args.args
        assert rows_of(recap) == [
            ("Title", "Help with X."), ("Objective", "Help with X."),
            ("Last run", "#1 · running"), ("Tokens", "0 in / 0 out"),
        ]
        assert said(app.session, "add_response") == ["Helped with X."]
        assert app.agent_view.instance == "helper-1"
        assert app.session.message == "helper-1> "
        assert status(app).startswith("helper-1 · idle")
        assert [(r.ref.run, r.status, r.started_by) for r in crew.records()] == [
            ("helper-1#1", RUN_COMPLETED, "user"),
        ]
        assert crew.requests("coordinator") == 0

    async def test_a_line_typed_while_its_first_run_works_is_given_to_that_run(self, make_crew):
        crew, app = await new_view(
            make_crew, helper=[calls(use("contract_hold", "h1")), says("X is 42; Z is 9.")],
        )

        await app.process_input("/agent new helper Hold on.")
        await asyncio.wait_for(HELD["started"].wait(), 10)
        await app.process_input("Also find Z.")
        HELD["go"].set()
        await settled_or_fail(app)

        assert said(app.session, "add_message")[-2:] == [
            "Also find Z.", STEERED.format(instance="helper-1"),
        ]
        assert crew.last_message("helper", -1) == "Also find Z."
        assert [r.ref.run for r in crew.records()] == ["helper-1#1"]
        assert said(app.session, "add_response")[-1] == "X is 42; Z is 9."

    async def test_ctrl_c_stops_its_first_run_the_users_stop(self, make_crew):
        crew, app = await new_view(
            make_crew, helper=[calls(use("contract_block", "h1")), says("Went on.")],
        )

        await app.process_input("/agent new helper Wait here.")
        await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        assert app.agent_view.stop() is True
        await settled_or_fail(app)

        (record,) = crew.records()
        assert (record.status, record.cancelled_by) == (RUN_CANCELLED, "user")
        assert crew.instances() == [("helper-1", INSTANCE_STOPPED)]
        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1
        assert app.session.message == "helper-1> "
        assert status(app).startswith("helper-1 · stopped")
        await typed(app, "Go on.")
        assert crew.instances() == [("helper-1", INSTANCE_IDLE)]
        assert said(app.session, "add_response")[-1] == "Went on."

    async def test_back_tells_the_main_agent_once(self, make_crew):
        crew, app = await new_view(
            make_crew, coordinator=[says("Noted: helped.")], helper=[says("Helped with X.")],
        )

        await typed(app, "/agent new helper Help with X.", "/back")

        latest = crew.manager.instance_result("helper-1", session_id=SESSION)
        assert latest["result"] == "Helped with X."
        assert said(app.session, "add_message")[-3:] == [
            "/back", BACK_RULE, TELLING.format(instance="helper-1"),
        ]
        assert crew.last_message("coordinator") == view_notice("helper-1", latest)
        assert said(app.session, "add_response")[-1] == "Noted: helped."
        assert app.session.message == ">>> "
        # Back in its view and out again: nothing new to tell.
        await typed(app, "/agent helper-1", "/back")
        assert crew.requests("coordinator") == 1

    async def test_a_type_you_cannot_start_opens_no_view(self, make_crew):
        crew, app = await new_view(make_crew)

        await typed(app, "/agent new researcher Find X.")

        assert said(app.session, "add_error")[-1] == (
            "You cannot start 'researcher' here. Types you can start: helper."
        )
        assert app.agent_view is None
        assert app.session.message == ">>> "
        assert crew.records() == []

    async def test_in_a_view_it_leaves_that_view_first(self, make_crew):
        crew, app = await in_a_view(
            make_crew, team=[*follow_team(), *user_team()[2:]], subagents=STARTING,
            coordinator=[says("Noted.")], researcher=[says("Y is 7.")], helper=[says("Helped.")],
        )

        await typed(app, "Now find Y.", "/agent new helper Help.")

        assert said(app.session, "add_message")[-4:] == [
            "/agent new helper Help.", BACK_RULE, TELLING.format(instance="researcher-1"),
            NEW_RULE,
        ]
        assert said(app.session, "add_response")[-2:] == ["Noted.", "Helped."]
        assert app.agent_view.instance == "helper-1"

    async def test_ctrl_c_before_its_view_opens_stops_the_run_and_opens_none(self, make_crew):
        """thinking-prompt's Ctrl+C cancels the command while the start waits
        for the store (before its run started): the run that starts is
        stopped by the user, said once, and no view opens."""
        crew, app = await new_view(make_crew, helper=[says("[never asked]")])
        delegations = crew.manager._delegations
        read, gate = asyncio.Event(), asyncio.Event()
        stored = delegations._stored

        async def slow(conversation):
            read.set()
            await gate.wait()
            return await stored(conversation)

        delegations._stored = slow

        command = asyncio.create_task(app.process_input("/agent new helper Go."))
        await asyncio.wait_for(read.wait(), 10)
        command.cancel()
        await asyncio.sleep(0.05)
        gate.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(command, 10)

        (record,) = crew.records()
        assert (record.status, record.cancelled_by) == (RUN_CANCELLED, "user")
        assert crew.instances() == [("helper-1", INSTANCE_STOPPED)]
        assert said(app.session, "add_message").count(CANCELLED_LINE) == 1
        assert app.agent_view is None
        assert crew.requests("helper") == 0

    async def test_an_interrupt_leaves_it_idle(self, make_crew):
        """What the app's exit does to the run in view (``AgentView.interrupt``):
        no stop of the user's."""
        crew, app = await new_view(
            make_crew, helper=[calls(use("contract_block", "h1")), says("Went on.")],
        )

        await app.process_input("/agent new helper Wait here.")
        await asyncio.wait_for(BLOCKING["started"].wait(), 10)
        assert app.agent_view.interrupt() is True
        await settled_or_fail(app)

        (record,) = crew.records()
        assert (record.status, record.cancelled_by) == (RUN_INTERRUPTED, None)
        assert crew.instances() == [("helper-1", INSTANCE_IDLE)]

    async def test_a_conversation_of_only_this_is_a_used_one_the_exit_names(self, make_crew):
        """Stored (the exit's ``--resume`` hint names a stored conversation) and
        no longer fresh: ``/new`` and a switch of backend treat it as used."""
        crew, app = await new_view(make_crew, store="sqlite", helper=[says("Helped.")])
        app._conversation.opened_empty = True  # as a new conversation is opened

        await typed(app, "/agent new helper Help with X.")

        assert await app._is_stored(app.workflow, app.session_id)
        assert app._conversation.written and not app._conversation.fresh
        assert not app._conversation.turned
        assert app.session_id in app._resume_hint()

    async def test_it_blocks_no_event_loop(self, make_crew):
        crew = make_crew(user_team(), subagents=STARTING, store="sqlite",
                         coordinator=[says("Noted.")], helper=[says("Helped.")])
        app = await app_on(crew)

        with no_blocking_calls():
            await typed(app, "/agent new helper Help.", "/back")

        assert said(app.session, "add_response") == ["Helped.", "Noted."]


class TestAGeneralPurposeInstancesView:
    async def test_it_says_the_model_and_context_of_the_type_that_started_it(self, make_crew):
        """A ``general_purpose`` instance runs on its spawner type's model: its view
        names that model, and its context is that model's."""
        from agentic_cli.workflow.agents import SubAgentPolicy

        crew = make_crew(
            [agent("coordinator", model=SMALL_MODEL)], models=SMALL_ENTRY,
            subagents=SubAgentPolicy(spawn_sources={"agent"}, general_purpose=True),
            coordinator=[calls(delegate("general_purpose", "Sum it up.", "c1")), says("Done.")],
            general_purpose=[says("Summed.", usage=(900, 30))],
        )
        app = await app_on(crew)

        await typed(app, "Sum it up.", "/agent general_purpose-1")
        bar = status(app)
        await typed(app, "/status")

        (panel,) = app.session.add_rich.call_args.args
        rows = dict(rows_of(panel.renderable))
        budget = crew.manager.context_budget("coordinator")
        assert rows["Sub-agent"] == "general_purpose-1 (general_purpose, idle)"
        assert (rows["Model"], rows["Summarize At"]) == (SMALL_MODEL, format_tokens(budget.trigger))
        assert UsageTracker(
            last_prompt_tokens=900, context_trigger=budget.trigger,
            window_known=budget.window_known,
        ).format_context() in bar
