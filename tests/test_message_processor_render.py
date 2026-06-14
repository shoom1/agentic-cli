"""Deterministic render tests for MessageProcessor (CLI output shape).

These assert *how a workflow event stream is rendered to the CLI* — which
``add_response``/``add_rich``/``add_warning``/thinking-box calls happen, with
what content, in what order — without any LLM. The event stream is fixed in
code (a "golden script"), fed through ``MessageProcessor.process()`` with a
``RecordingSession`` standing in for the real ``ThinkingPromptSession``.

This is the deterministic, CI-gateable half of agent testing: it pins the
rendering logic (tool-result formatting, the Tasks box lifecycle, the events
box, warnings) while the non-deterministic agent behavior is covered
separately by the live scenario tests in ``tests/integration``.

To snapshot a *real* run instead of the in-code script, dump it from a live
scenario test (``events_to_dicts`` → JSON) and load it here with
``events_from_dicts``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_cli.cli.message_processor import MessageProcessor
from agentic_cli.workflow.events import WorkflowEvent
from tests.event_replay import RecordingSession, ReplayController, ReplayWorkflow


def _settings(verbose_thinking: bool = False) -> SimpleNamespace:
    """Minimal settings stub — only the fields process() reads."""
    return SimpleNamespace(default_user="tester", verbose_thinking=verbose_thinking)


async def _render(
    events: list[WorkflowEvent],
    *,
    verbose_thinking: bool = False,
) -> tuple[RecordingSession, ReplayController, MessageProcessor]:
    """Render an event stream through MessageProcessor with a recording UI."""
    mp = MessageProcessor()
    ui = RecordingSession()
    ctrl = ReplayController(ReplayWorkflow(events))
    await mp.process(
        message="hi",
        workflow_controller=ctrl,
        ui=ui,
        settings=_settings(verbose_thinking),
    )
    return ui, ctrl, mp


# A representative "planning" turn: think → call save_plan → result → task box
# → text response → usage.
GOLDEN_PLANNING_TURN = [
    WorkflowEvent.thinking("Planning the research."),
    WorkflowEvent.tool_call("save_plan", {"content": "- [ ] Search arXiv"}),
    WorkflowEvent.tool_result(
        "save_plan", {"success": True}, success=True, duration_ms=8
    ),
    WorkflowEvent.task_progress(
        "[ ] Search arXiv",
        {"total": 1, "completed": 0},
        current_task_description="Search arXiv",
    ),
    WorkflowEvent.text("Here's the plan:\n- [ ] Search arXiv"),
    WorkflowEvent.llm_usage(
        "gemini-2.0-flash", prompt_tokens=120, completion_tokens=30, total_tokens=150
    ),
]


def _index_of(ui: RecordingSession, predicate) -> int:
    """Index of the first recorded call matching predicate, or -1."""
    for i, call in enumerate(ui.calls):
        if predicate(call):
            return i
    return -1


class TestRenderBasics:
    async def test_text_response_rendered_as_markdown(self):
        ui, _, _ = await _render([WorkflowEvent.text("hello **world**")])
        assert ui.responses() == ["hello **world**"]
        # TEXT is rendered with markdown=True
        assert ui.of("response")[0][2] is True

    async def test_events_box_started_and_finished(self):
        ui, _, _ = await _render([WorkflowEvent.text("done")])
        starts = [c for c in ui.of("start_thinking") if c[1] == "events"]
        finishes = [c for c in ui.of("ctx_finish") if c[1] == "events"]
        assert starts, "events thinking box was never started"
        assert finishes, "events thinking box was never finished"

    async def test_no_error_calls_on_clean_stream(self):
        ui, _, _ = await _render(GOLDEN_PLANNING_TURN)
        assert ui.errors() == []


class TestToolResultRendering:
    async def test_success_result_uses_green_plus_marker(self):
        ui, _, _ = await _render(GOLDEN_PLANNING_TURN)
        rich = ui.rich()
        assert any("save_plan" in r and "green" in r and "+" in r for r in rich), rich

    async def test_failed_result_uses_red_x_marker(self):
        events = [
            WorkflowEvent.tool_result(
                "kb_search",
                {"success": False, "error": "index missing"},
                success=False,
            ),
        ]
        ui, _, _ = await _render(events)
        rich = ui.rich()
        assert any("kb_search" in r and "red" in r and "x" in r for r in rich), rich
        # The failure message is surfaced, not swallowed.
        assert any("index missing" in r for r in rich), rich


class TestTaskBoxLifecycle:
    async def test_task_box_created_with_progress_title(self):
        ui, _, _ = await _render(GOLDEN_PLANNING_TURN)
        task_starts = [c for c in ui.of("start_thinking") if c[1].startswith("Tasks:")]
        assert task_starts, "Tasks box was not created on progress event"

    async def test_task_box_destroyed_when_all_complete(self):
        events = [
            WorkflowEvent.task_progress("[ ] step", {"total": 1, "completed": 0}),
            WorkflowEvent.task_progress("[x] step", {"total": 1, "completed": 1}),
        ]
        ui, _, _ = await _render(events)
        # The Tasks box that was created must also be finished.
        finishes = [c for c in ui.of("ctx_finish") if c[1].startswith("Tasks:")]
        assert finishes, "completed Tasks box was not finished"


class TestThinkingAndWarnings:
    async def test_thinking_hidden_by_default(self):
        ui, _, _ = await _render([WorkflowEvent.thinking("secret reasoning")])
        assert not any(
            c[0] == "message" and "secret reasoning" in c[2] for c in ui.calls
        )

    async def test_thinking_shown_when_verbose(self):
        ui, _, _ = await _render(
            [WorkflowEvent.thinking("visible reasoning")], verbose_thinking=True
        )
        assert any(
            c[0] == "message" and c[1] == "system" and "visible reasoning" in c[2]
            for c in ui.calls
        )

    async def test_context_trimmed_emits_warning(self):
        events = [
            WorkflowEvent.context_trimmed(
                messages_before=10, messages_after=4, source="langgraph"
            )
        ]
        ui, _, _ = await _render(events)
        assert any("6 messages removed" in w for w in ui.warnings()), ui.warnings()


class TestStatusBarAndOrder:
    async def test_status_bar_updated(self):
        _, ctrl, _ = await _render(GOLDEN_PLANNING_TURN)
        assert ctrl.status_bar_updates >= 1

    async def test_render_order_snapshot(self):
        """Key calls appear in the expected order for a full turn."""
        ui, _, _ = await _render(GOLDEN_PLANNING_TURN)

        i_events_start = _index_of(
            ui, lambda c: c[0] == "start_thinking" and c[1] == "events"
        )
        i_tool_rich = _index_of(ui, lambda c: c[0] == "rich" and "save_plan" in c[1])
        i_text = _index_of(ui, lambda c: c[0] == "response")
        i_events_finish = _index_of(
            ui, lambda c: c[0] == "ctx_finish" and c[1] == "events"
        )

        assert -1 not in (i_events_start, i_tool_rich, i_text, i_events_finish)
        assert i_events_start < i_tool_rich < i_text < i_events_finish


class TestRecordReplayRoundTrip:
    async def test_events_survive_serialization(self):
        """A serialized stream replays to the same rendered output."""
        from tests.event_replay import events_from_dicts, events_to_dicts

        ui_direct, _, _ = await _render(GOLDEN_PLANNING_TURN)
        round_tripped = events_from_dicts(events_to_dicts(GOLDEN_PLANNING_TURN))
        ui_replayed, _, _ = await _render(round_tripped)

        assert ui_direct.kinds() == ui_replayed.kinds()
        assert ui_direct.responses() == ui_replayed.responses()
