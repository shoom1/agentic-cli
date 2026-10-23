"""Tests for source-level context trimming detection.

Covers:
- WorkflowEvent.context_trimmed() factory, with what a summary or a trim did
- WorkflowEvent.context_summarizing(), and the CLI's status line for it
- MessageProcessor dispatch for CONTEXT_TRIMMED, and the warning it shows:
  what a summary did, or how many turns a request left out
- LLM_USAGE warns of nothing: a smaller prompt is no trim (the token-drop
  heuristic is gone)
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from agentic_cli.workflow.events import WorkflowEvent, EventType


# === WorkflowEvent.context_trimmed factory ===


class TestContextTrimmedEvent:
    def test_creates_correct_event_type(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        assert event.type == EventType.CONTEXT_TRIMMED

    def test_metadata_fields(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=20,
            messages_after=10,
            source="langgraph",
            agent="researcher",
        )
        assert event.metadata["messages_before"] == 20
        assert event.metadata["messages_after"] == 10
        assert event.metadata["messages_removed"] == 10
        assert event.metadata["source"] == "langgraph"
        assert event.metadata["agent"] == "researcher"

    def test_content_string(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=15, source="an_apps_trim"
        )
        assert "5 messages removed" in event.content
        assert "an_apps_trim" in event.content

    def test_agent_optional(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=10, messages_after=5, source="langgraph"
        )
        assert "agent" not in event.metadata

    def test_zero_messages_removed(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=10, messages_after=10, source="langgraph"
        )
        assert event.metadata["messages_removed"] == 0

    def test_what_a_summary_did(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=11, messages_after=2, source="summarization", agent="solo",
            messages_summarized=10, tokens_before=152_000, tokens_after=9_000,
        )

        assert event.metadata["messages_summarized"] == 10
        assert event.metadata["tokens_before"] == 152_000
        assert event.metadata["tokens_after"] == 9_000
        assert "turns_dropped" not in event.metadata

    def test_what_a_trim_did(self):
        event = WorkflowEvent.context_trimmed(
            messages_before=12, messages_after=6, source="fit_guard", turns_dropped=3
        )

        assert event.metadata["turns_dropped"] == 3
        for key in ("messages_summarized", "tokens_before", "tokens_after"):
            assert key not in event.metadata

    def test_the_counts_are_keyword_only(self):
        with pytest.raises(TypeError):
            WorkflowEvent.context_trimmed(11, 2, "summarization", "solo", 10)


# === WorkflowEvent.context_summarizing ===


class TestContextSummarizingEvent:
    def test_the_line_the_user_reads(self):
        event = WorkflowEvent.context_summarizing("solo")

        assert event.type is EventType.CONTEXT_SUMMARIZING
        assert EventType.CONTEXT_SUMMARIZING.value == "context_summarizing"
        assert event.content == "Summarizing the earlier conversation…"
        assert event.metadata == {"agent": "solo"}

    def test_the_agent_is_optional(self):
        assert WorkflowEvent.context_summarizing().metadata == {}


class TestHandleContextSummarizing:
    async def test_it_is_the_status_line_and_prints_nothing(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from tests.cli.dialog_fakes import autospec_session

        MessageProcessor._EVENT_DISPATCH = None
        handler = MessageProcessor._get_event_dispatch()[EventType.CONTEXT_SUMMARIZING]
        state = _EventProcessingState()
        ui = autospec_session()

        await handler(MessageProcessor(), WorkflowEvent.context_summarizing("solo"),
                      state, ui, MagicMock(), MagicMock())

        assert state.status_line == "Summarizing the earlier conversation…"
        assert ui.method_calls == []

    async def test_the_line_ends_with_the_summary(self):
        # The answer's model call runs next, and reports nothing until it ends.
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from tests.cli.dialog_fakes import autospec_session

        MessageProcessor._EVENT_DISPATCH = None
        dispatch = MessageProcessor._get_event_dispatch()
        state = _EventProcessingState()
        usage = WorkflowEvent.llm_usage(model="local:small", prompt_tokens=10, completion_tokens=5)
        usage.metadata["purpose"] = "summarization"

        for event in (WorkflowEvent.context_summarizing("solo"), usage):
            await dispatch[event.type](MessageProcessor(), event, state, autospec_session(),
                                       MagicMock(), MagicMock())

        assert state.status_line == "Processing..."

    async def test_other_usage_leaves_the_line_alone(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from tests.cli.dialog_fakes import autospec_session

        MessageProcessor._EVENT_DISPATCH = None
        state = _EventProcessingState(status_line="Calling: web_fetch")
        usage = WorkflowEvent.llm_usage(model="local:small", prompt_tokens=10, completion_tokens=5)

        await MessageProcessor._get_event_dispatch()[EventType.LLM_USAGE](
            MessageProcessor(), usage, state, autospec_session(), MagicMock(), MagicMock()
        )

        assert state.status_line == "Calling: web_fetch"

    async def test_a_turn_that_summarizes_shows_only_its_answer(self):
        from tests.test_message_processor_render import _render

        ui, _, _ = await _render([
            WorkflowEvent.context_summarizing("solo"),
            WorkflowEvent.text("The answer."),
        ])

        assert ui.responses() == ["The answer."]
        assert not ui.warnings() and not ui.of("message") and not ui.rich()


# === Dispatch table includes CONTEXT_TRIMMED ===


class TestDispatchTable:
    def test_context_trimmed_in_dispatch(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        # Clear cached dispatch table so it rebuilds
        MessageProcessor._EVENT_DISPATCH = None
        dispatch = MessageProcessor._get_event_dispatch()
        assert EventType.CONTEXT_TRIMMED in dispatch

    def test_handler_is_handle_context_trimmed(self):
        from agentic_cli.cli.message_processor import MessageProcessor

        MessageProcessor._EVENT_DISPATCH = None
        dispatch = MessageProcessor._get_event_dispatch()
        handler = dispatch[EventType.CONTEXT_TRIMMED]
        assert handler.__name__ == "_handle_context_trimmed"


# === _handle_context_trimmed behavior ===


class TestHandleContextTrimmed:
    @pytest.fixture
    def setup(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from agentic_cli.cli.usage_tracker import UsageTracker

        processor = MessageProcessor()
        state = _EventProcessingState(usage_tracker=UsageTracker())
        ui = MagicMock()
        settings = MagicMock()
        workflow = SimpleNamespace(root_agent_name="solo")  # the main agent's events
        return processor, state, ui, settings, workflow

    async def test_increments_trimmed_count(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        await processor._handle_context_trimmed(event, state, ui, settings, workflow)
        assert state.usage_tracker.context_trimmed_count == 1

    async def test_another_sources_trim_reads_as_before(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(
            messages_before=20, messages_after=10, source="langgraph"
        )
        await processor._handle_context_trimmed(event, state, ui, settings, workflow)
        ui.add_warning.assert_called_once_with("Context trimmed: 10 messages removed (langgraph)")

    @pytest.mark.parametrize("source", ["summarization", "compact"])
    async def test_a_summary_says_what_it_did(self, setup, source):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(
            49, 2, source, "solo",
            messages_summarized=48, tokens_before=152_000, tokens_after=9_000,
        )

        await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        ui.add_warning.assert_called_once_with("Context summarized: 48 messages, 152k → 9k tokens")
        assert state.usage_tracker.context_trimmed_count == 1

    async def test_a_summary_without_its_tokens_says_how_many_messages(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(49, 2, "compact", messages_summarized=48)

        await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        ui.add_warning.assert_called_once_with("Context summarized: 48 messages")

    async def test_a_request_trimmed_to_fit_says_how_many_turns(self, setup):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(12, 6, "fit_guard", "solo", turns_dropped=3)

        await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        ui.add_warning.assert_called_once_with(
            "Context: 3 earlier turns left out of this request to fit the window"
        )
        assert state.usage_tracker.context_trimmed_count == 1

    async def test_one_is_one(self, setup):
        processor, state, ui, settings, workflow = setup
        for event in (
            WorkflowEvent.context_trimmed(3, 2, "summarization", messages_summarized=1),
            WorkflowEvent.context_trimmed(4, 2, "fit_guard", turns_dropped=1),
        ):
            await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        assert [c.args[0] for c in ui.add_warning.call_args_list] == [
            "Context summarized: 1 message",
            "Context: 1 earlier turn left out of this request to fit the window",
        ]

    @pytest.mark.parametrize("event, warning", [
        (WorkflowEvent.context_trimmed(
            49, 2, "summarization", "researcher",
            messages_summarized=48, tokens_before=152_000, tokens_after=9_000,
        ), "researcher: context summarized: 48 messages, 152k → 9k tokens"),
        (WorkflowEvent.context_trimmed(12, 6, "fit_guard", "researcher", turns_dropped=3),
         "researcher: context: 3 earlier turns left out of this request to fit the window"),
        (WorkflowEvent.context_trimmed(12, 6, "an_apps_trim", "researcher"),
         "researcher: context trimmed: 6 messages removed (an_apps_trim)"),
    ])
    async def test_a_sub_agents_is_prefixed_with_its_name(self, setup, event, warning):
        # Its own context was made smaller, not the main conversation.
        processor, state, ui, settings, _ = setup
        workflow = SimpleNamespace(root_agent_name="coordinator")

        await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        ui.add_warning.assert_called_once_with(warning)

    @pytest.mark.parametrize("workflow", [
        SimpleNamespace(root_agent_name="solo"),
        SimpleNamespace(root_agent_name=None),  # a manager that names no root
        object(),  # one without root_agent_name
    ])
    async def test_the_main_agents_or_an_unknown_roots_is_not(self, setup, workflow):
        processor, state, ui, settings, _ = setup
        event = WorkflowEvent.context_trimmed(12, 6, "fit_guard", "solo", turns_dropped=3)

        await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        ui.add_warning.assert_called_once_with(
            "Context: 3 earlier turns left out of this request to fit the window"
        )

    @pytest.mark.parametrize("source", ["summarization", "fit_guard"])
    async def test_without_its_counts_it_reads_as_before(self, setup, source):
        processor, state, ui, settings, workflow = setup
        event = WorkflowEvent.context_trimmed(12, 6, source)

        await processor._handle_context_trimmed(event, state, ui, settings, workflow)

        ui.add_warning.assert_called_once_with(f"Context trimmed: 6 messages removed ({source})")


# === LLM_USAGE warns of nothing ===


class TestUsageWarnsNothing:
    @pytest.fixture
    def setup(self):
        from agentic_cli.cli.message_processor import (
            MessageProcessor,
            _EventProcessingState,
        )
        from agentic_cli.cli.usage_tracker import UsageTracker
        from tests.cli.dialog_fakes import autospec_session

        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 100_000, "completion_tokens": 100,
                        "context_trigger": 170_000, "window_known": True})
        controller = MagicMock()
        state = _EventProcessingState(usage_tracker=tracker, workflow_controller=controller)
        return MessageProcessor(), state, autospec_session(), controller

    async def test_a_smaller_prompt_is_no_trim(self, setup):
        processor, state, ui, controller = setup
        usage = WorkflowEvent.llm_usage(model="m", prompt_tokens=9_000, completion_tokens=200)
        usage.metadata.update(context_trigger=170_000, window_known=True)

        await processor._handle_llm_usage(usage, state, ui, MagicMock(), MagicMock())

        assert ui.method_calls == []
        assert state.usage_tracker.context_trimmed_count == 0
        assert state.usage_tracker.format_context() == "ctx 9k/170k (5%)"
        controller.update_status_bar.assert_called_once_with(ui)

    def test_the_turn_keeps_no_trim_flag(self):
        from agentic_cli.cli.message_processor import _EventProcessingState

        assert not hasattr(_EventProcessingState(), "_context_trimmed_this_invocation")

    async def test_a_turn_that_summarizes_warns_once_and_shows_the_answers_prompt(self):
        from agentic_cli.cli.message_processor import MessageProcessor
        from agentic_cli.cli.usage_tracker import UsageTracker
        from tests.event_replay import RecordingSession, ReplayController, ReplayWorkflow
        from tests.test_message_processor_render import _settings

        def usage(prompt: int, **extra) -> WorkflowEvent:
            event = WorkflowEvent.llm_usage(model="m", prompt_tokens=prompt, completion_tokens=50)
            event.metadata.update(agent="solo", context_trigger=170_000, window_known=True,
                                  **extra)
            return event

        tracker = UsageTracker()
        tracker.record({"prompt_tokens": 180_000, "context_trigger": 170_000,
                        "window_known": True})
        ui = RecordingSession()
        await MessageProcessor().process(
            message="hi",
            workflow_controller=ReplayController(ReplayWorkflow([
                WorkflowEvent.context_summarizing("solo"),
                usage(150_000, purpose="summarization"),
                WorkflowEvent.context_trimmed(
                    49, 2, "summarization", "solo",
                    messages_summarized=48, tokens_before=152_000, tokens_after=9_000,
                ),
                WorkflowEvent.text("The answer."),
                usage(9_500),
            ])),
            ui=ui,
            settings=_settings(False),
            usage_tracker=tracker,
        )

        assert ui.warnings() == ["Context summarized: 48 messages, 152k → 9k tokens"]
        assert ui.responses() == ["The answer."]
        assert tracker.format_context() == "ctx 9.5k/170k (6%)"
