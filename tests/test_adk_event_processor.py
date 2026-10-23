"""Tests for ADK event processor after ThinkingDetector removal."""

from unittest.mock import MagicMock

from google.adk.events import Event
from google.genai import types

from agentic_cli.workflow.adk.event_processor import ADKEventProcessor
from agentic_cli.workflow.events import EventType


class TestThinkingDetectionInline:
    """Verify inline part.thought check handles all cases."""

    def _make_processor(self):
        return ADKEventProcessor(model="gemini-3-flash")

    def _make_part(self, text, thought=None, has_thought_attr=True):
        if has_thought_attr:
            part = MagicMock()
            part.text = text
            part.thought = thought
        else:
            # spec= without "thought" ensures getattr falls through to default
            part = MagicMock(
                spec=["text", "function_call", "function_response",
                      "code_execution_result", "executable_code", "file_data"]
            )
            part.text = text
        part.function_call = None
        part.function_response = None
        part.code_execution_result = None
        part.executable_code = None
        part.file_data = None
        return part

    def test_thinking_part(self):
        event = self._make_processor().process_part(
            self._make_part("reasoning...", thought=True), "s1"
        )
        assert event.type == EventType.THINKING
        assert event.content == "reasoning..."

    def test_regular_text_part(self):
        event = self._make_processor().process_part(
            self._make_part("answer", thought=False), "s1"
        )
        assert event.type == EventType.TEXT
        assert event.content == "answer"

    def test_no_thought_attribute(self):
        """Older models without part.thought default to TEXT."""
        event = self._make_processor().process_part(
            self._make_part("text", has_thought_attr=False), "s1"
        )
        assert event.type == EventType.TEXT
        assert event.content == "text"


# --- a sub-agent's text is activity -----------------------------------------------


def _event(author: str, text: str = "X is 42.") -> Event:
    """An ADK event ``author`` wrote: its text and the usage of its call."""
    return Event(
        author=author,
        invocation_id="i1",
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=2, total_token_count=12
        ),
    )


async def _events(processor: ADKEventProcessor, author: str) -> dict[EventType, object]:
    events = [e async for e in processor.process_event(_event(author), "s1")]
    assert [e.type for e in events] == [EventType.LLM_USAGE, EventType.TEXT]
    return {e.type: e for e in events}


class TestSubAgentActivity:
    """Only the root replies to the user; a sub-agent's text is its work,
    reported to its parent, and its call's prompt is its brief."""

    async def test_a_sub_agents_text_is_marked_and_its_usage_is_the_sub_agents(self):
        events = await _events(ADKEventProcessor(model="m", root="coordinator"), "researcher")

        text = events[EventType.TEXT]
        assert (text.content, text.metadata["agent"], text.metadata["subagent"]) == (
            "X is 42.", "researcher", True
        )
        assert events[EventType.LLM_USAGE].metadata["purpose"] == "subagent"

    async def test_the_roots_text_is_the_reply(self):
        events = await _events(ADKEventProcessor(model="m", root="coordinator"), "coordinator")

        assert "subagent" not in events[EventType.TEXT].metadata
        assert "purpose" not in events[EventType.LLM_USAGE].metadata

    async def test_the_users_text_is_nobodys_activity(self):
        events = await _events(ADKEventProcessor(model="m", root="coordinator"), "user")

        assert "subagent" not in events[EventType.TEXT].metadata
        assert "purpose" not in events[EventType.LLM_USAGE].metadata

    async def test_a_sub_agents_failure_is_a_warning_naming_it(self):
        # The coordinator carries on (ADK hands it "Error running sub-agent:
        # …"), so the user is told why, as a warning.
        failed = Event(
            author="researcher", invocation_id="i1", error_code="BadRequestError",
            error_message="Error code: 400 - max_tokens: 100000 > 64000",
        )

        (event,) = [e async for e in ADKEventProcessor(model="m", root="coordinator")
                    .process_event(failed, "s1")]

        assert event.type is EventType.ERROR
        assert event.content == "researcher failed: Error code: 400 - max_tokens: 100000 > 64000"
        assert (event.metadata["error_code"], event.metadata["recoverable"],
                event.metadata["agent"]) == ("BadRequestError", True, "researcher")

    async def test_the_roots_failure_is_left_to_the_turn(self):
        # Its exception leaves the run, which reports it: no second report.
        failed = Event(author="coordinator", invocation_id="i1",
                       error_code="BadRequestError", error_message="bad")

        assert [e async for e in ADKEventProcessor(model="m", root="coordinator")
                .process_event(failed, "s1")] == []

    async def test_without_a_root_nothing_is_marked(self):
        # A native ADK YAML app: an agent it transferred to owns the reply.
        events = await _events(ADKEventProcessor(model="m"), "helper")

        assert "subagent" not in events[EventType.TEXT].metadata
        assert "purpose" not in events[EventType.LLM_USAGE].metadata
        failed = Event(author="helper", invocation_id="i1", error_code="E", error_message="x")
        assert [e async for e in ADKEventProcessor(model="m").process_event(failed, "s1")] == []
