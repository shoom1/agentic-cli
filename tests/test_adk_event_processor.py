"""Tests for ADK event processor after ThinkingDetector removal."""

from unittest.mock import MagicMock

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
