"""Tests for ThinkingDetector."""

from unittest.mock import MagicMock

from agentic_cli.workflow.thinking import ThinkingDetector, ThinkingResult


class TestThinkingResult:
    """Tests for ThinkingResult namedtuple."""

    def test_namedtuple_fields(self):
        """Test ThinkingResult has expected fields."""
        result = ThinkingResult(is_thinking=True, content="test")
        assert result.is_thinking is True
        assert result.content == "test"

    def test_is_thinking_true(self):
        """Test thinking result with is_thinking=True."""
        result = ThinkingResult(is_thinking=True, content="reasoning")
        assert result.is_thinking is True

    def test_is_thinking_false(self):
        """Test thinking result with is_thinking=False."""
        result = ThinkingResult(is_thinking=False, content="response")
        assert result.is_thinking is False


class TestThinkingDetector:
    """Tests for ThinkingDetector.detect_from_part()."""

    def test_detect_anthropic_thinking_block(self):
        """Test detection of Anthropic thinking block."""
        part = MagicMock()
        part.type = "thinking"
        part.thinking = "I should analyze this..."
        part.text = ""

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is True
        assert result.content == "I should analyze this..."

    def test_detect_anthropic_thinking_block_text_fallback(self):
        """Test Anthropic thinking block falls back to text attr."""
        part = MagicMock(spec=["type", "text"])
        part.type = "thinking"
        part.text = "thinking via text attr"

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is True
        assert result.content == "thinking via text attr"

    def test_detect_anthropic_text_block(self):
        """Test detection of Anthropic text block."""
        part = MagicMock()
        part.type = "text"
        part.text = "Hello, how can I help?"

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is False
        assert result.content == "Hello, how can I help?"

    def test_detect_gemini_thinking(self):
        """Test detection of Gemini thinking part (thought=True)."""
        part = MagicMock(spec=["thought", "text"])
        part.thought = True
        part.text = "Let me think about this..."

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is True
        assert result.content == "Let me think about this..."

    def test_detect_gemini_text(self):
        """Test detection of Gemini regular text (thought=False)."""
        part = MagicMock(spec=["thought", "text"])
        part.thought = False
        part.text = "Here is my answer."

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is False
        assert result.content == "Here is my answer."

    def test_detect_gemini_text_no_thought_attr(self):
        """Test Gemini part with text but no thought attribute."""
        part = MagicMock(spec=["text"])
        part.text = "Just plain text."

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is False
        assert result.content == "Just plain text."

    def test_detect_unknown_part(self):
        """Test fallback for unknown part type."""
        part = MagicMock(spec=[])

        result = ThinkingDetector.detect_from_part(part)

        assert result.is_thinking is False
        assert result.content == ""

    def test_is_thinking_part_shortcut(self):
        """Test the is_thinking_part convenience method."""
        thinking_part = MagicMock()
        thinking_part.type = "thinking"
        thinking_part.thinking = "reason"
        thinking_part.text = ""

        text_part = MagicMock(spec=["text"])
        text_part.text = "response"

        assert ThinkingDetector.is_thinking_part(thinking_part) is True
        assert ThinkingDetector.is_thinking_part(text_part) is False
