"""Unified thinking detection across LLM providers."""

from typing import Any, NamedTuple


class ThinkingResult(NamedTuple):
    """Result of thinking detection."""

    is_thinking: bool
    content: str


class ThinkingDetector:
    """Detects thinking/reasoning content from LLM responses.

    Provides unified interface for detecting thinking content across:
    - Gemini: part.thought attribute
    - Anthropic: thinking_blocks in response
    """

    @staticmethod
    def detect_from_part(part: Any, provider: str | None = None) -> ThinkingResult:
        """Detect if a response part contains thinking content.

        Args:
            part: Response part from LLM (provider-specific format)
            provider: Optional provider hint ("gemini", "anthropic")

        Returns:
            ThinkingResult with is_thinking flag and content
        """
        # Anthropic: Check for thinking block type first (has 'type' attribute)
        if hasattr(part, "type"):
            if part.type == "thinking":
                content = getattr(part, "thinking", "") or getattr(part, "text", "")
                return ThinkingResult(is_thinking=True, content=content)
            if part.type == "text":
                return ThinkingResult(is_thinking=False, content=getattr(part, "text", ""))

        # Gemini: Check thought attribute
        if hasattr(part, "thought") and part.thought:
            return ThinkingResult(is_thinking=True, content=part.text or "")

        # Gemini: Regular text (thought=False or absent)
        if hasattr(part, "text") and part.text:
            return ThinkingResult(is_thinking=False, content=part.text)

        # Fallback: Not thinking content
        return ThinkingResult(is_thinking=False, content="")

    @staticmethod
    def is_thinking_part(part: Any) -> bool:
        """Quick check if part is thinking content."""
        return ThinkingDetector.detect_from_part(part).is_thinking
