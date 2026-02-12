"""LLM summarizer protocol and utilities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMSummarizer(Protocol):
    """Protocol for LLM-based content summarization."""

    async def summarize(self, content: str, prompt: str) -> str:
        """Summarize content based on a prompt."""
        ...


FAST_MODEL_MAP: dict[str, str] = {
    # Anthropic
    "claude-opus-4-5-20251101": "claude-haiku-4-20251101",
    "claude-sonnet-4-20251101": "claude-haiku-4-20251101",
    # Google
    "gemini-3-pro": "gemini-3-flash",
    "gemini-3-ultra": "gemini-3-flash",
    "gemini-2.5-pro": "gemini-2.5-flash",
    # OpenAI
    "gpt-5.2": "gpt-5-mini",
    "gpt-5": "gpt-5-mini",
    "gpt-4.5-turbo": "gpt-4o-mini",
}


def get_fast_model(main_model: str) -> str | None:
    """Get the fast model for summarization based on main model."""
    return FAST_MODEL_MAP.get(main_model)


SUMMARIZE_PROMPT_TEMPLATE = """Web page content:
---
{content}
---

{prompt}

Provide a thorough and detailed response based only on the content above."""


def build_summarize_prompt(content: str, user_prompt: str) -> str:
    """Build the full prompt for summarization."""
    return SUMMARIZE_PROMPT_TEMPLATE.format(content=content, prompt=user_prompt)
