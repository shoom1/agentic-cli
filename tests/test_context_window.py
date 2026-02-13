"""Tests for context window management integration."""

from google.genai.types import ContextWindowCompressionConfig, SlidingWindow
from langchain_core.messages import HumanMessage, AIMessage, trim_messages


# --- Settings ---


def test_context_window_settings_defaults(mock_context):
    """Context window settings have correct defaults."""
    settings = mock_context.settings
    assert settings.context_window_enabled is False
    assert settings.context_window_trigger_tokens == 100_000
    assert settings.context_window_target_tokens == 80_000


def test_context_window_disabled_by_default(mock_context):
    """Context window management is off by default (no behavior change)."""
    assert not mock_context.settings.context_window_enabled


# --- ADK compression config ---


def test_adk_compression_config():
    """ADK compression config can be created with expected values."""
    config = ContextWindowCompressionConfig(
        trigger_tokens=100_000,
        sliding_window=SlidingWindow(target_tokens=80_000),
    )
    assert config.trigger_tokens == 100_000
    assert config.sliding_window.target_tokens == 80_000


def test_adk_compression_config_custom_values():
    """ADK compression config works with custom token values."""
    config = ContextWindowCompressionConfig(
        trigger_tokens=50_000,
        sliding_window=SlidingWindow(target_tokens=30_000),
    )
    assert config.trigger_tokens == 50_000
    assert config.sliding_window.target_tokens == 30_000


# --- LangGraph trim_messages ---


def test_langchain_trim_messages_trims():
    """trim_messages reduces message count when over limit."""
    messages = [HumanMessage(content="x" * 1000)] * 50
    trimmed = trim_messages(
        messages,
        max_tokens=100,
        strategy="last",
        token_counter="approximate",
        start_on="human",
    )
    assert len(trimmed) < len(messages)


def test_langchain_trim_messages_preserves_under_limit():
    """trim_messages returns all messages when under limit."""
    messages = [HumanMessage(content="hello"), AIMessage(content="hi")]
    trimmed = trim_messages(
        messages,
        max_tokens=100_000,
        strategy="last",
        token_counter="approximate",
        start_on="human",
    )
    assert len(trimmed) == len(messages)


def test_langchain_trim_messages_starts_on_human():
    """trim_messages ensures result starts with a human message."""
    messages = [
        AIMessage(content="stale"),
        HumanMessage(content="hello"),
        AIMessage(content="hi"),
    ]
    trimmed = trim_messages(
        messages,
        max_tokens=100_000,
        strategy="last",
        token_counter="approximate",
        start_on="human",
    )
    assert isinstance(trimmed[0], HumanMessage)
