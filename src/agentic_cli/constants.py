"""Shared constants for agentic-cli."""

# Content display truncation limits
CONTENT_PREVIEW_LENGTH = 200
TOOL_SUMMARY_MAX_LENGTH = 100


def truncate(text: str, max_length: int = CONTENT_PREVIEW_LENGTH) -> str:
    """Truncate text with ellipsis if it exceeds max_length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
