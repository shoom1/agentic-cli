"""Shared constants for agentic-cli."""

# Content display truncation limits
CONTENT_PREVIEW_LENGTH = 200
TOOL_SUMMARY_MAX_LENGTH = 100


def truncate(text: str, max_length: int = CONTENT_PREVIEW_LENGTH) -> str:
    """Truncate text with ellipsis if it exceeds max_length."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string (B, KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes / (1024 * 1024):.1f}MB"
