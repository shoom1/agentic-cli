"""Rate-limit detection.

These helpers are backend-agnostic and used by both the ADK event
processor and the CLI message processor.
"""

from __future__ import annotations


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a 429 rate-limit / RESOURCE_EXHAUSTED error."""
    # Gemini/genai errors expose ``.code``; anthropic.RateLimitError (and other
    # httpx-based SDKs) expose ``.status_code`` instead.
    if getattr(error, "code", None) == 429:
        return True
    if getattr(error, "status_code", None) == 429:
        return True
    if "RESOURCE_EXHAUSTED" in str(error):
        return True
    return False

