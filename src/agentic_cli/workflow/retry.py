"""Rate-limit detection and retry-delay parsing.

These helpers are backend-agnostic and used by both the ADK event
processor and the CLI message processor.
"""

from __future__ import annotations

import re


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a 429 rate-limit / RESOURCE_EXHAUSTED error."""
    if getattr(error, "code", None) == 429:
        return True
    if "RESOURCE_EXHAUSTED" in str(error):
        return True
    return False


def parse_retry_delay(error: Exception) -> float | None:
    """Extract retry delay in seconds from a rate-limit error.

    Looks for retryDelay in the structured error details first,
    then falls back to regex on the error message string.

    Returns:
        Delay in seconds, or None if unparseable.
    """
    # Try structured details: error.details["error"]["details"][*]["retryDelay"]
    details = getattr(error, "details", None)
    if isinstance(details, dict):
        inner_error = details.get("error", {})
        for detail_entry in inner_error.get("details", []):
            if isinstance(detail_entry, dict):
                retry_delay = detail_entry.get("retryDelay")
                if retry_delay:
                    match = re.search(r"([\d.]+)\s*s", str(retry_delay))
                    if match:
                        return float(match.group(1))

    # Fallback: regex on error message string
    match = re.search(r"retry\s+in\s+([\d.]+)\s*s", str(error), re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None
