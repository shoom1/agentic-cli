"""Rate-limit detection.

``is_rate_limit_error`` tells a provider's 429 from any other failure: a
failed turn's ``ERROR`` says so (``WorkflowEvent.turn_failed``'s
``rate_limited``), and the CLI message processor uses it for a manager that
raises, to report a turn that failed on one. It retries nothing and works for
every backend. The retries are the provider clients'
(``retry_max_attempts``, ``retry_initial_delay``), before a failure reaches
the turn.
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

