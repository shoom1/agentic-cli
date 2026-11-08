"""Whether a text looks like it holds a credential.

The memory store refuses to keep one: a memory is read in every later
conversation, by the model and by anyone who opens its folder. The patterns
are the keys and tokens most often pasted into a conversation, and cover at
least what agentic-cli's LLM log masks (a test outside the package feeds both
the same samples). A pattern is anchored so that a word merely containing one
(``task-management-…`` holds ``sk-``) is not taken for a key.
"""

from __future__ import annotations

import re

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("an API key", re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_\-]{16,}", re.IGNORECASE)),
    ("a Google API key", re.compile(r"(?<![A-Za-z0-9_])AIza[A-Za-z0-9_\-]{16,}")),
    ("a bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{10,}", re.IGNORECASE)),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("an AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("a Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("a JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    (
        "a password or key",
        re.compile(
            r"\b(?:password|passwd|pwd|api[_-]?key|secret|token)\s*[:=]\s*['\"]?"
            r"(?=[^\s'\"]*\d)[^\s'\"]{8,}",
            re.IGNORECASE,
        ),
    ),
)


def looks_like_secret(text: str) -> str | None:
    """The kind of credential ``text`` seems to hold, or None."""
    for kind, pattern in _PATTERNS:
        if pattern.search(text):
            return kind
    return None
