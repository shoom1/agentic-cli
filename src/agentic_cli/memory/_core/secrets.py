"""Whether a text looks like it holds a credential.

The memory store refuses to keep one: a memory is read in every later
conversation, by the model and by anyone who opens its folder. The patterns
are the keys and tokens most often pasted into a conversation, and cover what
agentic-cli's LLM log masks of them (a test outside the package feeds both the
same samples). A pattern is anchored so that a word merely containing one
(``task-management-…`` holds ``sk-``) is not taken for a key, and the two
loosest ones are tightened against ordinary notes:

- ``Bearer`` must be followed by something that looks like a token, not a word
  or a placeholder (``bearer authentication``, ``Bearer YOUR_API_TOKEN``);
- an assignment (``password=…``, or ``"password": …`` in JSON) is not flagged
  when its value is an environment reference, a path, a placeholder or a
  template (``$DB_PASSWORD``, ``/etc/app``, ``<your key>``, ``{{ token }}``).
"""

from __future__ import annotations

import re

_NAME_WORD = r"(?:password|passwd|api[_-]?key|secret|token)"
_NAME_CHARS = r"[A-Za-z0-9_.-]"
_BEARER_CHARS = r"[A-Za-z0-9._~+/=-]"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----")),
    ("an API key", re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_\-]{16,}")),
    ("a Stripe key", re.compile(r"(?<![A-Za-z0-9_])[sr]k_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("a Google API key", re.compile(r"(?<![A-Za-z0-9_])AIza[A-Za-z0-9_\-]{16,}")),
    (
        "a bearer token",
        # A word is no token, and neither is a placeholder (``YOUR_API_TOKEN``). Ten or more
        # token characters with a digit in them, or twenty or more with a mark of their own
        # (a period only inside the run: a sentence's last full stop is none) or mixed case.
        re.compile(
            rf"\b(?i:Bearer)\s+(?:(?={_BEARER_CHARS}*[0-9]){_BEARER_CHARS}{{10,}}"
            rf"|(?={_BEARER_CHARS}*(?:[~+/=]|\.{_BEARER_CHARS})){_BEARER_CHARS}{{20,}}"
            rf"|(?={_BEARER_CHARS}*[a-z])(?={_BEARER_CHARS}*[A-Z]){_BEARER_CHARS}{{20,}})"
        ),
    ),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("a GitLab token", re.compile(r"(?<![A-Za-z0-9_])glpat-[A-Za-z0-9_\-]{20,}")),
    ("a Hugging Face token", re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{30,}")),
    ("an AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("a Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("a JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    (
        "a password or key",
        # A name that holds one of the words (``DB_PASSWORD``, ``client_secret``,
        # ``aws_secret_access_key``), a quote when the name is a JSON or dict key, ``:`` or
        # ``=``, and a value of 8 or more characters.
        re.compile(
            rf"(?<!{_NAME_CHARS}){_NAME_CHARS}*{_NAME_WORD}(?:[_.-]{_NAME_CHARS}*)?"
            r"['\"]?\s*[:=]\s*['\"]?(?![$/~<{])[^\s'\"]{8,}",
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
