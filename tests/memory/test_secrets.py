"""What the memory store refuses to keep: text that looks like a credential."""

from __future__ import annotations

import importlib.util

import pytest

from agentic_cli.memory._core.secrets import looks_like_secret
from agentic_cli.memory._core.tokenize import TOKENIZER, tokenize

#: Secrets as people paste them. tests/workflow/test_secret_screens_agree.py
#: feeds the same list to the log redactor.
SECRET_SAMPLES = [
    "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv",
    "OPENAI_API_KEY=sk-proj-AbCdEfGhIjKlMnOp1234",
    "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz123456",
    "Authorization: Bearer abcdefghij.klmnopqrst",
    "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
    "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
    "AKIAIOSFODNN7EXAMPLE",
    "xoxb-1234567890-abcdefghij",
    "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAA\n-----END OPENSSH PRIVATE KEY-----",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "password=hunter2hunter2",
    "api_key: 9f8e7d6c5b4a3210",
]


@pytest.mark.parametrize("text", SECRET_SAMPLES)
def test_a_credential_is_flagged(text):
    assert looks_like_secret(f"The key is {text} for the staging box") is not None


@pytest.mark.parametrize("text", [
    "The task-management-feature-flags module owns the rollout",
    "Prefers pytest fixtures over unittest classes for new tests",
    "The auth flow rotates its refresh token every hour",
    "Use the token bucket limiter for the arXiv client",
    "password reset emails go through the notifications service",
])
def test_ordinary_text_is_not(text):
    assert looks_like_secret(text) is None


def test_the_kind_names_what_was_found():
    assert looks_like_secret("AKIAIOSFODNN7EXAMPLE") == "an AWS access key"


def test_the_tokenizer_lives_in_core_unchanged():
    assert TOKENIZER == "words-v3"
    assert tokenize("Otters 水獭住") == ["otters", "水獭", "獭住"]
    assert importlib.util.find_spec("agentic_cli.memory.kb._tokenize") is None
