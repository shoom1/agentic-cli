"""The memory store's secret screen flags at least what the LLM log masks."""

from __future__ import annotations

import pytest

from agentic_cli.memory import looks_like_secret
from agentic_cli.workflow.llm_log import redact_secrets
from tests.memory.test_secrets import SECRET_SAMPLES


@pytest.mark.parametrize("text", SECRET_SAMPLES)
def test_what_the_log_masks_the_store_refuses(text):
    if redact_secrets(text) != text:
        assert looks_like_secret(text) is not None
