"""The backend-neutral per-agent ``ModelSettings``.

How each backend turns them into its own request settings is tested with its
adapter (ADK: ``tests/workflow/adk/test_model_factory.py``).
"""

from __future__ import annotations

from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings


class TestModelSettingsModel:
    def test_defaults_all_none(self):
        ms = ModelSettings()
        assert (ms.temperature, ms.top_p, ms.top_k, ms.max_tokens, ms.stop_sequences) == (None,) * 5
        assert ms.thinking is None
        assert ms.extra == {}

    def test_thinking_defaults(self):
        ts = ThinkingSettings()
        assert ts.mode == "none"
        assert ts.budget_tokens is None
