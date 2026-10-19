"""Startup against a cached model list that predates a model.

A fresh cache entry spares startup the network, but it can predate a model the
provider now serves. The manager lists such a provider again, once, before a
cached listing stops it from starting; a live listing still decides.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("google.adk")

from agentic_cli.config import BaseSettings, SettingsValidationError  # noqa: E402
from agentic_cli.models import ModelCatalog  # noqa: E402
from agentic_cli.workflow import model_services  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.model_settings import ModelSettings, ThinkingSettings  # noqa: E402
from tests.models.fakes import ProviderAPI, recorded  # noqa: E402

NEW = "gemini-3.5-flash"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME with a Google key, and a Google listing cached an hour ago."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
    cache = tmp_path / "home" / ".agentic_cli" / "model_cache.json"
    cache.parent.mkdir(parents=True)
    models = [{"id": m} for m in ("gemini-2.5-flash", "gemini-2.5-pro")]
    cache.write_text(json.dumps({"version": 1, "providers": {"google": {
        "fetched_at": time.time() - 3600, "ok": True, "models": models,
    }}}))
    return tmp_path


def serve(monkeypatch, home: Path, api: ProviderAPI) -> None:
    cache = home / "home" / ".agentic_cli" / "model_cache.json"

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(endpoints=endpoints, cache_path=cache, transport=api.transport)

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)


def google_serving(model_id: str) -> dict:
    """Google's recorded listing, plus a model released after the cache was written."""
    payload = copy.deepcopy(recorded("google"))
    payload["models"].insert(0, dict(
        payload["models"][0], name=f"models/{model_id}", displayName=model_id,
    ))
    return payload


def manager(model: str, **writer) -> GoogleADKWorkflowManager:
    return GoogleADKWorkflowManager(
        agent_configs=[AgentConfig(name="writer", prompt="Write.", model=model, **writer)],
        settings=BaseSettings(session_store="memory"),
    )


async def test_a_model_released_after_the_cache_was_written_starts(home, monkeypatch):
    api = ProviderAPI(google=google_serving(NEW))
    serve(monkeypatch, home, api)

    started = manager(NEW)
    await started.initialize_services()
    try:
        assert str(started._resolved_models["writer"].ref) == f"google:{NEW}"
        assert api.calls == ["google"]
    finally:
        await started.cleanup()


async def test_a_model_the_provider_still_lacks_fails_after_one_listing(home, monkeypatch):
    api = ProviderAPI()
    serve(monkeypatch, home, api)

    with pytest.raises(SettingsValidationError, match="Available google models"):
        await manager(NEW).initialize_services()

    assert api.calls == ["google"]


async def test_another_startup_problem_lists_nothing(home, monkeypatch):
    api = ProviderAPI(google=google_serving(NEW))
    serve(monkeypatch, home, api)
    typo = ModelSettings(thinking=ThinkingSettings(mode="hihg"))

    with pytest.raises(SettingsValidationError, match="hihg"):
        await manager("gemini-2.5-flash", model_settings=typo).initialize_services()

    assert api.calls == []
