"""The workflow gives its knowledge bases a summarizer when the setting says so.

Before, a knowledge base found the workflow's summarizer only when some agent
had web_fetch (the one tool that declares ``requires="llm_summarizer"``), so an
app with knowledge-base tools and no web_fetch silently stored previews.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import (
    KB_MANAGER,
    LLM_SUMMARIZER,
    USER_KB_MANAGER,
    set_service_registry,
)

SIDECAR = "SUMMARY: Written by the workflow.\nCLAIMS:\n- A claim.\n"


class _Manager(BaseWorkflowManager):
    """A workflow whose model answers every prompt with SIDECAR."""

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        pass

    async def process(
        self, message: str, user_id: str, session_id: str | None = None
    ) -> AsyncGenerator[WorkflowEvent, None]:
        if False:
            yield  # type: ignore[misc]

    async def reinitialize(self, model: str | None = None, preserve_sessions: bool = True) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def _get_state_tools(self) -> list:
        return []

    async def generate_simple(self, prompt: str, max_tokens: int = 500) -> str:
        self.prompts.append(prompt)
        return SIDECAR


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTIC_KNOWLEDGE_BASE_SUMMARIZE", raising=False)
    monkeypatch.chdir(tmp_path / "project")
    return tmp_path / "project"


def _knowledge_bases(tmp_path, **settings):
    """Build the services of an app whose agents have KB tools and no web_fetch."""
    mgr = _Manager(
        agent_configs=[],
        settings=BaseSettings(
            workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True, **settings
        ),
    )
    mgr.prompts = []
    mgr._required_managers = {"kb_manager"}
    services = mgr._build_services()
    assert LLM_SUMMARIZER not in services
    return mgr, services[KB_MANAGER], services[USER_KB_MANAGER]


async def test_knowledge_bases_summarize_without_web_fetch(tmp_path):
    mgr, project_kb, user_kb = _knowledge_bases(tmp_path)

    for kb in (project_kb, user_kb):
        payload = await kb.generate_sidecar_payload("Body text.", title="T")
        assert payload["summary"] == "Written by the workflow."
    assert len(mgr.prompts) == 2


async def test_the_setting_turns_summaries_off_even_inside_a_turn(tmp_path):
    mgr, project_kb, _ = _knowledge_bases(tmp_path, knowledge_base_summarize=False)
    token = set_service_registry({LLM_SUMMARIZER: mgr})
    try:
        payload = await project_kb.generate_sidecar_payload("Body text.", title="T")
    finally:
        token.var.reset(token)

    assert payload["summary"] == "Body text."
    assert mgr.prompts == []


def test_summaries_are_on_by_default():
    assert BaseSettings().knowledge_base_summarize is True


def test_user_settings_can_turn_summaries_off(tmp_path):
    user_dir = tmp_path / "home" / f".{BaseSettings().app_name}"
    user_dir.mkdir()
    (user_dir / "settings.json").write_text(json.dumps({"knowledge_base_summarize": False}))

    assert BaseSettings().knowledge_base_summarize is False


def test_a_project_settings_file_cannot_turn_summaries_back_on(tmp_path, project):
    app_name = BaseSettings().app_name
    (tmp_path / "home" / f".{app_name}").mkdir()
    (tmp_path / "home" / f".{app_name}" / "settings.json").write_text(
        json.dumps({"knowledge_base_summarize": False})
    )
    (project / f".{app_name}").mkdir()
    (project / f".{app_name}" / "settings.json").write_text(
        json.dumps({"knowledge_base_summarize": True})
    )

    assert BaseSettings().knowledge_base_summarize is False
