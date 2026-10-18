"""Both knowledge bases and the memory store share one embedding service.

Each knowledge base loaded the embedding model when it opened (to size its
index), and the memory store loaded its own on first use: up to three
copies of one model per process. The workflow now builds one service, only
when a knowledge base or the memory store is needed and never in mock mode,
keeps it under a private key, and passes it to all three.
"""

from __future__ import annotations

import sys
import types
from typing import AsyncGenerator

import numpy as np
import pytest

from agentic_cli.config import BaseSettings
from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory.kb import SourceType
from agentic_cli.workflow import memory_services
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.events import WorkflowEvent
from agentic_cli.workflow.service_registry import (
    EMBEDDING_SERVICE,
    KB_MANAGER,
    MEMORY_STORE,
    USER_KB_MANAGER,
)

from tests.memory.kb_extra import require_kb_module


class _Manager(BaseWorkflowManager):
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


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AGENTIC_KNOWLEDGE_BASE_USE_MOCK", raising=False)
    monkeypatch.chdir(tmp_path / "project")


def _manager(tmp_path, required, **settings) -> _Manager:
    mgr = _Manager(agent_configs=[], settings=BaseSettings(workspace_dir=tmp_path / "workspace", **settings))
    mgr._required_managers = set(required)
    return mgr


@pytest.fixture
def made(monkeypatch) -> list:
    """``build_embedding_service`` hands out a mock service and records each one."""
    services: list = []

    def build(settings):
        services.append(MockEmbeddingService(embedding_dim=16))
        return services[-1]

    monkeypatch.setattr(memory_services, "build_embedding_service", build)
    return services


def test_one_service_reaches_both_scopes_and_the_memory_store(tmp_path, made):
    services = _manager(tmp_path, {"kb_manager", "memory_store"}, knowledge_base_use_mock=False)._build_services()

    [service] = made
    assert services[EMBEDDING_SERVICE] is service
    assert services[KB_MANAGER]._embedding_service is service
    assert services[USER_KB_MANAGER]._embedding_service is service
    assert services[MEMORY_STORE]._embedding_service is service


def test_no_service_without_a_knowledge_base_or_memory_store(tmp_path, made):
    services = _manager(tmp_path, {"arxiv_source"}, knowledge_base_use_mock=False)._build_services()

    assert made == []
    assert EMBEDDING_SERVICE not in services


def test_a_component_built_later_reuses_the_service(tmp_path, made):
    mgr = _manager(tmp_path, {"kb_manager"}, knowledge_base_use_mock=False)
    mgr._publish_services(mgr._build_services())
    mgr._required_managers.add("memory_store")

    later = mgr._build_services(frozenset(mgr._services))

    assert len(made) == 1
    assert later[MEMORY_STORE]._embedding_service is mgr._services[EMBEDDING_SERVICE]


def test_a_model_switch_carries_the_service(tmp_path, made):
    mgr = _manager(tmp_path, {"memory_store"}, knowledge_base_use_mock=False)
    mgr._publish_services(mgr._build_services())
    service = mgr._services[EMBEDDING_SERVICE]

    with mgr._carrying_services(True):
        mgr._cleanup_managers()

    assert mgr._services[EMBEDDING_SERVICE] is service


def test_mock_mode_builds_no_service(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("mock mode must not build an embedding service")

    monkeypatch.setattr(EmbeddingService, "is_available", staticmethod(lambda: True))
    monkeypatch.setattr(EmbeddingService, "__init__", refuse)

    services = _manager(tmp_path, {"kb_manager", "memory_store"}, knowledge_base_use_mock=True)._build_services()

    assert EMBEDDING_SERVICE not in services
    assert services[MEMORY_STORE]._embedding_service is None


def _install_fake_model(monkeypatch, loads: list[str]) -> None:
    """A sentence-transformers stand-in of dimension 8 that records each load."""

    class SentenceTransformer:
        def __init__(self, name, device=None):
            loads.append(name)

        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, texts, **kwargs):
            rows = np.full((1 if isinstance(texts, str) else len(texts), 8), 0.5, dtype="float32")
            return rows[0] if isinstance(texts, str) else rows

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=SentenceTransformer)
    )


def test_the_model_loads_once_for_both_scopes_and_the_memory_store(tmp_path, monkeypatch):
    loads: list[str] = []
    _install_fake_model(monkeypatch, loads)
    mgr = _manager(
        tmp_path, {"kb_manager", "memory_store"}, knowledge_base_use_mock=False, embedding_device="cpu"
    )
    services = mgr._build_services()

    services[MEMORY_STORE].store("The team mascot is an otter")
    services[KB_MANAGER].ingest_document(content="Basalt columns.", title="Basalt", source_type=SourceType.USER)
    services[USER_KB_MANAGER].ingest_document(content="Glacier notes.", title="Glacier", source_type=SourceType.USER)

    assert loads == [mgr.settings.embedding_model]


def test_both_knowledge_bases_hold_the_shared_service_with_faiss(tmp_path, monkeypatch):
    """With FAISS a real service reaches both knowledge bases (without it they fall back to keyword search)."""
    require_kb_module("faiss")
    _install_fake_model(monkeypatch, [])
    mgr = _manager(tmp_path, {"kb_manager"}, knowledge_base_use_mock=False, embedding_device="cpu")

    services = mgr._build_services()

    assert services[KB_MANAGER]._embedding_service is services[EMBEDDING_SERVICE]
    assert services[USER_KB_MANAGER]._embedding_service is services[EMBEDDING_SERVICE]
