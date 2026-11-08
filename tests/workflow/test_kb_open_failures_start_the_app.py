"""A workflow starts when part of its knowledge base or memory store cannot open.

A project knowledge base the framework cannot read (here, a database that is
a git-LFS pointer, as a repository can ship it) and an embedding model that
does not load (an offline first run) raised out of service construction, so an
app whose agents use the kb or memory tools did not start. The project scope is
now left out (the user scope takes the writes), the model's absence leaves
keyword and substring search, and the tools answer.
"""

from __future__ import annotations

from typing import AsyncGenerator

from agentic_cli.config import BaseSettings
from agentic_cli.tools.knowledge_tools import kb_ingest_text, kb_search
from agentic_cli.tools.memory_tools import save_memory, search_memory
from agentic_cli.workflow import memory_services
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.events import WorkflowEvent


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


class _OfflineModel:
    """An embedding service whose model cannot load: an offline first run."""

    model_name = "all-MiniLM-L6-v2"
    batch_size = 32

    def _fail(self):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load this model")

    @property
    def embedding_dim(self) -> int:
        self._fail()

    def embed_text(self, text):
        self._fail()

    def embed_batch(self, texts):
        self._fail()


async def test_a_project_database_that_does_not_open_leaves_the_user_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")  # initialization resolves a model
    monkeypatch.setattr(memory_services, "build_embedding_service", lambda settings: _OfflineModel())
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=False)
    project = memory_services.project_kb_dir(settings)
    project.mkdir(parents=True)
    (project / "kb.sqlite").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
    config = AgentConfig(name="a", prompt="p", tools=[kb_search, kb_ingest_text, save_memory, search_memory])
    mgr = _Manager(agent_configs=[config], settings=settings)

    await mgr.initialize_services(validate=False)

    tools = mgr._get_service_tool_map()
    assert mgr.knowledge_base.scopes == ("user",)
    ingested = await tools["kb_ingest_text"]("Otters nest in burrows along the river.", title="Otters")
    assert ingested["success"] is True, ingested
    found = tools["kb_search"]("otters")
    assert [(hit["document_title"], hit["scope"]) for hit in found["results"]] == [("Otters", "user")]
    assert tools["save_memory"]("The user prefers metric units")["success"] is True
    assert [item["content"] for item in tools["search_memory"]("metric")["items"]] == [
        "The user prefers metric units"
    ]


def test_a_model_that_does_not_load_is_tried_once(tmp_path, monkeypatch):
    """The memory store and the knowledge base each probe the shared service:
    each probe of a failed model went to the network again."""
    import sys
    import types

    loads: list[str] = []

    class SentenceTransformer:
        def __init__(self, name, device=None):
            loads.append(name)
            raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=SentenceTransformer)
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    settings = BaseSettings(
        workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=False, embedding_device="cpu"
    )
    mgr = _Manager(agent_configs=[], settings=settings)
    mgr._required_managers = {"knowledge_base", "memory_store"}

    services = mgr._build_services()

    assert len(loads) == 1
    assert services["memory_store"]._embedding_service is None
