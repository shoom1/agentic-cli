"""``/papers`` finds a match older than the 50 newest documents.

It took the 50 newest documents and only then filtered by the query, so a
matching document outside them was never listed. It now asks the knowledge
base for the catalogue, which filters first, as ``kb_list`` does. Its table
is pinned in ``tests/tools/test_kb_tool_contract.py``.
"""

from __future__ import annotations

from agentic_cli.cli.builtin_commands import PapersCommand
from agentic_cli.config import BaseSettings
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from agentic_cli.workflow import memory_services


class _Session:
    def __init__(self) -> None:
        self.tables: list = []
        self.messages: list[str] = []

    def add_rich(self, table) -> None:
        self.tables.append(table)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(content)


class _App:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.session = _Session()

    @property
    def workflow(self):
        raise RuntimeError("workflow not ready")


async def test_a_match_older_than_the_newest_fifty_is_listed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True)
    project = KnowledgeBaseManager(memory_services.project_kb_dir(settings), use_mock=True)
    project.ingest_document(content="Moorings.", title="Zeppelin mooring", source_type=SourceType.USER)
    for n in range(55):
        project.ingest_document(content=f"Note {n}.", title=f"Note {n}", source_type=SourceType.USER)
    app = _App(settings)

    await PapersCommand().execute("zeppelin", app)

    [table] = app.session.tables
    assert [str(cell) for cell in table.columns[1]._cells] == ["Zeppelin mooring"]


async def test_without_a_workflow_only_the_listed_scope_is_opened(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    settings = BaseSettings(workspace_dir=tmp_path / "workspace", knowledge_base_use_mock=True)

    await PapersCommand().execute("--global", _App(settings))

    assert not memory_services.project_kb_dir(settings).exists()
