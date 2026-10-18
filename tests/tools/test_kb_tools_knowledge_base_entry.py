"""The knowledge-base factories take a ``KnowledgeBase`` and refuse 0.6.3's managers.

What the tools return through the factories is pinned by ``test_kb_tool_contract.py``.
"""

from __future__ import annotations

import pytest

from tests.tools import test_kb_tool_contract as contract


def test_a_factory_refuses_0_6_3_managers(tmp_path):
    from agentic_cli.memory.kb import KnowledgeBaseManager
    from agentic_cli.tools.factories import make_ingest_arxiv_tool, make_kb_tools

    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    with pytest.raises(TypeError, match="KnowledgeBase"):
        make_kb_tools(manager)
    with pytest.raises(TypeError, match="KnowledgeBase"):
        make_ingest_arxiv_tool(contract._ArxivSource(), manager)
    with pytest.raises(TypeError):
        make_kb_tools(manager, manager)
    with pytest.raises(TypeError):
        make_kb_tools(kb_manager=manager)
    with pytest.raises(TypeError):
        make_ingest_arxiv_tool(contract._ArxivSource(), kb_manager=manager)


def test_a_factory_needs_a_knowledge_base():
    from agentic_cli.tools.factories import make_kb_tools

    with pytest.raises(TypeError):
        make_kb_tools()
