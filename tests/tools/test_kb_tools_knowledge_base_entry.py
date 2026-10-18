"""The tools return what 0.6.3 returned when given a ``KnowledgeBase``.

``test_kb_tool_contract.py`` runs the tools through the 0.6.3 entry points:
the registry keys ``kb_manager`` and ``user_kb_manager``, and the factories
given managers. Here the same scenarios run through the new ones, the
registry key ``knowledge_base`` and the factories given a ``KnowledgeBase``,
and must match the same recording, without a deprecation warning.
"""

from __future__ import annotations

import warnings

import pytest

from tests.tools import test_kb_tool_contract as contract


def _knowledge_base(env):
    from agentic_cli.memory.kb import KnowledgeBase

    managers = {"project": env.project}
    if env.user is not None:
        managers["user"] = env.user
    return KnowledgeBase(managers)


def knowledge_base_registry_entry(env, arxiv):
    from agentic_cli.tools import arxiv_tools, knowledge_tools

    registry = {"knowledge_base": _knowledge_base(env)}
    if arxiv is not None:
        registry["arxiv_source"] = arxiv
    tools = {name: getattr(knowledge_tools, name) for name in contract.KB_TOOLS}
    tools["ingest_arxiv_paper"] = arxiv_tools.ingest_arxiv_paper
    return registry, tools


def knowledge_base_factory_entry(env, arxiv):
    from agentic_cli.tools.factories import make_ingest_arxiv_tool, make_kb_tools

    kb = _knowledge_base(env)
    tools = {t.__name__: t for t in make_kb_tools(kb)}
    if arxiv is not None:
        tools["ingest_arxiv_paper"] = make_ingest_arxiv_tool(arxiv, kb)
    return {}, tools


NEW = {"knowledge_base_registry": knowledge_base_registry_entry, "knowledge_base_factory": knowledge_base_factory_entry}


@pytest.mark.parametrize("entry", list(NEW))
@pytest.mark.parametrize(("name", "layout"), contract._cases())
async def test_the_tool_returns_what_0_6_3_returned(name, layout, entry):
    assert await contract.run_scenario(name, layout, entry, NEW) == contract._golden()[f"{name}/{layout}"]


@pytest.mark.parametrize("entry", list(NEW))
def test_each_tool_keeps_its_call_style_and_arguments(entry):
    tools = contract.tools_of(entry, NEW)

    assert contract.call_styles(tools) == {**contract.KB_TOOLS, "ingest_arxiv_paper": "async"}
    assert contract.arguments(tools) == contract._golden()["arguments"]


def test_the_new_factory_arguments_do_not_warn(tmp_path):
    from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager
    from agentic_cli.tools.factories import make_ingest_arxiv_tool, make_kb_tools

    kb = KnowledgeBase({"project": KnowledgeBaseManager(tmp_path / "kb", use_mock=True)})

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        make_kb_tools(kb)
        make_ingest_arxiv_tool(contract._ArxivSource(), kb)


def test_the_0_6_3_factory_arguments_still_work_and_warn(tmp_path):
    from agentic_cli.memory.kb import KnowledgeBaseManager
    from agentic_cli.tools.factories import make_ingest_arxiv_tool, make_kb_tools

    manager = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    with pytest.warns(DeprecationWarning, match="0.7.0"):
        assert len(make_kb_tools(manager, manager)) == 7
    with pytest.warns(DeprecationWarning, match="0.7.0"):
        assert len(make_kb_tools(kb_manager=manager)) == 7
    with pytest.warns(DeprecationWarning, match="0.7.0"):
        assert callable(make_ingest_arxiv_tool(contract._ArxivSource(), manager))


def test_a_manager_given_twice_raises_as_in_0_6_3(tmp_path):
    from agentic_cli.memory.kb import KnowledgeBaseManager
    from agentic_cli.tools.factories import make_ingest_arxiv_tool, make_kb_tools

    one = KnowledgeBaseManager(tmp_path / "one", use_mock=True)
    two = KnowledgeBaseManager(tmp_path / "two", use_mock=True)

    with pytest.raises(TypeError, match="multiple values"):
        make_kb_tools(one, kb_manager=two)
    with pytest.raises(TypeError, match="multiple values"):
        make_ingest_arxiv_tool(contract._ArxivSource(), one, kb_manager=two)


def test_a_factory_needs_a_knowledge_base():
    from agentic_cli.tools.factories import make_kb_tools

    with pytest.raises(TypeError):
        make_kb_tools()
