"""The memory tools over the store (spec §7): both entry points, every result."""

from __future__ import annotations

import pytest

from agentic_cli.memory import MemoryStore
from agentic_cli.tools import memory_tools
from agentic_cli.tools.factories import make_memory_tools
from agentic_cli.tools.registry import get_registry
from agentic_cli.workflow.permissions import Capability
from agentic_cli.workflow.service_registry import MEMORY_STORE, set_service_registry
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn

NAMES = ("save_memory", "read_memory", "search_memory", "update_memory", "delete_memory")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "user", tmp_path / "project" / "memory")


@pytest.fixture(params=["registry", "factory"])
def tools(request, store):
    if request.param == "factory":
        yield {tool.__name__: tool for tool in make_memory_tools(store)}
        return
    token = set_service_registry({MEMORY_STORE: store})
    try:
        yield {name: getattr(memory_tools, name) for name in NAMES}
    finally:
        token.var.reset(token)


def test_the_five_tools_and_their_capabilities():
    registry = get_registry()
    reads = {"read_memory", "search_memory"}
    for name in NAMES:
        definition = registry.get(name)
        expected = "memory.read" if name in reads else "memory.write"
        assert definition.capabilities == [Capability(expected)]
        assert definition.requires == ("memory_store",)


def test_save_then_read(tools, store):
    saved = tools["save_memory"]("Prefers tabs", "Tabs, width 4.", "feedback", scope="user")
    assert saved == {"success": True, "item_id": saved["item_id"], "scope": "user",
                     "message": "Saved to user memory: Prefers tabs"}
    read = tools["read_memory"](saved["item_id"])
    assert read["success"] is True
    assert set(read["memory"]) == {"id", "description", "content", "type", "scope", "source", "created", "updated"}
    assert (read["memory"]["source"], read["memory"]["content"]) == ("agent", "Tabs, width 4.")
    assert read["memory"]["created"].endswith("+00:00")
    assert "path" not in read["memory"]


def test_the_default_scope_is_the_project(tools):
    assert tools["save_memory"]("d", "c", "project")["scope"] == "project"


def test_provenance_from_the_model_and_the_turn(tools, store):
    token = set_active_turn(SessionRef(app_name="app", user_id="u", session_id="conv-9"))
    try:
        saved = tools["save_memory"]("Uses metric", "km, kg.", "user", scope="user", user_requested=True)
    finally:
        reset_active_turn(token)
    memory = store.read(saved["item_id"])
    assert (memory.source, memory.conversation) == ("user", "conv-9")


@pytest.mark.parametrize("args, kwargs, error", [
    (("d", "c", "preference"), {}, "type: 'preference'"),
    (("d", "key AKIAIOSFODNN7EXAMPLE", "user"), {}, "looks like a secret"),
    (("", "c", "user"), {}, "description: empty"),
])
def test_save_failures_are_results(tools, args, kwargs, error):
    result = tools["save_memory"](*args, **kwargs)
    assert result["success"] is False and error in result["error"]


def test_a_duplicate_points_to_update(tools):
    first = tools["save_memory"]("Prefers tabs", "a", "feedback")
    result = tools["save_memory"]("prefers TABS", "b", "feedback")
    assert result == {"success": False,
                      "error": f"memory {first['item_id']} has the same description: use update_memory to change it"}


def test_no_project_scope(tmp_path):
    [save] = [t for t in make_memory_tools(MemoryStore(tmp_path / "user")) if t.__name__ == "save_memory"]
    result = save("d", "c", "user")
    assert result["success"] is False and "no project scope" in result["error"]


def test_search(tools):
    tools["save_memory"]("Otter facts", "They nest in burrows.", "reference")
    tools["save_memory"]("Heron facts", "They wade at dawn.", "reference")
    found = tools["search_memory"]("burrows")
    assert (found["success"], found["query"], found["count"]) == (True, "burrows", 1)
    assert found["items"][0]["description"] == "Otter facts"
    assert tools["search_memory"]("")["count"] == 2


def test_update(tools):
    saved = tools["save_memory"]("Prefers spaces", "Spaces.", "feedback")
    assert tools["update_memory"](saved["item_id"], description="Prefers tabs") == {
        "success": True, "updated": True, "message": "Updated memory: Prefers tabs"}
    assert tools["update_memory"](saved["item_id"])["error"] == "Give a new description, content or type"
    assert tools["update_memory"]("deadbeef", content="x") == {"success": False, "error": "No memory with id deadbeef"}


def test_delete(tools):
    saved = tools["save_memory"]("Prefers tabs", "Tabs.", "feedback")
    assert tools["delete_memory"](saved["item_id"]) == {
        "success": True, "deleted": True, "message": "Deleted memory: Prefers tabs"}
    assert tools["delete_memory"](saved["item_id"]) == {"success": False, "error": f"No memory with id {saved['item_id']}"}


def test_read_of_an_unknown_id(tools):
    assert tools["read_memory"]("../x") == {"success": False, "error": "No memory with id ../x"}


def test_without_a_store_the_registered_tool_says_so():
    token = set_service_registry({})
    try:
        result = memory_tools.save_memory("d", "c", "user")
    finally:
        token.var.reset(token)
    assert result["success"] is False
