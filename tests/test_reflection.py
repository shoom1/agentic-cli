"""Tests for tool reflection memory."""

import pytest

from agentic_cli.tools.reflection_tools import ReflectionStore, ToolReflection


class TestToolReflection:

    def test_dataclass_roundtrip(self):
        r = ToolReflection(
            tool_name="edit_file",
            error_summary="Content mismatch",
            heuristic="Always read the file first",
            created_at="2026-03-29T00:00:00",
        )
        data = r.to_dict()
        restored = ToolReflection.from_dict(data)
        assert restored.tool_name == "edit_file"
        assert restored.heuristic == "Always read the file first"


class TestReflectionStore:

    def test_save_and_get(self, mock_context):
        store = ReflectionStore(mock_context.settings)
        store.save("edit_file", "content mismatch", "read file first")
        reflections = store.get_for_tool("edit_file")
        assert len(reflections) == 1
        assert reflections[0].heuristic == "read file first"

    def test_bounded_buffer(self, mock_context):
        store = ReflectionStore(mock_context.settings, max_per_tool=2)
        store.save("tool_a", "err1", "hint1")
        store.save("tool_a", "err2", "hint2")
        store.save("tool_a", "err3", "hint3")
        reflections = store.get_for_tool("tool_a")
        assert len(reflections) == 2
        heuristics = [r.heuristic for r in reflections]
        assert "hint1" not in heuristics
        assert "hint2" in heuristics
        assert "hint3" in heuristics

    def test_separate_tools(self, mock_context):
        store = ReflectionStore(mock_context.settings, max_per_tool=2)
        store.save("tool_a", "err", "hint_a")
        store.save("tool_b", "err", "hint_b")
        assert len(store.get_for_tool("tool_a")) == 1
        assert len(store.get_for_tool("tool_b")) == 1

    def test_persistence(self, mock_context):
        store1 = ReflectionStore(mock_context.settings)
        store1.save("edit_file", "err", "hint")
        store2 = ReflectionStore(mock_context.settings)
        assert len(store2.get_for_tool("edit_file")) == 1

    def test_get_all(self, mock_context):
        store = ReflectionStore(mock_context.settings)
        store.save("tool_a", "err", "hint_a")
        store.save("tool_b", "err", "hint_b")
        all_reflections = store.get_all()
        assert "tool_a" in all_reflections
        assert "tool_b" in all_reflections

    def test_format_for_prompt(self, mock_context):
        store = ReflectionStore(mock_context.settings)
        store.save("edit_file", "mismatch", "read first")
        text = store.format_for_prompt("edit_file")
        assert "read first" in text
        assert "edit_file" in text

    def test_format_empty(self, mock_context):
        store = ReflectionStore(mock_context.settings)
        assert store.format_for_prompt("unknown_tool") == ""
