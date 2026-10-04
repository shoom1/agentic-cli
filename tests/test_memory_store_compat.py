"""``agentic_cli.tools.memory_tools.MemoryStore(settings)`` keeps working until 0.7.0."""

from __future__ import annotations

import pytest

import agentic_cli.memory as memory
from agentic_cli.config import BaseSettings
from agentic_cli.tools import memory_tools


def test_the_old_constructor_warns_and_uses_the_workspace_memory_dir(tmp_path):
    settings = BaseSettings(workspace_dir=tmp_path / "workspace")

    with pytest.warns(DeprecationWarning, match="agentic_cli.memory.MemoryStore"):
        old = memory_tools.MemoryStore(settings)
    old.store("written by the old class")

    new = memory.MemoryStore(settings.workspace_dir / "memory")
    assert [item.content for item in new.search("old class")] == ["written by the old class"]
    assert isinstance(old, memory.MemoryStore)


def test_the_item_types_are_re_exported_unchanged():
    assert memory_tools.MemoryItem is memory.MemoryItem
    assert memory_tools.ForgettingPolicy is memory.ForgettingPolicy
