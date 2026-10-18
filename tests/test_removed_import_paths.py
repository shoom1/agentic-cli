"""The import paths 0.7.0 removed fail instead of forwarding."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "agentic_cli.knowledge_base",
        "agentic_cli.knowledge_base.manager",
        "agentic_cli.tools.kb_convert",
    ],
)
def test_a_removed_module_cannot_be_imported(module, monkeypatch):
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_memory_tools_no_longer_exports_a_memory_store():
    with pytest.raises(ImportError):
        from agentic_cli.tools.memory_tools import MemoryStore  # noqa: F401
