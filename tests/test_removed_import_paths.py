"""The import paths 0.7.0 removed fail instead of forwarding."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["agentic_cli.knowledge_base", "agentic_cli.knowledge_base.manager"],
)
def test_a_removed_module_cannot_be_imported(module, monkeypatch):
    monkeypatch.delitem(sys.modules, module, raising=False)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
