"""``agentic_cli.tools.kb_convert`` forwards to ``agentic_cli.memory.kb`` until 0.7.0."""

from __future__ import annotations

import importlib
import sys

import pytest

import agentic_cli.memory.kb as kb


def test_the_old_module_warns_and_forwards(monkeypatch):
    monkeypatch.delitem(sys.modules, "agentic_cli.tools.kb_convert", raising=False)

    with pytest.warns(DeprecationWarning, match="0.7.0"):
        old = importlib.import_module("agentic_cli.tools.kb_convert")

    assert old.convert_document is kb.convert_document
    assert old.ConvertedDocument is kb.ConvertedDocument
    assert old.UnsupportedDocument is kb.UnsupportedDocument


def test_the_tools_do_not_use_the_old_module():
    from tests.memory.test_boundary import outside_violations

    assert not any(target == "agentic_cli.tools.kb_convert" for _path, target in outside_violations())
