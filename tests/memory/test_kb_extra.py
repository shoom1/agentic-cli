"""``require_kb_module``: tests that need the ``kb`` extra skip without it,
except in CI's kb job (``KB_REQUIRE_EXTRA=1``), where a missing library must
fail the run instead of reading as green."""

from __future__ import annotations

import json
import sys

import pytest

from tests.memory.kb_extra import require_kb_module


@pytest.fixture
def without_trafilatura(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", None)


def test_a_missing_library_skips_by_default(monkeypatch, without_trafilatura):
    monkeypatch.delenv("KB_REQUIRE_EXTRA", raising=False)

    with pytest.raises(pytest.skip.Exception):
        require_kb_module("trafilatura")


def test_a_missing_library_fails_when_the_extra_is_required(monkeypatch, without_trafilatura):
    monkeypatch.setenv("KB_REQUIRE_EXTRA", "1")

    with pytest.raises(pytest.fail.Exception, match="trafilatura.*KB_REQUIRE_EXTRA=1"):
        require_kb_module("trafilatura")


def test_an_installed_library_is_returned(monkeypatch):
    monkeypatch.setenv("KB_REQUIRE_EXTRA", "1")

    assert require_kb_module("json") is json
