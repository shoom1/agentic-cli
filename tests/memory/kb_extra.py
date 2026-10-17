"""Import a library from the ``kb`` extra (faiss, bm25s, trafilatura, ...).

Without the extra, the test skips, as ``pytest.importorskip`` would. CI's kb
job sets ``KB_REQUIRE_EXTRA=1``: there a missing or broken install fails the
test instead, since a skip reads as green.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import pytest


def require_kb_module(name: str) -> ModuleType:
    if os.environ.get("KB_REQUIRE_EXTRA") != "1":
        return pytest.importorskip(name)
    try:
        return importlib.import_module(name)
    except ImportError as e:
        pytest.fail(f"{name} cannot be imported with KB_REQUIRE_EXTRA=1: {e}", pytrace=False)
