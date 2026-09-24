"""Every framework tool must produce a model-facing declaration that
serializes, or every request from an agent holding it fails.

``update_memory`` used an ``object()`` sentinel as a parameter default; ADK
copied it into the declaration and the request could not be serialized, so
any ADK agent on Gemini that included the tool failed before reaching the
model.
"""

from __future__ import annotations

import warnings

import pytest

import agentic_cli.tools  # noqa: F401  (registers the built-in tools)
import agentic_cli.tools.memory_tools  # noqa: F401  (lazy module)
import agentic_cli.tools.sandbox  # noqa: F401  (lazy module)
from agentic_cli.tools.registry import get_registry

# shell_executor is registered but hard-disabled (never given to an agent), and
# its ShellSecurityConfig parameter is not declarable.
NOT_OFFERED = {"shell_executor"}


def _framework_tools():
    return sorted(
        (d.name, d.func)
        for d in get_registry().list_tools()
        if d.func is not None
        and getattr(d.func, "__module__", "").startswith("agentic_cli.")
        and d.name not in NOT_OFFERED
    )


def _serialize(func) -> str:
    from google.adk.tools import FunctionTool

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declaration = FunctionTool(func=func)._get_declaration()
    return declaration.model_dump_json(exclude_none=True)


@pytest.mark.parametrize("name, func", _framework_tools(), ids=lambda v: v if isinstance(v, str) else "")
def test_declaration_serializes(name, func):
    assert _serialize(func)


def test_service_bound_memory_tools_serialize():
    from unittest.mock import MagicMock
    from agentic_cli.tools.factories import make_memory_tools

    for tool in make_memory_tools(MagicMock()):
        assert _serialize(tool), tool.__name__
