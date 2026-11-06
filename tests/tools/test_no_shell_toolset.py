"""The shell toolset is gone; the OS sandbox it held lives on its own.

``tools/shell/`` was hard-disabled in every release (no setting could turn it
on), yet ``agentic_cli.tools`` imported and registered it for every app. A
future shell tool is the engine's ``shell.exec`` plus the OS sandbox, so the
package, its exports, its registration and its three settings are removed, and
``os_sandbox`` (live for ``execute_python``) is ``agentic_cli.tools.os_sandbox``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_the_shell_package_is_gone():
    """No ``agentic_cli.tools.shell`` package, and no source of one.

    A checkout that ran the old code keeps ``tools/shell/__pycache__/`` (git
    ignores it), and that folder imports as an empty namespace package. That
    is not the package: what must be gone is every module of it.
    """
    spec = importlib.util.find_spec("agentic_cli.tools.shell")
    if spec is None:
        return
    assert spec.origin is None, f"agentic_cli.tools.shell is a package again: {spec.origin}"
    sources = [
        p for loc in spec.submodule_search_locations or [] for p in Path(loc).rglob("*.py")
    ]
    assert sources == [], f"shell sources are back: {sources}"


def test_no_tool_named_shell_executor_is_registered():
    import agentic_cli.tools  # noqa: F401  (registers the built-in tools)
    import agentic_cli.tools.memory_tools  # noqa: F401  (lazy module)
    import agentic_cli.tools.sandbox  # noqa: F401  (lazy module)
    from agentic_cli.tools.registry import get_registry

    assert get_registry().get("shell_executor") is None
    assert "shell_executor" not in {d.name for d in get_registry().list_tools()}


@pytest.mark.parametrize("name", ["shell_executor", "is_shell_enabled"])
def test_the_tools_package_exports_no_shell_names(name):
    import agentic_cli.tools as tools

    assert name not in tools.__all__
    assert not hasattr(tools, name)


@pytest.mark.parametrize("field", ["shell_sandbox_type", "shell_docker_image", "shell_timeout"])
def test_the_unread_shell_settings_are_gone(field):
    from agentic_cli.config import BaseSettings

    assert field not in BaseSettings.model_fields


def test_the_os_sandbox_lives_in_tools():
    from agentic_cli.tools.os_sandbox import OSSandbox, OSSandboxPolicy, get_os_sandbox

    assert OSSandbox.__module__ == "agentic_cli.tools.os_sandbox.base"
    assert OSSandboxPolicy.__module__ == "agentic_cli.tools.os_sandbox.policy"
    assert callable(get_os_sandbox)
