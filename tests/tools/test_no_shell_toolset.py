"""The shell toolset is gone; the OS sandbox it held lives on its own.

``tools/shell/`` was hard-disabled in every release (no setting could turn it
on), yet ``agentic_cli.tools`` imported and registered it for every app. A
future shell tool is the engine's ``shell.exec`` plus the OS sandbox, so the
package, its exports, its registration and its three settings are removed, and
``os_sandbox`` (live for ``execute_python``) is ``agentic_cli.tools.os_sandbox``.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a parent package is missing
        return False


def test_the_shell_package_does_not_import():
    # find_spec also sees a namespace package, which is what a folder left
    # with only __pycache__ in it would import as.
    assert not _importable("agentic_cli.tools.shell")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agentic_cli.tools.shell")


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
