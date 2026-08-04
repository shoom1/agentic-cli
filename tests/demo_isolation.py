"""Construct fully-isolated ``ResearchDemoSettings`` for tests.

Plumbing shared by the headless application tests and the live scenario tests;
it defines no test classes, so pytest does not collect it.

``ResearchDemoSettings`` reads from three developer-owned places, and they are
*not* isolated by the same mechanism:

===========================  ==========================  ====================
source                       resolved                    isolated by
===========================  ==========================  ====================
``./.research_demo/…json``   ``Path.cwd()`` per call     ``monkeypatch.chdir``
``~/.research_demo/…json``   ``Path.home()`` per call    ``monkeypatch.setenv``
``~/.research_demo/.env``    ``Path.home()`` **at        an explicit
                             class-definition time**     ``_env_file=``
===========================  ==========================  ====================

The third is the trap: ``model_config["env_file"]`` is evaluated when the
module is imported — which, under pytest, is during collection, long before any
fixture runs. Redirecting ``HOME`` afterwards does nothing to it, so a test that
only patched ``HOME`` was still reading the developer's real dotenv (and any
API key in it). Passing ``_env_file`` explicitly is the only thing that moves
it, so every fixture here does.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# The real home, captured at import (collection time) before any fixture has
# had a chance to redirect HOME. This is what isolation is asserted *against*.
REAL_HOME = Path(os.path.expanduser("~")).resolve()

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.research_demo.settings import ResearchDemoSettings  # noqa: E402


def isolated_env_file(home: Path) -> Path:
    """Where an isolated run's dotenv lives (it need not exist)."""
    return home / ".research_demo" / ".env"


def make_isolated_settings(
    *, home: Path, workspace: Path, **kwargs: Any
) -> ResearchDemoSettings:
    """``ResearchDemoSettings`` that cannot read the developer's configuration.

    The caller is still responsible for redirecting ``HOME`` and ``cwd``
    (``monkeypatch``) — those cover the two JSON sources. This adds the third:
    an explicit ``_env_file`` under the temp home, overriding the class's
    import-time default.
    """
    kwargs.setdefault("_env_file", str(isolated_env_file(home)))
    return ResearchDemoSettings(workspace_dir=workspace, **kwargs)


def effective_env_file(**kwargs: Any) -> Path | None:
    """The dotenv path ``ResearchDemoSettings`` would *actually* read.

    Captured from the ``dotenv_settings`` source pydantic-settings hands to
    ``settings_customise_sources`` — the real value in effect, not the class
    default and not what the caller hoped it passed.

    Returns:
        The resolved path, or None when the dotenv source is disabled.
    """
    captured: list[Any] = []

    class _Probe(ResearchDemoSettings):
        @classmethod
        def settings_customise_sources(  # type: ignore[override]
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ):
            captured.append(getattr(dotenv_settings, "env_file", None))
            return super().settings_customise_sources(
                settings_cls,
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
            )

    _Probe(**kwargs)
    assert captured, "settings_customise_sources was never called"
    value = captured[0]
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        assert len(value) == 1, f"expected a single env_file, got {value}"
        value = value[0]
    return Path(value).resolve()


def assert_dotenv_isolated(env_file: Path | None, home: Path) -> None:
    """The effective dotenv must be inside the temp home, or disabled.

    Asserting "not under the real home" alone would pass for a path that is
    simply somewhere else on disk, so this pins the positive form too.
    """
    if env_file is None:
        return  # dotenv disabled outright — isolated by construction
    home = home.resolve()
    assert env_file.is_relative_to(home), (
        f"the effective dotenv is {env_file}, which is not under the test's "
        f"temporary home {home}"
    )
    assert not env_file.is_relative_to(REAL_HOME), (
        f"the effective dotenv resolves under the developer's real home: {env_file}"
    )


def suppress_global_logging_config(monkeypatch) -> None:
    """Stop app construction from reconfiguring logging process-wide.

    ``BaseCLIApp.__init__`` calls ``configure_logging()``, which reconfigures
    structlog globally with ``cache_logger_on_first_use=True``. That cannot be
    undone afterwards: by the time the constructor returns, the module-level
    loggers have already bound and *cached* a logger, so restoring the previous
    configuration leaves those caches in place and every later test using
    ``structlog.testing.capture_logs()`` silently captures nothing.

    So it is prevented rather than reverted. These tests assert on command
    routing and rendering, never on log output, so the application's logging
    configuration is not part of what they cover.
    """
    monkeypatch.setattr(
        "agentic_cli.cli.app.configure_logging", lambda *a, **k: None
    )


def assert_no_global_settings_leak() -> None:
    """No test may leave a global settings singleton behind.

    ``set_settings()`` writes a process-wide singleton with no teardown, so one
    test's settings would silently answer ``get_settings()`` for every later
    test that runs outside a manager's ``SettingsContext``.
    """
    from agentic_cli import config as _config

    assert _config._settings_instance is None, (
        "a test left a global settings singleton behind "
        f"({_config._settings_instance!r}); use the manager's SettingsContext "
        "instead of set_settings()"
    )
    assert _config._settings_context.get() is None, (
        "a test left settings in the context variable"
    )
