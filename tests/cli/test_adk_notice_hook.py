"""The CLI logs ADK's notices instead of printing them over the terminal UI.

ADK 2 announces each experimental feature it uses with a ``UserWarning``
("[EXPERIMENTAL] feature FeatureName.X is enabled.", or "[EXPERIMENTAL]
ClassName: This feature is experimental ..." for a class marked experimental),
and warns through stdlib logging, when the runner is built, that an app with
sub-agents has no ``context_cache_config`` (the ADK manager gives the app one
while prompt caching is on). Python prints both to stderr, which in the CLI
lands over the terminal UI. ``BaseCLIApp`` installs a ``warnings.showwarning`` hook
that logs the notices at debug and hands every other warning to the hook it
replaced, and a filter that does the same for the context-cache advice.
"""

from __future__ import annotations

import logging
import sys
import warnings

import pytest
from thinking_prompt import AppInfo

from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp, _install_adk_notice_hook
from agentic_cli.config import BaseSettings
from tests.conftest import MockContext
from tests.demo_isolation import suppress_global_logging_config

NOTICE = "[EXPERIMENTAL] feature FeatureName.JSON_SCHEMA_FOR_FUNC_DECL is enabled."
ADK_RUNNER_LOGGER = "google_adk.google.adk.runners"
ADK_CACHE_LOGGER = "google_adk.google.adk.models.gemini_context_cache_manager"


class _LogRecorder:
    """Stands in for ``cli.app``'s logger and records every call.

    ``structlog.testing.capture_logs()`` sees no debug call once an earlier test
    has run ``configure_logging()``: the module's logger is cached as a logger
    that filters at WARNING. Recording the calls keeps these tests independent
    of test order.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.events.append({"event": event, "log_level": level, **fields})

        return log


@pytest.fixture
def logged(monkeypatch) -> list[dict]:
    recorder = _LogRecorder()
    monkeypatch.setattr(app_module, "logger", recorder)
    return recorder.events


@pytest.fixture(autouse=True)
def _restore_adk_runner_filters(monkeypatch):
    """The hook adds filters to ADK's loggers; take them off after each test."""
    for name in (ADK_RUNNER_LOGGER, ADK_CACHE_LOGGER):
        adk_logger = logging.getLogger(name)
        monkeypatch.setattr(adk_logger, "filters", list(adk_logger.filters))


@pytest.fixture
def previous_hook(monkeypatch) -> list[str]:
    """Stand in for Python's own hook: it prints, and records what it printed."""
    shown: list[str] = []

    def showwarning(message, category, filename, lineno, file=None, line=None):
        shown.append(str(message))
        sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))

    monkeypatch.setattr(warnings, "showwarning", showwarning)
    return shown


def _warn(message: str, category: type[Warning] = UserWarning) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(message, category, stacklevel=2)


def test_an_experimental_notice_is_logged_not_printed(previous_hook, logged, capsys):
    _install_adk_notice_hook()

    _warn(NOTICE)

    assert previous_hook == []
    assert capsys.readouterr().err == ""
    assert {
        "event": "adk_experimental_feature",
        "feature": "JSON_SCHEMA_FOR_FUNC_DECL",
        "log_level": "debug",
    } in logged


def test_an_experimental_class_notice_is_logged_not_printed(
    previous_hook, logged, capsys, monkeypatch
):
    """ADK's other form of notice, for a class marked experimental: a Gemini
    agent's first turn with prompt caching on builds ADK's context cache
    manager, which gives it."""
    from google.adk.models.gemini_context_cache_manager import GeminiContextCacheManager
    from google.adk.utils import feature_decorator

    monkeypatch.setattr(feature_decorator, "_WARNED_MESSAGES", set())  # ADK warns once
    monkeypatch.delenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", raising=False)
    _install_adk_notice_hook()

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        GeminiContextCacheManager(genai_client=None)

    assert previous_hook == []
    assert capsys.readouterr().err == ""
    assert {
        "event": "adk_experimental_feature",
        "feature": "GeminiContextCacheManager",
        "log_level": "debug",
    } in logged


@pytest.mark.parametrize(
    "message, category",
    [
        ("Something else went wrong.", UserWarning),
        (NOTICE, DeprecationWarning),
        (f"Note: {NOTICE}", UserWarning),
        ("[EXPERIMENTAL] GeminiContextCacheManager: experimental.", DeprecationWarning),
    ],
)
def test_every_other_warning_reaches_the_previous_hook(previous_hook, capsys, message, category):
    _install_adk_notice_hook()

    _warn(message, category)

    assert previous_hook == [message]
    assert message in capsys.readouterr().err


def test_installing_twice_keeps_one_hook(previous_hook):
    _install_adk_notice_hook()
    installed = warnings.showwarning
    filters = list(logging.getLogger(ADK_RUNNER_LOGGER).filters)
    _install_adk_notice_hook()

    assert warnings.showwarning is installed
    assert logging.getLogger(ADK_RUNNER_LOGGER).filters == filters


def test_the_cli_app_installs_the_hook(previous_hook, tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    suppress_global_logging_config(monkeypatch)

    BaseCLIApp(
        app_info=AppInfo(name="Notice test", version="0"),
        agent_configs=[],
        settings=BaseSettings(workspace_dir=tmp_path / "workspace"),
    )
    _warn(NOTICE)

    assert previous_hook == []


async def test_adks_context_cache_advice_is_logged_not_printed(logged, tmp_path, monkeypatch, caplog):
    """An app whose agents transfer, with prompt caching off: ADK 2 advises a
    context cache. Only a native ADK YAML app's agents transfer (agents built
    from configs delegate), and with caching on the app has a cache and ADK
    gives no advice."""
    import google.adk.runners as adk_runners

    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
    from tests.workflow.test_adk_context_cache import _native_yaml

    monkeypatch.setattr(adk_runners, "_UNCACHED_TRANSFER_APPS", set())  # ADK warns once per app
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    _install_adk_notice_hook()

    with MockContext(
        google_api_key="test-key", session_store="memory", prompt_caching_enabled=False
    ) as ctx:
        manager = GoogleADKWorkflowManager(
            agent_configs=[],
            settings=ctx.settings,
            model="gemini-2.5-flash",
            adk_config_path=_native_yaml(tmp_path, "gemini-2.5-flash", helper=True),
        )
        try:
            with caplog.at_level(logging.WARNING, logger=ADK_RUNNER_LOGGER):
                await manager.initialize_services()
        finally:
            await manager.cleanup()

    printed = [r.getMessage() for r in caplog.records if "context_cache_config" in r.getMessage()]
    assert printed == []
    advice = [e for e in logged if e["event"] == "adk_uncached_agent_transfer"]
    assert [e["log_level"] for e in advice] == ["debug"]
    assert "has no context_cache_config" in advice[0]["message"]


# --- ADK's context-cache failures ----------------------------------------------------
#
# While prompt caching is on, ADK warns through stdlib logging when it cannot
# create a Gemini cache or delete an old one (an expired cache is already gone
# from Google's side, so deleting it fails). The turn goes on uncached; the
# warning would print over the UI mid-conversation.


@pytest.mark.parametrize(
    "template, value, event",
    [
        ("Failed to create cache: %s", "500 INTERNAL", "adk_cache_create_failed"),
        ("Failed to cleanup cache %s: %s", ("cachedContents/c1", "404 NOT_FOUND"),
         "adk_cache_cleanup_failed"),
    ],
)
def test_adks_cache_failures_are_logged_not_printed(logged, caplog, template, value, event):
    _install_adk_notice_hook()
    cache_logger = logging.getLogger(ADK_CACHE_LOGGER)
    args = value if isinstance(value, tuple) else (value,)

    with caplog.at_level(logging.WARNING, logger=ADK_CACHE_LOGGER):
        cache_logger.warning(template, *args)

    assert [r for r in caplog.records if r.name == ADK_CACHE_LOGGER] == []
    (entry,) = [e for e in logged if e["event"] == event]
    assert entry["log_level"] == "debug"
    assert entry["message"] == template % args


def test_other_cache_warnings_still_reach_their_handlers(logged, caplog):
    _install_adk_notice_hook()

    with caplog.at_level(logging.WARNING, logger=ADK_CACHE_LOGGER):
        logging.getLogger(ADK_CACHE_LOGGER).warning("Something else went wrong")

    assert [r.getMessage() for r in caplog.records if r.name == ADK_CACHE_LOGGER] == [
        "Something else went wrong"
    ]
