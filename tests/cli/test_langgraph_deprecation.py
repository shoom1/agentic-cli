"""The LangGraph backend says it is deprecated, where each of its users looks.

0.7.0 deprecates the LangGraph backend and 0.8.0 removes it. A CLI user hears
it from the CLI, through the session: one warning at startup when the
orchestrator is ``langgraph``, and one more on a ``/settings`` switch to it. A
library user gets a ``DeprecationWarning`` from ``LangGraphWorkflowManager()``.
Python's default filters show that warning only when the user's own
``__main__`` builds the manager; the CLI builds it through the factory, so it
never prints over the terminal UI. The last two tests check that on the real
console process, over a pty.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from thinking_prompt import AppInfo

from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.settings import OrchestratorType
from tests.examples.console_smoke import (
    assert_clean_session,
    child_env,
    platform_skip_reason,
    run_console_smoke,
)

CLI_NOTICE = (
    "The LangGraph backend is deprecated and will be removed in 0.8.0. Set "
    "orchestrator to deepagents (pip install 'agentic-cli[deepagents]'); "
    "sessions saved by LangGraph do not carry over."
)
LIBRARY_NOTICE = (
    "LangGraphWorkflowManager is deprecated and will be removed in agentic-cli "
    "0.8.0; use the deepagents orchestrator."
)
KEY_VARIABLES = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
CONFIGS = [AgentConfig(name="assistant", prompt="Help.")]
ADK_RUNNER_LOGGER = "google_adk.google.adk.runners"
ADK_CACHE_LOGGER = "google_adk.google.adk.models.gemini_context_cache_manager"
_REPO_ROOT = Path(__file__).resolve().parents[2]

with warnings.catch_warnings():
    # pydantic notes that the field overrides the mixin's; it does, on purpose.
    warnings.simplefilter("ignore")

    class PlainStringSettings(BaseSettings):
        """The orchestrator as a plain string, as ``examples/hello_langgraph.py``
        declares it."""

        orchestrator: str = "langgraph"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd and HOME with a Google key, and an app that leaves no
    global hook, filter or logging configuration behind."""
    import logging

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(warnings, "showwarning", warnings.showwarning)
    for name in (ADK_RUNNER_LOGGER, ADK_CACHE_LOGGER):
        adk_logger = logging.getLogger(name)
        monkeypatch.setattr(adk_logger, "filters", list(adk_logger.filters))
    return tmp_path


class FakeManager:
    """What the app's startup and an orchestrator swap touch, for any backend."""

    catalog = None

    def __init__(self, backend: str, fails: bool = False) -> None:
        self.backend_type = backend
        self.model = "google:gemini-2.5-flash"
        self.agent_models: dict = {}
        self.is_initialized = False
        self._fails = fails

    async def initialize_services(self) -> None:
        if self._fails:
            raise RuntimeError(f"the {self.backend_type} backend cannot start")
        self.is_initialized = True

    async def reinitialize(self, model=None, preserve_sessions=True) -> None:
        self.is_initialized = True

    async def load_session(self, session_id, user_id=None) -> bool:
        return False

    async def cleanup(self) -> None:
        self.is_initialized = False


@pytest.fixture
def backends(monkeypatch) -> list[FakeManager]:
    """Every manager the controller builds, in order: a fake of the backend the
    settings name. ``backends.failing`` names backends that fail to start."""
    built: list[FakeManager] = []
    failing: set[str] = set()

    def create(**kwargs) -> FakeManager:
        backend = OrchestratorType(kwargs["settings"].orchestrator).value
        manager = FakeManager(backend, fails=backend in failing)
        built.append(manager)
        return manager

    monkeypatch.setattr(
        "agentic_cli.cli.workflow_controller.create_workflow_manager_from_settings", create
    )
    return SimpleNamespace(built=built, failing=failing)


class Session:
    """The terminal session: records what the app shows, and plays the user's
    inputs (and their ``/settings`` dialog answers) while it runs."""

    def __init__(self, inputs=(), dialogs=()) -> None:
        self.shown: list[tuple[str, str]] = []
        self._inputs = list(inputs)
        self._dialogs = list(dialogs)
        self._handler = None

    def add_warning(self, text: str) -> None:
        self.shown.append(("warning", text))

    def add_error(self, text: str) -> None:
        self.shown.append(("error", text))

    def add_success(self, text: str) -> None:
        self.shown.append(("success", text))

    def add_message(self, role: str, text: str) -> None:
        self.shown.append((role, text))

    def add_response(self, text: str, markdown: bool = True) -> None:
        self.shown.append(("response", text))

    def set_status(self, text: str) -> None:
        pass

    def start_thinking(self, *args, **kwargs):
        return SimpleNamespace(finish=lambda **kw: None)

    def on_input(self, handler):
        self._handler = handler
        return handler

    async def show_dialog(self, dialog) -> dict:
        return self._dialogs.pop(0)

    async def run_async(self) -> None:
        for text in self._inputs:
            await self._handler(text)

    def exit(self) -> None:
        pass

    @property
    def notices(self) -> list[str]:
        return [text for kind, text in self.shown if kind == "warning" and text == CLI_NOTICE]


class App(BaseCLIApp):
    """An app that offers the orchestrator in ``/settings``."""

    def get_ui_setting_keys(self) -> list[str]:
        return [*super().get_ui_setting_keys(), "orchestrator"]


def _app(settings: BaseSettings, session: Session | None = None) -> App:
    app = App(
        app_info=AppInfo(name="Deprecation test", version="0"),
        agent_configs=CONFIGS,
        settings=settings,
    )
    app.session = session or Session()
    return app


def _settings(**fields) -> BaseSettings:
    return BaseSettings(session_store="memory", **fields)


# --- The CLI: at startup ---------------------------------------------------------


@pytest.mark.parametrize(
    "orchestrator, notices",
    [("langgraph", 1), (OrchestratorType.LANGGRAPH, 1), ("deepagents", 0), ("adk", 0)],
    ids=["langgraph", "enum", "deepagents", "adk"],
)
async def test_startup_warns_once_on_langgraph_and_not_otherwise(
    backends, capsys, orchestrator, notices
):
    app = _app(_settings(orchestrator=orchestrator))

    await app.run()

    assert app.session.notices == [CLI_NOTICE] * notices
    assert [m.backend_type for m in backends.built] == [OrchestratorType(orchestrator).value]
    # Through the session alone: nothing printed, no Python warning.
    captured = capsys.readouterr()
    assert "deprecated" not in captured.out + captured.err


async def test_startup_warns_for_an_orchestrator_held_as_a_plain_string(backends):
    app = _app(PlainStringSettings(session_store="memory"))

    await app.run()

    assert app.session.notices == [CLI_NOTICE]


async def test_startup_warns_even_when_langgraph_cannot_start(backends):
    """The backend's extra may be missing, or its key: the user still hears it."""
    backends.failing.add("langgraph")
    app = _app(_settings(orchestrator="langgraph"))

    await app.run()

    assert app.session.notices == [CLI_NOTICE]
    (failed,) = [
        i for i, (kind, text) in enumerate(app.session.shown)
        if kind == "error" and "cannot start" in text
    ]
    assert app.session.shown.index(("warning", CLI_NOTICE)) < failed


# --- The CLI: a /settings switch -------------------------------------------------


async def test_a_settings_switch_to_langgraph_warns_again(backends):
    session = Session(
        inputs=["/settings", "/settings"],
        dialogs=[{"orchestrator": "adk"}, {"orchestrator": "langgraph"}],
    )
    app = _app(_settings(orchestrator="langgraph"), session)

    await app.run()

    assert [m.backend_type for m in backends.built] == ["langgraph", "adk", "langgraph"]
    assert session.notices == [CLI_NOTICE, CLI_NOTICE]
    switched = session.shown.index(("success", "Orchestrator: langgraph"))
    assert session.shown[switched + 1] == ("warning", CLI_NOTICE)
    assert session.shown.index(("success", "Orchestrator: adk")) > session.shown.index(
        ("warning", CLI_NOTICE)
    )


async def test_a_change_made_on_langgraph_does_not_repeat_it(backends):
    session = Session(inputs=["/settings"], dialogs=[{"thinking_effort": "high"}])
    app = _app(_settings(orchestrator="langgraph"), session)

    await app.run()

    assert ("success", "Thinking Effort: high") in session.shown
    assert session.notices == [CLI_NOTICE]  # the startup one alone


async def test_a_switch_to_another_backend_says_nothing(backends):
    session = Session(inputs=["/settings"], dialogs=[{"orchestrator": "deepagents"}])
    app = _app(_settings(), session)

    await app.run()

    assert [m.backend_type for m in backends.built] == ["adk", "deepagents"]
    assert session.notices == []


async def test_a_switch_that_is_rolled_back_says_nothing(backends):
    backends.failing.add("langgraph")
    session = Session(inputs=["/settings"], dialogs=[{"orchestrator": "langgraph"}])
    app = _app(_settings(), session)

    await app.run()

    assert app.settings.orchestrator is OrchestratorType.ADK
    assert any(
        kind == "error" and text.startswith("Failed to reinitialize workflow")
        for kind, text in session.shown
    )
    assert session.notices == []


async def test_a_switch_before_the_workflow_is_ready_warns_too():
    """The next initialization builds LangGraph: the switch is made now."""
    app = _app(_settings())

    assert await app.apply_settings({"orchestrator": "langgraph"})

    assert app.session.shown == [
        ("success", "Orchestrator: langgraph"),
        ("warning", CLI_NOTICE),
    ]


def test_the_orchestrator_setting_says_langgraph_is_deprecated():
    """What the ``/settings`` picker shows under the orchestrator."""
    description = BaseSettings.model_fields["orchestrator"].description

    assert description == (
        "Workflow orchestrator backend: adk (the default), deepagents "
        "(pip install 'agentic-cli[deepagents]') or langgraph (deprecated, "
        "removed in 0.8.0)"
    )


# --- The library warning ---------------------------------------------------------


def _library_warnings(caught) -> list[warnings.WarningMessage]:
    return [w for w in caught if "LangGraphWorkflowManager is deprecated" in str(w.message)]


def test_building_the_manager_warns_library_users():
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.langgraph import LangGraphWorkflowManager

    with pytest.warns(DeprecationWarning) as caught:
        LangGraphWorkflowManager(agent_configs=CONFIGS, settings=_settings())

    (warning,) = _library_warnings(caught)
    assert warning.category is DeprecationWarning
    assert str(warning.message) == LIBRARY_NOTICE
    # Blamed on the code that built it (stacklevel=2), so Python's filters
    # judge the caller's module, not the backend's.
    assert warning.filename == __file__


def test_built_through_the_factory_the_factory_is_blamed():
    """Python's default filters show a ``DeprecationWarning`` only for code in
    ``__main__``; the CLI's managers come from the factory."""
    pytest.importorskip("langgraph")
    from agentic_cli.workflow.factory import create_workflow_manager_from_settings

    with pytest.warns(DeprecationWarning) as caught:
        create_workflow_manager_from_settings(CONFIGS, _settings(orchestrator="langgraph"))

    (warning,) = _library_warnings(caught)
    assert Path(warning.filename).as_posix().endswith("agentic_cli/workflow/factory.py")


def test_the_other_backends_are_not_deprecated():
    pytest.importorskip("deepagents")
    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager
    from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GoogleADKWorkflowManager(agent_configs=CONFIGS, settings=_settings())
        DeepAgentsWorkflowManager(agent_configs=CONFIGS, settings=_settings())

    assert [str(w.message) for w in caught if "will be removed" in str(w.message)] == []


# --- The real console process ----------------------------------------------------

_SKIP = platform_skip_reason()
_HELLO_LANGGRAPH = [sys.executable, str(_REPO_ROOT / "examples" / "hello_langgraph.py")]


@pytest.fixture
def console(tmp_path) -> tuple[Path, Path]:
    pytest.importorskip("langgraph")
    if _SKIP is not None:
        pytest.skip(_SKIP)
    home, cwd = tmp_path / "console-home", tmp_path / "console-cwd"
    home.mkdir()
    cwd.mkdir()
    return home, cwd


def test_the_console_shows_the_notice_once_and_no_python_warning(console):
    """Startup → /help → /exit on the example LangGraph app, with Python's
    default warning filters (the child inherits no ``PYTHONWARNINGS``)."""
    home, cwd = console

    result = run_console_smoke(_HELLO_LANGGRAPH, cwd=cwd, home=home)

    assert_clean_session(result)
    assert result.transcript.count("The LangGraph backend is deprecated") == 1
    assert "[WARN] The LangGraph backend is deprecated" in result.transcript
    assert "LangGraphWorkflowManager is deprecated" not in result.transcript
    assert "DeprecationWarning" not in result.transcript


def test_the_python_warning_would_show_if_the_filters_let_it(console):
    """The check above can see the warning: with every ``DeprecationWarning``
    shown, it prints over the UI."""
    home, cwd = console
    env = {**child_env(home), "PYTHONWARNINGS": "always::DeprecationWarning"}

    result = run_console_smoke(_HELLO_LANGGRAPH, cwd=cwd, home=home, env=env)

    assert_clean_session(result)
    assert f"DeprecationWarning: {LIBRARY_NOTICE}" in result.transcript
