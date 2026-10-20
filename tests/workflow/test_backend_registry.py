"""The factory's backend registry: one lazy entry per orchestrator.

``workflow/factory.py`` maps each orchestrator to the ``"module:Class"`` that
implements it and the extra that installs it. Only the import is guarded: a
backend whose extra is missing fails with that extra's install hint, while an
``ImportError`` raised by a backend's own constructor is that backend's error
and reaches the caller as itself. Constructor arguments that only one backend
takes come from that backend's entry.
"""

from __future__ import annotations

import importlib
import importlib.util
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langgraph")

from agentic_cli.workflow import factory  # noqa: E402
from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.factory import (  # noqa: E402
    create_workflow_manager_from_settings,
)
from agentic_cli.workflow.langgraph import LangGraphWorkflowManager  # noqa: E402
from agentic_cli.workflow.settings import OrchestratorType  # noqa: E402

ADK_CLASS = "agentic_cli.workflow.adk.manager.GoogleADKWorkflowManager"
LANGGRAPH_CLASS = "agentic_cli.workflow.langgraph.LangGraphWorkflowManager"
DEEPAGENTS_CLASS = "agentic_cli.workflow.deepagents.DeepAgentsWorkflowManager"

#: A case that imports the Deep Agents backend needs its extra.
_needs_deepagents = pytest.mark.skipif(
    importlib.util.find_spec("deepagents") is None, reason="needs the deepagents extra"
)


def _each(orchestrators) -> list:
    return [
        pytest.param(o, marks=_needs_deepagents) if o is OrchestratorType.DEEPAGENTS else o
        for o in orchestrators
    ]


NOT_ADK = _each(o for o in OrchestratorType if o is not OrchestratorType.ADK)


@pytest.fixture
def configs() -> list[AgentConfig]:
    return [AgentConfig(name="agent", prompt="You are a test agent.")]


def _settings(orchestrator=OrchestratorType.ADK, session_store="memory") -> MagicMock:
    settings = MagicMock()
    settings.orchestrator = orchestrator
    settings.session_store = session_store
    settings.app_name = "test-app"
    return settings


class _LogRecorder:
    """Stands in for the factory's logger and records every call.

    ``structlog.testing.capture_logs()`` depends on whether an earlier test
    configured logging; recording the calls keeps the test order-independent.
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
    monkeypatch.setattr(factory, "logger", recorder)
    return recorder.events


def _missing(monkeypatch, module_name: str) -> ModuleNotFoundError:
    """Make importing ``module_name`` fail as a missing extra does."""
    error = ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")
    real_import = importlib.import_module

    def import_module(name, package=None):
        if name == module_name:
            raise error
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    return error


class TestEntries:
    def test_every_orchestrator_has_an_entry(self):
        assert set(factory._BACKENDS) == set(OrchestratorType)

    def test_entries_name_the_class_and_the_extra(self):
        assert factory._BACKENDS[OrchestratorType.ADK] == (
            "agentic_cli.workflow.adk.manager:GoogleADKWorkflowManager",
            "",
        )
        assert factory._BACKENDS[OrchestratorType.LANGGRAPH] == (
            "agentic_cli.workflow.langgraph:LangGraphWorkflowManager",
            "langgraph",
        )
        assert factory._BACKENDS[OrchestratorType.DEEPAGENTS] == (
            "agentic_cli.workflow.deepagents:DeepAgentsWorkflowManager",
            "deepagents",
        )

    @pytest.mark.parametrize(
        "orchestrator, cls",
        [
            (OrchestratorType.ADK, GoogleADKWorkflowManager),
            (OrchestratorType.LANGGRAPH, LangGraphWorkflowManager),
        ],
    )
    def test_each_orchestrator_resolves_to_its_class(self, orchestrator, cls):
        assert factory._load_backend(orchestrator) is cls

    @_needs_deepagents
    def test_deepagents_resolves_to_its_class(self):
        from agentic_cli.workflow.deepagents import DeepAgentsWorkflowManager

        assert factory._load_backend(OrchestratorType.DEEPAGENTS) is DeepAgentsWorkflowManager


class TestOrchestratorValue:
    """Settings objects built outside pydantic may hold plain values."""

    def test_a_plain_string_routes_like_the_enum(self, monkeypatch, configs):
        cls = MagicMock()
        monkeypatch.setattr(LANGGRAPH_CLASS, cls)

        manager = create_workflow_manager_from_settings(configs, _settings("langgraph"))

        assert manager is cls.return_value

    def test_an_unknown_value_runs_on_adk_as_before(self, monkeypatch, configs):
        cls = MagicMock()
        monkeypatch.setattr(ADK_CLASS, cls)

        manager = create_workflow_manager_from_settings(configs, _settings(object()))

        assert manager is cls.return_value


class TestInstallHints:
    def test_a_missing_extra_gives_its_install_hint(self, monkeypatch):
        error = _missing(monkeypatch, "agentic_cli.workflow.langgraph")

        with pytest.raises(ImportError) as exc:
            factory._load_backend(OrchestratorType.LANGGRAPH)

        assert str(exc.value) == (
            "The langgraph orchestrator needs the 'langgraph' extra. "
            "Install with: pip install 'agentic-cli[langgraph]'\n"
            f"Original error: {error}"
        )
        assert exc.value.__cause__ is error

    def test_a_missing_deepagents_extra_gives_its_install_hint(self, monkeypatch):
        error = _missing(monkeypatch, "agentic_cli.workflow.deepagents")

        with pytest.raises(ImportError) as exc:
            factory._load_backend(OrchestratorType.DEEPAGENTS)

        assert str(exc.value) == (
            "The deepagents orchestrator needs the 'deepagents' extra. "
            "Install with: pip install 'agentic-cli[deepagents]'\n"
            f"Original error: {error}"
        )
        assert exc.value.__cause__ is error

    def test_the_factory_reports_the_missing_extra(self, monkeypatch, configs):
        _missing(monkeypatch, "agentic_cli.workflow.langgraph")

        with pytest.raises(ImportError, match=r"pip install 'agentic-cli\[langgraph\]'"):
            create_workflow_manager_from_settings(
                configs, _settings(OrchestratorType.LANGGRAPH)
            )

    def test_a_backend_without_an_extra_raises_the_import_error_itself(self, monkeypatch):
        """ADK is a core dependency: there is no extra to suggest."""
        error = _missing(monkeypatch, "agentic_cli.workflow.adk.manager")

        with pytest.raises(ImportError) as exc:
            factory._load_backend(OrchestratorType.ADK)

        assert exc.value is error

    def test_an_import_error_inside_a_constructor_is_not_an_install_hint(
        self, monkeypatch, configs
    ):
        error = ImportError("cannot import name 'thing' from 'somewhere'")
        monkeypatch.setattr(LANGGRAPH_CLASS, MagicMock(side_effect=error))

        with pytest.raises(ImportError) as exc:
            create_workflow_manager_from_settings(
                configs, _settings(OrchestratorType.LANGGRAPH)
            )

        assert exc.value is error


class TestConstructorArguments:
    @pytest.mark.parametrize("store", ["sqlite", "postgres", "memory"])
    def test_langgraph_gets_the_session_store_as_its_checkpointer(
        self, monkeypatch, configs, store
    ):
        cls = MagicMock()
        monkeypatch.setattr(LANGGRAPH_CLASS, cls)

        manager = create_workflow_manager_from_settings(
            configs, _settings(OrchestratorType.LANGGRAPH, session_store=store), model="m"
        )

        assert manager is cls.return_value
        cls.assert_called_once()
        assert cls.call_args.kwargs["checkpointer"] == store
        assert cls.call_args.kwargs["agent_configs"] is configs
        assert cls.call_args.kwargs["model"] == "m"

    def test_adk_gets_no_checkpointer(self, monkeypatch, configs):
        cls = MagicMock()
        monkeypatch.setattr(ADK_CLASS, cls)

        create_workflow_manager_from_settings(
            configs, _settings(OrchestratorType.ADK, session_store="sqlite")
        )

        assert "checkpointer" not in cls.call_args.kwargs

    def test_backend_kwargs_per_orchestrator(self):
        settings = _settings(session_store="sqlite")

        assert factory._backend_kwargs(OrchestratorType.LANGGRAPH, settings) == {
            "checkpointer": "sqlite"
        }
        assert factory._backend_kwargs(OrchestratorType.ADK, settings) == {}
        # Deep Agents opens its own checkpointer from the settings.
        assert factory._backend_kwargs(OrchestratorType.DEEPAGENTS, settings) == {}

    @_needs_deepagents
    def test_deepagents_gets_no_checkpointer(self, monkeypatch, configs):
        cls = MagicMock()
        monkeypatch.setattr(DEEPAGENTS_CLASS, cls)

        manager = create_workflow_manager_from_settings(
            configs, _settings(OrchestratorType.DEEPAGENTS, session_store="sqlite"), model="m"
        )

        assert manager is cls.return_value
        assert "checkpointer" not in cls.call_args.kwargs
        assert cls.call_args.kwargs["model"] == "m"

    def test_caller_kwargs_reach_the_backend(self, monkeypatch, configs):
        cls = MagicMock()
        monkeypatch.setattr(ADK_CLASS, cls)
        hook = object()

        create_workflow_manager_from_settings(configs, _settings(), on_event=hook)

        assert cls.call_args.kwargs["on_event"] is hook


class TestNativeAdkConfig:
    @pytest.mark.parametrize("orchestrator", NOT_ADK)
    def test_native_config_forces_adk_with_a_warning(
        self, monkeypatch, configs, logged, orchestrator
    ):
        adk = MagicMock()
        other = MagicMock()
        monkeypatch.setattr(ADK_CLASS, adk)
        module, _, name = factory._BACKENDS[orchestrator][0].partition(":")
        monkeypatch.setattr(f"{module}.{name}", other)

        manager = create_workflow_manager_from_settings(
            configs,
            _settings(orchestrator, session_store="sqlite"),
            adk_config_path="agent.yaml",
            adk_config_mode="native",
        )

        assert manager is adk.return_value
        other.assert_not_called()
        assert adk.call_args.kwargs["adk_config_path"] == "agent.yaml"
        assert "checkpointer" not in adk.call_args.kwargs
        assert [e["event"] for e in logged if e["log_level"] == "warning"] == [
            "adk_native_config_forces_adk"
        ]

    def test_native_config_on_adk_does_not_warn(self, monkeypatch, configs, logged):
        adk = MagicMock()
        monkeypatch.setattr(ADK_CLASS, adk)

        create_workflow_manager_from_settings(
            configs, _settings(), adk_config_path="agent.yaml", adk_config_mode="native"
        )

        adk.assert_called_once()
        assert not [e for e in logged if e["log_level"] == "warning"]

    @pytest.mark.parametrize("orchestrator", _each(OrchestratorType))
    def test_translate_mode_routes_to_the_configured_backend(
        self, monkeypatch, orchestrator
    ):
        translated = [AgentConfig(name="translated", prompt="hi")]
        monkeypatch.setattr(
            "agentic_cli.workflow.adk_config_bridge.translate_adk_yaml",
            lambda path: translated,
        )
        cls = MagicMock()
        module, _, name = factory._BACKENDS[orchestrator][0].partition(":")
        monkeypatch.setattr(f"{module}.{name}", cls)

        manager = create_workflow_manager_from_settings(
            [], _settings(orchestrator), adk_config_path="agent.yaml",
            adk_config_mode="translate",
        )

        assert manager is cls.return_value
        assert cls.call_args.kwargs["agent_configs"] is translated
        assert "adk_config_path" not in cls.call_args.kwargs
