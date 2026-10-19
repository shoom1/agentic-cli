"""``/settings`` has a model picker per role, and says what thinking does.

The dialog offers three model pickers: the default model (``default_model``)
and the ``fast`` and ``strong`` roles (``model_roles``, through the synthetic
keys ``model_role:fast`` and ``model_role:strong``). Each offers exactly the
references the model catalog can run (``ModelCatalog.picker_refs``). Picking a
role writes ``model_roles[<role>]`` to the project settings file and
reinitializes the workflow with its sessions kept; a role the workflow cannot
run is rolled back like any other setting. The thinking-effort picker's
description says what the effort means for the default model.

``model_endpoints`` stays out of the dialog: it holds the addresses a key is
sent to, and the dialog may only edit what a project file may set.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import structlog

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.settings_command import SettingsCommand
from agentic_cli.cli.workflow_controller import WorkflowController
from agentic_cli.config import BaseSettings, validate_settings
from agentic_cli.models import ModelCatalog, ModelRef, ProviderKind, load_table
from agentic_cli.workflow import model_services
from agentic_cli.workflow.config import AgentConfig
from tests.models.fakes import ProviderAPI

PICKERS = ["model", "model_role:fast", "model_role:strong"]
KEY_VARIABLES = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch) -> Path:
    """A hermetic cwd (the project) and HOME, and no provider key in the environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def _app(settings: BaseSettings, workflow=None, keys: list[str] | None = None) -> BaseCLIApp:
    """The app's dialog side only; ``workflow`` stands for a ready one."""
    cls = BaseCLIApp
    if keys is not None:

        class _App(BaseCLIApp):
            def get_ui_setting_keys(self) -> list[str]:
                return keys

        cls = _App
    app = cls.__new__(cls)
    app._settings = settings
    app._workflow_controller = SimpleNamespace(is_ready=workflow is not None, workflow=workflow)
    return app


def _items(app: BaseCLIApp) -> dict:
    return {item.key: item for item in app._build_ui_items()}


def _refs(kind: ProviderKind) -> set[str]:
    return {f"{kind.value}:{model}" for model in load_table().fallback_models(kind)}


# --- the pickers ------------------------------------------------------------------


def test_the_dialog_has_a_model_picker_per_role():
    items = _items(_app(BaseSettings(google_api_key="k-g")))

    assert list(items) == [*PICKERS, "thinking_effort"]
    assert [items[key].label for key in PICKERS] == ["Default Model", "Fast Model", "Strong Model"]


def test_every_model_picker_offers_exactly_the_runnable_models():
    """The keyed providers' models, plus the OpenRouter models a role maps or
    the user pinned; nothing of a provider without a key."""
    settings = BaseSettings(
        google_api_key="k-g",
        anthropic_api_key="k-a",
        openrouter_api_key="k-r",
        model_roles={"strong": "openrouter:qwen/qwen3-235b-a22b"},
        pinned_models=["openrouter:meta-llama/llama-3.3-70b-instruct"],
    )

    items = _items(_app(settings))

    catalog = ModelCatalog(endpoints=model_services.endpoints_from_settings(settings))
    runnable = catalog.picker_refs(
        referenced=["openrouter:qwen/qwen3-235b-a22b"], pinned=settings.pinned_models
    )
    assert set(runnable) == (
        _refs(ProviderKind.GOOGLE)
        | _refs(ProviderKind.ANTHROPIC)
        | {"openrouter:qwen/qwen3-235b-a22b", "openrouter:meta-llama/llama-3.3-70b-instruct"}
    )
    for key in PICKERS:
        assert items[key].options == runnable, key


async def test_once_the_workflow_is_ready_the_pickers_offer_its_catalog():
    settings = BaseSettings(google_api_key="k-g")
    catalog = ModelCatalog(
        endpoints=model_services.endpoints_from_settings(settings),
        transport=ProviderAPI().transport,
    )
    await catalog.load()
    workflow = SimpleNamespace(catalog=catalog, model="google:gemini-3-flash-preview")

    items = _items(_app(settings, workflow))

    runnable = catalog.picker_refs(referenced=["google:gemini-3-flash-preview"])
    assert "google:gemini-2.0-flash" in runnable  # from the listing, not the table
    for key in PICKERS:
        assert items[key].options == runnable, key
    assert items["model_role:fast"].default == "google:gemini-2.5-flash-lite"  # listed
    assert items["model"].default == "google:gemini-3-flash-preview"


async def test_a_picker_s_menu_shows_every_model_whole():
    """With listings in, a picker offers many references; its open menu must
    show each one whole, or the models cannot be told apart."""
    from thinking_prompt.settings_dialog import DropdownControl, DropdownItem

    settings = BaseSettings(google_api_key="k-g", anthropic_api_key="k-a")
    catalog = ModelCatalog(
        endpoints=model_services.endpoints_from_settings(settings),
        transport=ProviderAPI().transport,
    )
    await catalog.load()
    workflow = SimpleNamespace(catalog=catalog, model="google:gemini-3-flash-preview")

    items = _items(_app(settings, workflow))

    for key in PICKERS:
        item = items[key]
        assert isinstance(item, DropdownItem), key
        control = DropdownControl(item)
        control._build_menu()
        menu = control._menu_window
        width = menu.width - len(menu.right_margins)  # the scrollbar takes a column
        content = control._menu_control.create_content(width, len(item.options))
        shown = [
            "".join(text for _, text in content.get_line(i)).strip()
            for i in range(content.line_count)
        ]
        assert shown == item.options, key


def test_a_mapped_role_shows_its_model():
    settings = BaseSettings(
        google_api_key="k-g",
        anthropic_api_key="k-a",
        model_roles={"fast": "claude-sonnet-4-5"},
    )

    assert _items(_app(settings))["model_role:fast"].default == "anthropic:claude-sonnet-4-5"


def test_an_unmapped_role_shows_the_model_it_resolves_to():
    items = _items(_app(BaseSettings(google_api_key="k-g")))

    assert items["model_role:fast"].default == "google:gemini-3.1-flash-lite-preview"
    assert items["model_role:strong"].default == "google:gemini-3.1-pro-preview"


def test_without_any_provider_there_are_no_model_pickers():
    items = _items(_app(BaseSettings()))

    assert not set(PICKERS) & set(items)


def test_the_role_pickers_pass_the_project_scope_guard_and_model_endpoints_does_not():
    app = _app(
        BaseSettings(google_api_key="k-g"),
        keys=["model_role:fast", "model_role:strong", "model_endpoints"],
    )

    with structlog.testing.capture_logs() as logs:
        items = _items(app)

    assert list(items) == ["model_role:fast", "model_role:strong"]
    excluded = [e["key"] for e in logs if e.get("event") == "user_scoped_setting_excluded_from_ui"]
    assert excluded == ["model_endpoints"]
    assert "model_endpoints" not in BaseCLIApp.get_ui_setting_keys(app)


# --- the thinking effort's description ------------------------------------------------


@pytest.mark.parametrize(
    ("values", "effort", "description"),
    [
        (
            {"google_api_key": "k-g", "default_model": "gemini-2.5-flash"},
            "high",
            "high → google:gemini-2.5-flash: budget 24576",
        ),
        (
            {"anthropic_api_key": "k-a", "default_model": "claude-sonnet-4-6"},
            "medium",
            "medium → anthropic:claude-sonnet-4-6: adaptive, effort medium",
        ),
        (
            {"google_api_key": "k-g", "default_model": "gemini-2.5-flash"},
            "none",
            "none → google:gemini-2.5-flash: off",
        ),
        (
            # It cannot turn thinking off: none is its lowest effort.
            {"anthropic_api_key": "k-a", "default_model": "claude-sonnet-5-5"},
            "none",
            "none → anthropic:claude-sonnet-5-5: adaptive, effort low",
        ),
        (
            {"openai_api_key": "k-o", "default_model": "gpt-4.1"},
            "high",
            "high → openai:gpt-4.1: ignored (no thinking knob)",
        ),
    ],
)
def test_the_thinking_effort_says_what_it_means_for_the_default_model(values, effort, description):
    settings = BaseSettings(thinking_effort=effort, **values)

    assert _items(_app(settings))["thinking_effort"].description == description


def test_a_ready_workflow_describes_the_model_it_runs():
    """The manager's own model, which a model switch may have set."""
    settings = BaseSettings(
        google_api_key="k-g", anthropic_api_key="k-a",
        default_model="gemini-2.5-flash", thinking_effort="high",
    )
    catalog = ModelCatalog(endpoints=model_services.endpoints_from_settings(settings))
    workflow = SimpleNamespace(catalog=catalog, model="anthropic:claude-sonnet-4-6")

    description = _items(_app(settings, workflow))["thinking_effort"].description

    assert description == "high → anthropic:claude-sonnet-4-6: adaptive, effort high"


def test_without_a_default_model_the_thinking_effort_keeps_its_own_description():
    field = BaseSettings.model_fields["thinking_effort"]

    assert _items(_app(BaseSettings()))["thinking_effort"].description == field.description


# --- update_setting("model_role:<role>") ------------------------------------------------


class TestUpdateSettingForARole:
    def test_it_maps_the_role_and_keeps_the_others(self):
        settings = BaseSettings(model_roles={"strong": "google:gemini-2.5-pro"})
        before = settings.model_roles

        settings.update_setting("model_role:fast", "anthropic:claude-haiku-4-5")

        assert settings.model_roles == {
            "strong": "google:gemini-2.5-pro",
            "fast": "anthropic:claude-haiku-4-5",
        }
        # A new mapping, so a rollback that puts the old object back undoes it.
        assert before == {"strong": "google:gemini-2.5-pro"}

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("model_role:fast", "strong", r"model_roles\['fast'\]"),
            ("model_role:fast", "anthropic:", r"model_roles\['fast'\]"),
            ("model_role:default", "google:gemini-2.5-flash", "default_model"),
            ("model_role:Fast", "google:gemini-2.5-flash", r"model_roles\['Fast'\]"),
        ],
    )
    def test_a_bad_role_or_model_is_refused(self, key, value, match):
        settings = BaseSettings()

        with pytest.raises(ValueError, match=match):
            settings.update_setting(key, value)

        assert settings.model_roles == {}


# --- applying and saving a role ----------------------------------------------------------


class FakeManager:
    """Reinitializes as a real manager does: from the live settings, and
    transactionally (a failure leaves it uninitialized)."""

    def __init__(self, settings: BaseSettings, refuses: str | None = None) -> None:
        self.settings = settings
        self.model = "google:gemini-2.5-flash"
        self.backend_type = "adk"
        self.is_initialized = True
        self.refuses = refuses
        self.reinits: list[tuple[str | None, bool, dict[str, str]]] = []
        self.cleanup = AsyncMock()

    async def reinitialize(self, model=None, preserve_sessions=True):
        self.reinits.append((model, preserve_sessions, dict(self.settings.model_roles)))
        if model is not None:
            self.model = model
        if self.refuses in self.settings.model_roles.values():
            self.is_initialized = False
            raise RuntimeError(f"Model {self.refuses} is not available")
        self.is_initialized = True


def _dialog_app(settings: BaseSettings, workflow, configs=None, dialog_result=None) -> BaseCLIApp:
    """A real app's settings path (apply, save) around a real controller."""
    controller = WorkflowController(configs or [AgentConfig(name="a", prompt="p")], settings)
    controller._workflow = workflow
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._settings = settings
    app._workflow_controller = controller
    app.errors, app.successes = [], []
    app.session = SimpleNamespace(
        show_dialog=AsyncMock(return_value=dialog_result),
        add_message=lambda *a: None,
        add_error=app.errors.append,
        add_success=app.successes.append,
        add_warning=lambda *a: None,
    )
    app._build_ui_items = lambda: ["item"]
    return app


def _project_file(home: Path) -> Path:
    return home / ".agentic_cli" / "settings.json"


def _user_file(home: Path) -> Path:
    return home / "home" / ".agentic_cli" / "settings.json"


async def _run_dialog(app: BaseCLIApp) -> None:
    with patch("agentic_cli.cli.settings_command.SettingsDialog", lambda **kw: None):
        await SettingsCommand().execute("", app)


async def test_picking_a_role_saves_it_to_the_project_and_reinitializes_keeping_sessions(home):
    settings = BaseSettings(google_api_key="k-g")
    manager = FakeManager(settings)
    app = _dialog_app(settings, manager, dialog_result={"model_role:fast": "google:gemini-2.5-pro"})

    await _run_dialog(app)

    assert manager.reinits == [(None, True, {"fast": "google:gemini-2.5-pro"})]
    assert json.loads(_project_file(home).read_text()) == {
        "model_roles": {"fast": "google:gemini-2.5-pro"}
    }
    assert not _user_file(home).exists()
    assert "Fast Model: google:gemini-2.5-pro" in app.successes
    assert not app.errors


# A pick saves only itself: the settings know roles from the user file, the
# environment and the app's defaults too, merged, and none of those belongs in
# a project file a teammate pulls. These tests load the settings from files.


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


async def test_a_pick_saves_only_that_role_to_the_project_file(home, monkeypatch):
    _write(_user_file(home), {
        "model_endpoints": {"local-vllm": {"base_url": "http://gpu-box:8000/v1"}},
        "model_roles": {"strong": "local-vllm:Qwen/Qwen3-32B"},
    })
    _write(_project_file(home), {"model_roles": {"fast": "google:gemini-2.5-flash-lite"}})
    monkeypatch.setenv("GOOGLE_API_KEY", "k-g")
    settings = BaseSettings()
    assert settings.model_roles == {
        "strong": "local-vllm:Qwen/Qwen3-32B", "fast": "google:gemini-2.5-flash-lite",
    }
    app = _dialog_app(
        settings, FakeManager(settings), dialog_result={"model_role:fast": "google:gemini-2.5-pro"}
    )

    await _run_dialog(app)

    assert json.loads(_project_file(home).read_text()) == {
        "model_roles": {"fast": "google:gemini-2.5-pro"}
    }
    assert settings.model_roles["strong"] == "local-vllm:Qwen/Qwen3-32B"  # still in use here
    # A teammate who pulls the project file, without the user's endpoint, starts.
    _user_file(home).unlink()
    validate_settings(BaseSettings())


async def test_a_role_from_the_environment_is_not_made_permanent(home, monkeypatch):
    monkeypatch.setenv("AGENTIC_MODEL_ROLES", json.dumps({"strong": "anthropic:claude-opus-4-6"}))
    monkeypatch.setenv("GOOGLE_API_KEY", "k-g")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k-a")
    settings = BaseSettings()
    app = _dialog_app(
        settings, FakeManager(settings), dialog_result={"model_role:fast": "google:gemini-2.5-pro"}
    )

    await _run_dialog(app)

    assert json.loads(_project_file(home).read_text()) == {
        "model_roles": {"fast": "google:gemini-2.5-pro"}
    }


async def test_a_role_the_workflow_cannot_run_is_rolled_back(home):
    settings = BaseSettings(google_api_key="k-g", model_roles={"strong": "google:gemini-2.5-pro"})
    manager = FakeManager(settings, refuses="google:gemini-0-missing")
    app = _dialog_app(settings, manager, dialog_result={"model_role:fast": "google:gemini-0-missing"})

    await _run_dialog(app)

    assert settings.model_roles == {"strong": "google:gemini-2.5-pro"}
    assert not _project_file(home).exists()
    # Brought back on the model and the roles it had.
    assert manager.reinits[-1] == ("google:gemini-2.5-flash", True, {"strong": "google:gemini-2.5-pro"})
    assert app._workflow_controller.is_ready
    assert any("gemini-0-missing" in error for error in app.errors)


async def test_a_malformed_role_never_reaches_the_workflow(home):
    settings = BaseSettings(google_api_key="k-g")
    manager = FakeManager(settings)
    app = _dialog_app(settings, manager, dialog_result={"model_role:fast": "fast"})

    await _run_dialog(app)

    assert manager.reinits == []
    assert settings.model_roles == {}
    assert not _project_file(home).exists()
    assert any("Fast Model" in error for error in app.errors)


# --- a real manager ------------------------------------------------------------------------


@pytest.fixture
def listed(monkeypatch) -> ProviderAPI:
    """Every catalog a manager builds lists the recorded payloads."""
    api = ProviderAPI()

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(endpoints=endpoints, transport=api.transport)

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)
    return api


async def _adk_with_a_session(monkeypatch):
    """An initialized ADK manager whose writer runs on the ``fast`` role, and
    a conversation (``kept``) in its in-memory session store."""
    pytest.importorskip("google.adk")
    from google.adk.events import Event
    from google.genai import types

    from agentic_cli.workflow.adk.manager import GoogleADKWorkflowManager

    monkeypatch.setenv("GOOGLE_API_KEY", "g-test-key")
    settings = BaseSettings(session_store="memory")
    configs = [
        AgentConfig(name="coordinator", prompt="Coordinate.", sub_agents=["writer"]),
        AgentConfig(name="writer", prompt="Write.", model="fast"),
    ]
    manager = GoogleADKWorkflowManager(agent_configs=configs, settings=settings)
    await manager.initialize_services()
    service = manager._session_service
    session = await service.create_session(app_name=manager.app_name, user_id="u", session_id="kept")
    await service.append_event(session, Event(
        author="user",
        content=types.Content(role="user", parts=[types.Part.from_text(text="Hello")]),
    ))
    return settings, configs, manager


async def test_a_role_change_reaches_the_agents_and_keeps_the_conversation(listed, monkeypatch):
    settings, configs, manager = await _adk_with_a_session(monkeypatch)
    try:
        assert manager._resolved_models["writer"].ref == ModelRef("google", "gemini-2.5-flash-lite")
        app = _dialog_app(settings, manager, configs)

        assert await app.apply_settings({"model_role:fast": "google:gemini-2.5-pro"})

        assert manager.is_initialized
        assert manager._resolved_models["writer"].ref == ModelRef("google", "gemini-2.5-pro")
        assert await manager.session_exists("kept", user_id="u")
    finally:
        await manager.cleanup()


async def test_a_role_the_listing_lacks_is_rolled_back_on_a_real_manager(listed, monkeypatch):
    settings, configs, manager = await _adk_with_a_session(monkeypatch)
    try:
        app = _dialog_app(settings, manager, configs)

        assert not await app.apply_settings({"model_role:fast": "google:gemini-0-missing"})

        assert settings.model_roles == {}
        assert app._workflow_controller.is_ready
        assert manager._resolved_models["writer"].ref == ModelRef("google", "gemini-2.5-flash-lite")
        assert await manager.session_exists("kept", user_id="u")
        assert any("gemini-0-missing" in error for error in app.errors)
    finally:
        await manager.cleanup()
