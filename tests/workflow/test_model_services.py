"""workflow/model_services.py is where settings become models-package arguments."""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path

import httpx
import pytest
from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict

from agentic_cli.config import BaseSettings, SettingsValidationError, validate_settings
from agentic_cli.models import (
    ModelCatalog,
    ModelRef,
    ModelSpecError,
    ProviderKind,
    ProviderSpec,
    RoleMap,
)
from agentic_cli.settings_mixins import AppSettingsMixin, CLISettingsMixin
from agentic_cli.workflow import model_services
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.model_services import build_catalog
from agentic_cli.workflow.settings import WorkflowSettingsMixin
from tests.models.fakes import openrouter_spec

_KEY_VARS = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
LAB = ProviderSpec(base_url="http://gpu-box:8000/v1", api_key_env="LAB_KEY")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A temp home (no real user settings), a temp project, and no keys in the env."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for var in (
        *_KEY_VARS, "LAB_KEY", "AGENTIC_DEFAULT_MODEL", "AGENTIC_MODELS", "AGENTIC_MODEL_PROVIDERS"
    ):
        monkeypatch.delenv(var, raising=False)


def _lab(**extra) -> dict:
    return {"lab": {"base_url": "http://gpu-box:8000/v1", "api_key_env": "LAB_KEY", **extra}}


@pytest.fixture
def real_catalogs(monkeypatch):
    """The catalog class ``build_catalog`` builds, as shipped: the suite's guard
    (conftest ``_no_model_listings``) keeps a manager's catalog unloaded."""
    monkeypatch.setattr(model_services, "ModelCatalog", ModelCatalog)


def _agents(*models: str | None) -> list[AgentConfig]:
    return [AgentConfig(name=f"agent{i}", prompt="p", model=m) for i, m in enumerate(models)]


def _app_with_env_files(env_file) -> type[BaseSettings]:
    """An app whose settings class reads ``env_file`` and defines the ``lab`` provider."""

    class AppSettings(BaseSettings):
        model_config = {**BaseSettings.model_config, "env_file": env_file}
        model_providers: dict[str, ProviderSpec] = Field(default_factory=lambda: {"lab": LAB})

    return AppSettings


def test_provider_keys_reads_the_three_key_settings():
    settings = BaseSettings(google_api_key="k-g", anthropic_api_key="k-a", openai_api_key="k-o")

    assert model_services.provider_keys(settings) == {
        "google": "k-g", "anthropic": "k-a", "openai": "k-o",
    }


def test_provider_keys_without_keys():
    assert model_services.provider_keys(BaseSettings()) == dict.fromkeys(
        ("google", "anthropic", "openai")
    )


def test_providers_from_settings():
    settings = BaseSettings(anthropic_api_key="k-a", model_providers=_lab())

    providers = model_services.providers_from_settings(settings, env={"LAB_KEY": "k-lab"})

    assert list(providers) == ["google", "anthropic", "openai", "lab"]
    assert providers["anthropic"].api_key == "k-a"
    assert not providers["google"].has_credentials
    assert providers["lab"].kind is ProviderKind.OPENAI_COMPATIBLE
    assert providers["lab"].base_url == "http://gpu-box:8000/v1"
    assert (providers["lab"].api_key, providers["lab"].key_env) == ("k-lab", "LAB_KEY")


def test_a_configured_key_comes_from_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("LAB_KEY", "k-from-env")
    settings = BaseSettings(openai_api_key="k-o", model_providers=_lab())

    providers = model_services.providers_from_settings(settings)

    assert providers["lab"].api_key == "k-from-env"
    assert providers["openai"].api_key == "k-o"


class TestUserLevelEnvFiles:
    """A configured provider's key is read from the environment, then from the
    settings class's user-level env files (absolute once ``~`` is expanded),
    never from one relative to the working directory, which a cloned
    repository can ship."""

    def test_the_key_is_read_from_the_user_level_env_file(self, tmp_path):
        env_file = tmp_path / "home" / ".app" / ".env"
        env_file.parent.mkdir()
        env_file.write_text("LAB_KEY=k-from-file\n")

        settings = _app_with_env_files(str(env_file))()

        lab = model_services.providers_from_settings(settings)["lab"]
        assert (lab.api_key, lab.has_credentials) == ("k-from-file", True)

    def test_the_environment_wins_over_the_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / "home" / ".env"
        env_file.write_text("LAB_KEY=k-from-file\n")
        monkeypatch.setenv("LAB_KEY", "k-from-env")

        settings = _app_with_env_files(str(env_file))()

        assert model_services.providers_from_settings(settings)["lab"].api_key == "k-from-env"

    def test_a_cwd_env_file_never_supplies_a_provider_key(self, tmp_path):
        (tmp_path / "project" / ".env").write_text("LAB_KEY=k-from-the-repo\n")

        for env_file in (".env", ["./.env"], (".env",)):
            settings = _app_with_env_files(env_file)()

            assert model_services.user_env_files(settings) == ()
            lab = model_services.providers_from_settings(settings)["lab"]
            assert (lab.api_key, lab.has_credentials) == (None, False)

    def test_the_absolute_entries_in_order_with_home_expanded(self, tmp_path):
        home = tmp_path / "home"
        settings = _app_with_env_files(
            ["~/.app/.env", ".env", str(tmp_path / "shared.env"), Path("~/.app/more.env")]
        )()

        assert model_services.user_env_files(settings) == (
            home / ".app" / ".env", tmp_path / "shared.env", home / ".app" / "more.env",
        )
        assert model_services.user_env_files(BaseSettings()) == ()

    def test_a_later_file_wins_and_the_environment_wins_over_all(self, tmp_path, monkeypatch):
        first, second = tmp_path / "first.env", tmp_path / "second.env"
        first.write_text("A_KEY=first\nB_KEY=first\nC_KEY=first\nNO_VALUE\n")
        second.write_text("B_KEY=second\nC_KEY=second\n")
        monkeypatch.setenv("C_KEY", "environment")

        env = model_services.provider_env(
            _app_with_env_files([str(first), str(tmp_path / "missing.env"), str(second)])()
        )

        assert (env["A_KEY"], env["B_KEY"], env["C_KEY"]) == ("first", "second", "environment")
        assert "NO_VALUE" not in env
        assert env["HOME"] == str(tmp_path / "home")  # the whole process environment

    def test_an_explicit_env_file_replaces_the_class_s(self, tmp_path):
        """As it does for the settings themselves: a test that isolates an app's
        dotenv with ``_env_file`` isolates its provider keys too."""
        class_file, given = tmp_path / "class.env", tmp_path / "given.env"
        class_file.write_text("LAB_KEY=k-class\n")
        given.write_text("LAB_KEY=k-given\n")
        app = _app_with_env_files(str(class_file))

        assert model_services.user_env_files(app(_env_file=str(given))) == (given,)
        assert model_services.providers_from_settings(app(_env_file=None))["lab"].api_key is None
        assert model_services.providers_from_settings(app())["lab"].api_key == "k-class"

    def test_a_malformed_line_is_not_reported_again(self, tmp_path, caplog):
        """The settings' own load reported it once, before the UI; the keys are
        read again at every validation, /settings and /models."""
        env_file = tmp_path / "app.env"
        env_file.write_text("LAB_KEY=k-from-file\nthis is not a statement\n")
        settings = _app_with_env_files(str(env_file))()

        with caplog.at_level(logging.WARNING, logger="dotenv.main"):
            caplog.clear()
            first = model_services.provider_env(settings)
            model_services.provider_env(settings)

        assert first["LAB_KEY"] == "k-from-file"
        assert [record.getMessage() for record in caplog.records] == []

    def test_an_explicit_env_overrides_every_file(self, tmp_path):
        env_file = tmp_path / "app.env"
        env_file.write_text("LAB_KEY=k-from-file\n")
        settings = _app_with_env_files(str(env_file))()

        providers = model_services.providers_from_settings(settings, env={})

        assert providers["lab"].api_key is None

    @pytest.mark.parametrize(
        "form", ["no env_file", "None", "one string", "a Path", "~", "a list"]
    )
    def test_settings_composed_from_the_mixins_read_their_class_s_env_files(
        self, tmp_path, form
    ):
        """Settings composed from the exported mixins over pydantic-settings' own
        BaseSettings have no ``env_files``: their class's ``env_file`` is read
        (its user-level entries), in each form it takes, and they validate:
        without a file to read the key from, the key is named, nothing crashes.
        (Their sources, filters included, are the mixins': test_config_trust.)"""
        env_file = tmp_path / "home" / ".app" / ".env"
        env_file.parent.mkdir()
        env_file.write_text("LAB_KEY=k-from-file\n")
        other = tmp_path / "other.env"
        config, expected = {
            "no env_file": ({}, ()),
            "None": ({"env_file": None}, ()),
            "one string": ({"env_file": str(env_file)}, (env_file,)),
            "a Path": ({"env_file": env_file}, (env_file,)),
            "~": ({"env_file": "~/.app/.env"}, (env_file,)),
            "a list": ({"env_file": [".env", str(other), str(env_file)]}, (other, env_file)),
        }[form]

        class Composed(
            WorkflowSettingsMixin, AppSettingsMixin, CLISettingsMixin, PydanticBaseSettings
        ):
            model_config = SettingsConfigDict(env_prefix="COMPOSED_", extra="ignore", **config)
            model_providers: dict[str, ProviderSpec] = Field(default_factory=lambda: {"lab": LAB})

        settings = Composed(default_model="lab:qwen3-32b")

        assert model_services.user_env_files(settings) == expected
        lab = model_services.providers_from_settings(settings)["lab"]
        if expected:
            assert lab.api_key == "k-from-file"
            validate_settings(settings)
        else:
            assert lab.api_key is None
            with pytest.raises(SettingsValidationError, match="set LAB_KEY in the environment"):
                validate_settings(settings)


def test_entries_from_settings_by_reference_in_their_order():
    settings = BaseSettings(models={
        "openrouter:qwen/qwen3-235b-a22b:free": {},
        "anthropic:claude-sonnet-4-6": {"context_window": 200_000},
    })

    entries = model_services.entries_from_settings(settings)

    assert list(entries) == [
        ModelRef("openrouter", "qwen/qwen3-235b-a22b:free"),
        ModelRef("anthropic", "claude-sonnet-4-6"),
    ]
    assert entries[ModelRef("anthropic", "claude-sonnet-4-6")].context_window == 200_000


def test_role_map_from_settings():
    settings = BaseSettings(default_model="gemini-2.5-flash", model_roles={"fast": "claude-haiku-4-5"})

    roles = model_services.role_map_from_settings(settings, manager_model="openai:gpt-5.5")

    assert isinstance(roles, RoleMap)
    assert roles.spec_for("default") == ModelRef("openai", "gpt-5.5")
    assert roles.spec_for("fast") == ModelRef("anthropic", "claude-haiku-4-5")
    assert model_services.role_map_from_settings(settings).spec_for("default") == ModelRef(
        "google", "gemini-2.5-flash"
    )


def test_a_default_model_that_names_a_role_is_an_error():
    """Settings reject it when they load; the role map rejects a manager's own."""
    with pytest.raises(ValueError, match="default_model"):
        BaseSettings(default_model="fast")
    with pytest.raises(ModelSpecError, match="manager's model"):
        model_services.role_map_from_settings(BaseSettings(), manager_model="fast")


def test_model_cache_path_is_in_the_user_folder(tmp_path):
    settings = BaseSettings(app_name="research_demo")

    assert model_services.model_cache_path(settings) == (
        Path(tmp_path / "home") / ".research_demo" / "model_cache.json"
    )


async def test_build_catalog_caches_listings_in_the_user_folder(real_catalogs):
    settings = BaseSettings(google_api_key="k-g")
    providers = model_services.providers_from_settings(settings)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": []}))

    catalog = build_catalog(settings, providers, transport=transport)
    await catalog.load()

    cache = model_services.model_cache_path(settings)
    assert json.loads(cache.read_text())["providers"]["google"]["ok"] is True
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    assert b"k-g" not in cache.read_bytes()
    assert catalog.is_authoritative("google")


def test_build_catalog_takes_the_settings_entries(real_catalogs):
    settings = BaseSettings(
        google_api_key="k-g", models={"google:gemini-2.5-flash": {"max_output": 1_000}}
    )

    catalog = build_catalog(settings, model_services.providers_from_settings(settings))

    record = catalog.record(ModelRef("google", "gemini-2.5-flash"))
    assert (record.max_output, record.sources["max_output"]) == (1_000, "config")


def test_build_catalog_takes_what_the_settings_the_agents_and_the_manager_name(
    real_catalogs, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k-r")
    settings = BaseSettings(
        google_api_key="k-g",
        anthropic_api_key="k-a",
        openai_api_key="k-o",
        model_providers={"openrouter": openrouter_spec()},
        models={"google:gemini-2.5-flash": {}},
    )

    catalog = build_catalog(
        settings,
        model_services.providers_from_settings(settings),
        agent_configs=_agents("openrouter:qwen/qwen3-235b-a22b"),
        manager_model="anthropic:claude-sonnet-4-6",
    )

    # The entry's provider, the agent's and the manager's; OpenAI is in none.
    assert catalog.in_use == ("google", "anthropic", "openrouter")
    assert catalog.offered() == [
        "google:gemini-2.5-flash", "anthropic:claude-sonnet-4-6", "openrouter:qwen/qwen3-235b-a22b",
    ]


def test_build_catalog_takes_the_roles_computed_from_the_providers(real_catalogs, monkeypatch):
    """No default model is set, so the default role is computed: from a
    built-in provider with a key, which is then in use too."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k-r")
    settings = BaseSettings(
        google_api_key="k-g",
        model_providers={"openrouter": openrouter_spec()},
        models={"openrouter:qwen/qwen3-235b-a22b": {}},
    )

    catalog = build_catalog(settings, model_services.providers_from_settings(settings))

    assert catalog.in_use == ("google", "openrouter")


def test_a_computed_role_the_list_can_fill_needs_no_other_provider(real_catalogs):
    """No default model is set, and the list has a default-tier model: the
    default role takes it, so no other provider is in use."""
    settings = BaseSettings(
        google_api_key="k-g", anthropic_api_key="k-a", models={"anthropic:claude-sonnet-4-6": {}}
    )

    catalog = build_catalog(settings, model_services.providers_from_settings(settings))

    assert catalog.in_use == ("anthropic",)


def test_role_models_names_each_role_s_model(real_catalogs):
    """The list's role models, for the pickers and /models alike: default,
    fast and strong, mapped or computed, and each mapped app role."""
    settings = BaseSettings(
        google_api_key="k-g",
        anthropic_api_key="k-a",
        model_roles={"strong": "anthropic:claude-opus-4-6", "critic": "anthropic:claude-opus-4-6"},
    )
    providers = model_services.providers_from_settings(settings)
    catalog = build_catalog(settings, providers)

    assert model_services.role_models(settings, catalog, providers) == {
        "google:gemini-3.8-flash": ["default"],
        "google:gemini-3.5-flash-lite": ["fast"],
        "anthropic:claude-opus-4-6": ["strong", "critic"],
    }
    # A manager's own model is the default role's.
    default = model_services.role_models(
        settings, catalog, providers, default="anthropic:claude-sonnet-4-6"
    )
    assert default["anthropic:claude-sonnet-4-6"] == ["default"]


def test_role_models_leave_out_a_role_nothing_can_fill():
    settings = BaseSettings()  # no key: no role is computed
    providers = model_services.providers_from_settings(settings)

    catalog = build_catalog(settings, providers)

    assert model_services.role_models(settings, catalog, providers) == {}


# --- what the catalog is told: the references and the roles computed ---------------


def test_model_references_in_order_once_each():
    settings = BaseSettings(
        default_model="gemini-2.5-flash",
        model_roles={"fast": "google:gemini-2.5-flash-lite", "critic": "anthropic:claude-sonnet-4-6"},
    )
    agents = _agents(
        "openrouter:qwen/qwen3-235b-a22b",
        "critic",  # a mapped role: its model, listed already
        "strong",  # unmapped: computed, so no reference
        "claude-opus-4-6",  # a bare id
        "qwen3-32b",  # malformed: validation reports it
        None,  # the default model's
        "google:gemini-2.5-flash",
    )

    references = model_services.model_references(
        settings, agents, manager_model="openai:gpt-5.5"
    )

    assert references == (
        ModelRef("openai", "gpt-5.5"),
        ModelRef("google", "gemini-2.5-flash"),
        ModelRef("google", "gemini-2.5-flash-lite"),
        ModelRef("anthropic", "claude-sonnet-4-6"),
        ModelRef("openrouter", "qwen/qwen3-235b-a22b"),
        ModelRef("anthropic", "claude-opus-4-6"),
    )


def test_reference_settings_say_where_each_reference_is_named():
    """For the warning that a provider does not serve a model: the setting, the
    role or the agent that names it, each that does."""
    settings = BaseSettings(
        default_model="gemini-2.5-flash",
        model_roles={"fast": "google:gemini-2.5-flash-lite", "critic": "anthropic:claude-sonnet-4-6"},
    )
    agents = [
        AgentConfig(name="writer", prompt="p", model="google:gemini-2.5-flash"),
        AgentConfig(name="critic", prompt="p", model="critic"),  # the role's, named there
        AgentConfig(name="bad", prompt="p", model="qwen3-32b"),  # malformed: left out
    ]

    named = model_services.reference_settings(settings, agents, manager_model="openai:gpt-5.5")

    assert named == {
        ModelRef("openai", "gpt-5.5"): "the manager's model",
        ModelRef("google", "gemini-2.5-flash"): "default_model and agent 'writer'",
        ModelRef("google", "gemini-2.5-flash-lite"): "model_roles['fast']",
        ModelRef("anthropic", "claude-sonnet-4-6"): "model_roles['critic']",
    }
    assert tuple(named) == model_services.model_references(
        settings, agents, manager_model="openai:gpt-5.5"
    )


def test_model_references_skip_what_does_not_parse():
    """Validation reports each of them, before any catalog is built."""
    agents = _agents("x y", "Lab:qwen3", "lab:", "")

    assert model_services.model_references(BaseSettings(), agents, manager_model="x y") == ()


@pytest.mark.parametrize(
    ("values", "manager_model", "models", "expected"),
    [
        # Nothing names the default model: it is computed.
        ({}, None, [None], ("default",)),
        ({"default_model": "google:gemini-2.5-flash"}, None, [None], ()),
        ({}, "google:gemini-2.5-flash", [None], ()),
        # A built-in role an agent names, unless model_roles maps it.
        (
            {"default_model": "google:gemini-2.5-flash", "model_roles": {"fast": "gpt-5.4-mini"}},
            None,
            ["strong", "fast", "critic", "default", "strong"],
            ("strong",),
        ),
        ({}, None, ["strong", "fast"], ("default", "fast", "strong")),
    ],
)
def test_computed_roles(values, manager_model, models, expected):
    settings = BaseSettings(**values)

    assert model_services.computed_roles(settings, _agents(*models), manager_model) == expected
