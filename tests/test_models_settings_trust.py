"""The model settings and the trust model.

``model_endpoints`` holds base URLs, so it is user-only: a project
``./.{app}/settings.json`` could otherwise send the user's key to its own
server. ``model_roles`` and ``models`` only choose among providers and
endpoints the user defined (an entry's numbers are hints), so a project may
set them. ``models`` is the app's list (its settings class default) with the
user's, the project's and the environment's entries merged in, field by
field, and no save writes it: each file keeps its own entries as they are. The
two new provider keys are secrets like the others: never persisted, never in
``repr``, and (R5) never exported to the process environment, since the
adapters pass every key explicitly.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest
import structlog
from pydantic import Field, ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.models import EndpointSpec, ModelEntry
from agentic_cli.settings_persistence import SettingsPersistence
from agentic_cli.workflow.model_settings import ThinkingSettings

APP = "agentic_cli"
ENDPOINTS = {
    "local-vllm": {
        "kind": "openai_compatible",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "LOCAL_VLLM_KEY",
    }
}
_KEY_VARS = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
#: The two entries ``_app_settings()``'s class ships, as an app's settings class does.
SHIPPED = {
    "anthropic:claude-sonnet-4-6": ModelEntry(context_window=1_000_000, max_output=64_000),
    "local-vllm:Qwen/Qwen3-32B": {"thinking": "chat_template"},
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A temp home and project, and no keys or model settings from the environment."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for var in (*_KEY_VARS, "AGENTIC_MODEL_ENDPOINTS", "AGENTIC_MODEL_ROLES", "AGENTIC_MODELS"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _project_file(root: Path) -> Path:
    return root / "project" / f".{APP}" / "settings.json"


def _user_file(root: Path) -> Path:
    return root / "home" / f".{APP}" / "settings.json"


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _app_settings() -> type[BaseSettings]:
    """A settings class that ships ``SHIPPED``: one entry an object, one a dict."""

    class AppSettings(BaseSettings):
        models: dict[str, ModelEntry] = Field(default_factory=lambda: dict(SHIPPED))

    return AppSettings


def _set(models: dict[str, ModelEntry]) -> dict[str, dict]:
    """Each entry's fields that some source set."""
    return {key: entry.model_dump(exclude_unset=True) for key, entry in models.items()}


class TestModelEndpointsAreUserOnly:
    def test_a_project_file_cannot_define_an_endpoint(self, isolated):
        _write(
            _project_file(isolated),
            {"model_endpoints": ENDPOINTS, "default_model": "gemini-2.5-flash"},
        )

        with structlog.testing.capture_logs() as logs:
            settings = BaseSettings()

        assert settings.model_endpoints == {}
        assert settings.default_model == "gemini-2.5-flash"
        assert any(
            e.get("event") == "untrusted_project_setting_ignored" and e.get("key") == "model_endpoints"
            for e in logs
        )

    def test_a_cwd_dotenv_cannot_define_an_endpoint(self, isolated):
        (isolated / "project" / ".env").write_text(f"AGENTIC_MODEL_ENDPOINTS={json.dumps(ENDPOINTS)}\n")

        class _Settings(BaseSettings):
            model_config = {**BaseSettings.model_config, "env_file": ".env"}

        assert _Settings().model_endpoints == {}

    def test_the_user_file_defines_endpoints(self, isolated):
        _write(_user_file(isolated), {"model_endpoints": ENDPOINTS})

        endpoints = BaseSettings().model_endpoints

        assert isinstance(endpoints["local-vllm"], EndpointSpec)
        assert endpoints["local-vllm"].base_url == "http://localhost:8000/v1"

    def test_a_save_never_writes_endpoints_to_the_project_file(self, isolated):
        settings = BaseSettings(model_endpoints=ENDPOINTS)

        SettingsPersistence(APP).save(settings)

        assert "model_endpoints" not in _read(_project_file(isolated))
        saved = _read(_user_file(isolated))["model_endpoints"]
        assert saved["local-vllm"]["base_url"] == "http://localhost:8000/v1"
        assert BaseSettings().model_endpoints["local-vllm"].api_key_env == "LOCAL_VLLM_KEY"

    def test_a_keyed_save_never_writes_endpoints_to_the_project_file(self, isolated):
        settings = BaseSettings(model_endpoints=ENDPOINTS)

        result = SettingsPersistence(APP).save(settings, keys=["model_endpoints"])

        assert result.project_keys == ()
        assert "model_endpoints" not in _read(_project_file(isolated))
        assert "local-vllm" in _read(_user_file(isolated))["model_endpoints"]


class TestRolesAndModelsAreProjectSettable:
    def test_they_load_from_the_project_file(self, isolated):
        _write(
            _project_file(isolated),
            {
                "model_roles": {"fast": "google:gemini-3.5-flash"},
                "models": {"openrouter:x/y:free": {"context_window": 40960}},
            },
        )

        settings = BaseSettings()

        assert settings.model_roles == {"fast": "google:gemini-3.5-flash"}
        assert _set(settings.models) == {"openrouter:x/y:free": {"context_window": 40960}}

    def test_roles_save_to_the_project_file(self, isolated):
        settings = BaseSettings(model_roles={"strong": "local-vllm:Qwen/Qwen3-32B"})

        SettingsPersistence(APP).save(settings)

        project = _read(_project_file(isolated))
        assert project["model_roles"] == {"strong": "local-vllm:Qwen/Qwen3-32B"}
        assert "model_roles" not in _read(_user_file(isolated))

    def test_a_keyed_save_writes_them_to_the_project_file(self, isolated):
        settings = BaseSettings(model_roles={"fast": "claude-haiku-4-5"})

        result = SettingsPersistence(APP).save(settings, keys=["model_roles"])

        assert result.project_keys == ("model_roles",)
        assert _read(_project_file(isolated))["model_roles"] == {"fast": "claude-haiku-4-5"}


class TestTheModelsList:
    """The app's entries, then the user file's, the project file's and the
    environment's, merged field by field (pydantic-settings alone would let
    any source's value replace the class default)."""

    def test_the_app_s_entries_then_the_user_s_project_s_and_environment_s(
        self, isolated, monkeypatch
    ):
        _write(_user_file(isolated), {"models": {
            "anthropic:claude-sonnet-4-6": {"max_output": 32_000},
            "google:gemini-2.5-flash": {},
        }})
        _write(_project_file(isolated), {"models": {
            "local-vllm:Qwen/Qwen3-32B": {"context_window": 32_768},
            "openai:gpt-5.5": {},
        }})
        monkeypatch.setenv("AGENTIC_MODELS", json.dumps({"openrouter:qwen/qwen3-235b-a22b": {}}))

        settings = _app_settings()()

        assert list(settings.models) == [
            "anthropic:claude-sonnet-4-6",
            "local-vllm:Qwen/Qwen3-32B",
            "google:gemini-2.5-flash",
            "openai:gpt-5.5",
            "openrouter:qwen/qwen3-235b-a22b",
        ]
        assert _set(settings.models)["anthropic:claude-sonnet-4-6"] == {
            "context_window": 1_000_000, "max_output": 32_000,  # the app's window, the user's limit
        }
        assert _set(settings.models)["local-vllm:Qwen/Qwen3-32B"] == {
            "thinking": "chat_template", "context_window": 32_768,
        }

    def test_without_an_app_list_the_files_merge_field_by_field(self, isolated):
        _write(_user_file(isolated), {"models": {"x:m": {"context_window": 1_000, "tools": True}}})
        _write(_project_file(isolated), {"models": {"x:m": {"max_output": 100, "tools": False}}})

        settings = BaseSettings()

        assert _set(settings.models) == {
            "x:m": {"context_window": 1_000, "max_output": 100, "tools": False},
        }

    def test_the_app_s_entries_validate_as_objects_or_dicts(self):
        settings = _app_settings()()

        assert _set(settings.models) == {
            "anthropic:claude-sonnet-4-6": {"context_window": 1_000_000, "max_output": 64_000},
            "local-vllm:Qwen/Qwen3-32B": {"thinking": "chat_template"},
        }
        assert all(isinstance(entry, ModelEntry) for entry in settings.models.values())

    def test_no_source_removes_an_app_entry(self, isolated):
        _write(_user_file(isolated), {"models": {}})

        assert list(_app_settings()().models) == list(SHIPPED)

    def test_an_assignment_through_pydantic_leaves_the_entries_as_they_are(self):
        """On an assignment pydantic hands the model validator the whole
        object, and takes what it returns for the other fields unvalidated:
        merging there would turn every entry into a dict."""
        settings = _app_settings()()

        settings.update_setting("orchestrator", "deepagents")  # validate_assignment

        assert list(settings.models) == list(SHIPPED)
        assert all(isinstance(entry, ModelEntry) for entry in settings.models.values())

    def test_an_app_that_overrides_the_defaults_is_warned_of_nothing(self):
        """The mixin declares the fields in Annotated form, so it holds no
        class attribute for an app's override to shadow."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class AppSettings(BaseSettings):
                models: dict[str, ModelEntry] = Field(default_factory=lambda: dict(SHIPPED))
                model_roles: dict[str, str] = Field(
                    default_factory=lambda: {"fast": "google:gemini-2.5-flash"}
                )

        assert [str(warning.message) for warning in caught] == []
        settings = AppSettings()
        assert list(settings.models) == list(SHIPPED)
        assert settings.model_roles == {"fast": "google:gemini-2.5-flash"}

    def test_an_assigned_list_is_validated_and_set_as_given(self):
        settings = _app_settings()()

        settings.update_setting("models", {"google:gemini-2.5-flash": {"max_output": 1_000}})

        assert _set(settings.models) == {"google:gemini-2.5-flash": {"max_output": 1_000}}
        assert isinstance(settings.models["google:gemini-2.5-flash"], ModelEntry)
        with pytest.raises(ValueError, match="write it as anthropic:claude-sonnet-4-5"):
            settings.update_setting("models", {"claude-sonnet-4-5": {}})


class TestModelsAreHandEdited:
    """No save writes ``models``: it would copy the app's entries, and the
    other files', into the file it writes."""

    def test_a_full_save_leaves_each_file_s_models_as_it_is(self, isolated):
        user_models = {"google:gemini-2.5-flash": {"max_output": 1_000}}
        project_models = {"openai:gpt-5.5": {}}
        _write(_user_file(isolated), {"models": user_models})
        _write(_project_file(isolated), {"models": project_models})
        settings = _app_settings()(model_roles={"fast": "google:gemini-2.5-flash"})

        SettingsPersistence(APP).save(settings)

        project = _read(_project_file(isolated))
        assert project["models"] == project_models
        assert project["model_roles"] == {"fast": "google:gemini-2.5-flash"}
        assert _read(_user_file(isolated))["models"] == user_models

    def test_a_full_save_adds_models_to_no_file(self, isolated):
        SettingsPersistence(APP).save(_app_settings()())

        assert "models" not in _read(_project_file(isolated))
        assert "models" not in _read(_user_file(isolated))

    def test_a_keyed_save_never_writes_models(self, isolated):
        settings = _app_settings()(model_roles={"fast": "google:gemini-2.5-flash"})

        result = SettingsPersistence(APP).save(
            settings, keys=["models", "models.anthropic:claude-sonnet-4-6", "model_roles.fast"]
        )

        assert result.project_keys == ("model_roles",)
        assert _read(_project_file(isolated)) == {
            "model_roles": {"fast": "google:gemini-2.5-flash"}
        }
        assert "models" not in _read(_user_file(isolated))

    def test_a_keyed_save_keeps_the_project_s_own_models(self, isolated):
        _write(_project_file(isolated), {"models": {"openai:gpt-5.5": {}}})
        settings = BaseSettings(model_roles={"fast": "google:gemini-2.5-flash"})

        SettingsPersistence(APP).save(settings, keys=["model_roles.fast"])

        assert _read(_project_file(isolated)) == {
            "models": {"openai:gpt-5.5": {}},
            "model_roles": {"fast": "google:gemini-2.5-flash"},
        }

    def test_an_explicit_path_dump_leaves_models_out(self, isolated):
        legacy = isolated / "legacy.json"

        SettingsPersistence(APP).save(_app_settings()(), path=legacy)

        assert "models" not in json.loads(legacy.read_text())


class TestValidation:
    def test_a_role_cannot_map_a_role(self):
        with pytest.raises(ValidationError, match=r"model_roles\['fast'\]"):
            BaseSettings(model_roles={"fast": "strong"})

    def test_model_roles_cannot_map_default(self):
        with pytest.raises(ValidationError, match="default_model"):
            BaseSettings(model_roles={"default": "gemini-2.5-flash"})

    def test_a_models_key_is_a_full_reference(self):
        with pytest.raises(ValidationError, match="write it as anthropic:claude-sonnet-4-5"):
            BaseSettings(models={"google:gemini-2.5-flash": {}, "claude-sonnet-4-5": {}})

    def test_a_builtin_model_s_entry_cannot_set_its_thinking(self, isolated):
        _write(
            _project_file(isolated),
            {"models": {"anthropic:claude-sonnet-4-5": {"thinking": "effort"}}},
        )

        with pytest.raises(ValidationError, match="only for a model on a model_providers provider"):
            BaseSettings()

    def test_an_endpoint_cannot_reuse_a_builtin_name(self):
        with pytest.raises(ValidationError, match="built-in"):
            BaseSettings(model_endpoints={"openai": {"base_url": "http://x:1/v1"}})

    def test_update_setting_validates_the_roles(self):
        settings = BaseSettings()

        with pytest.raises(ValueError, match="model_roles"):
            settings.update_setting("model_roles", {"fast": "fast"})
        settings.update_setting("model_roles", {"fast": "google:gemini-3.5-flash-lite"})

        assert settings.model_roles == {"fast": "google:gemini-3.5-flash-lite"}

    def test_an_endpoint_error_never_echoes_a_secret(self):
        """A key written inline, or a password in the URL, stays out of the error."""
        for entry in (
            {"base_url": "http://x:1/v1", "api_key": "sk-inline-secret"},
            {"base_url": "http://user:sk-inline-secret@x:1/v1"},
        ):
            with pytest.raises(ValidationError) as exc:
                BaseSettings(model_endpoints={"lab": entry})
            assert "sk-inline-secret" not in str(exc.value)


class TestTheNewKeys:
    def test_they_bind_by_field_name_and_by_environment_variable(self, monkeypatch):
        assert BaseSettings(openai_api_key="k-o", openrouter_api_key="k-r").openai_api_key == "k-o"
        monkeypatch.setenv("OPENROUTER_API_KEY", "k-env")
        assert BaseSettings().openrouter_api_key == "k-env"

    def test_they_never_reach_a_persisted_file(self, isolated):
        settings = BaseSettings(
            openai_api_key="k-secret-openai",
            openrouter_api_key="k-secret-openrouter",
            model_endpoints=ENDPOINTS,
            model_roles={"fast": "gpt-5.4-mini"},
        )
        persistence = SettingsPersistence(APP)

        persistence.save(settings)
        persistence.save(settings, keys=["openai_api_key", "openrouter_api_key", "model_roles"])
        legacy = isolated / "legacy.json"
        persistence.save(settings, path=legacy)

        for path in (_project_file(isolated), _user_file(isolated), legacy):
            text = path.read_text()
            assert "k-secret" not in text, path
            assert "openai_api_key" not in text and "openrouter_api_key" not in text, path

    def test_they_never_appear_in_repr(self):
        settings = BaseSettings(openai_api_key="k-secret-openai", openrouter_api_key="k-secret-openrouter")

        assert "k-secret" not in repr(settings)
        assert "k-secret" not in str(settings)

    def test_a_project_file_cannot_inject_them(self, isolated):
        _write(
            _project_file(isolated),
            {"openai_api_key": "from-repo", "OPENAI_API_KEY": "from-repo", "openrouter_api_key": "from-repo"},
        )

        settings = BaseSettings()

        assert settings.openai_api_key is None
        assert settings.openrouter_api_key is None

    def test_they_are_not_exported_to_the_environment(self, monkeypatch):
        """R5: the adapters pass these keys explicitly; nothing reads them from the env."""
        settings = BaseSettings(
            google_api_key="k-google", openai_api_key="k-openai", openrouter_api_key="k-openrouter"
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "")  # restored by monkeypatch after the export

        settings.export_api_keys_to_env()

        assert os.environ.get("GOOGLE_API_KEY") == "k-google"
        assert "OPENAI_API_KEY" not in os.environ
        assert "OPENROUTER_API_KEY" not in os.environ


class TestThinkingModes:
    @pytest.mark.parametrize("mode", ["none", "low", "medium", "high", "budget", "xhigh", "max", "minimal"])
    def test_a_lowercase_name_is_accepted(self, mode):
        assert ThinkingSettings(mode=mode).mode == mode

    @pytest.mark.parametrize("mode", ["High!", "HIGH", "x high", "", "1low", "-low"])
    def test_anything_else_is_rejected(self, mode):
        with pytest.raises(ValidationError):
            ThinkingSettings(mode=mode)

    def test_the_default_is_none(self):
        assert ThinkingSettings().mode == "none"
