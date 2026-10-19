"""The model settings and the trust model.

``model_endpoints`` holds base URLs, so it is user-only: a project
``./.{app}/settings.json`` could otherwise send the user's key to its own
server. ``model_roles`` and ``pinned_models`` only choose among providers and
endpoints the user defined, so a project may set them. The two new provider
keys are secrets like the others: never persisted, never in ``repr``, and (R5)
never exported to the process environment, since the adapters pass every key
explicitly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import structlog
from pydantic import ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.models import EndpointSpec
from agentic_cli.settings_persistence import SettingsPersistence
from agentic_cli.workflow.model_settings import ThinkingSettings

APP = "agentic_cli"
ENDPOINTS = {
    "local-vllm": {
        "kind": "openai_compatible",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "LOCAL_VLLM_KEY",
        "models": {"Qwen/Qwen3-32B": {"context_window": 32768, "thinking": "chat_template"}},
    }
}
_KEY_VARS = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A temp home and project, and no keys or model settings from the environment."""
    (tmp_path / "home").mkdir()
    (tmp_path / "project").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "project")
    for var in (*_KEY_VARS, "AGENTIC_MODEL_ENDPOINTS", "AGENTIC_MODEL_ROLES", "AGENTIC_PINNED_MODELS"):
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


class TestModelEndpointsAreUserOnly:
    def test_a_project_file_cannot_define_an_endpoint(self, isolated):
        _write(_project_file(isolated), {"model_endpoints": ENDPOINTS, "default_model": "kept"})

        with structlog.testing.capture_logs() as logs:
            settings = BaseSettings()

        assert settings.model_endpoints == {}
        assert settings.default_model == "kept"
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
        assert endpoints["local-vllm"].models["Qwen/Qwen3-32B"].context_window == 32768

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


class TestRolesAndPinsAreProjectSettable:
    def test_they_load_from_the_project_file(self, isolated):
        _write(
            _project_file(isolated),
            {"model_roles": {"fast": "google:gemini-3.5-flash"}, "pinned_models": ["openrouter:x/y:free"]},
        )

        settings = BaseSettings()

        assert settings.model_roles == {"fast": "google:gemini-3.5-flash"}
        assert settings.pinned_models == ["openrouter:x/y:free"]

    def test_they_save_to_the_project_file(self, isolated):
        settings = BaseSettings(
            model_roles={"strong": "local-vllm:Qwen/Qwen3-32B"}, pinned_models=["gpt-5.5"]
        )

        SettingsPersistence(APP).save(settings)

        project = _read(_project_file(isolated))
        assert project["model_roles"] == {"strong": "local-vllm:Qwen/Qwen3-32B"}
        assert project["pinned_models"] == ["gpt-5.5"]
        user = _read(_user_file(isolated))
        assert "model_roles" not in user and "pinned_models" not in user

    def test_a_keyed_save_writes_them_to_the_project_file(self, isolated):
        settings = BaseSettings(model_roles={"fast": "claude-haiku-4-5"})

        result = SettingsPersistence(APP).save(settings, keys=["model_roles"])

        assert result.project_keys == ("model_roles",)
        assert _read(_project_file(isolated))["model_roles"] == {"fast": "claude-haiku-4-5"}


class TestValidation:
    def test_a_role_cannot_map_a_role(self):
        with pytest.raises(ValidationError, match=r"model_roles\['fast'\]"):
            BaseSettings(model_roles={"fast": "strong"})

    def test_model_roles_cannot_map_default(self):
        with pytest.raises(ValidationError, match="default_model"):
            BaseSettings(model_roles={"default": "gemini-2.5-flash"})

    def test_a_pinned_model_is_a_reference(self):
        with pytest.raises(ValidationError, match=r"pinned_models\[1\]"):
            BaseSettings(pinned_models=["gpt-5.5", "fast"])

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
