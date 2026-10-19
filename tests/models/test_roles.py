"""Roles: ``default``, ``fast``, ``strong`` and an app's own, mapped or computed."""

from __future__ import annotations

import logging

import pytest

from agentic_cli.models import (
    ModelRef,
    ModelResolutionError,
    ModelSpecError,
    ProviderKind,
    RoleMap,
)
from tests.models.fakes import FakeCatalog, endpoints, table_records

G, A, O = ProviderKind.GOOGLE, ProviderKind.ANTHROPIC, ProviderKind.OPENAI

GOOGLE = table_records(
    G, "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-pro-preview",
)
ANTHROPIC = table_records(
    A, "claude-haiku-4-5", "claude-sonnet-4-5", "claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-5-5",
)
OPENAI = table_records(O, "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-5.6", "gpt-5.5-pro", "o3")
CATALOG = FakeCatalog([*GOOGLE, *ANTHROPIC, *OPENAI])


def _roles(**kwargs) -> RoleMap:
    return RoleMap(default_model=kwargs.pop("default_model", None),
                   model_roles=kwargs.pop("model_roles", {}), **kwargs)


class TestValidation:
    def test_a_role_cannot_point_at_a_role(self):
        """Review Focus 5: a role's value is a model, never another role (R6)."""
        with pytest.raises(ModelSpecError, match=r"model_roles\['fast'\]"):
            _roles(model_roles={"fast": "strong"})
        with pytest.raises(ModelSpecError, match="default_model"):
            _roles(default_model="fast")

    def test_model_roles_cannot_map_default(self):
        with pytest.raises(ModelSpecError, match=r"model_roles\['default'\].*default_model"):
            _roles(model_roles={"default": "google:gemini-2.5-flash"})

    @pytest.mark.parametrize("role", ["Fast", "gpt5", "o3", "with space"])
    def test_a_malformed_role_name_is_an_error(self, role):
        with pytest.raises(ModelSpecError, match="model_roles"):
            _roles(model_roles={role: "google:gemini-2.5-flash"})

    def test_a_malformed_reference_is_an_error_naming_the_key(self):
        with pytest.raises(ModelSpecError, match=r"model_roles\['strong'\]"):
            _roles(model_roles={"strong": "anthropic:"})

    def test_the_manager_model_cannot_be_a_role(self):
        with pytest.raises(ModelSpecError, match="manager"):
            _roles(manager_model="strong")


class TestMappedRoles:
    def test_spec_for(self):
        roles = _roles(
            default_model="claude-sonnet-4-5",
            model_roles={"fast": "google:gemini-2.5-flash-lite", "vision": "openai:gpt-5.5"},
        )

        assert roles.spec_for("default") == ModelRef("anthropic", "claude-sonnet-4-5")
        assert roles.spec_for("fast") == ModelRef("google", "gemini-2.5-flash-lite")
        assert roles.spec_for("vision") == ModelRef("openai", "gpt-5.5")
        assert roles.spec_for("strong") is None
        assert roles.spec_for("review") is None
        assert _roles().spec_for("default") is None

    def test_the_manager_model_wins_for_default(self):
        roles = _roles(default_model="google:gemini-2.5-flash", manager_model="gpt-5.5")

        assert roles.spec_for("default") == ModelRef("openai", "gpt-5.5")
        assert roles.resolve("default", catalog=CATALOG, endpoints=endpoints()) == ModelRef("openai", "gpt-5.5")

    def test_default_is_default_model_when_set(self):
        roles = _roles(default_model="anthropic:claude-opus-5-5")

        assert roles.resolve("default", catalog=CATALOG, endpoints=endpoints(google="k")) == ModelRef(
            "anthropic", "claude-opus-5-5"
        )

    def test_a_role_mapped_to_openrouter(self):
        roles = _roles(model_roles={"strong": "openrouter:qwen/qwen3-235b-a22b:free"})

        assert roles.resolve("strong", catalog=CATALOG, endpoints=endpoints(google="k")) == ModelRef(
            "openrouter", "qwen/qwen3-235b-a22b:free"
        )

    def test_referenced_lists_every_reference_once(self):
        roles = _roles(
            default_model="gemini-2.5-flash",
            manager_model="anthropic:claude-sonnet-4-5",
            model_roles={"fast": "google:gemini-2.5-flash", "strong": "openrouter:x/y:free"},
        )

        assert roles.referenced() == (
            "anthropic:claude-sonnet-4-5", "google:gemini-2.5-flash", "openrouter:x/y:free",
        )


class TestComputedRoles:
    @pytest.mark.parametrize(
        ("keys", "role", "expected"),
        [
            ({"google": "k"}, "default", "google:gemini-3.5-flash"),
            ({"google": "k"}, "fast", "google:gemini-3.5-flash-lite"),
            ({"google": "k"}, "strong", "google:gemini-3.1-pro-preview"),
            ({"anthropic": "k"}, "default", "anthropic:claude-sonnet-4-6"),
            ({"anthropic": "k"}, "fast", "anthropic:claude-haiku-4-5"),
            ({"anthropic": "k"}, "strong", "anthropic:claude-opus-5-5"),
            ({"openai": "k"}, "default", "openai:gpt-5.6"),
            ({"openai": "k"}, "fast", "openai:gpt-5.4-mini"),
            ({"openai": "k"}, "strong", "openai:gpt-5.5-pro"),
        ],
    )
    def test_tiers_per_provider(self, keys, role, expected):
        resolved = _roles().resolve(role, catalog=CATALOG, endpoints=endpoints(**keys))

        assert str(resolved) == expected

    @pytest.mark.parametrize(
        ("keys", "provider"),
        [
            ({"google": "k", "anthropic": "k", "openai": "k"}, "google"),
            ({"anthropic": "k", "openai": "k"}, "anthropic"),
            ({"openai": "k", "openrouter": "k"}, "openai"),
        ],
    )
    def test_the_provider_order_is_google_anthropic_openai(self, keys, provider):
        resolved = _roles().resolve("default", catalog=CATALOG, endpoints=endpoints(**keys))

        assert resolved.provider == provider

    def test_a_provider_without_the_tier_is_skipped(self):
        catalog = FakeCatalog([*table_records(G, "gemini-2.5-flash"), *ANTHROPIC])

        resolved = _roles().resolve(
            "strong", catalog=catalog, endpoints=endpoints(google="k", anthropic="k")
        )

        assert resolved == ModelRef("anthropic", "claude-opus-5-5")

    def test_the_newest_wins_and_a_preview_loses_a_tie(self):
        catalog = FakeCatalog(
            table_records(G, "gemini-2.5-flash", "gemini-3.5-flash-preview", "gemini-3.5-flash")
        )

        resolved = _roles().resolve("default", catalog=catalog, endpoints=endpoints(google="k"))

        assert resolved == ModelRef("google", "gemini-3.5-flash")

    def test_a_tie_on_version_is_settled_by_the_id(self):
        """gpt-5.4-mini and gpt-5.4-nano are both fast and both 5.4."""
        catalog = FakeCatalog(table_records(O, "gpt-5.4-nano", "gpt-5.4-mini"))

        resolved = _roles().resolve("fast", catalog=catalog, endpoints=endpoints(openai="k"))

        assert resolved == ModelRef("openai", "gpt-5.4-mini")

    def test_openrouter_and_endpoints_never_fill_a_role_on_their_own(self):
        catalog = FakeCatalog(
            [*table_records(ProviderKind.OPENROUTER, "qwen/qwen3-235b-a22b")]
        )

        with pytest.raises(ModelResolutionError) as exc:
            _roles().resolve("default", catalog=catalog, endpoints=endpoints(openrouter="k"))

        assert "default" in str(exc.value)
        assert "default_model" in str(exc.value)

    def test_no_key_is_an_error_naming_the_role_and_the_fix(self):
        with pytest.raises(ModelResolutionError) as exc:
            _roles().resolve("fast", catalog=CATALOG, endpoints=endpoints())

        message = str(exc.value)
        assert "'fast'" in message
        assert "model_roles" in message
        assert "GOOGLE_API_KEY" in message


class TestAppRoles:
    def test_a_mapped_app_role(self):
        roles = _roles(model_roles={"vision": "google:gemini-2.5-pro"})

        assert roles.resolve("vision", catalog=CATALOG, endpoints=endpoints()) == ModelRef(
            "google", "gemini-2.5-pro"
        )

    def test_an_unmapped_app_role_falls_back_to_default_with_one_warning(self, caplog):
        roles = _roles(default_model="anthropic:claude-sonnet-4-5")

        with caplog.at_level(logging.WARNING, logger="agentic_cli.models.roles"):
            first = roles.resolve("vision", catalog=CATALOG, endpoints=endpoints())
            second = roles.resolve("vision", catalog=CATALOG, endpoints=endpoints())

        assert first == second == ModelRef("anthropic", "claude-sonnet-4-5")
        warnings = [r for r in caplog.records if r.name == "agentic_cli.models.roles"]
        assert len(warnings) == 1
        assert "vision" in warnings[0].getMessage()
        assert "model_roles" in warnings[0].getMessage()

    def test_an_invalid_role_name_is_an_error(self):
        with pytest.raises(ModelSpecError):
            _roles().resolve("Vision", catalog=CATALOG, endpoints=endpoints())
