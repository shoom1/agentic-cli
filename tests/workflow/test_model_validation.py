"""Model/provider validation: ordered, complete, and off the event loop.

Three defects are covered:

1. ``validate_settings`` ran *before* the registry refresh, so a model that
   exists but predates the static fallback list was rejected at startup.
2. Per-agent ``AgentConfig.model`` overrides were never checked, so an
   override pointing at a provider with no credential failed mid-run.
3. ``ModelRegistry._fetch_*_models`` are async but called blocking provider
   SDKs, stalling the CLI's event loop during startup.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from agentic_cli.config import SettingsValidationError, validate_settings
from agentic_cli.workflow.base_manager import BaseWorkflowManager
from agentic_cli.workflow.config import AgentConfig
from agentic_cli.workflow.models import ModelFamily, ModelInfo, ModelRegistry
from tests.conftest import MockContext

# A well-formed Gemini id that is deliberately absent from FALLBACK_GOOGLE.
DISCOVERED = "gemini-9.9-flash-preview"


class _TestManager(BaseWorkflowManager):
    """Minimal concrete manager; backend init is a no-op."""

    def _get_state_tools(self):
        return []

    @property
    def backend_type(self) -> str:
        return "test"

    async def _do_initialize(self) -> None:
        return None

    async def process(self, message, user_id, session_id=None):
        raise NotImplementedError

    async def reinitialize(self, model=None, preserve_sessions=True):
        return None

    async def cleanup(self):
        return None


def _discovering_registry(*models: ModelInfo) -> ModelRegistry:
    """A registry whose provider fetches return fixed models."""
    registry = ModelRegistry()

    async def _google(api_key):
        return [m for m in models if m.family is ModelFamily.GEMINI]

    async def _anthropic(api_key):
        return [m for m in models if m.family is ModelFamily.CLAUDE]

    registry._fetch_google_models = _google
    registry._fetch_anthropic_models = _anthropic
    return registry


class TestDiscoveryBeforeValidation:
    async def test_dynamically_discovered_model_is_accepted(self):
        """A model absent from the static list but present in the API listing."""
        with MockContext(google_api_key="k", default_model=DISCOVERED) as ctx:
            assert DISCOVERED not in ModelRegistry.FALLBACK_GOOGLE
            mgr = _TestManager(agent_configs=[], settings=ctx.settings)
            mgr._model_registry = _discovering_registry(
                ModelInfo(id=DISCOVERED, family=ModelFamily.GEMINI)
            )

            await mgr.initialize_services()

            assert mgr.is_initialized

    async def test_refreshed_registry_still_rejects_unknown_model(self):
        """Once discovery succeeded, its list is authoritative."""
        with MockContext(google_api_key="k", default_model="gemini-not-real") as ctx:
            mgr = _TestManager(agent_configs=[], settings=ctx.settings)
            mgr._model_registry = _discovering_registry(
                ModelInfo(id=DISCOVERED, family=ModelFamily.GEMINI)
            )

            with pytest.raises(SettingsValidationError, match="not available"):
                await mgr.initialize_services()


class TestPerAgentOverrideValidation:
    def test_override_without_provider_credential_fails(self):
        """A Claude override with only a Google key must fail loudly."""
        with MockContext(google_api_key="k") as ctx:
            configs = [
                AgentConfig(name="root", prompt="p"),
                AgentConfig(name="claude_helper", prompt="p", model="claude-sonnet-4-6"),
            ]
            with pytest.raises(SettingsValidationError) as exc:
                validate_settings(ctx.settings, agent_configs=configs)

        message = str(exc.value)
        assert "claude_helper" in message
        assert "ANTHROPIC_API_KEY" in message
        # Errors must name the missing credential, never its value.
        assert "k" != message and "api_key=" not in message

    def test_override_with_credential_passes(self):
        with MockContext(google_api_key="k", anthropic_api_key="a") as ctx:
            configs = [
                AgentConfig(name="root", prompt="p"),
                AgentConfig(name="helper", prompt="p", model="claude-sonnet-4-6"),
            ]
            validate_settings(ctx.settings, agent_configs=configs)

    def test_unknown_provider_in_override_fails(self):
        with MockContext(google_api_key="k") as ctx:
            configs = [AgentConfig(name="odd", prompt="p", model="mystery-model-1")]
            with pytest.raises(SettingsValidationError, match="provider cannot be determined"):
                validate_settings(ctx.settings, agent_configs=configs)


class TestDiscoveryUnavailable:
    """Offline behaviour stays deterministic: no discovery, no false rejections."""

    def test_unrecognised_but_well_formed_model_is_not_rejected(self):
        with MockContext(google_api_key="k", default_model=DISCOVERED) as ctx:
            assert ctx.settings._get_registry().is_refreshed is False
            validate_settings(ctx.settings)  # warns, does not raise

    def test_missing_credential_still_fails_offline(self):
        with MockContext(google_api_key="k", default_model="claude-sonnet-4-6") as ctx:
            with pytest.raises(SettingsValidationError, match="ANTHROPIC_API_KEY"):
                validate_settings(ctx.settings)

    async def test_failed_discovery_does_not_make_fallbacks_authoritative(
        self, monkeypatch
    ):
        """A provider outage must not turn the stale fallback list into truth."""
        def _boom(api_key=None):
            raise RuntimeError("provider down")

        monkeypatch.setattr("google.genai.Client", _boom)

        with MockContext(google_api_key="k", default_model=DISCOVERED) as ctx:
            registry = ModelRegistry()
            mgr = _TestManager(agent_configs=[], settings=ctx.settings)
            mgr._model_registry = registry

            await mgr.initialize_services()

            assert mgr.is_initialized
            # Fallbacks were substituted, so the list is not authoritative.
            assert registry.is_refreshed is True
            assert registry.discovery_complete is False


class TestBlockingFetchOffEventLoop:
    """Provider listings are blocking SDK calls; they must not run on the loop."""

    async def test_google_listing_runs_on_worker_thread(self, monkeypatch):
        calling_thread: list[threading.Thread] = []

        class _FakeModels:
            def list(self):
                calling_thread.append(threading.current_thread())
                return []

        class _FakeClient:
            def __init__(self, api_key=None):
                self.models = _FakeModels()

        monkeypatch.setattr("google.genai.Client", _FakeClient)

        registry = ModelRegistry()
        await registry._fetch_google_models("key")

        assert calling_thread, "the SDK listing was never called"
        assert calling_thread[0] is not threading.main_thread()

    async def test_anthropic_listing_runs_on_worker_thread(self, monkeypatch):
        calling_thread: list[threading.Thread] = []

        class _FakeModels:
            def list(self, limit=None):
                calling_thread.append(threading.current_thread())
                return MagicMock(data=[])

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.models = _FakeModels()

        monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)

        registry = ModelRegistry()
        await registry._fetch_anthropic_models("key")

        assert calling_thread, "the SDK listing was never called"
        assert calling_thread[0] is not threading.main_thread()

    async def test_event_loop_stays_responsive_during_refresh(self, monkeypatch):
        """A slow provider listing must not stall other coroutines."""
        release = threading.Event()
        ticks = 0

        class _FakeModels:
            def list(self):
                release.wait(timeout=5)
                return []

        class _FakeClient:
            def __init__(self, api_key=None):
                self.models = _FakeModels()

        monkeypatch.setattr("google.genai.Client", _FakeClient)

        registry = ModelRegistry()
        fetch = asyncio.create_task(registry._fetch_google_models("key"))
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks += 1
        release.set()
        await fetch

        assert ticks == 3  # the loop kept running while the SDK call blocked


class TestPerProviderDiscoveryAuthority:
    """Providers fail independently; authority is tracked per family."""

    def _registry(self, *, google_ok: bool, anthropic_ok: bool) -> ModelRegistry:
        registry = ModelRegistry()

        async def _google(api_key):
            if google_ok:
                return [ModelInfo(id=DISCOVERED, family=ModelFamily.GEMINI)]
            registry._degraded_families.add(ModelFamily.GEMINI)
            return [
                ModelInfo(id=m, family=ModelFamily.GEMINI)
                for m in ModelRegistry.FALLBACK_GOOGLE
            ]

        async def _anthropic(api_key):
            if anthropic_ok:
                return [ModelInfo(id="claude-real-9", family=ModelFamily.CLAUDE)]
            registry._degraded_families.add(ModelFamily.CLAUDE)
            return [
                ModelInfo(id=m, family=ModelFamily.CLAUDE)
                for m in ModelRegistry.FALLBACK_ANTHROPIC
            ]

        registry._fetch_google_models = _google
        registry._fetch_anthropic_models = _anthropic
        return registry

    async def test_google_success_anthropic_failure_states(self):
        from agentic_cli.workflow.models import DiscoveryState

        registry = self._registry(google_ok=True, anthropic_ok=False)
        await registry.refresh(google_api_key="g", anthropic_api_key="a")

        assert registry.authority_for(ModelFamily.GEMINI) is DiscoveryState.SUCCEEDED
        assert registry.authority_for(ModelFamily.CLAUDE) is DiscoveryState.DEGRADED
        assert registry.is_authoritative_for(DISCOVERED) is True
        assert registry.is_authoritative_for("claude-sonnet-4-6") is False

    async def test_google_success_rejects_unknown_gemini_despite_anthropic_outage(self):
        registry = self._registry(google_ok=True, anthropic_ok=False)
        await registry.refresh(google_api_key="g", anthropic_api_key="a")

        with pytest.raises(ValueError, match="not available"):
            registry.resolve_model("gemini-does-not-exist")

    async def test_anthropic_outage_does_not_make_its_fallbacks_authoritative(self):
        registry = self._registry(google_ok=True, anthropic_ok=False)
        await registry.refresh(google_api_key="g", anthropic_api_key="a")

        # Not in the (fallback) Claude list, but the listing failed: accept it.
        assert registry.resolve_model("claude-brand-new-1") == "claude-brand-new-1"

    async def test_unattempted_provider_is_never_authoritative(self):
        from agentic_cli.workflow.models import DiscoveryState

        registry = self._registry(google_ok=True, anthropic_ok=True)
        await registry.refresh(google_api_key="g")  # no Anthropic key

        assert registry.authority_for(ModelFamily.CLAUDE) is DiscoveryState.UNATTEMPTED
        assert registry.resolve_model("claude-anything-1") == "claude-anything-1"

    async def test_empty_listing_is_treated_as_degraded(self):
        from agentic_cli.workflow.models import DiscoveryState

        registry = ModelRegistry()

        async def _empty(api_key):
            return []

        registry._fetch_google_models = _empty
        await registry.refresh(google_api_key="g")

        assert registry.authority_for(ModelFamily.GEMINI) is DiscoveryState.DEGRADED
        assert registry.resolve_model("gemini-whatever") == "gemini-whatever"


class TestSetterValidatorConsistency:
    """set_model() and validate_settings() must never disagree."""

    def _settings_with(self, ctx, registry):
        ctx.settings.set_model_registry(registry)
        return ctx.settings

    async def test_setter_accepts_what_the_validator_accepts(self):
        with MockContext(google_api_key="k") as ctx:
            registry = _discovering_registry(
                ModelInfo(id=DISCOVERED, family=ModelFamily.GEMINI)
            )
            await registry.refresh(google_api_key="k")
            settings = self._settings_with(ctx, registry)

            settings.set_model(DISCOVERED)
            assert settings.default_model == DISCOVERED
            validate_settings(settings)  # must not raise

    async def test_setter_rejects_what_the_validator_rejects(self):
        with MockContext(google_api_key="k") as ctx:
            registry = _discovering_registry(
                ModelInfo(id=DISCOVERED, family=ModelFamily.GEMINI)
            )
            await registry.refresh(google_api_key="k")
            settings = self._settings_with(ctx, registry)

            with pytest.raises(ValueError, match="not available"):
                settings.set_model("gemini-nope")

            object.__setattr__(settings, "default_model", "gemini-nope")
            with pytest.raises(SettingsValidationError, match="not available"):
                validate_settings(settings)

    def test_setter_rejects_a_model_without_its_credential(self):
        with MockContext(google_api_key="k") as ctx:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ctx.settings.set_model("claude-sonnet-4-6")

    def test_setter_accepts_an_unknown_model_when_not_authoritative(self):
        """Offline, the static list cannot disprove a well-formed model id."""
        with MockContext(google_api_key="k") as ctx:
            ctx.settings.set_model(DISCOVERED)
            assert ctx.settings.default_model == DISCOVERED
            validate_settings(ctx.settings)

    async def test_deprecated_alias_is_upgraded_by_both(self):
        with MockContext(google_api_key="k") as ctx:
            registry = ModelRegistry()
            registry._models = {
                "gemini-old": ModelInfo(
                    id="gemini-old", family=ModelFamily.GEMINI, deprecated=True
                ),
                "gemini-2.5-pro": ModelInfo(
                    id="gemini-2.5-pro", family=ModelFamily.GEMINI
                ),
            }
            registry._refreshed = True
            settings = self._settings_with(ctx, registry)

            settings.set_model("gemini-old")
            assert settings.default_model == "gemini-2.5-pro"
            validate_settings(settings)


class TestDeprecatedAliasesApplyAtRuntime:
    """A deprecated alias must be *replaced*, not merely warned about.

    ``check_model()`` returns the live replacement, but only ``set_model()``
    was writing it back. A ``default_model`` loaded from settings.json/env, and
    every ``AgentConfig.model`` override, kept the dead id and was then sent to
    the provider. None of these tests call ``set_model()``.
    """

    @staticmethod
    def _deprecating_registry() -> ModelRegistry:
        registry = ModelRegistry()
        registry._models = {
            "gemini-old": ModelInfo(
                id="gemini-old", family=ModelFamily.GEMINI, deprecated=True
            ),
            "gemini-2.5-pro": ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI),
        }
        registry._refreshed = True
        return registry

    def test_configured_default_is_replaced(self):
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            settings.set_model_registry(self._deprecating_registry())
            # As if loaded from settings.json / GOOGLE_MODEL, not via set_model().
            object.__setattr__(settings, "default_model", "gemini-old")

            validate_settings(settings)

            assert settings.default_model == "gemini-2.5-pro"
            assert settings.get_model() == "gemini-2.5-pro"

    def test_agent_override_is_replaced(self):
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            settings.set_model_registry(self._deprecating_registry())
            config = AgentConfig(name="a", prompt="p", model="gemini-old")

            validate_settings(settings, agent_configs=[config])

            assert config.model == "gemini-2.5-pro"

    def test_live_model_is_left_alone(self):
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            settings.set_model_registry(self._deprecating_registry())
            object.__setattr__(settings, "default_model", "gemini-2.5-pro")
            config = AgentConfig(name="a", prompt="p", model="gemini-2.5-pro")

            validate_settings(settings, agent_configs=[config])

            assert settings.default_model == "gemini-2.5-pro"
            assert config.model == "gemini-2.5-pro"

    async def test_manager_runs_on_the_replacement(self):
        """End to end: what the backend actually uses after initialization."""
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            object.__setattr__(settings, "default_model", "gemini-old")
            config = AgentConfig(name="a", prompt="p", model="gemini-old")

            manager = _TestManager(agent_configs=[config], settings=settings)
            manager._model_registry = self._deprecating_registry()
            # refresh() is a no-op for a directly-seeded registry.
            manager._model_registry.refresh = lambda **kw: asyncio.sleep(0)

            await manager.initialize_services()

            assert manager.model == "gemini-2.5-pro"
            assert config.model == "gemini-2.5-pro"


class TestManagerModelIsValidated:
    """Every model the *runtime* will actually send must be validated.

    ``validate_settings`` covered ``settings.default_model`` and the per-agent
    overrides, but a manager's own model — ``GoogleADKWorkflowManager(model=...)``,
    ``reinitialize(model=...)``, or one cached from an earlier
    ``settings.get_model()`` — bypassed it entirely: an unusable id reached the
    provider, and a deprecated one was never swapped for its replacement.
    """

    @staticmethod
    def _deprecating_registry() -> ModelRegistry:
        registry = ModelRegistry()
        registry._models = {
            "gemini-old": ModelInfo(
                id="gemini-old", family=ModelFamily.GEMINI, deprecated=True
            ),
            "gemini-2.5-pro": ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI),
        }
        registry._refreshed = True
        return registry

    def _manager(self, settings, cls=_TestManager, **kwargs):
        manager = cls(agent_configs=[], settings=settings, **kwargs)
        manager._model_registry = self._deprecating_registry()
        manager._model_registry.refresh = lambda **kw: asyncio.sleep(0)
        return manager

    async def test_explicit_constructor_model_is_normalized(self):
        with MockContext(google_api_key="k") as ctx:
            manager = self._manager(ctx.settings, model="gemini-old")
            await manager.initialize_services()
            assert manager.model == "gemini-2.5-pro"

    async def test_cached_model_is_normalized(self):
        """A model resolved before discovery may since have been deprecated."""
        with MockContext(google_api_key="k") as ctx:
            manager = self._manager(ctx.settings)
            manager._model = "gemini-old"
            manager._model_resolved = True

            await manager.initialize_services()

            assert manager.model == "gemini-2.5-pro"


    async def test_explicit_model_without_a_credential_fails(self):
        with MockContext(google_api_key="k") as ctx:
            manager = self._manager(ctx.settings, model="claude-sonnet-4-6")
            with pytest.raises(SettingsValidationError, match="ANTHROPIC_API_KEY"):
                await manager.initialize_services()

    async def test_unknown_explicit_model_is_rejected(self):
        with MockContext(google_api_key="k") as ctx:
            manager = self._manager(ctx.settings, model="gemini-nope")
            with pytest.raises(SettingsValidationError, match="not available"):
                await manager.initialize_services()


class TestValidateSettingsReturnContract:
    """``validate_settings()`` is a checker: it returns None, always.

    Extra-model resolutions are an internal need of the workflow manager and
    must not change what the public function hands back.
    """

    def test_returns_none(self):
        with MockContext(google_api_key="k") as ctx:
            assert validate_settings(ctx.settings) is None

    def test_returns_none_with_agent_configs(self):
        with MockContext(google_api_key="k") as ctx:
            config = AgentConfig(name="a", prompt="p", model="gemini-2.5-flash")
            assert validate_settings(ctx.settings, agent_configs=[config]) is None

    def test_public_signature_takes_no_extra_models(self):
        import inspect

        params = inspect.signature(validate_settings).parameters
        assert list(params) == ["settings", "agent_configs"]


class TestRewritesAreAllOrNothing:
    """A deprecated-alias rewrite must not land when validation later fails.

    Rewrites were applied as each model was checked, so a bad agent override
    left ``settings.default_model`` already mutated by a validation that raised
    — the next attempt then validated a different configuration than the user
    wrote.
    """

    @staticmethod
    def _registry() -> ModelRegistry:
        registry = ModelRegistry()
        registry._models = {
            "gemini-old": ModelInfo(
                id="gemini-old", family=ModelFamily.GEMINI, deprecated=True
            ),
            "gemini-2.5-pro": ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI),
        }
        registry._refreshed = True
        return registry

    def test_default_model_is_untouched_when_an_override_fails(self):
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            settings.set_model_registry(self._registry())
            object.__setattr__(settings, "default_model", "gemini-old")
            bad = AgentConfig(name="a", prompt="p", model="claude-sonnet-4-6")

            with pytest.raises(SettingsValidationError):
                validate_settings(settings, agent_configs=[bad])

            assert settings.default_model == "gemini-old", (
                "a rewrite was applied by a validation that failed"
            )

    def test_agent_override_is_untouched_when_the_default_fails(self):
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            settings.set_model_registry(self._registry())
            object.__setattr__(settings, "default_model", "claude-sonnet-4-6")
            good = AgentConfig(name="a", prompt="p", model="gemini-old")

            with pytest.raises(SettingsValidationError):
                validate_settings(settings, agent_configs=[good])

            assert good.model == "gemini-old"

    def test_all_rewrites_land_when_everything_validates(self):
        with MockContext(google_api_key="k") as ctx:
            settings = ctx.settings
            settings.set_model_registry(self._registry())
            object.__setattr__(settings, "default_model", "gemini-old")
            config = AgentConfig(name="a", prompt="p", model="gemini-old")

            validate_settings(settings, agent_configs=[config])

            assert settings.default_model == "gemini-2.5-pro"
            assert config.model == "gemini-2.5-pro"


class TestProviderClientRelease:
    """Listing clients are one-shot; their connection pools must be released."""

    async def test_google_client_is_closed(self, monkeypatch):
        closed: list[bool] = []

        class _FakeClient:
            def __init__(self, api_key=None):
                self.models = MagicMock(list=lambda: [])

            def close(self):
                closed.append(True)

        monkeypatch.setattr("google.genai.Client", _FakeClient)
        await ModelRegistry()._fetch_google_models("k")
        assert closed == [True]

    async def test_anthropic_client_is_closed(self, monkeypatch):
        closed: list[bool] = []

        class _FakeAnthropic:
            def __init__(self, api_key=None):
                self.models = MagicMock(list=lambda limit=None: MagicMock(data=[]))

            def close(self):
                closed.append(True)

        monkeypatch.setattr("anthropic.Anthropic", _FakeAnthropic)
        await ModelRegistry()._fetch_anthropic_models("k")
        assert closed == [True]

    async def test_client_is_closed_even_when_the_listing_fails(self, monkeypatch):
        closed: list[bool] = []

        class _FakeClient:
            def __init__(self, api_key=None):
                self.models = MagicMock(
                    list=MagicMock(side_effect=RuntimeError("provider down"))
                )

            def close(self):
                closed.append(True)

        monkeypatch.setattr("google.genai.Client", _FakeClient)
        await ModelRegistry()._fetch_google_models("k")
        assert closed == [True]
