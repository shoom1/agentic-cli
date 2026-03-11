"""Tests for ModelRegistry."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_cli.workflow.models import ModelFamily, ModelInfo, ModelRegistry


class TestModelFamily:
    """Tests for ModelFamily enum."""

    def test_values(self):
        assert ModelFamily.GEMINI == "gemini"
        assert ModelFamily.CLAUDE == "claude"
        assert ModelFamily.GPT == "gpt"


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_defaults(self):
        info = ModelInfo(id="test-model", family=ModelFamily.GEMINI)
        assert info.display_name == "test-model"
        assert not info.supports_thinking
        assert not info.deprecated

    def test_custom_display_name(self):
        info = ModelInfo(id="test", family=ModelFamily.CLAUDE, display_name="Test Model")
        assert info.display_name == "Test Model"


class TestGetFamily:
    """Tests for family detection."""

    def test_gemini_prefix(self):
        reg = ModelRegistry()
        assert reg.get_family("gemini-2.5-pro") == ModelFamily.GEMINI
        assert reg.get_family("gemini-3-flash-preview") == ModelFamily.GEMINI

    def test_claude_prefix(self):
        reg = ModelRegistry()
        assert reg.get_family("claude-sonnet-4-5") == ModelFamily.CLAUDE
        assert reg.get_family("claude-opus-4") == ModelFamily.CLAUDE

    def test_gpt_prefix(self):
        reg = ModelRegistry()
        assert reg.get_family("gpt-4o") == ModelFamily.GPT

    def test_unknown_prefix(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="Cannot determine"):
            reg.get_family("llama-3")

    def test_cached_model(self):
        reg = ModelRegistry()
        reg._models["custom-model"] = ModelInfo(
            id="custom-model", family=ModelFamily.CLAUDE
        )
        assert reg.get_family("custom-model") == ModelFamily.CLAUDE


class TestSupportsThinking:
    """Tests for thinking support detection."""

    def test_claude_always_supports(self):
        reg = ModelRegistry()
        assert reg.supports_thinking("claude-sonnet-4-5")
        assert reg.supports_thinking("claude-opus-4")

    def test_gemini_25_supports(self):
        reg = ModelRegistry()
        assert reg.supports_thinking("gemini-2.5-pro")
        assert reg.supports_thinking("gemini-2.5-flash")

    def test_gemini_3_supports(self):
        reg = ModelRegistry()
        assert reg.supports_thinking("gemini-3-flash-preview")

    def test_cached_model_info(self):
        reg = ModelRegistry()
        reg._models["test-model"] = ModelInfo(
            id="test-model", family=ModelFamily.GEMINI, supports_thinking=True
        )
        assert reg.supports_thinking("test-model")

    def test_unknown_model(self):
        reg = ModelRegistry()
        assert not reg.supports_thinking("unknown-model")


class TestFallbackBehavior:
    """Tests for pre-refresh fallback behavior."""

    def test_not_refreshed_initially(self):
        reg = ModelRegistry()
        assert not reg.is_refreshed

    def test_all_fallback_models(self):
        reg = ModelRegistry()
        models = reg.get_available_models()
        for fb in ModelRegistry.FALLBACK_GOOGLE + ModelRegistry.FALLBACK_ANTHROPIC:
            assert fb in models

    def test_fallback_by_family(self):
        reg = ModelRegistry()
        google = reg.get_available_models(ModelFamily.GEMINI)
        assert all("gemini" in m for m in google)
        assert len(google) == len(ModelRegistry.FALLBACK_GOOGLE)

        claude = reg.get_available_models(ModelFamily.CLAUDE)
        assert all("claude" in m for m in claude)
        assert len(claude) == len(ModelRegistry.FALLBACK_ANTHROPIC)

    def test_fallback_by_key(self):
        reg = ModelRegistry()
        models = reg.get_available_models(google_key=True)
        assert "gemini-2.5-pro" in models
        assert "claude-sonnet-4-5" not in models

        models = reg.get_available_models(anthropic_key=True)
        assert "claude-sonnet-4-5" in models
        assert "gemini-2.5-pro" not in models

    def test_fallback_both_keys(self):
        reg = ModelRegistry()
        models = reg.get_available_models(google_key=True, anthropic_key=True)
        assert "gemini-2.5-pro" in models
        assert "claude-sonnet-4-5" in models

    def test_default_models(self):
        reg = ModelRegistry()
        assert reg.get_default(ModelFamily.GEMINI) == "gemini-2.5-flash"
        assert reg.get_default(ModelFamily.CLAUDE) == "claude-sonnet-4-6"

    def test_set_default(self):
        reg = ModelRegistry()
        reg.set_default(ModelFamily.GEMINI, "gemini-2.5-pro")
        assert reg.get_default(ModelFamily.GEMINI) == "gemini-2.5-pro"


class TestResolveModel:
    """Tests for model resolution."""

    def test_resolve_before_refresh_accepts_any(self):
        reg = ModelRegistry()
        assert reg.resolve_model("any-model") == "any-model"

    def test_resolve_exact_match(self):
        reg = ModelRegistry()
        reg._models["gemini-2.5-pro"] = ModelInfo(
            id="gemini-2.5-pro", family=ModelFamily.GEMINI
        )
        reg._refreshed = True
        assert reg.resolve_model("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_resolve_deprecated_finds_replacement(self):
        reg = ModelRegistry()
        reg._models["gemini-2.0-pro"] = ModelInfo(
            id="gemini-2.0-pro", family=ModelFamily.GEMINI, deprecated=True
        )
        reg._models["gemini-2.5-pro"] = ModelInfo(
            id="gemini-2.5-pro", family=ModelFamily.GEMINI
        )
        reg._refreshed = True

        resolved = reg.resolve_model("gemini-2.0-pro")
        assert resolved == "gemini-2.5-pro"

    def test_resolve_missing_finds_closest(self):
        reg = ModelRegistry()
        reg._models["gemini-2.5-flash"] = ModelInfo(
            id="gemini-2.5-flash", family=ModelFamily.GEMINI
        )
        reg._refreshed = True

        # Missing pro model → falls back to flash (only available)
        resolved = reg.resolve_model("gemini-3-pro-preview")
        assert resolved == "gemini-2.5-flash"

    def test_resolve_missing_unknown_family_raises(self):
        reg = ModelRegistry()
        reg._refreshed = True
        with pytest.raises(ValueError, match="cannot be determined"):
            reg.resolve_model("llama-3")

    def test_resolve_missing_no_candidates_raises(self):
        reg = ModelRegistry()
        reg._models["claude-sonnet-4-5"] = ModelInfo(
            id="claude-sonnet-4-5", family=ModelFamily.CLAUDE
        )
        reg._refreshed = True

        with pytest.raises(ValueError, match="not available"):
            reg.resolve_model("gemini-2.5-pro")

    def test_resolve_same_tier_prefers_newest(self):
        reg = ModelRegistry()
        reg._models["gemini-2.0-pro"] = ModelInfo(
            id="gemini-2.0-pro", family=ModelFamily.GEMINI
        )
        reg._models["gemini-2.5-pro"] = ModelInfo(
            id="gemini-2.5-pro", family=ModelFamily.GEMINI
        )
        reg._models["gemini-3-pro-preview"] = ModelInfo(
            id="gemini-3-pro-preview", family=ModelFamily.GEMINI
        )
        reg._refreshed = True

        # Deprecated pro → newest pro
        reg._models["gemini-1.5-pro"] = ModelInfo(
            id="gemini-1.5-pro", family=ModelFamily.GEMINI, deprecated=True
        )
        resolved = reg.resolve_model("gemini-1.5-pro")
        assert resolved == "gemini-3-pro-preview"


class TestNormalization:
    """Tests for model ID normalization."""

    def test_normalize_anthropic_dated(self):
        assert ModelRegistry._normalize_anthropic_id(
            "claude-sonnet-4-5-20250514"
        ) == "claude-sonnet-4-5"

    def test_normalize_anthropic_old_format(self):
        assert ModelRegistry._normalize_anthropic_id(
            "claude-3-5-sonnet-20241022"
        ) == "claude-3-5-sonnet"

    def test_normalize_anthropic_no_date(self):
        assert ModelRegistry._normalize_anthropic_id(
            "claude-sonnet-4-5"
        ) == "claude-sonnet-4-5"

    def test_normalize_at_date(self):
        assert ModelRegistry._normalize_anthropic_id(
            "claude-sonnet-4-5@2025-05-14"
        ) == "claude-sonnet-4-5"


class TestTierExtraction:
    """Tests for tier extraction helper."""

    def test_pro_tier(self):
        assert ModelRegistry._extract_tier("gemini-2.5-pro") == "pro"

    def test_flash_tier(self):
        assert ModelRegistry._extract_tier("gemini-2.5-flash") == "flash"

    def test_sonnet_tier(self):
        assert ModelRegistry._extract_tier("claude-sonnet-4-5") == "sonnet"

    def test_opus_tier(self):
        assert ModelRegistry._extract_tier("claude-opus-4") == "opus"

    def test_haiku_tier(self):
        assert ModelRegistry._extract_tier("claude-haiku-3-5") == "haiku"

    def test_no_tier(self):
        assert ModelRegistry._extract_tier("some-model") == ""


class TestVersionExtraction:
    """Tests for version extraction helper."""

    def test_gemini_version(self):
        assert ModelRegistry._extract_version("gemini-2.5-pro") == (2, 5)

    def test_claude_version(self):
        assert ModelRegistry._extract_version("claude-sonnet-4-5") == (4, 5)

    def test_no_version(self):
        assert ModelRegistry._extract_version("model") == (0,)


class TestRefreshWithMocks:
    """Tests for async refresh with mocked API clients."""

    @pytest.mark.asyncio
    async def test_refresh_google_success(self):
        """Test Google model fetching with mocked client."""
        reg = ModelRegistry()

        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.5-pro"
        mock_model.display_name = "Gemini 2.5 Pro"
        mock_model.supported_actions = ["generateContent"]
        mock_model.thinking = True

        mock_client = MagicMock()
        mock_client.models.list.return_value = [mock_model]

        with patch("agentic_cli.workflow.models.genai", create=True) as mock_genai:
            mock_genai.Client.return_value = mock_client
            # Patch the import inside the method
            with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
                with patch("agentic_cli.workflow.models.ModelRegistry._fetch_google_models") as mock_fetch:
                    mock_fetch.return_value = [
                        ModelInfo(
                            id="gemini-2.5-pro",
                            family=ModelFamily.GEMINI,
                            display_name="Gemini 2.5 Pro",
                            supports_thinking=True,
                        )
                    ]
                    await reg.refresh(google_api_key="test-key")

        assert reg.is_refreshed
        assert "gemini-2.5-pro" in reg._models

    @pytest.mark.asyncio
    async def test_refresh_anthropic_success(self):
        """Test Anthropic model fetching with mocked client."""
        reg = ModelRegistry()

        with patch.object(reg, "_fetch_anthropic_models") as mock_fetch:
            mock_fetch.return_value = [
                ModelInfo(
                    id="claude-sonnet-4-5",
                    family=ModelFamily.CLAUDE,
                    display_name="Claude Sonnet 4.5",
                    supports_thinking=True,
                ),
                ModelInfo(
                    id="claude-opus-4",
                    family=ModelFamily.CLAUDE,
                    display_name="Claude Opus 4",
                    supports_thinking=True,
                ),
            ]
            await reg.refresh(anthropic_api_key="test-key")

        assert reg.is_refreshed
        assert "claude-sonnet-4-5" in reg._models
        assert "claude-opus-4" in reg._models

    @pytest.mark.asyncio
    async def test_refresh_failure_stays_fallback(self):
        """Test that failed refresh keeps fallback mode."""
        reg = ModelRegistry()

        # No keys → no fetching → stays in fallback mode
        await reg.refresh()
        assert not reg.is_refreshed

    @pytest.mark.asyncio
    async def test_refresh_both_providers(self):
        """Test fetching from both providers."""
        reg = ModelRegistry()

        with patch.object(reg, "_fetch_google_models") as mock_google, \
             patch.object(reg, "_fetch_anthropic_models") as mock_anthropic:
            mock_google.return_value = [
                ModelInfo(id="gemini-2.5-flash", family=ModelFamily.GEMINI),
            ]
            mock_anthropic.return_value = [
                ModelInfo(id="claude-sonnet-4-5", family=ModelFamily.CLAUDE, supports_thinking=True),
            ]
            await reg.refresh(
                google_api_key="google-key",
                anthropic_api_key="anthropic-key",
            )

        assert reg.is_refreshed
        assert "gemini-2.5-flash" in reg._models
        assert "claude-sonnet-4-5" in reg._models

    @pytest.mark.asyncio
    async def test_refresh_updates_available_models(self):
        """Test that after refresh, get_available_models returns fetched models."""
        reg = ModelRegistry()

        with patch.object(reg, "_fetch_google_models") as mock_fetch:
            mock_fetch.return_value = [
                ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI),
                ModelInfo(id="gemini-2.5-flash", family=ModelFamily.GEMINI),
                ModelInfo(id="gemini-3-pro", family=ModelFamily.GEMINI),
            ]
            await reg.refresh(google_api_key="test-key")

        models = reg.get_available_models(google_key=True)
        assert "gemini-2.5-pro" in models
        assert "gemini-2.5-flash" in models
        assert "gemini-3-pro" in models

    @pytest.mark.asyncio
    async def test_auto_select_defaults(self):
        """Test that defaults are auto-selected after refresh."""
        reg = ModelRegistry()

        with patch.object(reg, "_fetch_google_models") as mock_google, \
             patch.object(reg, "_fetch_anthropic_models") as mock_anthropic:
            mock_google.return_value = [
                ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI),
                ModelInfo(id="gemini-2.5-flash", family=ModelFamily.GEMINI),
            ]
            mock_anthropic.return_value = [
                ModelInfo(id="claude-sonnet-4-5", family=ModelFamily.CLAUDE, supports_thinking=True),
                ModelInfo(id="claude-opus-4", family=ModelFamily.CLAUDE, supports_thinking=True),
            ]
            await reg.refresh(
                google_api_key="g-key",
                anthropic_api_key="a-key",
            )

        # Flash is preferred for Gemini, sonnet for Claude
        assert reg.get_default(ModelFamily.GEMINI) == "gemini-2.5-flash"
        assert reg.get_default(ModelFamily.CLAUDE) == "claude-sonnet-4-5"

    @pytest.mark.asyncio
    async def test_deprecated_models_excluded(self):
        """Test that deprecated models are excluded from available list."""
        reg = ModelRegistry()

        with patch.object(reg, "_fetch_google_models") as mock_fetch:
            mock_fetch.return_value = [
                ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI),
                ModelInfo(id="gemini-1.5-pro", family=ModelFamily.GEMINI, deprecated=True),
            ]
            await reg.refresh(google_api_key="test-key")

        models = reg.get_available_models(google_key=True)
        assert "gemini-2.5-pro" in models
        assert "gemini-1.5-pro" not in models


class TestFilterByKeys:
    """Tests for key-based model filtering after refresh."""

    @pytest.mark.asyncio
    async def test_filter_google_only(self):
        reg = ModelRegistry()
        with patch.object(reg, "_fetch_google_models") as mg, \
             patch.object(reg, "_fetch_anthropic_models") as ma:
            mg.return_value = [ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI)]
            ma.return_value = [ModelInfo(id="claude-sonnet-4-5", family=ModelFamily.CLAUDE)]
            await reg.refresh(google_api_key="g", anthropic_api_key="a")

        # Only google key
        models = reg.get_available_models(google_key=True)
        assert "gemini-2.5-pro" in models
        assert "claude-sonnet-4-5" not in models

    @pytest.mark.asyncio
    async def test_filter_no_keys_returns_all(self):
        reg = ModelRegistry()
        with patch.object(reg, "_fetch_google_models") as mg, \
             patch.object(reg, "_fetch_anthropic_models") as ma:
            mg.return_value = [ModelInfo(id="gemini-2.5-pro", family=ModelFamily.GEMINI)]
            ma.return_value = [ModelInfo(id="claude-sonnet-4-5", family=ModelFamily.CLAUDE)]
            await reg.refresh(google_api_key="g", anthropic_api_key="a")

        # No key filter → all models
        models = reg.get_available_models()
        assert "gemini-2.5-pro" in models
        assert "claude-sonnet-4-5" in models


class TestGoogleExcludePatterns:
    """Tests for Google model filtering patterns."""

    def test_excludes_point_releases(self):
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-2.0-flash-001")
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-2.0-flash-002")

    def test_excludes_latest_aliases(self):
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-flash-latest")
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-pro-latest")

    def test_excludes_dated_snapshots(self):
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-2.5-flash-lite-preview-09-2025")

    def test_excludes_customtools(self):
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-3.1-pro-preview-customtools")

    def test_excludes_experimental(self):
        assert ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search("gemini-2.0-flash-exp-image-generation")

    def test_keeps_main_models(self):
        for model_id in [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
        ]:
            assert not ModelRegistry._GOOGLE_EXCLUDE_PATTERNS.search(model_id), \
                f"{model_id} should NOT be excluded"


class TestThinkingEffortLevels:
    """Tests for THINKING_EFFORT_LEVELS constant."""

    def test_levels_present(self):
        assert "none" in ModelRegistry.THINKING_EFFORT_LEVELS
        assert "low" in ModelRegistry.THINKING_EFFORT_LEVELS
        assert "medium" in ModelRegistry.THINKING_EFFORT_LEVELS
        assert "high" in ModelRegistry.THINKING_EFFORT_LEVELS
