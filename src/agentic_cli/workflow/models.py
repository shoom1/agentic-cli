"""Dynamic model registry with API-based discovery.

Replaces static model lists with a registry that fetches available models
from Google and Anthropic APIs, selects per-family defaults, and handles
model deprecation gracefully.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """Model provider families."""

    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT = "gpt"


@dataclass
class ModelInfo:
    """Metadata for a single model."""

    id: str
    family: ModelFamily
    display_name: str = ""
    supports_thinking: bool = False
    deprecated: bool = False

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.id


class ModelRegistry:
    """Single source of truth for model discovery and selection.

    Works in two modes:
    - Before refresh(): uses hardcoded fallback lists
    - After refresh(): uses models fetched from provider APIs

    All sync methods (get_available_models, get_family, etc.) work
    without refresh(), enabling pre-init usage in settings.
    """

    # Hardcoded fallbacks (used when API fetch fails or before refresh)
    FALLBACK_GOOGLE = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    FALLBACK_ANTHROPIC = [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
    ]
    FALLBACK_DEFAULTS = {
        ModelFamily.GEMINI: "gemini-2.5-flash",
        ModelFamily.CLAUDE: "claude-sonnet-4-6",
    }

    # Thinking effort levels
    THINKING_EFFORT_LEVELS = ["none", "low", "medium", "high"]

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}
        self._defaults: dict[ModelFamily, str] = dict(self.FALLBACK_DEFAULTS)
        self._refreshed = False

    @property
    def is_refreshed(self) -> bool:
        """Whether the registry has been populated from APIs."""
        return self._refreshed

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def get_available_models(
        self,
        family: ModelFamily | None = None,
        *,
        google_key: bool = False,
        anthropic_key: bool = False,
    ) -> list[str]:
        """Return model IDs, optionally filtered by family and available keys.

        Args:
            family: Filter to a specific model family.
            google_key: Whether a Google API key is available.
            anthropic_key: Whether an Anthropic API key is available.

        Returns:
            List of model ID strings.
        """
        if self._refreshed:
            models = list(self._models.values())
            if family:
                models = [m for m in models if m.family == family]
            else:
                # Filter by available keys
                models = self._filter_by_keys(models, google_key, anthropic_key)
            return [m.id for m in models if not m.deprecated]

        # Fallback mode
        result: list[str] = []
        if family == ModelFamily.GEMINI or (family is None and google_key):
            result.extend(self.FALLBACK_GOOGLE)
        if family == ModelFamily.CLAUDE or (family is None and anthropic_key):
            result.extend(self.FALLBACK_ANTHROPIC)
        # If no keys specified and no family, return all fallbacks
        if family is None and not google_key and not anthropic_key:
            result = list(self.FALLBACK_GOOGLE) + list(self.FALLBACK_ANTHROPIC)
        return result

    def _filter_by_keys(
        self,
        models: list[ModelInfo],
        google_key: bool,
        anthropic_key: bool,
    ) -> list[ModelInfo]:
        """Filter models to those with available API keys."""
        if not google_key and not anthropic_key:
            return models
        result = []
        for m in models:
            if m.family == ModelFamily.GEMINI and google_key:
                result.append(m)
            elif m.family == ModelFamily.CLAUDE and anthropic_key:
                result.append(m)
            elif m.family == ModelFamily.GPT:
                result.append(m)
        return result

    def get_default(self, family: ModelFamily) -> str:
        """Get the default model for a family."""
        return self._defaults.get(family, self.FALLBACK_DEFAULTS.get(family, ""))

    def set_default(self, family: ModelFamily, model_id: str) -> None:
        """Set the default model for a family."""
        self._defaults[family] = model_id

    def get_family(self, model_id: str) -> ModelFamily:
        """Detect model family from model ID.

        Args:
            model_id: Model identifier string.

        Returns:
            ModelFamily enum value.

        Raises:
            ValueError: If family cannot be determined.
        """
        # Check cached models first
        if model_id in self._models:
            return self._models[model_id].family

        # Prefix-based detection
        if model_id.startswith("gemini"):
            return ModelFamily.GEMINI
        if model_id.startswith("claude"):
            return ModelFamily.CLAUDE
        if model_id.startswith("gpt"):
            return ModelFamily.GPT

        raise ValueError(f"Cannot determine model family for '{model_id}'")

    def supports_thinking(self, model_id: str) -> bool:
        """Check if a model supports thinking/reasoning configuration.

        Args:
            model_id: Model identifier string.

        Returns:
            True if the model supports thinking effort.
        """
        # Check cached info
        if model_id in self._models:
            return self._models[model_id].supports_thinking

        # Heuristic fallback
        try:
            family = self.get_family(model_id)
        except ValueError:
            return False

        if family == ModelFamily.CLAUDE:
            return True
        if family == ModelFamily.GEMINI:
            return "2.5" in model_id or "3" in model_id
        return False

    def resolve_model(self, model_id: str) -> str:
        """Validate a model ID against the registry.

        If the model exists, returns it as-is. If deprecated or missing,
        attempts to find the closest match in the same family and tier.

        Args:
            model_id: Model identifier to resolve.

        Returns:
            Resolved model ID (may differ from input if deprecated).

        Raises:
            ValueError: If no suitable model can be found.
        """
        # Exact match
        if model_id in self._models:
            info = self._models[model_id]
            if not info.deprecated:
                return model_id
            # Deprecated — find replacement
            replacement = self._find_closest_match(model_id, info.family)
            if replacement:
                logger.warning(
                    "Model '%s' is deprecated, using '%s' instead",
                    model_id,
                    replacement,
                )
                return replacement

        # Not in registry — if not refreshed, accept anything
        if not self._refreshed:
            return model_id

        # Try to find closest match
        try:
            family = self.get_family(model_id)
        except ValueError:
            raise ValueError(
                f"Model '{model_id}' is not available and its family cannot be determined."
            )

        replacement = self._find_closest_match(model_id, family)
        if replacement:
            logger.warning(
                "Model '%s' is not available, using '%s' instead",
                model_id,
                replacement,
            )
            return replacement

        available = self.get_available_models(family)
        raise ValueError(
            f"Model '{model_id}' is not available. "
            f"Available {family.value} models: {', '.join(available) if available else 'none'}"
        )

    def _find_closest_match(
        self, model_id: str, family: ModelFamily
    ) -> str | None:
        """Find the closest available model in the same family and tier.

        Tier matching: pro→pro, flash→flash, opus→opus, sonnet→sonnet, haiku→haiku.
        Prefers the newest version within the same tier.

        Args:
            model_id: Original model ID.
            family: Model family to search within.

        Returns:
            Closest matching model ID, or None.
        """
        tier = self._extract_tier(model_id)
        candidates = [
            m for m in self._models.values()
            if m.family == family and not m.deprecated
        ]

        if not candidates:
            return None

        # Prefer same tier
        same_tier = [c for c in candidates if self._extract_tier(c.id) == tier]
        if same_tier:
            # Return the one with the highest version number
            return max(same_tier, key=lambda m: self._extract_version(m.id)).id

        # Fall back to default for this family
        default = self._defaults.get(family)
        if default and default in self._models and not self._models[default].deprecated:
            return default

        # Return any available model in the family
        return candidates[0].id

    @staticmethod
    def _extract_tier(model_id: str) -> str:
        """Extract tier from model ID (e.g., 'pro', 'flash', 'sonnet')."""
        for tier in ("pro", "flash", "opus", "sonnet", "haiku"):
            if tier in model_id:
                return tier
        return ""

    @staticmethod
    def _extract_version(model_id: str) -> tuple[int, ...]:
        """Extract version tuple for sorting (e.g., 'gemini-2.5-pro' → (2, 5))."""
        numbers = re.findall(r"(\d+)", model_id)
        return tuple(int(n) for n in numbers) if numbers else (0,)

    # ------------------------------------------------------------------
    # Async API fetch
    # ------------------------------------------------------------------

    async def refresh(
        self,
        google_api_key: str | None = None,
        anthropic_api_key: str | None = None,
    ) -> None:
        """Fetch available models from provider APIs.

        Falls back to hardcoded lists on failure, logging a warning.

        Args:
            google_api_key: Google API key for listing Gemini models.
            anthropic_api_key: Anthropic API key for listing Claude models.
        """
        models: dict[str, ModelInfo] = {}

        if google_api_key:
            google_models = await self._fetch_google_models(google_api_key)
            for m in google_models:
                models[m.id] = m

        if anthropic_api_key:
            anthropic_models = await self._fetch_anthropic_models(anthropic_api_key)
            for m in anthropic_models:
                models[m.id] = m

        # If we got models from at least one provider, use them
        if models:
            self._models = models
            self._refreshed = True
            # Auto-select defaults from fetched models
            self._auto_select_defaults()
            logger.info(
                "Model registry refreshed: %d models from %d families",
                len(models),
                len({m.family for m in models.values()}),
            )
        else:
            # No models fetched — stay in fallback mode
            logger.warning(
                "No models fetched from APIs, using fallback lists"
            )

    # Patterns to exclude from Google model listings
    _GOOGLE_EXCLUDE_PATTERNS = re.compile(
        r"-\d{3}$"           # point releases: -001, -002
        r"|-latest$"         # alias: gemini-flash-latest
        r"|-\d{2}-\d{4}$"   # dated snapshots: -09-2025
        r"|-customtools"     # variant: -customtools
        r"|-exp-"            # experimental: -exp-image-generation
    )

    async def _fetch_google_models(self, api_key: str) -> list[ModelInfo]:
        """Fetch text-generation Gemini models from Google GenAI API.

        Filters out non-text models (image/video/audio/embedding),
        open-source models (Gemma), specialized previews, and
        duplicate aliases.
        """
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            models: list[ModelInfo] = []

            for model in client.models.list():
                model_id = model.name
                if model_id.startswith("models/"):
                    model_id = model_id[len("models/"):]

                # Must be a Gemini model (not Gemma, not nano-banana, etc.)
                if not model_id.startswith("gemini-"):
                    continue

                # Must support both generateContent AND createCachedContent.
                # This filters out image-gen (Nano Banana), TTS, native-audio,
                # robotics, computer-use, and deep-research models.
                supported = set(getattr(model, "supported_actions", None) or [])
                if not {"generateContent", "createCachedContent"} <= supported:
                    continue

                # Skip duplicates and variants
                if self._GOOGLE_EXCLUDE_PATTERNS.search(model_id):
                    continue

                has_thinking = bool(getattr(model, "thinking", None))
                display = getattr(model, "display_name", model_id) or model_id

                models.append(ModelInfo(
                    id=model_id,
                    family=ModelFamily.GEMINI,
                    display_name=display,
                    supports_thinking=has_thinking,
                ))

            logger.debug("Fetched %d Google models", len(models))
            return models

        except Exception as exc:
            logger.warning("Failed to fetch Google models: %s", exc)
            # Return fallback models
            return [
                ModelInfo(id=mid, family=ModelFamily.GEMINI, supports_thinking="2.5" in mid or "3" in mid)
                for mid in self.FALLBACK_GOOGLE
            ]

    async def _fetch_anthropic_models(self, api_key: str) -> list[ModelInfo]:
        """Fetch models from Anthropic API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            page = client.models.list(limit=1000)
            models: list[ModelInfo] = []

            for model in page.data:
                model_id = model.id
                display = getattr(model, "display_name", model_id) or model_id

                # Normalize dated IDs to short aliases
                normalized = self._normalize_anthropic_id(model_id)

                models.append(ModelInfo(
                    id=normalized,
                    family=ModelFamily.CLAUDE,
                    display_name=display,
                    supports_thinking=True,  # All Claude models support thinking
                ))

            # Deduplicate (dated variants → same alias)
            seen: dict[str, ModelInfo] = {}
            for m in models:
                if m.id not in seen:
                    seen[m.id] = m
            models = list(seen.values())

            logger.debug("Fetched %d Anthropic models", len(models))
            return models

        except Exception as exc:
            logger.warning("Failed to fetch Anthropic models: %s", exc)
            # Return fallback models
            return [
                ModelInfo(id=mid, family=ModelFamily.CLAUDE, supports_thinking=True)
                for mid in self.FALLBACK_ANTHROPIC
            ]

    @staticmethod
    def _normalize_anthropic_id(model_id: str) -> str:
        """Normalize a dated Anthropic model ID to its short alias.

        E.g., 'claude-sonnet-4-5-20250514' → 'claude-sonnet-4-5'
             'claude-3-5-sonnet-20241022' → 'claude-3-5-sonnet'

        If no date pattern matches, returns the ID as-is.
        """
        # Strip trailing date pattern (YYYYMMDD)
        normalized = re.sub(r"-\d{8}$", "", model_id)
        # Also strip @YYYY-MM-DD patterns
        normalized = re.sub(r"@\d{4}-\d{2}-\d{2}$", "", normalized)
        return normalized

    def _auto_select_defaults(self) -> None:
        """Auto-select default models from fetched models.

        Prefers the current default if still available, otherwise picks
        the best candidate per family.
        """
        for family in (ModelFamily.GEMINI, ModelFamily.CLAUDE):
            current_default = self._defaults.get(family)

            # Keep current default if it's still available
            if current_default and current_default in self._models:
                info = self._models[current_default]
                if not info.deprecated:
                    continue

            # Find best candidate: prefer flash/sonnet tier, newest version
            candidates = [
                m for m in self._models.values()
                if m.family == family and not m.deprecated
            ]
            if not candidates:
                continue

            # Prefer specific tiers per family
            preferred_tier = "flash" if family == ModelFamily.GEMINI else "sonnet"
            preferred = [c for c in candidates if preferred_tier in c.id]
            if preferred:
                best = max(preferred, key=lambda m: self._extract_version(m.id))
            else:
                best = max(candidates, key=lambda m: self._extract_version(m.id))

            self._defaults[family] = best.id
