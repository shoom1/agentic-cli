"""Resolvers for model and path configuration.

Extracted from BaseSettings to follow Single Responsibility Principle.
These classes handle specific resolution logic independently.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings


# Available models by provider
GOOGLE_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

ANTHROPIC_MODELS = [
    "claude-opus-4-5",
    "claude-opus-4",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
]

ALL_MODELS = GOOGLE_MODELS + ANTHROPIC_MODELS

# Thinking effort levels (for models that support it)
THINKING_EFFORT_LEVELS = ["none", "low", "medium", "high"]


class ModelResolver:
    """Resolves model selection based on configuration and available API keys.

    Separates model resolution logic from settings storage.
    """

    DEFAULT_GOOGLE_MODEL = "gemini-3-flash-preview"
    DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        google_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """Initialize the model resolver.

        Args:
            google_api_key: Google API key (if available)
            anthropic_api_key: Anthropic API key (if available)
            default_model: Explicitly configured default model
        """
        self._google_api_key = google_api_key
        self._anthropic_api_key = anthropic_api_key
        self._default_model = default_model

    @classmethod
    def from_settings(cls, settings: "BaseSettings") -> "ModelResolver":
        """Create a ModelResolver from settings instance."""
        return cls(
            google_api_key=settings.google_api_key,
            anthropic_api_key=settings.anthropic_api_key,
            default_model=settings.default_model,
        )

    @property
    def has_google_key(self) -> bool:
        """Check if Google API key is available."""
        return bool(self._google_api_key)

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is available."""
        return bool(self._anthropic_api_key)

    @property
    def has_any_api_key(self) -> bool:
        """Check if any API key is available."""
        return self.has_google_key or self.has_anthropic_key

    def get_model(self) -> str:
        """Get the model to use based on configuration and available keys.

        Resolution order:
        1. Explicitly configured default_model
        2. Google model (if Google API key available)
        3. Anthropic model (if Anthropic API key available)

        Returns:
            Model name string

        Raises:
            RuntimeError: If no API keys are available
        """
        if self._default_model:
            return self._default_model

        if self.has_google_key:
            return self.DEFAULT_GOOGLE_MODEL
        if self.has_anthropic_key:
            return self.DEFAULT_ANTHROPIC_MODEL

        raise RuntimeError(
            "No API keys found. Please set GOOGLE_API_KEY or ANTHROPIC_API_KEY."
        )

    def get_available_models(self) -> list[str]:
        """Get list of models available based on configured API keys."""
        models = []
        if self.has_google_key:
            models.extend(GOOGLE_MODELS)
        if self.has_anthropic_key:
            models.extend(ANTHROPIC_MODELS)
        return models

    def is_google_model(self, model: str | None = None) -> bool:
        """Check if the given model (or resolved model) is a Google model."""
        model = model or self.get_model()
        return model in GOOGLE_MODELS or model.startswith("gemini")

    def is_anthropic_model(self, model: str | None = None) -> bool:
        """Check if the given model (or resolved model) is an Anthropic model."""
        model = model or self.get_model()
        return model in ANTHROPIC_MODELS or model.startswith("claude")

    def supports_thinking_effort(self, model: str | None = None) -> bool:
        """Check if the model supports thinking effort configuration."""
        model = model or self.get_model()
        return (
            self.is_anthropic_model(model)
            or "gemini-2.5" in model
            or "gemini-3" in model
        )

    def validate_model(self, model: str) -> None:
        """Validate that a model is available.

        Args:
            model: Model name to validate

        Raises:
            ValueError: If model is not available
        """
        available = self.get_available_models()
        if model not in available:
            raise ValueError(
                f"Model '{model}' is not available. "
                f"Available models: {', '.join(available)}"
            )

    def get_api_key_status(self) -> dict[str, bool]:
        """Get status of all API keys."""
        return {
            "google": self.has_google_key,
            "anthropic": self.has_anthropic_key,
        }


class PathResolver:
    """Resolves workspace paths for the application.

    Provides consistent path resolution for all storage locations.
    """

    def __init__(self, workspace_dir: Path) -> None:
        """Initialize the path resolver.

        Args:
            workspace_dir: Base workspace directory
        """
        self._workspace_dir = workspace_dir

    @classmethod
    def from_settings(cls, settings: "BaseSettings") -> "PathResolver":
        """Create a PathResolver from settings instance."""
        return cls(workspace_dir=settings.workspace_dir)

    @property
    def workspace_dir(self) -> Path:
        """Base workspace directory."""
        return self._workspace_dir

    @property
    def sessions_dir(self) -> Path:
        """Directory for session storage."""
        return self._workspace_dir / "sessions"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for artifact storage."""
        return self._workspace_dir / "workspace"

    @property
    def templates_dir(self) -> Path:
        """Directory for report templates."""
        return self._workspace_dir / "templates"

    @property
    def reports_dir(self) -> Path:
        """Directory for generated reports."""
        return self._workspace_dir / "reports"

    @property
    def knowledge_base_dir(self) -> Path:
        """Directory for knowledge base storage."""
        return self._workspace_dir / "knowledge_base"

    @property
    def knowledge_base_documents_dir(self) -> Path:
        """Directory for knowledge base documents."""
        return self.knowledge_base_dir / "documents"

    @property
    def knowledge_base_embeddings_dir(self) -> Path:
        """Directory for knowledge base embeddings."""
        return self.knowledge_base_dir / "embeddings"

    def ensure_workspace_exists(self) -> None:
        """Create workspace directory if it doesn't exist."""
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

    def ensure_all_dirs_exist(self) -> None:
        """Create all standard directories if they don't exist."""
        for dir_path in [
            self.sessions_dir,
            self.artifacts_dir,
            self.templates_dir,
            self.reports_dir,
            self.knowledge_base_dir,
            self.knowledge_base_documents_dir,
            self.knowledge_base_embeddings_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
