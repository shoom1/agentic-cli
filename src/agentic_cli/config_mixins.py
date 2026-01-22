"""Configuration mixins for optional features.

These mixins allow domain applications to opt-in to specific features
without inheriting unnecessary configuration from BaseSettings.

This addresses the Interface Segregation Principle (ISP) - clients should
not be forced to depend on interfaces they don't use.

Usage:
    # Basic app - just CLI + workflow
    class MySettings(BaseSettings):
        pass

    # App with knowledge base support
    class MySettings(KnowledgeBaseMixin, BaseSettings):
        pass

    # App with full features
    class MySettings(KnowledgeBaseMixin, PersistenceMixin, BaseSettings):
        pass
"""

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

if TYPE_CHECKING:
    pass


class KnowledgeBaseMixin:
    """Mixin for knowledge base configuration.

    Adds settings for semantic search, embeddings, and document management.
    Use this mixin when your application needs the knowledge base feature.
    """

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )
    knowledge_base_use_mock: bool = Field(
        default=False,
        description="Use mock knowledge base (no ML dependencies required)",
    )
    serper_api_key: str | None = Field(
        default=None,
        description="Serper.dev API key for web search",
        validation_alias="SERPER_API_KEY",
    )


class PythonExecutorMixin:
    """Mixin for Python code execution configuration.

    Adds settings for safe Python code execution.
    Use this mixin when your application needs to execute Python code.
    """

    python_executor_timeout: int = Field(
        default=30,
        description="Default timeout for Python execution (seconds)",
    )


class PersistenceMixin:
    """Mixin for session persistence configuration.

    Adds settings and path resolution for session storage.
    Use this mixin when your application needs session save/load.
    """

    @property
    def sessions_dir(self) -> Path:
        """Directory for session storage."""
        # Assumes workspace_dir is defined in BaseSettings
        return getattr(self, "workspace_dir") / "sessions"


class ArtifactsMixin:
    """Mixin for artifact management configuration.

    Adds settings and path resolution for artifact storage.
    Use this mixin when your application generates artifacts.
    """

    @property
    def artifacts_dir(self) -> Path:
        """Directory for artifact storage."""
        return getattr(self, "workspace_dir") / "workspace"

    @property
    def templates_dir(self) -> Path:
        """Directory for report templates."""
        return getattr(self, "workspace_dir") / "templates"

    @property
    def reports_dir(self) -> Path:
        """Directory for generated reports."""
        return getattr(self, "workspace_dir") / "reports"


class FullFeaturesMixin(
    KnowledgeBaseMixin,
    PythonExecutorMixin,
    PersistenceMixin,
    ArtifactsMixin,
):
    """Convenience mixin that includes all optional features.

    For applications that need the complete feature set.

    Usage:
        class MySettings(FullFeaturesMixin, BaseSettings):
            pass
    """

    pass
