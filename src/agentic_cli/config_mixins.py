"""Configuration mixins for optional features.

NOTE: As of v0.3.0, most settings have been consolidated into BaseSettings
with organized mixins (WorkflowSettingsMixin, CLISettingsMixin). These mixins
are kept for backward compatibility but may be deprecated in a future version.

The recommended approach is now:
    - Use BaseSettings directly (includes all common settings)
    - Override get_ui_setting_keys() in your app to control which settings appear in UI

Legacy Usage:
    # Basic app - just CLI + workflow
    class MySettings(BaseSettings):
        pass

    # App with knowledge base support (backward compatible)
    class MySettings(KnowledgeBaseMixin, BaseSettings):
        pass
"""

from pathlib import Path


class KnowledgeBaseMixin:
    """Mixin for knowledge base configuration.

    NOTE: These fields are now included in BaseSettings by default.
    This mixin is kept for backward compatibility only.

    The following fields are available in BaseSettings:
    - embedding_model
    - embedding_batch_size
    - knowledge_base_use_mock
    - serper_api_key
    """

    pass  # Fields now in BaseSettings


class PythonExecutorMixin:
    """Mixin for Python code execution configuration.

    NOTE: These fields are now included in WorkflowSettingsMixin
    which is composed into BaseSettings by default.
    This mixin is kept for backward compatibility only.

    The following fields are available in BaseSettings:
    - python_executor_timeout
    """

    pass  # Fields now in WorkflowSettingsMixin -> BaseSettings


class PersistenceMixin:
    """Mixin for session persistence configuration.

    NOTE: These properties are now available in BaseSettings via PathResolver.
    This mixin is kept for backward compatibility only.

    The following properties are available in BaseSettings:
    - sessions_dir
    """

    @property
    def sessions_dir(self) -> Path:
        """Directory for session storage."""
        # Assumes workspace_dir is defined in BaseSettings
        return getattr(self, "workspace_dir") / "sessions"


class ArtifactsMixin:
    """Mixin for artifact management configuration.

    NOTE: These properties are now available in BaseSettings via PathResolver.
    This mixin is kept for backward compatibility only.

    The following properties are available in BaseSettings:
    - artifacts_dir
    - templates_dir
    - reports_dir
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

    NOTE: All these features are now included in BaseSettings by default.
    This mixin is kept for backward compatibility only.

    Usage (deprecated):
        class MySettings(FullFeaturesMixin, BaseSettings):
            pass

    Recommended:
        class MySettings(BaseSettings):
            pass  # All features included by default
    """

    pass
