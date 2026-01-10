"""Shared test fixtures and utilities for agentic-cli tests.

Provides:
- MockContext for isolating tests from global state
- Temporary workspace fixtures
- Settings fixtures with various configurations
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from agentic_cli.config import (
    BaseSettings,
    reload_settings,
    set_context_settings,
    set_settings,
)


class MockContext:
    """Context manager for isolating tests from global state.

    Handles:
    - Resetting global settings singleton
    - Providing temporary workspace directories
    - Cleaning up after tests

    Usage:
        with MockContext() as ctx:
            # Tests run in isolation
            settings = ctx.settings
            workspace = ctx.workspace_dir
    """

    def __init__(
        self,
        google_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        **settings_kwargs,
    ):
        """Initialize mock context.

        Args:
            google_api_key: Optional Google API key for testing
            anthropic_api_key: Optional Anthropic API key for testing
            **settings_kwargs: Additional settings overrides
        """
        self._google_api_key = google_api_key
        self._anthropic_api_key = anthropic_api_key
        self._settings_kwargs = settings_kwargs
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._settings: BaseSettings | None = None
        self._original_env: dict[str, str | None] = {}

    def __enter__(self) -> "MockContext":
        """Enter the mock context."""
        # Create temporary workspace
        self._temp_dir = tempfile.TemporaryDirectory()
        workspace_dir = Path(self._temp_dir.name)

        # Preserve and clear relevant environment variables
        env_vars = ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "SERPER_API_KEY"]
        for var in env_vars:
            self._original_env[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        # Set test API keys in environment if provided
        if self._google_api_key:
            os.environ["GOOGLE_API_KEY"] = self._google_api_key
        if self._anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self._anthropic_api_key

        # Create settings with temporary workspace
        self._settings = BaseSettings(
            workspace_dir=workspace_dir,
            **self._settings_kwargs,
        )
        set_settings(self._settings)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the mock context and clean up."""
        # Clear context settings first
        set_context_settings(None)

        # Restore environment variables
        for var, value in self._original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

        # Reset global settings
        reload_settings()

        # Clean up temp directory
        if self._temp_dir:
            self._temp_dir.cleanup()

    @property
    def settings(self) -> BaseSettings:
        """Get the test settings instance."""
        if self._settings is None:
            raise RuntimeError("MockContext not entered")
        return self._settings

    @property
    def workspace_dir(self) -> Path:
        """Get the temporary workspace directory."""
        if self._temp_dir is None:
            raise RuntimeError("MockContext not entered")
        return Path(self._temp_dir.name)


@pytest.fixture
def mock_context() -> Generator[MockContext, None, None]:
    """Fixture providing an isolated test context."""
    with MockContext() as ctx:
        yield ctx


@pytest.fixture
def mock_context_with_google_key() -> Generator[MockContext, None, None]:
    """Fixture providing context with Google API key."""
    with MockContext(google_api_key="test-google-key") as ctx:
        yield ctx


@pytest.fixture
def mock_context_with_anthropic_key() -> Generator[MockContext, None, None]:
    """Fixture providing context with Anthropic API key."""
    with MockContext(anthropic_api_key="test-anthropic-key") as ctx:
        yield ctx


@pytest.fixture
def mock_context_with_both_keys() -> Generator[MockContext, None, None]:
    """Fixture providing context with both API keys."""
    with MockContext(
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    ) as ctx:
        yield ctx


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Fixture providing a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


@pytest.fixture
def settings_no_keys(temp_workspace: Path) -> BaseSettings:
    """Fixture providing settings with no API keys."""
    # Clear environment variables temporarily
    with patch.dict(os.environ, {}, clear=True):
        return BaseSettings(workspace_dir=temp_workspace)


@pytest.fixture
def settings_google_only(temp_workspace: Path) -> BaseSettings:
    """Fixture providing settings with only Google API key."""
    with patch.dict(
        os.environ,
        {"GOOGLE_API_KEY": "test-google-key"},
        clear=True,
    ):
        return BaseSettings(workspace_dir=temp_workspace)


@pytest.fixture
def settings_anthropic_only(temp_workspace: Path) -> BaseSettings:
    """Fixture providing settings with only Anthropic API key."""
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-anthropic-key"},
        clear=True,
    ):
        return BaseSettings(workspace_dir=temp_workspace)


@pytest.fixture
def settings_both_keys(temp_workspace: Path) -> BaseSettings:
    """Fixture providing settings with both API keys."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "test-google-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
        },
        clear=True,
    ):
        return BaseSettings(workspace_dir=temp_workspace)
