"""Settings persistence utilities.

Provides functionality to save settings to JSON files for layered configuration.
Settings are saved to ./settings.json (project config) by default.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings


# Fields that should never be saved to JSON (secrets)
SECRET_FIELDS = frozenset({
    "google_api_key",
    "anthropic_api_key",
    "serper_api_key",
})


class SettingsPersistence:
    """Manages loading and saving settings to JSON files.

    Settings are saved to project config (./settings.json) by default.
    Users can manually create ~/.appname/settings.json for user-level defaults.

    Priority (highest to lowest):
        1. Environment variables
        2. Project config (./settings.json)
        3. User config (~/.appname/settings.json)
        4. .env file
        5. Default values
    """

    def __init__(self, app_name: str = "agentic"):
        """Initialize persistence manager.

        Args:
            app_name: Application name used for user config directory
        """
        self.app_name = app_name

    @property
    def project_config_path(self) -> Path:
        """Get path to project config file (./settings.json)."""
        return Path.cwd() / "settings.json"

    @property
    def user_config_path(self) -> Path:
        """Get path to user config file (~/.appname/settings.json)."""
        return Path.home() / f".{self.app_name}" / "settings.json"

    def save(
        self,
        settings: "BaseSettings",
        exclude_defaults: bool = True,
        path: Path | None = None,
    ) -> Path:
        """Save settings to JSON config file.

        By default saves to project config (./settings.json).
        Secrets (API keys) are never saved.

        Args:
            settings: Settings instance to save
            exclude_defaults: If True, only save non-default values
            path: Optional custom path (defaults to project_config_path)

        Returns:
            Path to the saved config file
        """
        target_path = path or self.project_config_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Get settings as dict, excluding secrets
        data = settings.model_dump(
            exclude=SECRET_FIELDS,
            exclude_defaults=exclude_defaults,
            exclude_none=True,
        )

        # Convert Path objects to strings for JSON serialization
        data = self._serialize_paths(data)

        # Write to file
        with open(target_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return target_path

    def load(self, path: Path | None = None) -> dict[str, Any]:
        """Load settings from JSON config file.

        Args:
            path: Optional custom path (defaults to project_config_path)

        Returns:
            Dictionary of settings from file, or empty dict if file doesn't exist
        """
        target_path = path or self.project_config_path

        if not target_path.exists():
            return {}

        with open(target_path) as f:
            return json.load(f)

    def _serialize_paths(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert Path objects to strings recursively.

        Args:
            data: Dictionary that may contain Path objects

        Returns:
            Dictionary with Path objects converted to strings
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_paths(value)
            else:
                result[key] = value
        return result
