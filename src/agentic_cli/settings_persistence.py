"""Settings persistence utilities.

Provides functionality to save settings to JSON files for layered configuration.

Saving splits by trust (mirroring the P0-1 load-side allowlist in
``agentic_cli.config``): allowlisted keys go to the project
``./.{app_name}/settings.json``; every other (user-scoped) key goes to the
trusted user ``~/.{app_name}/settings.json``. Without the split, a key the
loader refuses to read from the project file would be written there and
silently dropped on the next start.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings


# Fields that should never be saved to JSON (secrets)
SECRET_FIELDS = frozenset({
    "google_api_key",
    "anthropic_api_key",
    "tavily_api_key",
    "brave_api_key",
    "postgres_uri",  # connection string embeds user:password@host
})

# Deny-by-default allowlist: the ONLY keys a project ./.{app}/settings.json (or
# a cwd-relative .env) may set. A cloned/untrusted repo must not be able to flip
# a security boundary — executor backend, container image/user, bind mounts,
# outputs dir, OS-sandbox policy, shell backend, raw LLM logging, workspace dir,
# permission rules, or secrets. Every entry below is a benign field that cannot
# select code execution, filesystem/mount scope, container identity/image,
# network policy, secrets, or sensitive logging. Anything not clearly benign —
# and any new field — is excluded automatically. Real environment variables and
# the user ~/.{app}/settings.json remain fully trusted.
#
# Used by BOTH sides of persistence: the load-side filter
# (config._AllowlistFilterSource) and the save-side split
# (SettingsPersistence.save), so writer and reader cannot drift apart.
PROJECT_SETTABLE_KEYS = frozenset({
    # model / behavior
    "default_model", "thinking_effort", "orchestrator",
    "context_window_trigger_tokens", "context_window_target_tokens",
    # retry / request timeouts (not code paths)
    "retry_max_attempts", "retry_initial_delay", "retry_backoff_factor",
    "anthropic_request_timeout", "python_executor_timeout", "sandbox_timeout",
    # sandbox RESOURCE limits (not backend / image / mounts / user / network)
    "sandbox_max_sessions", "sandbox_memory_mb", "sandbox_cpus", "sandbox_pids_limit",
    # non-exec tool config
    "search_backend",
    "webfetch_cache_ttl_seconds", "webfetch_max_content_bytes", "webfetch_max_pdf_bytes",
    # persistence backend selection (NOT the credential-bearing postgres_uri)
    "session_store",
    # display / logging verbosity (NOT raw_llm_logging)
    "log_level", "log_format", "verbose_thinking",
})

# Identity fields set by the application, not the user
IDENTITY_FIELDS = frozenset({
    "app_name",
    "workspace_dir",
})


def get_project_config_path(app_name: str) -> Path:
    """Get path to project config file (./.{app_name}/settings.json)."""
    return Path.cwd() / f".{app_name}" / "settings.json"


def get_user_config_path(app_name: str) -> Path:
    """Get path to user config file (~/.{app_name}/settings.json)."""
    return Path.home() / f".{app_name}" / "settings.json"


@dataclass(frozen=True)
class SettingsSaveResult:
    """Where a settings save landed.

    ``project_path`` always receives the allowlisted keys (or, with an
    explicit ``path=``, the full legacy dump). ``user_path`` is set only when
    user-scoped (non-allowlisted) keys were written to the user config;
    ``user_scoped_keys`` lists the keys added/updated/removed there.
    """

    project_path: Path
    user_path: Path | None = None
    user_scoped_keys: tuple[str, ...] = ()


def get_user_project_grants_path(app_name: str) -> Path:
    """Path to interactively-granted permission rules
    (``~/.{app_name}/project_grants.json``), keyed by resolved project path.

    Lives in USER config — never in the repo — so a cloned workspace carries no
    "Allow always" grants: a clone at a different path simply has no entry and
    the user re-grants. Structure::

        { "<resolved project root>": {"permissions": {"allow": [...], "deny": [...]}} }
    """
    return Path.home() / f".{app_name}" / "project_grants.json"



class SettingsPersistence:
    """Manages loading and saving settings to JSON files.

    Settings are saved to project config (./.{app_name}/settings.json) by default.
    User config (~/.{app_name}/settings.json) serves as a fallback for loading.

    Loading priority (highest to lowest):
        1. Environment variables
        2. Project config (./.{app_name}/settings.json)
        3. User config (~/.{app_name}/settings.json)
        4. .env file
        5. Default values

    Saving: Always saves to project config (./.{app_name}/settings.json).
    """

    def __init__(self, app_name: str = "agentic"):
        """Initialize persistence manager.

        Args:
            app_name: Application name used for config directories
        """
        self.app_name = app_name

    @property
    def project_config_path(self) -> Path:
        """Get path to project config file (./.{app_name}/settings.json)."""
        return get_project_config_path(self.app_name)

    @property
    def user_config_path(self) -> Path:
        """Get path to user config file (~/.{app_name}/settings.json)."""
        return get_user_config_path(self.app_name)

    def save(
        self,
        settings: "BaseSettings",
        path: Path | None = None,
    ) -> SettingsSaveResult:
        """Save settings, split by trust to match the load-side allowlist.

        Default save writes two files:

        - Project config (./.{app_name}/settings.json) receives ONLY
          allowlisted (``PROJECT_SETTABLE_KEYS``) fields, always all of them —
          regardless of whether they match the schema default, because
          subclasses may have different effective defaults. The file is fully
          rewritten, which also heals stale pre-P0-1 files holding keys the
          loader now ignores.
        - User config (~/.{app_name}/settings.json) receives the remaining
          user-scoped fields, but only those differing from the settings
          class default (so code-default changes keep applying for untouched
          fields, and the user file stays minimal). A user-scoped field back
          at its default is REMOVED from the file. Keys this method does not
          manage (hand-stored secrets, unknown/domain keys) are preserved
          verbatim; a malformed user file raises rather than being clobbered.

        Secrets (API keys) and identity fields are never written anywhere.
        With an explicit ``path=``, the legacy behavior is kept: one full
        dump (minus secrets/identity) to that file, no split.

        Args:
            settings: Settings instance to save
            path: Optional custom path (single-file legacy dump)

        Returns:
            SettingsSaveResult with the written path(s)
        """
        from agentic_cli.file_utils import atomic_write_text

        # Get settings as dict, excluding secrets and identity fields
        data = settings.model_dump(
            exclude=SECRET_FIELDS | IDENTITY_FIELDS,
            exclude_none=True,
        )

        # Convert Path objects to strings for JSON serialization
        data = self._serialize_paths(data)

        if path is not None:
            # Explicit target: legacy single-file full dump.
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, json.dumps(data, indent=2, default=str))
            return SettingsSaveResult(project_path=path)

        project_data = {k: v for k, v in data.items() if k in PROJECT_SETTABLE_KEYS}
        user_updates, user_removals = self._split_user_scoped(settings, data)

        project_path = self.project_config_path
        project_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            project_path, json.dumps(project_data, indent=2, default=str)
        )

        user_path = self.user_config_path
        # Merge-write: never clobber keys we don't manage (e.g. hand-stored
        # API keys). A malformed user file raises instead of being replaced.
        existing: dict[str, Any] = {}
        if user_path.exists():
            existing = json.loads(user_path.read_text())
            if not isinstance(existing, dict):
                raise ValueError(
                    f"User settings file is not a JSON object: {user_path}"
                )
        merged = dict(existing)
        removed = tuple(k for k in sorted(user_removals) if k in existing)
        for key in removed:
            del merged[key]
        merged.update(user_updates)

        if merged == existing:
            return SettingsSaveResult(project_path=project_path)

        user_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(user_path, json.dumps(merged, indent=2, default=str))
        return SettingsSaveResult(
            project_path=project_path,
            user_path=user_path,
            user_scoped_keys=tuple(sorted(user_updates)) + removed,
        )

    def _split_user_scoped(
        self, settings: "BaseSettings", data: dict[str, Any]
    ) -> tuple[dict[str, Any], set[str]]:
        """Partition non-allowlisted dumped fields by deviation from default.

        Returns (updates, removals): ``updates`` maps user-scoped keys whose
        live value differs from the settings class default to their dumped
        value; ``removals`` holds user-scoped keys back at their default,
        whose stale entries should leave the user file. Comparison uses the
        instance's own class fields, so subclass default overrides are
        respected.
        """
        from pydantic_core import PydanticUndefined

        updates: dict[str, Any] = {}
        removals: set[str] = set()
        fields = type(settings).model_fields
        for key, value in data.items():
            if key in PROJECT_SETTABLE_KEYS:
                continue
            field = fields.get(key)
            if field is None:
                continue
            default = field.get_default(call_default_factory=True)
            if default is not PydanticUndefined and getattr(settings, key) == default:
                removals.add(key)
            else:
                updates[key] = value
        return updates, removals

    def load(self, path: Path | None = None) -> dict[str, Any]:
        """Load settings from JSON config file.

        If no path is specified, tries project config first, then user config.

        Args:
            path: Optional custom path (if not specified, uses fallback order)

        Returns:
            Dictionary of settings from file, or empty dict if no file exists
        """
        if path is not None:
            if not path.exists():
                return {}
            with open(path) as f:
                return json.load(f)

        # Try project config first, then user config
        for config_path in [self.project_config_path, self.user_config_path]:
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)

        return {}

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
