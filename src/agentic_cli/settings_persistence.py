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
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings


#: A settings field whose name matches this is a credential. It is never saved,
#: whichever application defines it (see ``is_secret_field``). ``token`` counts
#: only as a whole word of the name (``github_token``, ``hf_token``): a token
#: count (``context_max_tokens``) or a tokenizer is not a credential.
CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(api_?key|secret|password|credential|(?:^|_)token(?:_|$))"
)

# Fields that should never be saved to JSON (secrets). Any field whose name
# matches CREDENTIAL_KEY_RE or whose type is SecretStr is excluded as well.
SECRET_FIELDS = frozenset({
    "google_api_key",
    "anthropic_api_key",
    "openai_api_key",
    "openrouter_api_key",
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
    # model / behavior. model_roles and pinned_models choose among the providers
    # and endpoints the user defined; model_endpoints (base URLs a key is sent
    # to) is deliberately absent.
    "default_model", "model_roles", "pinned_models", "thinking_effort", "orchestrator",
    # when a conversation is summarized (workflow/context.py)
    "context_summarize_at", "context_max_tokens",
    # when a tool result is cut (workflow/tool_output.py; the folder is fixed)
    "tool_result_max_chars",
    # retry / request timeouts (not code paths)
    "retry_max_attempts", "retry_initial_delay", "retry_backoff_factor",
    "anthropic_request_timeout", "python_executor_timeout", "sandbox_timeout",
    # sandbox RESOURCE limits (not backend / image / mounts / user / network)
    "sandbox_max_sessions", "sandbox_memory_mb", "sandbox_cpus", "sandbox_pids_limit",
    # non-exec tool config
    "search_backend",
    "webfetch_cache_ttl_seconds", "webfetch_max_content_bytes", "webfetch_max_pdf_bytes",
    "webfetch_max_download_bytes", "webfetch_saved_max_age_days", "webfetch_saved_max_mb",
    # persistence backend selection (NOT the credential-bearing postgres_uri)
    "session_store",
    # display / logging verbosity (NOT raw_llm_logging)
    "log_level", "log_format", "verbose_thinking",
})

# The user's preferences: how the CLI looks, which is the user's choice in
# every project. They are user-scoped like every key not in
# PROJECT_SETTABLE_KEYS (a project file or a cwd .env cannot set them, and a
# save writes them to the user file), and shown in /settings only when an app
# lists them (BaseCLIApp._build_ui_items), unlike any other user-scoped key.
USER_PREFERENCE_KEYS = frozenset({"ui_theme", "dialog_placement"})

# Sections of the project file that are not settings: the permission engine
# reads its deny rules from the file itself (workflow/permissions/engine.py),
# so the loader passes over them and every save keeps them as they are.
PROJECT_PRESERVED_KEYS = frozenset({"permissions"})

# Identity fields set by the application, not the user
IDENTITY_FIELDS = frozenset({
    "app_name",
    "workspace_dir",
})


def is_secret_field(name: str, field: Any | None = None) -> bool:
    """Whether a settings field holds a credential and must never be saved.

    True for ``SECRET_FIELDS``, for any name that looks like a credential
    (``CREDENTIAL_KEY_RE``: an application's own ``mistral_api_key``, say) and
    for any field typed ``SecretStr``/``SecretBytes``.
    """
    if name in SECRET_FIELDS or CREDENTIAL_KEY_RE.search(name):
        return True
    return field is not None and _mentions_secret_type(getattr(field, "annotation", None))


def _mentions_secret_type(annotation: Any) -> bool:
    from pydantic import SecretBytes, SecretStr

    if annotation in (SecretStr, SecretBytes):
        return True
    return any(_mentions_secret_type(arg) for arg in get_args(annotation))


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
    #: For a ``keys=`` save: the keys written to the project file (empty when
    #: the project file was not touched). ``None`` for a full save.
    project_keys: tuple[str, ...] | None = None


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
        *,
        keys: Iterable[str] | None = None,
    ) -> SettingsSaveResult:
        """Save settings, split by trust to match the load-side allowlist.

        With ``keys``, only those fields are written, each merged into its
        file (the ``/settings`` command passes the fields the user changed).
        A key ``field.entry`` names one entry of a project-settable dict field
        (``model_roles.fast``): that entry is merged into the project file's
        own dict, and the rest of the live dict, which may hold entries from
        the user file, the environment or the app's defaults, is left out.
        Without it, the whole live settings object is written as described
        below, which also persists values that came from the environment or
        were derived at startup.

        Default save writes two files:

        - Project config (./.{app_name}/settings.json) receives ONLY
          allowlisted (``PROJECT_SETTABLE_KEYS``) fields, always all of them —
          regardless of whether they match the schema default, because
          subclasses may have different effective defaults. The file is fully
          rewritten, which also heals stale pre-P0-1 files holding keys the
          loader now ignores; its ``PROJECT_PRESERVED_KEYS`` sections (the
          project's deny rules) are kept as they are, so a malformed project
          file raises rather than being clobbered.
        - User config (~/.{app_name}/settings.json) receives the remaining
          user-scoped fields, but only those differing from the settings
          class default (so code-default changes keep applying for untouched
          fields, and the user file stays minimal). A user-scoped field back
          at its default is REMOVED from the file. Keys this method does not
          manage (hand-stored secrets, unknown/domain keys) are preserved
          verbatim; a malformed user file raises rather than being clobbered.

        Secrets and identity fields are never written anywhere: see
        ``is_secret_field``.
        With an explicit ``path=``, the legacy behavior is kept: one full
        dump (minus secrets/identity) to that file, no split.

        Args:
            settings: Settings instance to save
            path: Optional custom path (single-file legacy dump)

        Returns:
            SettingsSaveResult with the written path(s)
        """
        from agentic_cli.file_utils import atomic_write_text

        fields = type(settings).model_fields
        secret = {name for name, f in fields.items() if is_secret_field(name, f)}
        secret |= SECRET_FIELDS

        if keys is not None:
            return self._save_keys(settings, set(keys) - secret - IDENTITY_FIELDS)

        # Get settings as dict, excluding secrets and identity fields
        data = settings.model_dump(
            exclude=secret | IDENTITY_FIELDS,
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
        existing = self._read_object(project_path)
        project_data.update(
            {k: existing[k] for k in sorted(PROJECT_PRESERVED_KEYS) if k in existing}
        )
        project_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            project_path, json.dumps(project_data, indent=2, default=str)
        )

        user_path, user_keys = self._merge_user(user_updates, user_removals)
        return SettingsSaveResult(
            project_path=project_path, user_path=user_path, user_scoped_keys=user_keys,
        )

    def _save_keys(self, settings: "BaseSettings", keys: set[str]) -> SettingsSaveResult:
        """Write only ``keys``: project-settable ones merged into the project
        file, the rest updated in (or, back at default, removed from) the
        user file. Nothing else is touched (the project file's
        ``PROJECT_PRESERVED_KEYS`` sections are kept)."""
        from pydantic_core import PydanticUndefined

        from agentic_cli.file_utils import atomic_write_text

        fields = type(settings).model_fields
        entries: dict[str, set[str]] = {}
        for key in keys:
            field, dot, entry = key.partition(".")
            if dot and entry and field in PROJECT_SETTABLE_KEYS and field in fields:
                entries.setdefault(field, set()).add(entry)
        keys = {k for k in keys if k in fields}
        entry_fields = sorted(set(entries) - keys)
        dump = self._serialize_paths(settings.model_dump(include=keys | set(entry_fields)))

        project_path = self.project_config_path
        project_updates = {k: dump[k] for k in sorted(keys) if k in PROJECT_SETTABLE_KEYS}
        if project_updates or entry_fields:
            existing = self._read_object(project_path)
            # Keep the file's other allowlisted entries and its deny rules;
            # drop stale ones the loader would ignore anyway.
            merged = {
                k: v for k, v in existing.items()
                if k in PROJECT_SETTABLE_KEYS or k in PROJECT_PRESERVED_KEYS
            }
            merged.update(project_updates)
            for field in entry_fields:
                own = merged.get(field)
                live = dump[field] or {}
                merged[field] = {
                    **(own if isinstance(own, dict) else {}),
                    **{entry: live[entry] for entry in sorted(entries[field]) if entry in live},
                }
            project_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(project_path, json.dumps(merged, indent=2, default=str))

        user_updates: dict[str, Any] = {}
        user_removals: set[str] = set()
        for key in keys - PROJECT_SETTABLE_KEYS:
            default = fields[key].get_default(call_default_factory=True)
            if default is not PydanticUndefined and getattr(settings, key) == default:
                user_removals.add(key)
            else:
                user_updates[key] = dump[key]
        user_path, user_keys = self._merge_user(user_updates, user_removals)
        return SettingsSaveResult(
            project_path=project_path,
            user_path=user_path,
            user_scoped_keys=user_keys,
            project_keys=(*project_updates, *entry_fields),
        )

    def _merge_user(
        self, updates: dict[str, Any], removals: set[str]
    ) -> tuple[Path | None, tuple[str, ...]]:
        """Merge-write the user file; never clobber keys it does not manage
        (e.g. hand-stored API keys). Returns (path written or None, keys)."""
        from agentic_cli.file_utils import atomic_write_text

        user_path = self.user_config_path
        existing = self._read_object(user_path)
        merged = dict(existing)
        removed = tuple(k for k in sorted(removals) if k in existing)
        for key in removed:
            del merged[key]
        merged.update(updates)
        if merged == existing:
            return None, ()
        user_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(user_path, json.dumps(merged, indent=2, default=str))
        return user_path, tuple(sorted(updates)) + removed

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        """A settings file's JSON object, ``{}`` if absent. A malformed file
        raises instead of being replaced."""
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Settings file is not a JSON object: {path}")
        return data

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
