"""Settings persistence utilities.

Saves the settings a user changed to JSON files for layered configuration.
Settings are read by constructing the settings class (``settings_sources``),
never here; ``SettingsPersistence`` is the one writer.

Saving splits by trust (mirroring the P0-1 load-side allowlist in
``agentic_cli.settings_sources``): allowlisted keys go to the project
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
    "tavily_api_key",
    "brave_api_key",
    "postgres_uri",  # connection string embeds user:password@host
})

# Deny-by-default allowlist: the ONLY keys a project ./.{app}/settings.json (or
# a cwd-relative .env) may set. A cloned/untrusted repo must not be able to flip
# a security boundary — executor backend, container image/user, bind mounts,
# outputs dir, OS-sandbox policy, raw LLM logging, workspace dir, permission
# rules, or secrets. Every entry below is a benign field that cannot
# select code execution, filesystem/mount scope, container identity/image,
# network policy, secrets, or sensitive logging. Anything not clearly benign —
# and any new field — is excluded automatically. Real environment variables and
# the user ~/.{app}/settings.json remain fully trusted.
#
# Used by BOTH sides of persistence: the load-side filter
# (settings_sources._AllowlistFilterSource) and the save-side split
# (SettingsPersistence.save), so writer and reader cannot drift apart.
PROJECT_SETTABLE_KEYS = frozenset({
    # model / behavior. model_roles and models choose among the providers the
    # app and the user defined (an entry's numbers are hints); model_providers
    # (base URLs a key is sent to) is deliberately absent.
    "default_model", "model_roles", "models", "thinking_effort", "orchestrator",
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
    "arxiv_max_pdf_bytes",
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

# When a change made while the app runs takes effect (BaseCLIApp.apply_settings,
# which /settings calls). Every key /settings may show, PROJECT_SETTABLE_KEYS
# and USER_PREFERENCE_KEYS, is in exactly one of the four sets, and a new one
# is placed when it is added (tests/cli/test_settings_take_effect.py fails
# until it is). An app's own key, in none of them, applies at once.
#
# Read when the workflow builds its agents and their models: the models and
# their thinking, each agent's summary thresholds and result cap, and what
# each model's client is built with (its retries and request timeout). A
# change rebuilds the workflow, keeping its sessions and its backend.
REBUILD_KEYS = frozenset({
    "default_model", "model_roles", "models", "thinking_effort",
    "context_summarize_at", "context_max_tokens", "tool_result_max_chars",
    "retry_max_attempts", "retry_initial_delay", "retry_backoff_factor",
    "anthropic_request_timeout",
})

# Read when a conversation starts: the backend it runs on, for its whole life
# (its history is in that backend's store alone). A change applies at once
# to a conversation with nothing stored yet, which is left for a new one on
# the new backend; else it is saved and applies to the next new conversation
# (/new).
NEW_CONVERSATION_KEYS = frozenset({"orchestrator"})

# Read only as the app starts: logging is configured once, in the app's
# constructor, and a rebuild keeps the session store it has. A change is
# saved, and takes effect at the next start.
RESTART_KEYS = frozenset({"session_store", "log_level", "log_format"})

# Read each time they are used (a tool call, a sandbox or a container started,
# a dialog, an event shown): a change takes effect at once.
LIVE_KEYS = frozenset({
    "python_executor_timeout", "sandbox_timeout",
    "sandbox_max_sessions", "sandbox_memory_mb", "sandbox_cpus", "sandbox_pids_limit",
    "search_backend",
    "webfetch_cache_ttl_seconds", "webfetch_max_content_bytes", "webfetch_max_pdf_bytes",
    "webfetch_max_download_bytes", "webfetch_saved_max_age_days", "webfetch_saved_max_mb",
    "arxiv_max_pdf_bytes",
    "verbose_thinking", "ui_theme", "dialog_placement",
})

# Sections of the project file that are not settings: the permission engine
# reads its deny rules from the file itself (workflow/permissions/store.py,
# load_project_denies), so the loader passes over them and every save keeps
# them as they are.
PROJECT_PRESERVED_KEYS = frozenset({"permissions"})

# Settings written by hand, or shipped by the app in its settings class, and
# never by save(): their live value merges the app's entries with every
# file's (permissions: the union of every trusted source's rules, the
# environment's and the constructor's too), so a save would copy the app's
# shipped entries (and the other sources') into the file it writes. Every
# save leaves each file's own section as it is (a project file's only where
# something reads it: its models, and its permissions, PROJECT_PRESERVED_KEYS).
HAND_EDITED_KEYS = frozenset({"models", "model_providers", "permissions"})

# Identity fields set by the application, not the user. No save writes them;
# app_name chooses the files, so no source but the constructor and the class
# sets it either (settings_sources.SettingsSourcesMixin).
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

    ``project_path`` is where allowlisted keys go. ``user_path`` is set only
    when user-scoped (non-allowlisted) keys changed the user config;
    ``user_scoped_keys`` lists the keys written there.
    """

    project_path: Path
    user_path: Path | None = None
    user_scoped_keys: tuple[str, ...] = ()
    #: The keys written to the project file (empty when the project file was
    #: not touched). ``None`` in a result that does not say, such as an app's
    #: own ``save_settings``.
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
    """The one writer of settings files: what the user changed in ``/settings``.

    Each key goes to the file the loader reads it from: the project config
    (./.{app_name}/settings.json) for the allowlisted keys, the user config
    (~/.{app_name}/settings.json) for the rest. Reading is not done here:
    the settings class's sources read the files, the environment and the
    constructor's arguments (``settings_sources``).
    """

    def __init__(self, app_name: str):
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

    def save(self, settings: "BaseSettings", *, keys: Iterable[str]) -> SettingsSaveResult:
        """Write ``keys`` of ``settings``, each into its file, and nothing else.

        ``/settings`` passes the fields the user changed. A key written is
        the value as the user picked it: a save of the whole live object
        would also persist what came from the environment, the app's
        defaults or the start-up.

        An allowlisted key (``PROJECT_SETTABLE_KEYS``) is merged into the
        project config. The file's other allowlisted entries and its
        ``PROJECT_PRESERVED_KEYS`` sections (the project's deny rules) and
        ``models`` are kept as they are; stale entries the loader ignores
        are dropped. A key ``field.entry`` names one entry of such a dict
        field (``model_roles.fast``): that entry is merged into the project
        file's own dict, and the rest of the live dict, which may hold
        entries from the user file, the environment or the app's defaults,
        is left out.

        Any other key is merged into the user config, which keeps every key
        this method does not manage (hand-stored secrets, unknown or domain
        keys). The pick is written, even a value equal to the class default,
        and no save removes a key. What an app starts on without the user's
        file can be set by its ``model_post_init``, a validator, its
        ``__init__``, an env file below the user file or the environment, and
        the settings class cannot tell which: a key removed on a guess would
        bring the app's value back on the next start, though the user chose
        another. A user who wants a key to follow the app again deletes it
        from the file.

        Secrets and identity fields are never written anywhere: see
        ``is_secret_field``. Nor are ``HAND_EDITED_KEYS`` (``models``,
        ``model_providers``, ``permissions``): each file keeps its own
        section of them as it is. A malformed file raises rather than being
        clobbered.

        Args:
            settings: Settings instance to save from
            keys: The fields to write (``field`` or ``field.entry``)

        Returns:
            SettingsSaveResult with the written path(s)

        Raises:
            TypeError: ``keys`` is None.
        """
        from agentic_cli.file_utils import atomic_write_text

        if keys is None:
            raise TypeError(
                "save() needs the keys to write (the fields changed); a full save "
                "of the live settings was removed in 0.7.0"
            )

        fields = type(settings).model_fields
        secret = {name for name, f in fields.items() if is_secret_field(name, f)}
        secret |= SECRET_FIELDS
        keys = set(keys) - secret - IDENTITY_FIELDS
        keys = {k for k in keys if k.partition(".")[0] not in HAND_EDITED_KEYS}
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

        user_updates = {k: dump[k] for k in sorted(keys - PROJECT_SETTABLE_KEYS)}
        user_path, user_scoped = self._merge_user(user_updates) if user_updates else (None, ())
        return SettingsSaveResult(
            project_path=project_path,
            user_path=user_path,
            user_scoped_keys=user_scoped,
            project_keys=(*project_updates, *entry_fields),
        )

    def _merge_user(self, updates: dict[str, Any]) -> tuple[Path | None, tuple[str, ...]]:
        """Merge-write the user file; never clobber keys it does not manage
        (e.g. hand-stored API keys). Returns (path written or None, keys)."""
        from agentic_cli.file_utils import atomic_write_text

        user_path = self.user_config_path
        existing = self._read_object(user_path)
        merged = {**existing, **updates}
        if merged == existing:
            return None, ()
        user_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(user_path, json.dumps(merged, indent=2, default=str))
        return user_path, tuple(sorted(updates))

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
