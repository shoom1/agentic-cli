"""Where settings are read from, and which of those sources are trusted.

``SettingsSourcesMixin`` holds the one ``settings_customise_sources``. The
settings mixins that carry the fields it protects (``WorkflowSettingsMixin``,
``AppSettingsMixin``) inherit it, so the trust boundary goes with the fields:
``agentic_cli.BaseSettings`` and a class composed from those mixins over
pydantic-settings' own ``BaseSettings`` read the same sources, in the same
order, through the same filters. The mixins must come before
pydantic-settings' class, as in ``BaseSettings``: otherwise its own sources
would win, so a settings class built in that order is refused when it is
defined. A class that defines ``settings_customise_sources`` itself chooses
its own sources.

This module imports neither ``agentic_cli.config`` nor the mixins, which
import it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import PydanticBaseSettingsSource

from agentic_cli.logging import Loggers
from agentic_cli.settings_persistence import (
    PROJECT_PRESERVED_KEYS,
    PROJECT_SETTABLE_KEYS,
    get_project_config_path,
    get_user_config_path,
)

logger = Loggers.config()

#: The app name the files are chosen by when neither the constructor nor the
#: class names one (``AppSettingsMixin.app_name``'s default).
DEFAULT_APP_NAME = "agentic_cli"

#: The key each source files its ``permissions`` value under, by its rank
#: among the sources (0 is the highest priority): ``_RulesBySource``.
RULES_BY_SOURCE = "__by_source__"


class SettingsSourcesMixin:
    """The settings' sources, highest priority first.

    1. The constructor's arguments.
    2. The environment.
    3. The project file, ``./.{app_name}/settings.json``, if it exists:
       untrusted (a cloned repository can ship it), so only
       ``PROJECT_SETTABLE_KEYS`` are read from it.
    4. The user file, ``~/.{app_name}/settings.json``, if it exists.
    5. The env files (the constructor's ``_env_file``, else the class's
       ``env_file``): filtered like the project file when any of them is
       cwd-relative.

    ``app_name`` chooses those files, so it comes only from the constructor
    or the class: the environment, both files and the env files cannot set it,
    and one that names another app is reported (``settings_app_name_ignored``).
    The one exception is an app's own: one that enables pydantic-settings'
    ``cli_parse_args`` gets a command-line source above all of these (added
    by pydantic-settings, outside this method), which can set it.

    pydantic-settings' ``secrets_dir`` is never read; one configured (the
    class's, or the constructor's ``_secrets_dir``) is reported
    (``settings_secrets_dir_ignored``).

    A settings class (a subclass of pydantic-settings' ``BaseSettings``) gets
    these sources only when the mixins come before pydantic-settings' class in
    its bases; one that lists that class first, or between the mixins, is
    refused with a ``TypeError`` when it is defined, since its sources, and so
    every filter here, would be pydantic-settings' own. A class that defines
    ``settings_customise_sources`` itself is its author's choice and is
    accepted in any order.

    Each source's ``permissions`` is filed under its rank
    (``_RulesBySource``): pydantic-settings merges a dict setting across the
    sources key by key but replaces a list inside it, which would leave only
    one source's ``allow`` and ``deny`` lists. Filed apart, every source's
    rules reach ``WorkflowSettingsMixin``, which joins them.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not issubclass(cls, PydanticBaseSettings):
            return  # a mixin of mixins: not a settings class yet
        for base in cls.__mro__:
            if "settings_customise_sources" in vars(base):
                if base is PydanticBaseSettings:
                    raise TypeError(
                        f"{cls.__name__}: list the settings mixins (WorkflowSettingsMixin, "
                        "AppSettingsMixin) before pydantic-settings' BaseSettings in its "
                        "bases; in this order its sources are pydantic-settings' own, "
                        "and the trust filters on the project file and a cwd .env, and "
                        "the user settings file, would not be read."
                    )
                return

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[PydanticBaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """init, environment, project file, user file, env files (see the class)."""
        app_name = _files_app_name(settings_cls, init_settings)
        secrets_dir = getattr(file_secret_settings, "secrets_dir", None)
        if secrets_dir is not None:
            logger.warning("settings_secrets_dir_ignored", setting="secrets_dir", value=secrets_dir)

        def without_app_name(source: PydanticBaseSettingsSource, label: str):
            return _AppNameFilterSource(settings_cls, source, label, app_name)

        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            without_app_name(env_settings, "environment"),
        ]

        # The project file. Untrusted: a cloned repo can ship it.
        project_json = _get_json_config_source(
            settings_cls, get_project_config_path(app_name), untrusted=True, app_name=app_name
        )
        if project_json:
            sources.append(project_json)

        # The user file.
        user_json = _get_json_config_source(
            settings_cls, get_user_config_path(app_name), app_name=app_name
        )
        if user_json:
            sources.append(user_json)

        # dotenv: a cwd-relative env_file is an untrusted project source (a
        # cloned repo can ship ./.env), so filter it like project settings.json.
        # We inspect EVERY configured env_file (a str/Path or a list of them):
        # if ANY entry is cwd-relative after ~ expansion, the whole dotenv
        # source is filtered. DotEnvSettingsSource merges all files into one
        # dict, so we cannot filter per-file; over-filtering fails safe. An
        # absolute/user-level env_file (including a "~/..." path) and real
        # environment variables stay trusted. Consequence: secrets/keys placed
        # in a cwd .env are dropped — put them in real env vars or an
        # absolute/user-level file. The files judged are those the dotenv
        # source reads: the constructor's _env_file when given, else the
        # class's.
        env_file = getattr(
            dotenv_settings, "env_file", settings_cls.model_config.get("env_file")
        )
        if env_file is None:
            entries: list = []
        elif isinstance(env_file, (list, tuple)):
            entries = list(env_file)
        else:
            entries = [env_file]
        has_cwd_relative = any(
            e is not None and not Path(e).expanduser().is_absolute() for e in entries
        )
        if has_cwd_relative:
            sources.append(
                _AllowlistFilterSource(
                    settings_cls, without_app_name(dotenv_settings, "cwd .env"), "cwd .env"
                )
            )
        else:
            sources.append(without_app_name(dotenv_settings, "env file"))

        return tuple(
            _RulesBySource(settings_cls, source, rank) for rank, source in enumerate(sources)
        )


def _files_app_name(
    settings_cls: type[PydanticBaseSettings], init_settings: PydanticBaseSettingsSource
) -> str:
    """The app name the settings files are chosen by: the constructor's, else
    the class's default (a value or a ``default_factory``), else
    ``DEFAULT_APP_NAME``."""
    app_name = getattr(init_settings, "init_kwargs", {}).get("app_name")
    if not app_name:
        field_info = settings_cls.model_fields.get("app_name")
        if field_info is not None:
            default = field_info.default
            if (
                default is PydanticUndefined
                and field_info.default_factory is not None
                # (pydantic 2.10+: a factory may take the other fields' values)
                and not getattr(field_info, "default_factory_takes_validated_data", False)
            ):
                default = field_info.default_factory()
            if isinstance(default, str):
                app_name = default
    return app_name or DEFAULT_APP_NAME


class _FilterSource(PydanticBaseSettingsSource):
    """A source that filters what another source read (``__call__``).

    pydantic-settings hands each source the values read so far; a filter
    passes them on, so the source it wraps sees what it would unwrapped.
    """

    def __init__(
        self, settings_cls: type[PydanticBaseSettings], inner: PydanticBaseSettingsSource
    ) -> None:
        super().__init__(settings_cls)
        self._inner = inner

    def _set_current_state(self, state: dict[str, Any]) -> None:
        super()._set_current_state(state)
        self._inner._set_current_state(state)

    def _set_settings_sources_data(self, states: dict[str, dict[str, Any]]) -> None:
        super()._set_settings_sources_data(states)
        self._inner._set_settings_sources_data(states)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Unused: __call__ is overridden to filter the inner source's output.
        return None, field_name, False


class _RulesBySource(_FilterSource):
    """Pass a source's values on, its ``permissions`` filed under its rank.

    As ``{RULES_BY_SOURCE: {"<rank>": value}}``, the value survives
    pydantic-settings' merge beside every other source's (the ranks differ),
    where its lists would replace theirs. The source keeps its own name, the
    one pydantic-settings files its values under for the other sources.
    """

    def __init__(
        self, settings_cls: type[PydanticBaseSettings], inner: PydanticBaseSettingsSource, rank: int
    ) -> None:
        super().__init__(settings_cls, inner)
        self._rank = rank
        self.__name__ = getattr(inner, "__name__", type(inner).__name__)

    def __call__(self) -> dict[str, Any]:
        values = self._inner()
        if "permissions" not in values:
            return values
        filed = {RULES_BY_SOURCE: {str(self._rank): values["permissions"]}}
        return {**values, "permissions": filed}


class _AppNameFilterSource(_FilterSource):
    """Wrap a source, leaving ``app_name`` out of what it read.

    The files were chosen by the constructor's or the class's app name; a
    source that names another would have the settings read one app's files
    and save to another's. A different name is reported once per source; the
    same name is no conflict.
    """

    def __init__(
        self,
        settings_cls: type[PydanticBaseSettings],
        inner: PydanticBaseSettingsSource,
        label: str,
        app_name: str,
    ) -> None:
        super().__init__(settings_cls, inner)
        self._label = label
        self._app_name = app_name

    def __call__(self) -> dict[str, Any]:
        values = dict(self._inner())
        if "app_name" in values:
            given = values.pop("app_name")
            if given != self._app_name:
                logger.warning(
                    "settings_app_name_ignored",
                    source=self._label,
                    value=given,
                    app_name=self._app_name,
                )
        return values


class _AllowlistFilterSource(_FilterSource):
    """Wrap an untrusted settings source, keeping only allowlisted keys.

    Applied to the project ``settings.json`` and a cwd-relative ``.env``. Any
    non-allowlisted key is dropped and logged (one warning per key) so a cloned
    repo cannot flip a security boundary. Drops (never raises) a non-allowlisted
    key, so a repo cannot flip a boundary by *adding* keys. (Malformed JSON or a
    bad-typed allowlisted value is still rejected upstream, as before P0-1 —
    this narrows, not removes, that pre-existing surface.)

    A key that names no setting at all (neither a field nor an alias) could
    set nothing anyway, so it is dropped without a log: a project file saved
    by an older version still holds settings since removed (0.6's
    ``context_window_*``), and settings are built before logging is
    configured, when structlog prints even a debug log above the UI. The
    ``passed_over`` keys are dropped without a log too: the project file's
    ``permissions`` section (``PROJECT_PRESERVED_KEYS``) sets no setting, and
    the permission engine reads its deny rules from the file itself (no one
    reads one from a ``.env``, which is still reported).
    """

    def __init__(
        self,
        settings_cls: type[PydanticBaseSettings],
        inner: PydanticBaseSettingsSource,
        label: str,
        passed_over: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(settings_cls, inner)
        self._label = label
        self._passed_over = passed_over

    def __call__(self) -> dict[str, Any]:
        kept: dict[str, Any] = {}
        settings = _accepted_input_names(self.settings_cls)
        for key, value in self._inner().items():
            if key in PROJECT_SETTABLE_KEYS:
                kept[key] = value
            elif key in self._passed_over:
                continue  # read by others (the project file's deny rules)
            elif key in settings:
                logger.warning(
                    "untrusted_project_setting_ignored", key=key, source=self._label
                )
        return kept


def _accepted_input_names(settings_cls: type[PydanticBaseSettings]) -> set[str]:
    """Every name the model accepts for a field: its own name and any aliases."""
    names: set[str] = set()
    for field_name, field in settings_cls.model_fields.items():
        names.add(field_name)
        alias = field.validation_alias
        if isinstance(alias, str):
            names.add(alias)
        elif isinstance(alias, AliasChoices):
            names.update(c for c in alias.choices if isinstance(c, str))
        if isinstance(field.alias, str):
            names.add(field.alias)
    return names


def _get_json_config_source(
    settings_cls: type[PydanticBaseSettings],
    json_file: Path,
    *,
    app_name: str,
    untrusted: bool = False,
) -> PydanticBaseSettingsSource | None:
    """Create a JSON config source if the file exists.

    Args:
        settings_cls: The settings class
        json_file: Path to JSON config file
        untrusted: When True (the project file), keep only allowlisted
            (non-security) keys so a cloned workspace cannot flip a security
            boundary.
        app_name: The app name the file was chosen by: the file's own
            ``app_name`` is left out, and reported when it differs.

    Returns:
        JsonConfigSettingsSource if file exists, None otherwise
    """
    if not json_file.exists():
        return None

    try:
        from pydantic_settings import JsonConfigSettingsSource
    except ImportError:
        # Older pydantic-settings without JsonConfigSettingsSource
        return None

    label = "project settings.json" if untrusted else "user settings.json"
    source = _AppNameFilterSource(
        settings_cls, JsonConfigSettingsSource(settings_cls, json_file=json_file), label, app_name
    )
    if not untrusted:
        return source

    # Untrusted (the project file): keep only allowlisted keys so a cloned
    # workspace cannot flip a security boundary.
    return _AllowlistFilterSource(
        settings_cls,
        source,
        label,
        # The permission engine reads this file's deny rules itself.
        passed_over=PROJECT_PRESERVED_KEYS,
    )
