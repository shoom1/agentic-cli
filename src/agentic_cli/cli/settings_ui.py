"""The ``/settings`` dialog's fields and how a Save takes effect (``SettingsUI``).

``SettingsUI`` builds the dialog's fields from the keys the app lists
(``get_ui_setting_keys``): a model picker for the default model and for each
role, from the model catalog, and each other key from its settings field
(``settings_introspection``). It applies a Save, all or nothing, each change
taking effect as ``settings_persistence`` declares for its field
(``REBUILD_KEYS``, ``NEW_CONVERSATION_KEYS``, ``RESTART_KEYS``,
``LIVE_KEYS``), and saves the keys a Save changed, split by trust.

It acts on the app it is given and keeps nothing of its own: the app's
settings, workflow controller, session and conversation are read where they
are used. ``BaseCLIApp``'s ``_build_ui_items``, ``apply_settings`` and
``save_settings`` each delegate to one, so ``/settings``
(``SettingsCommand``) still goes through the app's methods, and an app's
override of one of them, or of a hook called from here
(``get_ui_setting_keys``, ``get_theme``, ``start_new_session``), still
applies.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic_cli.cli.conversation import Hold, Place, Switch
from agentic_cli.cli.message_processor import _says_so_when_stopped
from agentic_cli.cli.workflow_controller import WorkflowController, WorkflowState
from agentic_cli.config import MODEL_ROLE_KEY_PREFIX, BaseSettings
from agentic_cli.logging import Loggers
from agentic_cli.settings_persistence import (
    HAND_EDITED_KEYS,
    NEW_CONVERSATION_KEYS,
    PROJECT_SETTABLE_KEYS,
    REBUILD_KEYS,
    RESTART_KEYS,
    USER_PREFERENCE_KEYS,
)
from agentic_cli.workflow.factory import check_backend, orchestrator_of
from agentic_cli.workflow.settings import OrchestratorType

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from thinking_prompt import Field, ThinkingPromptSession

    from agentic_cli.cli.app import BaseCLIApp
    from agentic_cli.models import CatalogView, ModelRef, Provider, ResolvedModel, RoleMap
    from agentic_cli.settings_persistence import SettingsSaveResult

logger = Loggers.cli()

# Synthetic dialog keys → the settings field their setter actually writes.
# "model" is not a field; update_setting() routes it via set_model() to
# default_model. Every "model_role:<role>" key (a role's model picker) writes
# model_roles, through set_model_role() (see _ui_key_target_field). A
# synthetic key without an entry here is treated as writing a field of the
# same name.
_UI_KEY_TARGET_FIELDS = {"model": "default_model"}


def _ui_key_target_field(key: str) -> str:
    """The settings field a ``/settings`` dialog key writes."""
    if key.startswith(MODEL_ROLE_KEY_PREFIX):
        return "model_roles"
    return _UI_KEY_TARGET_FIELDS.get(key, key)


def _is_model_picker(key: str) -> bool:
    return key == "model" or key.startswith(MODEL_ROLE_KEY_PREFIX)


def _setting_label(key: str, settings: BaseSettings) -> str:
    """How messages name a dialog key, as the dialog labels it: by the field's
    title (``UI Theme``, ``Max Retry Attempts``), ``Fast Model`` for a role's
    model picker, and otherwise by the key title-cased (``Model``)."""
    if key.startswith(MODEL_ROLE_KEY_PREFIX):
        return _role_picker_label(key)
    field = type(settings).model_fields.get(key)
    if field is not None and field.title:
        return field.title
    return key.replace("_", " ").title()


def _role_picker_label(key: str) -> str:
    """A role's model picker, ``model_role:fast``, is ``Fast Model``."""
    return f"{key.removeprefix(MODEL_ROLE_KEY_PREFIX).title()} Model"


@dataclass(frozen=True)
class _ModelChoices:
    """What the ``/settings`` model pickers offer and show, from one catalog.

    Attributes:
        options: The runnable references every model picker offers.
        current: Each role's model now, by role (``default`` included); None
            when nothing can fill it.
        thinking: What the thinking effort means for the default model, as
            ``high → provider:model: <label>``; None when it cannot be said.
        left_out: Why the list leaves out each model the settings name that
            it leaves out (``ModelCatalog.not_offered``), by reference.
    """

    options: list[str]
    current: Mapping[str, str | None]
    thinking: str | None
    left_out: Mapping[str, str] = dataclasses.field(default_factory=dict)


# === Deprecated orchestrators ===

# What the CLI tells a user of the LangGraph backend, deprecated in 0.7.0: once
# at startup, and again on a /settings switch to it. The manager's own
# DeprecationWarning is for library users; Python's default filters keep it off
# the terminal, so this is what a CLI user reads.
_LANGGRAPH_DEPRECATION = (
    "The LangGraph backend is deprecated and will be removed in 0.8.0. Set "
    "orchestrator to deepagents (pip install 'agentic-cli[deepagents]'); "
    "sessions saved by LangGraph do not carry over."
)


def _runs_langgraph(settings: BaseSettings) -> bool:
    """Whether ``settings`` choose the LangGraph backend.

    Read as the factory reads it: an ``OrchestratorType``, or its plain value
    on a settings class that declares ``orchestrator: str``.
    """
    return getattr(settings, "orchestrator", None) == OrchestratorType.LANGGRAPH


def _warn_if_langgraph(settings: BaseSettings, session: ThinkingPromptSession) -> None:
    """Warn, through the session, when ``settings`` choose the LangGraph backend."""
    if _runs_langgraph(settings):
        session.add_warning(_LANGGRAPH_DEPRECATION)


# === When a Save takes effect ===

#: How ``/settings`` reports a change of a key read only as the app starts
#: (``settings_persistence.RESTART_KEYS``).
_AT_NEXT_START = "(takes effect at the next start)"

#: How ``/settings`` reports a change of the backend of new conversations
#: (``settings_persistence.NEW_CONVERSATION_KEYS``) that the open one keeps.
_AT_NEXT_NEW = "(applies to the next /new)"

#: The keys besides the rebuild keys whose change waits for a start in
#: progress, as a rebuild does: whether an orchestrator change starts a new
#: conversation, or moves the run's, needs that start's outcome. The session
#: store is fixed for the run as its first start begins
#: (``WorkflowController.store``), and ``log_level`` and ``log_format`` are
#: read by the app's constructor only: a change of them does not wait.
_READ_BY_THE_START = frozenset({"orchestrator"})

#: What ``_move_here`` returns when it finds, under the turn lock, that the
#: move no longer applies, decided as for a running workflow: something of
#: the conversation is stored by now, or the next start no longer builds
#: afresh (it stays; the change applies to the next ``/new``) ...
_STAYS = "stays"
#: ... or the workflow came up meanwhile, nothing of it stored (a new
#: conversation on the new backend, ``start_new_session``).
_STARTS_ANEW = "starts anew"


class SettingsUI:
    """``/settings`` on ``app``: the dialog's fields, a Save's effect, and its save.

    Args:
        app: The app whose settings, workflow controller, session and
            conversation it acts on, and whose hooks it calls.
    """

    def __init__(self, app: "BaseCLIApp") -> None:
        self.app = app

    # --- the dialog's fields ---------------------------------------------------

    def fields(self) -> list[Field]:
        """The settings dialog's fields (``BaseCLIApp._build_ui_items``).

        The app's ``get_ui_setting_keys()`` says which keys to show; each is
        converted to a form field from its settings field's type annotation
        and metadata. The model pickers come from the model catalog
        (``model_choices``), and the thinking effort's description says what
        it means for the default model. thinking_prompt's fields are frozen,
        so each is built complete.

        Returns:
            The form fields, sorted by ``ui_order``.
        """
        from agentic_cli.cli.settings_introspection import field_to_ui_item, get_ui_order

        app = self.app
        keys = app.get_ui_setting_keys()
        roles = [
            key.removeprefix(MODEL_ROLE_KEY_PREFIX)
            for key in keys
            if key.startswith(MODEL_ROLE_KEY_PREFIX)
        ]
        choices: _ModelChoices | None = None  # one catalog for every picker
        items: list[tuple[int, Field]] = []
        shown = (PROJECT_SETTABLE_KEYS | USER_PREFERENCE_KEYS) - HAND_EDITED_KEYS

        for key in keys:
            # Project-scope guard: the dialog may only expose settings the
            # project file can persist (PROJECT_SETTABLE_KEYS), and the user's
            # preferences (USER_PREFERENCE_KEYS), which are meant to apply in
            # every project. Any other user-scoped key would save to the user
            # ~/.{app}/settings.json and leak the change across all projects
            # of the app. Nor does it expose a hand-edited key
            # (HAND_EDITED_KEYS: models, and model_providers, which holds the
            # addresses a key is sent to): no save writes one, so an edit
            # would be gone at the next start.
            target = _ui_key_target_field(key)
            if target not in shown:
                if target in HAND_EDITED_KEYS:
                    logger.warning("hand_edited_setting_excluded_from_ui", key=key)
                else:
                    logger.warning("user_scoped_setting_excluded_from_ui", key=key)
                continue

            if _is_model_picker(key):
                choices = choices or self.model_choices(roles)
                if not choices.options:
                    continue  # no provider to pick from
                # The default model first, then the roles in the app's order.
                items.append((10 if key == "model" else 11, _model_picker(key, choices)))
                continue

            # Handle regular fields from settings
            # Access model_fields from class to avoid Pydantic deprecation warning
            settings_cls = type(app._settings)
            if key in settings_cls.model_fields:
                field = settings_cls.model_fields[key]
                current = getattr(app._settings, key, None)
                description = None  # the field's own
                if key == "thinking_effort":
                    choices = choices or self.model_choices(roles)
                    description = choices.thinking or None
                item = field_to_ui_item(key, field, current, description=description)
                order = get_ui_order(field)
                items.append((order, item))

        # Sort by order and return items only
        return [item for _, item in sorted(items, key=lambda x: x[0])]

    def model_choices(self, roles: "Sequence[str]" = ()) -> _ModelChoices:
        """The model pickers' options, each role's model, and the thinking description.

        The options are the list (``ModelCatalog.offered``), with each role's
        model (``model_services.role_models``, as ``/models`` has it). Once
        the workflow is ready, from its catalog (listings
        included) and its default model; before that, from the catalog it
        would load, unloaded (the entries and the shipped table), and the
        default model the settings name (or compute). A role shows the model
        it maps, else the one it resolves to (an app's own role runs on the
        default model).

        Args:
            roles: The roles other than ``default`` whose model to show.
        """
        from agentic_cli.models import BUILTIN_ROLES
        from agentic_cli.workflow import model_services

        app = self.app
        settings = app._settings
        try:
            role_map = model_services.role_map_from_settings(settings)
        except ValueError:
            return _ModelChoices(options=[], current={}, thinking=None)
        providers = model_services.providers_from_settings(settings)
        controller = app._workflow_controller
        workflow = controller.workflow if controller.is_ready else None
        if workflow is not None and workflow.catalog is not None:
            catalog, default = workflow.catalog, workflow.model
        else:
            # Not loaded: the entries, what the settings and agents name, and
            # the shipped table.
            catalog = model_services.build_catalog(
                settings, providers, agent_configs=app.agent_configs
            )
            default = _resolved_role(role_map, "default", catalog, providers)

        current: dict[str, str | None] = {"default": default}
        for role in roles:
            mapped = role_map.spec_for(role)
            if mapped is not None:
                current[role] = str(mapped)
            elif role in BUILTIN_ROLES:
                current[role] = _resolved_role(role_map, role, catalog, providers)
            else:
                current[role] = default
        # Every role's model, whichever pickers the app shows: /models offers
        # the same list.
        role_models = model_services.role_models(settings, catalog, providers, default)
        not_offered = getattr(catalog, "not_offered", None)  # an app's own catalog may lack it
        return _ModelChoices(
            options=catalog.offered(extra=list(role_models)),
            current=current,
            thinking=self._thinking_meaning(default, catalog, providers),
            left_out=not_offered() if not_offered is not None else {},
        )

    def _thinking_meaning(
        self, model: str | None, catalog: "CatalogView", providers: "Mapping[str, Provider]"
    ) -> str | None:
        """``high → anthropic:claude-sonnet-5-5: adaptive, effort high``: the
        thinking effort, resolved on ``model`` as the workflow resolves it."""
        from agentic_cli.models import ModelResolutionError, RoleMap, resolve_model

        if model is None:
            return None
        effort = self.app._settings.thinking_effort
        try:
            resolved = resolve_model(
                agent=None,
                spec=model,
                generation=None,
                thinking_effort=effort,
                roles=RoleMap(default_model=None, model_roles={}),  # a reference needs none
                catalog=catalog,
                providers=providers,
            )
        except ModelResolutionError:
            return None
        meaning = resolved.thinking.label if resolved.thinking else "ignored (no thinking knob)"
        return f"{effort} → {resolved.ref}: {meaning}"

    # --- a Save ----------------------------------------------------------------

    async def apply(self, changes: dict[str, Any]) -> bool:
        """Apply a Save's ``changes``, all or nothing (``BaseCLIApp.apply_settings``,
        whose docstring says when each change takes effect).

        Ctrl+C stops it wherever it waits, as it stops a turn, and
        "Operation cancelled..." is said once: by ``_says_so_when_stopped``
        around each wait of its own (a start in progress, the turn lock, the
        move, the rebuild), and by the new conversation's own
        (``start_new_session``, as ``/new`` says it) around that one.

        Returns:
            True if the changes were applied, False if they were rolled back.
        """
        app = self.app
        if not changes:
            app.session.add_message("system", "No changes made.")
            return True

        # When each change takes effect (settings_persistence): a key the
        # workflow reads as it builds its agents and models rebuilds it; the
        # backend of new conversations starts one on it, or waits for /new;
        # one read only as the app starts is said to take effect at the next
        # start; the rest apply at once, read where they are used.
        needs_reinit = any(_ui_key_target_field(key) in REBUILD_KEYS for key in changes)
        at_next_start = {key for key in changes if _ui_key_target_field(key) in RESTART_KEYS}

        read_by_the_start = needs_reinit or any(
            _ui_key_target_field(key) in _READ_BY_THE_START for key in changes
        )

        settings = app._settings
        controller = app._workflow_controller
        if read_by_the_start and controller.state is WorkflowState.INITIALIZING:
            # The start in progress read the settings as they are (its models
            # first): wait for it, as a turn does, so the rebuild below runs
            # on the new values, and whether an orchestrator change starts a
            # new conversation or moves the open one follows its outcome.
            # Ctrl+C stops the wait before anything changed. A start that
            # fails leaves the workflow down, and the change is applied for
            # the next start, as for any failed workflow.
            async with _says_so_when_stopped(app.session):
                await controller.ensure_initialized(app.session)

        new_model = changes.get("model")
        snapshot = _snapshot_settings(settings)
        # The backend the conversation runs on, before the change.
        backend_before = controller.backend if "orchestrator" in changes else None
        was_langgraph = _runs_langgraph(settings)
        theme_before = settings.ui_theme

        for key, value in changes.items():
            try:
                settings.update_setting(key, value)
            except ValueError as e:
                _restore_settings(settings, snapshot)
                label = _setting_label(key, settings)
                app.session.add_error(f"Failed to set {label}: {e}")
                return False

        # The backend new conversations run on, and whether it is another than
        # the open conversation's. A conversation never changes backend once
        # anything of it is stored (ConversationCore.fresh), read under the
        # turn lock: after a startup resume or an app's task that holds it.
        # With nothing stored, a running workflow leaves it for a new
        # conversation there at once (no rebuild first: the new manager is
        # built on the new values), and a workflow that is down, whose next
        # start builds afresh, has that start run it there (below); else the
        # change waits for /new, and the conversation stays where it is.
        target = orchestrator_of(settings).value
        if "orchestrator" in changes:
            try:  # no backend that cannot run is taken, for now or the next /new
                check_backend(target)
            except ImportError as error:
                _restore_settings(settings, snapshot)
                label = _setting_label("orchestrator", settings)
                app.session.add_error(f"Failed to set {label}: {_first_line(error)}")
                return False
        moves_backend = "orchestrator" in changes and target != backend_before
        movable = False
        if moves_backend:
            try:
                async with _says_so_when_stopped(app.session):
                    async with app._turn_lock:
                        movable = app._conversation.fresh
            except asyncio.CancelledError:  # waiting for it: nothing changes
                _restore_settings(settings, snapshot)
                raise
        starts_anew = movable and controller.is_ready
        moves_here = movable and not controller.is_ready and controller.next_start_builds
        waits_for_new = {
            key for key in changes if _ui_key_target_field(key) in NEW_CONVERSATION_KEYS
        }
        at_next_new = waits_for_new if moves_backend and not (starts_anew or moves_here) else set()

        # The workflow is down, its next start builds afresh, and nothing of
        # the conversation is stored: that start runs it on the new backend,
        # where it is held (from its first write, when nothing holds it yet).
        # Decided again under the turn lock, which the move waits for: what
        # it finds then is decided as for a running workflow.
        if moves_here:
            place = dataclasses.replace(app._place(), backend=target)
            try:
                async with _says_so_when_stopped(app.session):
                    moved = await self._move_here(place)
            except asyncio.CancelledError:  # it stays where it was
                _restore_settings(settings, snapshot)
                raise
            except Exception as error:
                _restore_settings(settings, snapshot)
                app.session.add_error(
                    f"Could not set Orchestrator: {str(error) or type(error).__name__}. "
                    "Settings were not changed."
                )
                return False
            if moved == _STAYS:  # a turn wrote to it meanwhile, say: it stays
                at_next_new = waits_for_new
            elif moved == _STARTS_ANEW:  # the workflow came up meanwhile
                starts_anew = True
            elif not moved.made:  # open there in another process: nothing changes
                _restore_settings(settings, snapshot)
                app.session.add_error(
                    f"Could not set Orchestrator: conversation '{app._session_id}' is open "
                    f"in another process ({moved.holder}) on {place.backend} ({place.store}). "
                    "Settings were not changed."
                )
                return False

        # Reinitialize workflow if needed
        switched: list[str] = []
        interrupted: asyncio.CancelledError | None = None
        if needs_reinit and controller.is_ready and not starts_anew:
            previous = controller.workflow
            previous_model = previous.model
            previous_models = dict(getattr(previous, "agent_models", {}))
            try:
                async with _says_so_when_stopped(app.session):
                    await controller.reinitialize(model=new_model)
            except asyncio.CancelledError:
                # Ctrl+C stopped /settings mid-restart, which is said as
                # wherever else the Save waits. Nothing of the Save stays, as
                # on a failure.
                _restore_settings(settings, snapshot)
                if not controller.is_ready:
                    try:
                        await controller.reinitialize(model=previous_model)
                    except Exception as restore_error:
                        # Down, on the old model: the next request retries,
                        # and reports a failure then.
                        logger.warning("settings_restore_failed", error=str(restore_error))
                raise
            except Exception as e:
                _restore_settings(settings, snapshot)
                app.session.add_error(
                    f"Failed to reinitialize workflow: {e}. Settings were not changed."
                )
                # A failed reinitialize leaves the manager down, holding the
                # new model. Bring it back on the model it had.
                if not controller.is_ready:
                    try:
                        await controller.reinitialize(model=previous_model)
                    except Exception as restore_error:
                        app.session.add_error(
                            f"Could not restart the workflow on {previous_model}: "
                            f"{restore_error}"
                        )
                return False
            switched = _provider_switches(previous_models, getattr(previous, "agent_models", {}))
            _show_new_context(controller, app.session)

        if starts_anew:
            try:
                # The app's own /new (an override applies), on the settings'
                # orchestrator, which names the new backend now.
                await app.start_new_session()
            except asyncio.CancelledError as cancel:
                if controller.backend != target:
                    # Stopped before the new backend ran: nothing of the Save
                    # stays ("Operation cancelled...", as /new says it).
                    _restore_settings(settings, snapshot)
                    raise
                # It runs, and its conversation is open: too late to undo. The
                # change is reported (and saved), then the cancel goes on.
                interrupted = cancel
            except Exception as error:
                _restore_settings(settings, snapshot)
                reason = _first_line(error)
                app.session.add_error(
                    f"Could not start a new conversation on {target}: {reason}. "
                    "Settings were not changed."
                )
                return False

        # Report changes
        for key, value in changes.items():
            line = f"{_setting_label(key, settings)}: {value}"
            if key in at_next_start:
                line += f" {_AT_NEXT_START}"
            elif key in at_next_new:
                line += f" {_AT_NEXT_NEW}"
            app.session.add_success(line)
        for notice in switched:
            app.session.add_message("system", notice)
        if not was_langgraph and not starts_anew:  # a switch to it (the new
            _warn_if_langgraph(settings, app.session)  # conversation said it)
        if settings.ui_theme != theme_before:
            # Repaint: text printed before would keep the old theme's colors.
            # The transcript is printed again, the lines above included.
            theme = app.get_theme(settings.ui_theme)
            app.session.set_theme(theme, repaint=True)
        if interrupted is not None:
            raise interrupted
        return True

    async def _move_here(self, place: Place) -> Switch | str:
        """Have the next start run the conversation on ``place``'s backend.

        Decided again under the turn lock, which it waits for (an app's task
        may hold it, and run a turn or start the workflow meanwhile): it
        moves only while the workflow is down, its next start builds a
        manager afresh, and nothing of the conversation is stored
        (``ConversationCore.fresh``): one with anything stored never changes
        backend. Else it returns what it found, decided as for a running
        workflow: ``_STAYS`` (something of it is stored, or the next start
        revives a manager kept on its backend: the change waits for
        ``/new``) or ``_STARTS_ANEW`` (the workflow came up, nothing of it
        stored: the caller starts a new conversation there, as on a running
        workflow; this one stays where it ran).

        One this run made is held from its first write, so mostly there is
        no hold to move; when one holds it (a line typed while the workflow
        was down), its lock is taken there. The controller is told the
        backend its next start builds (``switch_backend``), then the switch
        to the same id there is made at once (no workflow runs: no await),
        letting go of where it was held. A cancel while it waits (for the
        turn lock, or for the controller) leaves it where it was, on the
        backend it had.

        Returns:
            How the switch ended: made, or refused (another process has it
            open there, and nothing changed); else ``_STAYS`` or
            ``_STARTS_ANEW`` (nothing changed).
        """
        app = self.app
        core, controller = app._conversation, app._workflow_controller
        async with app._turn_lock:
            if not core.fresh:
                return _STAYS
            if controller.is_ready:
                return _STARTS_ANEW
            if not controller.next_start_builds:
                return _STAYS
            if core.deferred:  # nothing holds it yet: none to move
                hold: Hold | int = Hold(app._session_id, place, None)
            else:
                hold = core.acquire(app._session_id, place)
            if isinstance(hold, int):
                return Switch(made=False, holder=hold)
            try:
                await controller.switch_backend(place.backend)
            except BaseException:
                core.abandon(hold)
                raise
            return await core.switch(
                app._session_id, app._switch_parts(), place=place, propose=False, hold=hold
            )

    async def save(self, keys: "Iterable[str]") -> "SettingsSaveResult":
        """Save the settings ``keys`` name, split by trust
        (``BaseCLIApp.save_settings``), on a worker thread.

        Raises:
            TypeError: ``keys`` is None.
        """
        from agentic_cli.settings_persistence import SettingsPersistence

        settings = self.app._settings
        persistence = SettingsPersistence(settings.app_name)
        return await asyncio.to_thread(persistence.save, settings, keys=keys)


def _first_line(error: BaseException) -> str:
    """What a line says of ``error``: its first line (a backend's missing extra
    adds the import's own error after it), or its type when it has no words."""
    return (str(error) or type(error).__name__).split("\n")[0]


def _model_picker(key: str, choices: _ModelChoices) -> Field:
    """The dialog field of one model picker: ``model`` or ``model_role:<role>``.

    It offers the list, and shows its role's model now. When the list leaves
    that model out (its provider is not defined, has no key, or does not
    serve it), the picker still offers it, first, as ``<ref> (current; not
    offered)``, and its description starts with why: the field's value is
    then that option, so a Save that leaves the picker as it opened, or picks
    it again, changes nothing (``SettingsCommand`` passes on only a value
    that differs from the field's own), and a user who moved off it can come
    back to it. The reason is in the description, not the option: the
    option would not fit the dialog's row beside its label.
    """
    from pydantic.fields import FieldInfo

    from agentic_cli.cli.settings_introspection import field_to_ui_item

    if key == "model":
        role, title = "default", "Default Model"
        description = "Agents without a model of their own run on it"
    else:
        role = key.removeprefix(MODEL_ROLE_KEY_PREFIX)
        title = _role_picker_label(key)
        description = f"Agents whose model is {role!r} run on it"
    current, options = choices.current.get(role), choices.options
    if current is not None and current not in options:
        why = choices.left_out.get(current)
        if why is not None:
            description = f"Not offered ({why}); {description[0].lower()}{description[1:]}"
        current = _NOT_OFFERED.format(current)
        options = [current, *options]
    field = FieldInfo(default=None, title=title, description=description)
    return field_to_ui_item(
        key=key,
        field=field,
        current_value=current,
        dynamic_options=options,
        max_width=_MODEL_PICKER_MAX_WIDTH,
    )


#: How a model picker offers its role's model when the list leaves it out.
_NOT_OFFERED = "{} (current; not offered)"


# The widest a model picker's dropdown grows: a reference and its padding. A
# dropdown's default width cuts a reference to its first 14 characters
# ("google:gemini-"), so the models could not be told apart. SettingsCommand's
# dialog leaves room for it beside the longest label.
_MODEL_PICKER_MAX_WIDTH = 48


def _resolved_role(
    roles: "RoleMap", role: str, catalog: "CatalogView", providers: "Mapping[str, Provider]"
) -> str | None:
    """The reference ``role`` resolves to, or None when nothing can fill it."""
    from agentic_cli.models import ModelResolutionError

    try:
        return str(roles.resolve(role, catalog=catalog, providers=providers))
    except ModelResolutionError:
        return None


def _show_new_context(controller: WorkflowController, session: Any) -> None:
    """Show ``ctx`` against the rebuilt agents' trigger now, not after the next call.

    A reinitialize may move the main agent's trigger (a model of another
    window, ``context_summarize_at``, ``context_max_tokens``), and the last
    prompt may already be past the new one; a switch of conversation starts
    one whose trigger the workflow knows. The usage tracker takes the
    trigger and the window flag from ``workflow.context_budget()``: none when
    the manager manages no context (``KeyError``), has no such method, or
    is not ready (it failed to start; the status bar then says so). The
    status bar is then drawn again.

    Args:
        controller: The app's workflow controller, after the reinitialize.
        session: The UI session whose status bar is drawn.
    """
    tracker = controller.usage_tracker
    if tracker is not None:
        budget = None
        # A workflow that is not ready (it failed to start) has no budget,
        # and the controller hands out none: the status bar says why.
        if controller.is_ready:
            context_budget = getattr(controller.workflow, "context_budget", None)
            try:
                budget = context_budget() if context_budget is not None else None
            except (KeyError, RuntimeError):
                budget = None
        tracker.context_trigger = budget.trigger if budget is not None else None
        tracker.window_known = budget.window_known if budget is not None else None
    controller.update_status_bar(session)


def _provider_switches(
    before: Mapping[str, ResolvedModel], after: Mapping[str, ResolvedModel]
) -> list[str]:
    """What each switch to another provider's model costs, one line per switch.

    The conversation is kept, but no model is sent another provider's
    reasoning (both backends leave it out), and the new provider has none of
    the old one's prompt cache. A switch within one provider loses neither.
    """
    agents: dict[tuple[ModelRef, ModelRef], list[str]] = {}
    for name, model in after.items():
        old = before.get(name)
        if old is not None and old.record.kind is not model.record.kind:
            agents.setdefault((old.ref, model.ref), []).append(name)
    return [
        f"{', '.join(names)}: {old} → {new}. The conversation is kept, but earlier "
        f"reasoning from {old.provider} is not sent to the new model, and the prompt "
        "cache starts over."
        for (old, new), names in agents.items()
    ]


def _snapshot_settings(settings: BaseSettings) -> tuple[dict[str, Any], set[str]]:
    """Capture every field value (and which were set) so a change can be undone."""
    return dict(settings.__dict__), set(settings.__pydantic_fields_set__)


def _restore_settings(
    settings: BaseSettings, snapshot: tuple[dict[str, Any], set[str]]
) -> None:
    """Put back the values captured by :func:`_snapshot_settings`.

    Written directly, like the dedicated setters do: they were valid when
    captured, and revalidating could itself fail.
    """
    values, fields_set = snapshot
    for key, value in values.items():
        if settings.__dict__.get(key) is not value:
            object.__setattr__(settings, key, value)
    object.__setattr__(settings, "__pydantic_fields_set__", fields_set)
