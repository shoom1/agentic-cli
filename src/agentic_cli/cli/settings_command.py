"""Settings command for configuring application settings.

Uses introspection to automatically generate form fields from Pydantic fields,
shown in a thinking_prompt Dialog with Save and Cancel buttons. Apps can
customize which settings appear by overriding get_ui_setting_keys().
"""

import inspect
from typing import TYPE_CHECKING

from thinking_prompt import ButtonConfig, Dialog

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp


class SettingsCommand(Command):
    """Open settings dialog to configure application settings.

    Uses the app's get_ui_setting_keys() method to determine which settings
    to display, and _build_ui_items() to generate the appropriate form fields
    based on field types and metadata. The dialog's values hold every field;
    the ones that differ from the field's starting value are the changes
    passed to apply_settings().
    """

    def __init__(self) -> None:
        super().__init__(
            name="settings",
            description="Configure application settings",
            aliases=["set", "config"],
            category=CommandCategory.SETTINGS,
        )

    async def execute(self, args: str, app: "BaseCLIApp") -> None:
        """Open interactive settings dialog."""
        # Build form fields using introspection
        fields = app._build_ui_items()

        if not fields:
            app.session.add_error(
                "No settings available. Please configure API keys first."
            )
            return

        dialog = Dialog(
            title="Settings",
            fields=fields,
            buttons=[ButtonConfig("Save", result=True), ButtonConfig("Cancel")],
            # Room for a model picker's row: its label and a whole reference.
            width=80,
        )
        # Save returns True; Cancel and Escape return None.
        if not await app.session.show_dialog(dialog):
            app.session.add_message("system", _UNCHANGED)
            return

        # A field's default is the value it started at: what differs from it
        # is what the user changed.
        values = dialog.values
        changes = {f.key: values[f.key] for f in fields if values[f.key] != f.default}
        if not changes:
            app.session.add_message("system", _UNCHANGED)
            return

        # Apply, then persist exactly the fields that changed. Saving the whole
        # live object also wrote values that came from the environment or were
        # derived at startup, making a one-run override permanent.
        before = app.settings.model_dump()
        if await app.apply_settings(changes) is False:
            return  # rolled back (an override returning None is still diffed)
        after = app.settings.model_dump()
        changed = {k for k, v in after.items() if before.get(k) != v}
        if not changed:
            return
        # A dict the dialog edits entry by entry saves only the entries that
        # changed: its live value merges entries from the user file, the
        # environment and the app's defaults, none of which belongs in the
        # project file a teammate pulls.
        for key in _SAVED_BY_ENTRY & changed:
            changed.discard(key)
            old, new = before.get(key) or {}, after.get(key) or {}
            changed |= {f"{key}.{entry}" for entry, value in new.items() if old.get(entry) != value}

        try:
            if _accepts_keyword(app.save_settings, "keys"):
                saved = await app.save_settings(keys=changed)
            else:  # an app override predating keys=: its own full save
                saved = await app.save_settings()
            app.session.add_success(_saved_message(saved))
        except Exception as e:
            app.session.add_warning(f"Settings applied but not saved: {e}")


# Dict settings the dialog edits one entry at a time (a role's model picker).
_SAVED_BY_ENTRY = {"model_roles"}

# What a dialog closed without a change says: Cancel, Escape, or a Save with
# nothing changed.
_UNCHANGED = "Settings unchanged."


def _saved_message(saved) -> str:
    """Describe where a save landed."""
    parts = []
    project_keys = getattr(saved, "project_keys", None)
    if project_keys is None or project_keys:
        parts.append(f"Settings saved to {saved.project_path}")
    if saved.user_path is not None:
        keys = ", ".join(saved.user_scoped_keys)
        parts.append(f"user-scoped ({keys}) saved to {saved.user_path}")
    return "; ".join(parts) or "Settings saved"


def _accepts_keyword(func, name: str) -> bool:
    """Whether ``func`` can be called with keyword ``name``."""
    params = inspect.signature(func).parameters.values()
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)
