"""Settings command for configuring application settings.

Uses introspection to automatically generate UI controls from Pydantic fields.
Apps can customize which settings appear by overriding get_ui_setting_keys().
"""

import inspect
from typing import TYPE_CHECKING

from thinking_prompt import SettingsDialog

from agentic_cli.cli.commands import Command, CommandCategory

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp


class SettingsCommand(Command):
    """Open settings dialog to configure application settings.

    Uses the app's get_ui_setting_keys() method to determine which settings
    to display, and _build_ui_items() to generate the appropriate UI controls
    based on field types and metadata.
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
        # Build UI items using introspection
        items = app._build_ui_items()

        if not items:
            app.session.add_error(
                "No settings available. Please configure API keys first."
            )
            return

        dialog = SettingsDialog(
            title="Settings",
            items=items,
            can_cancel=True,
        )
        result = await app.session.show_dialog(dialog)

        if result is None:
            app.session.add_message("system", "Settings unchanged.")
            return

        # Apply, then persist exactly the fields that changed. Saving the whole
        # live object also wrote values that came from the environment or were
        # derived at startup, making a one-run override permanent.
        before = app.settings.model_dump()
        await app.apply_settings(result)
        after = app.settings.model_dump()
        changed = {k for k, v in after.items() if before.get(k) != v}
        if not changed:
            return

        try:
            if _accepts_keyword(app.save_settings, "keys"):
                saved = await app.save_settings(keys=changed)
            else:  # an app override predating keys=: its own full save
                saved = await app.save_settings()
            app.session.add_success(_saved_message(saved))
        except Exception as e:
            app.session.add_warning(f"Settings applied but not saved: {e}")


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
