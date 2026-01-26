"""Settings command for configuring application settings.

Uses introspection to automatically generate UI controls from Pydantic fields.
Apps can customize which settings appear by overriding get_ui_setting_keys().
"""

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

        # Apply settings changes
        await app.apply_settings(result)

        # Save settings to project config file
        try:
            path = await app.save_settings()
            app.session.add_success(f"Settings saved to {path}")
        except Exception as e:
            app.session.add_warning(f"Settings applied but not saved: {e}")
