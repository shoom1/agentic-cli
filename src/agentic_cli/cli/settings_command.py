"""Settings command for configuring model and thinking effort."""

from typing import TYPE_CHECKING, Any

from thinking_prompt import (
    DropdownItem,
    InlineSelectItem,
    SettingsDialog,
)

from agentic_cli.cli.commands import Command, CommandCategory
from agentic_cli.config import THINKING_EFFORT_LEVELS

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp


class SettingsCommand(Command):
    """Open settings dialog to configure model and thinking effort."""

    def __init__(self) -> None:
        super().__init__(
            name="settings",
            description="Configure model and thinking effort settings",
            aliases=["set", "config"],
            category=CommandCategory.SETTINGS,
        )

    async def execute(self, args: str, app: "BaseCLIApp") -> None:
        """Open interactive settings dialog."""
        settings = app.settings
        available_models = list(settings.get_available_models())
        current_model = settings.get_model()
        current_effort = settings.thinking_effort

        if not available_models:
            app.session.add_error("No models available. Please configure API keys.")
            return

        items = [
            DropdownItem(
                key="model",
                label="Model",
                description="Select the AI model to use",
                options=available_models,
                default=current_model,
            ),
            InlineSelectItem(
                key="thinking_effort",
                label="Thinking Effort",
                description="Controls depth of reasoning",
                options=list(THINKING_EFFORT_LEVELS),
                default=current_effort,
            ),
        ]

        dialog = SettingsDialog(
            title="Settings",
            items=items,
            can_cancel=True,
        )
        result = await app.session.show_dialog(dialog)

        if result is None:
            app.session.add_message("system", "Settings unchanged.")
            return

        await app.apply_settings(result)
