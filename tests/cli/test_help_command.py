"""``/help`` lists the commands by category, and ``/help <command>`` shows one.

Every command carries a usage line, examples and a category, and the README
promised "usage information" from ``/help``, yet ``/help`` ignored its argument
and printed one flat table. Now the list is grouped by category (in the order
the categories are declared), and ``/help settings``, ``/help /settings`` or an
alias (``/help set``) shows that command's description, aliases, usage and
examples.
"""

from __future__ import annotations

import io
from typing import Any

from rich.console import Console

from agentic_cli.cli.builtin_commands import ClearCommand, HelpCommand
from agentic_cli.cli.commands import Command, CommandCategory, CommandRegistry
from agentic_cli.cli.settings_command import SettingsCommand

from tests.event_replay import RecordingSession


class _Screen(RecordingSession):
    """Records what is printed as the terminal would show it."""

    def add_rich(self, renderable: Any) -> None:
        console = Console(file=io.StringIO(), record=True, width=120)
        console.print(renderable)
        self.calls.append(("rich", console.export_text()))

    def printed(self) -> str:
        return "\n".join(str(part) for call in self.calls for part in call[1:])


class _Papers(Command):
    def __init__(self) -> None:
        super().__init__(
            name="papers",
            description="Search papers",
            aliases=["p"],
            usage="/papers <query> [--max=N]",
            examples=["/papers attention", "/papers rag --max=5"],
            category=CommandCategory.SEARCH,
        )

    async def execute(self, args: str, app: Any) -> None:
        return None


class _App:
    def __init__(self) -> None:
        self.session = _Screen()
        self.command_registry = CommandRegistry()
        for command in (HelpCommand(), ClearCommand(), SettingsCommand(), _Papers()):
            self.command_registry.register(command)


async def _help(args: str) -> str:
    app = _App()
    await HelpCommand().execute(args, app)
    return app.session.printed()


async def test_the_list_is_grouped_by_category_in_their_order():
    text = await _help("")

    general, settings, search = (text.index(heading) for heading in ("General", "Settings", "Search"))
    assert general < settings < search
    assert general < text.index("/clear") < text.index("/help") < settings
    assert settings < text.index("/settings") < search < text.index("/papers")


async def test_a_command_s_help_shows_its_usage_and_examples():
    text = await _help("papers")

    assert "/papers" in text and "Search papers" in text
    assert "/p" in text  # its alias
    assert "/papers <query> [--max=N]" in text
    assert "/papers attention" in text and "/papers rag --max=5" in text
    assert "/settings" not in text  # one command, not the list


async def test_a_slash_or_an_alias_names_the_same_command():
    by_slash = await _help("/settings")
    by_alias = await _help("set")

    for text in (by_slash, by_alias):
        assert "Configure application settings" in text
        assert "/set" in text and "/config" in text
        assert "/papers" not in text


async def test_a_command_without_examples_shows_its_usage_only():
    text = await _help("clear")

    assert "Clear the screen" in text
    assert "/clear" in text
    assert "Examples" not in text


async def test_an_unknown_name_is_a_warning_that_points_to_the_list():
    app = _App()

    await HelpCommand().execute("nope", app)

    (warning,) = [call for call in app.session.calls if call[0] == "warning"]
    assert "No command /nope" in warning[1]
    assert "/help" in warning[1]


class _Odd(Command):
    """An app's command whose category is no CommandCategory member, and whose
    texts look like Rich markup."""

    def __init__(self) -> None:
        super().__init__(
            name="odd",
            description="[bold]not bold[/bold]",
            aliases=["o[/x]"],
            usage="/odd [id]",
            examples=["/odd [red]1"],
            category="custom",  # type: ignore[arg-type]
        )

    async def execute(self, args: str, app: Any) -> None:
        return None


async def test_a_command_of_another_category_is_still_listed():
    app = _App()
    app.command_registry.register(_Odd())

    await HelpCommand().execute("", app)

    text = app.session.printed()
    assert text.index("Other") < text.index("/odd")


async def test_markup_like_text_prints_as_written_in_the_list():
    app = _App()
    app.command_registry.register(_Odd())

    await HelpCommand().execute("", app)

    text = app.session.printed()
    assert "[bold]not bold[/bold]" in text
    assert "/o[/x]" in text


async def test_markup_like_text_prints_as_written_in_one_command_s_help():
    app = _App()
    app.command_registry.register(_Odd())

    await HelpCommand().execute("odd", app)

    text = app.session.printed()
    assert "[bold]not bold[/bold]" in text
    assert "/o[/x]" in text
    assert "/odd [id]" in text and "/odd [red]1" in text
