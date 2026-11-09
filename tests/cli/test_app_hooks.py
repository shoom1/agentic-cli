"""The construction hooks see a built app, and the completer the commands as they are.

``register_commands()`` (and ``_register_builtin_commands()``) run last in
the constructor, once the settings, the usage tracker, the workflow
controller and the session exist; ``get_theme()`` runs while the session is
built, since its theme is the session's, with everything else in place.
The hooks run during construction: an attribute of an app's own that a
hook reads is set before ``super().__init__()``.

The slash-command completer reads the registry each time it completes, so a
command registered after construction completes too.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from prompt_toolkit.document import Document

from agentic_cli.cli import AppInfo
from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp, SlashCommandCompleter
from agentic_cli.cli.builtin_commands import HelpCommand
from agentic_cli.cli.commands import Command, CommandRegistry
from agentic_cli.config import BaseSettings
from agentic_cli.workflow.config import AgentConfig
from tests.cli.dialog_fakes import autospec_session


@pytest.fixture(autouse=True)
def hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)


def build(cls: type[BaseCLIApp], **kwargs) -> BaseCLIApp:
    return cls(
        app_info=AppInfo(name="Hooks test", version="0"),
        agent_configs=[AgentConfig(name="a", prompt="p")],
        settings=BaseSettings(session_store="memory"),
        **kwargs,
    )


def built_with_completer(cls: type[BaseCLIApp]) -> tuple[BaseCLIApp, SlashCommandCompleter]:
    """The app, and the completer its session was built with."""
    with patch.object(app_module, "ThinkingPromptSession") as session_cls:
        app = build(cls)
    return app, session_cls.call_args.kwargs["completer"]


class Files(Command):
    def __init__(self, name: str = "files") -> None:
        super().__init__(name=name, description="List the files")

    async def execute(self, args: str, app) -> None:
        app.session.add_message("system", "files")


def completions(completer: SlashCommandCompleter, text: str) -> list[str]:
    return [c.text for c in completer.get_completions(Document(text), None)]


class TestRegisterCommands:
    def test_sees_the_built_app(self):
        seen: dict = {}

        class ReadsTheApp(BaseCLIApp):
            def register_commands(self) -> None:
                seen.update(
                    settings=self.settings, tracker=self.usage_tracker,
                    agents=self.agent_configs, session=self.session,
                    help=self.command_registry.get("help"),
                )
                self.command_registry.register(Files())

        app, completer = built_with_completer(ReadsTheApp)

        assert seen["settings"] is app.settings
        assert seen["tracker"] is app.usage_tracker
        assert [a.name for a in seen["agents"]] == ["a"]
        assert seen["session"] is app.session
        assert isinstance(seen["help"], HelpCommand)  # the built-ins come first
        assert completions(completer, "/fi") == ["/files"]

    def test_an_attribute_set_before_super_init_is_seen_by_both_hooks(self):
        """What README shows for an app's own state a hook reads."""

        class Verbose(BaseCLIApp):
            def __init__(self, verbose: bool) -> None:
                self.verbose = verbose
                super().__init__(
                    app_info=AppInfo(name="Hooks test", version="0"),
                    agent_configs=[AgentConfig(name="a", prompt="p")],
                    settings=BaseSettings(session_store="memory"),
                )

            def get_theme(self, name: str):
                return "light" if self.verbose else super().get_theme(name)

            def register_commands(self) -> None:
                if self.verbose:
                    self.command_registry.register(Files())

        app = Verbose(verbose=True)

        assert app.command_registry.get("files") is not None


class TestTheCompleter:
    def test_a_command_registered_after_construction_completes(self):
        app, completer = built_with_completer(BaseCLIApp)

        app.command_registry.register(Files("later"))

        assert completions(completer, "/lat") == ["/later"]

    def test_it_reads_the_registry_it_is_given(self):
        registry = CommandRegistry()
        completer = SlashCommandCompleter(registry)
        registry.register(Files())

        assert completer.commands == ["files"]

    def test_a_list_of_names_still_works(self):
        completer = SlashCommandCompleter(["quit", "help"])

        assert completer.commands == ["help", "quit"]
        assert completions(completer, "/h") == ["/help"]


async def test_help_lists_a_command_registered_after_construction():
    app = build(BaseCLIApp)
    app.session = autospec_session()
    app.command_registry.register(Files("later"))

    await app.process_input("/help later")

    shown = " ".join(str(c) for c in app.session.method_calls)
    assert "later" in shown and "Unknown command" not in shown
