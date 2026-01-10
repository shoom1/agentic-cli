"""Tests for command registry and base command class."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_cli.cli.commands import (
    Command,
    CommandCategory,
    CommandRegistry,
    ParsedArgs,
    create_simple_command,
)


class MockCommand(Command):
    """Mock command for testing."""

    def __init__(
        self,
        name: str = "test",
        description: str = "Test command",
        aliases: list[str] | None = None,
        category: CommandCategory = CommandCategory.GENERAL,
    ):
        super().__init__(name, description, aliases, category=category)
        self.executed = False
        self.last_args = None
        self.last_app = None

    async def execute(self, args: str, app) -> None:
        """Record execution details."""
        self.executed = True
        self.last_args = args
        self.last_app = app


class TestCommand:
    """Tests for Command base class."""

    def test_command_initialization(self):
        """Test command initialization."""
        cmd = MockCommand("help", "Show help", aliases=["h", "?"])

        assert cmd.name == "help"
        assert cmd.description == "Show help"
        assert cmd.aliases == ["h", "?"]

    def test_command_no_aliases(self):
        """Test command with no aliases."""
        cmd = MockCommand("exit", "Exit the application")

        assert cmd.aliases == []

    @pytest.mark.asyncio
    async def test_command_execute(self):
        """Test command execution."""
        cmd = MockCommand()
        app = MagicMock()

        await cmd.execute("arg1 arg2", app)

        assert cmd.executed
        assert cmd.last_args == "arg1 arg2"
        assert cmd.last_app == app


class TestCommandRegistry:
    """Tests for CommandRegistry class."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = CommandRegistry()

        assert registry.all_commands() == []
        assert registry.get("nonexistent") is None
        assert registry.get_completions() == []

    def test_register_command(self):
        """Test registering a command."""
        registry = CommandRegistry()
        cmd = MockCommand("help", "Show help")

        registry.register(cmd)

        assert registry.get("help") == cmd
        assert "help" in registry.get_completions()

    def test_register_with_aliases(self):
        """Test registering command with aliases."""
        registry = CommandRegistry()
        cmd = MockCommand("help", "Show help", aliases=["h", "?"])

        registry.register(cmd)

        # All names should resolve to same command
        assert registry.get("help") == cmd
        assert registry.get("h") == cmd
        assert registry.get("?") == cmd

        # All should appear in completions
        completions = registry.get_completions()
        assert "help" in completions
        assert "h" in completions
        assert "?" in completions

    def test_all_commands_no_duplicates(self):
        """Test all_commands returns unique commands."""
        registry = CommandRegistry()
        cmd1 = MockCommand("help", "Help", aliases=["h"])
        cmd2 = MockCommand("exit", "Exit", aliases=["q", "quit"])

        registry.register(cmd1)
        registry.register(cmd2)

        all_cmds = registry.all_commands()

        # Should have exactly 2 unique commands
        assert len(all_cmds) == 2
        assert cmd1 in all_cmds
        assert cmd2 in all_cmds

    def test_register_multiple_commands(self):
        """Test registering multiple commands."""
        registry = CommandRegistry()

        commands = [
            MockCommand("help", "Show help"),
            MockCommand("exit", "Exit"),
            MockCommand("clear", "Clear screen"),
        ]

        for cmd in commands:
            registry.register(cmd)

        assert len(registry.all_commands()) == 3
        for cmd in commands:
            assert registry.get(cmd.name) == cmd

    def test_get_completions(self):
        """Test getting command completions."""
        registry = CommandRegistry()
        registry.register(MockCommand("help", "Help", aliases=["h"]))
        registry.register(MockCommand("exit", "Exit"))

        completions = registry.get_completions()

        assert set(completions) == {"help", "h", "exit"}

    def test_command_override(self):
        """Test that registering same name overrides."""
        registry = CommandRegistry()
        cmd1 = MockCommand("help", "First help")
        cmd2 = MockCommand("help", "Second help")

        registry.register(cmd1)
        registry.register(cmd2)

        # Second registration should override
        assert registry.get("help") == cmd2

        # But both are in all_commands (they're different instances)
        all_cmds = registry.all_commands()
        assert len(all_cmds) == 1  # Same name, last one wins

    def test_unregister_command(self):
        """Test unregistering a command."""
        registry = CommandRegistry()
        cmd = MockCommand("test", "Test", aliases=["t"])

        registry.register(cmd)
        assert registry.get("test") is not None
        assert registry.get("t") is not None

        registry.unregister("test")

        assert registry.get("test") is None
        assert registry.get("t") is None

    def test_by_category(self):
        """Test getting commands by category."""
        registry = CommandRegistry()
        cmd1 = MockCommand("help", "Help", category=CommandCategory.GENERAL)
        cmd2 = MockCommand("save", "Save", category=CommandCategory.SESSION)
        cmd3 = MockCommand("settings", "Settings", category=CommandCategory.SETTINGS)

        registry.register(cmd1)
        registry.register(cmd2)
        registry.register(cmd3)

        general = registry.by_category(CommandCategory.GENERAL)
        session = registry.by_category(CommandCategory.SESSION)
        settings = registry.by_category(CommandCategory.SETTINGS)

        assert cmd1 in general
        assert cmd2 in session
        assert cmd3 in settings
        assert len(general) == 1
        assert len(session) == 1
        assert len(settings) == 1


class TestParsedArgs:
    """Tests for ParsedArgs class."""

    def test_get_option_string(self):
        """Test getting string option."""
        args = ParsedArgs(positional="query", options={"max": "10"})

        assert args.get_option("max") == "10"
        assert args.get_option("missing") is None

    def test_get_option_default(self):
        """Test getting option with default."""
        args = ParsedArgs(positional="query", options={})

        assert args.get_option("max", 5) == 5
        assert args.get_option("format", "json") == "json"

    def test_get_option_type_converter(self):
        """Test type conversion."""
        args = ParsedArgs(positional="query", options={"max": "20", "verbose": "true"})

        assert args.get_option("max", 10, int) == 20
        assert args.get_option("verbose", False, lambda x: x == "true") is True

    def test_get_option_invalid_conversion(self):
        """Test fallback to default on conversion error."""
        args = ParsedArgs(positional="", options={"max": "not_a_number"})

        assert args.get_option("max", 10, int) == 10

    def test_has_flag(self):
        """Test flag detection."""
        args = ParsedArgs(positional="", options={"verbose": "true", "quiet": "true"})

        assert args.has_flag("verbose") is True
        assert args.has_flag("debug") is False


class TestCommandParsing:
    """Tests for command argument parsing."""

    def test_parse_simple_args(self):
        """Test parsing simple positional args."""
        cmd = MockCommand()
        parsed = cmd.parse_args("hello world")

        assert parsed.positional == "hello world"
        assert parsed.options == {}

    def test_parse_key_equals_value(self):
        """Test parsing --key=value format."""
        cmd = MockCommand()
        parsed = cmd.parse_args("query --max=10 --format=json")

        assert parsed.positional == "query"
        assert parsed.options["max"] == "10"
        assert parsed.options["format"] == "json"

    def test_parse_key_space_value(self):
        """Test parsing --key value format."""
        cmd = MockCommand()
        parsed = cmd.parse_args("query --max 10 --format json")

        assert parsed.positional == "query"
        assert parsed.options["max"] == "10"
        assert parsed.options["format"] == "json"

    def test_parse_flag(self):
        """Test parsing boolean flags."""
        cmd = MockCommand()
        parsed = cmd.parse_args("query --verbose --debug")

        assert parsed.positional == "query"
        assert parsed.options["verbose"] == "true"
        assert parsed.options["debug"] == "true"

    def test_parse_short_options(self):
        """Test parsing short -k format."""
        cmd = MockCommand()
        parsed = cmd.parse_args("query -n 5 -v")

        assert parsed.positional == "query"
        assert parsed.options["n"] == "5"
        assert parsed.options["v"] == "true"

    def test_parse_quoted_strings(self):
        """Test parsing quoted strings in positional args."""
        cmd = MockCommand()
        parsed = cmd.parse_args('"hello world" --name John')

        assert parsed.positional == "hello world"
        assert parsed.options["name"] == "John"

    def test_parse_quoted_positional(self):
        """Test quoted positional argument."""
        cmd = MockCommand()
        parsed = cmd.parse_args('"search query with spaces"')

        assert parsed.positional == "search query with spaces"

    def test_parse_empty_args(self):
        """Test parsing empty args."""
        cmd = MockCommand()
        parsed = cmd.parse_args("")

        assert parsed.positional == ""
        assert parsed.options == {}

    def test_parse_mixed_args(self):
        """Test parsing mixed positional and options."""
        cmd = MockCommand()
        parsed = cmd.parse_args("machine learning --max=20 papers --verbose")

        assert parsed.positional == "machine learning papers"
        assert parsed.options["max"] == "20"
        assert parsed.options["verbose"] == "true"


class TestCommandHelp:
    """Tests for command help generation."""

    def test_get_help_basic(self):
        """Test basic help generation."""
        cmd = MockCommand("test", "A test command")
        help_text = cmd.get_help()

        assert "**/test**" in help_text
        assert "A test command" in help_text

    def test_get_help_with_usage(self):
        """Test help with usage string."""

        class UsageCommand(Command):
            def __init__(self):
                super().__init__(
                    name="search",
                    description="Search for items",
                    usage="/search <query> [--max=N]",
                )

            async def execute(self, args, app):
                pass

        cmd = UsageCommand()
        help_text = cmd.get_help()

        assert "/search <query> [--max=N]" in help_text

    def test_get_help_with_examples(self):
        """Test help with examples."""

        class ExampleCommand(Command):
            def __init__(self):
                super().__init__(
                    name="search",
                    description="Search",
                    examples=["/search hello", "/search world --max=5"],
                )

            async def execute(self, args, app):
                pass

        cmd = ExampleCommand()
        help_text = cmd.get_help()

        assert "/search hello" in help_text
        assert "/search world --max=5" in help_text


class TestCreateSimpleCommand:
    """Tests for create_simple_command factory."""

    @pytest.mark.asyncio
    async def test_sync_handler(self):
        """Test simple command with sync handler."""
        result = []

        def handler(args, app):
            result.append(args)

        cmd = create_simple_command("test", "Test", handler)
        await cmd.execute("hello", None)

        assert result == ["hello"]

    @pytest.mark.asyncio
    async def test_async_handler(self):
        """Test simple command with async handler."""
        result = []

        async def handler(args, app):
            result.append(args)

        cmd = create_simple_command("test", "Test", handler)
        await cmd.execute("world", None)

        assert result == ["world"]

    def test_command_properties(self):
        """Test simple command has correct properties."""
        cmd = create_simple_command(
            name="greet",
            description="Greet someone",
            handler=lambda a, b: None,
            aliases=["g"],
            usage="/greet <name>",
            examples=["/greet Alice"],
            category=CommandCategory.GENERAL,
        )

        assert cmd.name == "greet"
        assert cmd.description == "Greet someone"
        assert cmd.aliases == ["g"]
        assert cmd.usage == "/greet <name>"
        assert cmd.examples == ["/greet Alice"]
        assert cmd.category == CommandCategory.GENERAL


class TestCommandDiscovery:
    """Tests for command auto-discovery."""

    def test_discover_from_nonexistent_module(self):
        """Test discovery handles missing module."""
        registry = CommandRegistry()
        discovered = registry.discover_commands("nonexistent.module")

        assert discovered == []

    def test_discover_from_module(self):
        """Test discovery finds Command subclasses."""
        # This tests the discover mechanism with the builtin_commands module
        registry = CommandRegistry()
        discovered = registry.discover_commands("agentic_cli.cli.builtin_commands")

        # Should find several commands
        assert len(discovered) > 0

        # Should be able to get them from registry
        assert registry.get("help") is not None
        assert registry.get("exit") is not None
