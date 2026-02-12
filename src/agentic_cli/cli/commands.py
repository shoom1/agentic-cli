"""Slash command registry and base command class.

Provides the foundation for CLI commands in agentic applications.

Example of creating a custom command:

    from agentic_cli.cli.commands import Command, ParsedArgs

    class SearchPapersCommand(Command):
        '''Search for academic papers.'''

        def __init__(self):
            super().__init__(
                name="papers",
                description="Search for academic papers",
                aliases=["p", "search"],
                usage="/papers <query> [--max=N]",
                examples=[
                    "/papers machine learning",
                    "/papers transformer architecture --max=20",
                ],
                category=CommandCategory.SEARCH,
            )

        async def execute(self, args: str, app: Any) -> None:
            parsed = self.parse_args(args)
            query = parsed.positional
            max_results = parsed.get_option("max", 10, int)

            # Perform search...
            app.session.add_message("system", f"Searching for: {query}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, TypeVar
import re

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    from agentic_cli.cli.app import BaseCLIApp

logger = Loggers.cli()
T = TypeVar("T")


class CommandCategory(Enum):
    """Categories for organizing commands."""

    GENERAL = "general"
    SESSION = "session"
    SETTINGS = "settings"
    SEARCH = "search"
    WORKFLOW = "workflow"
    DEBUG = "debug"


@dataclass
class ParsedArgs:
    """Parsed command arguments.

    Provides easy access to positional arguments and options.
    """

    positional: str
    """Positional arguments (everything not an option)."""

    options: dict[str, str] = field(default_factory=dict)
    """Named options (--key=value or --flag)."""

    def get_option(
        self,
        name: str,
        default: T = None,
        type_converter: Callable[[str], T] = str,
    ) -> T:
        """Get an option value with type conversion.

        Args:
            name: Option name (without --)
            default: Default value if option not provided
            type_converter: Function to convert string value to desired type

        Returns:
            Option value converted to specified type, or default
        """
        value = self.options.get(name)
        if value is None:
            return default
        try:
            return type_converter(value)
        except (ValueError, TypeError):
            return default

    def has_flag(self, name: str) -> bool:
        """Check if a flag option is present.

        Args:
            name: Flag name (without --)

        Returns:
            True if flag is present
        """
        return name in self.options


class Command(ABC):
    """Base class for slash commands.

    Subclass this to create custom commands. Override execute() to
    implement command behavior.

    Example:
        class MyCommand(Command):
            def __init__(self):
                super().__init__(
                    name="mycommand",
                    description="Does something useful",
                    aliases=["mc"],
                    usage="/mycommand <arg>",
                    examples=["/mycommand foo"],
                )

            async def execute(self, args: str, app: Any) -> None:
                app.session.add_message("system", f"Args: {args}")
    """

    def __init__(
        self,
        name: str,
        description: str,
        aliases: list[str] | None = None,
        usage: str | None = None,
        examples: list[str] | None = None,
        category: CommandCategory = CommandCategory.GENERAL,
        silent: bool = False,
    ) -> None:
        """Initialize the command.

        Args:
            name: Command name (used as /name)
            description: Short description of what the command does
            aliases: Alternative names for the command
            usage: Usage string showing syntax (e.g., "/cmd <arg> [--opt]")
            examples: List of example usages
            category: Category for organizing in help
            silent: If True, suppress "Executing command:" echo
        """
        self.name = name
        self.description = description
        self.aliases = aliases or []
        self.usage = usage or f"/{name}"
        self.examples = examples or []
        self.category = category
        self.silent = silent

    @abstractmethod
    async def execute(self, args: str, app: Any) -> None:
        """Execute the command with given arguments.

        Args:
            args: Command arguments string (everything after the command name)
            app: The CLI application instance
        """
        pass

    def parse_args(self, args: str) -> ParsedArgs:
        """Parse command arguments into structured form.

        Parses options in the forms:
        - --key=value
        - --key value
        - --flag (boolean flag)

        Everything else is treated as positional arguments.

        Args:
            args: Raw argument string

        Returns:
            ParsedArgs with positional arguments and options
        """
        options: dict[str, str] = {}
        positional_parts: list[str] = []

        # Split by whitespace, respecting quotes
        parts = self._tokenize(args)

        i = 0
        while i < len(parts):
            part = parts[i]

            if part.startswith("--"):
                key = part[2:]

                if "=" in key:
                    # --key=value format
                    key, value = key.split("=", 1)
                    options[key] = value
                elif i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    # --key value format
                    options[key] = parts[i + 1]
                    i += 1
                else:
                    # --flag format (boolean)
                    options[key] = "true"
            elif part.startswith("-") and len(part) == 2:
                # -k format (short option)
                key = part[1]
                if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    options[key] = parts[i + 1]
                    i += 1
                else:
                    options[key] = "true"
            else:
                positional_parts.append(part)

            i += 1

        return ParsedArgs(
            positional=" ".join(positional_parts),
            options=options,
        )

    def _tokenize(self, args: str) -> list[str]:
        """Tokenize argument string respecting quotes.

        Args:
            args: Raw argument string

        Returns:
            List of tokens
        """
        # Match quoted strings or non-whitespace sequences
        pattern = r'"[^"]*"|\'[^\']*\'|\S+'
        tokens = re.findall(pattern, args)

        # Remove surrounding quotes from quoted strings
        def strip_quotes(token: str) -> str:
            if len(token) >= 2 and token[0] in "\"'" and token[0] == token[-1]:
                return token[1:-1]
            return token

        return [strip_quotes(token) for token in tokens]

    def get_help(self) -> str:
        """Get detailed help text for this command.

        Returns:
            Formatted help string
        """
        lines = [
            f"**/{self.name}**",
            f"  {self.description}",
            "",
            f"**Usage:** `{self.usage}`",
        ]

        if self.aliases:
            lines.append(f"**Aliases:** {', '.join(f'/{a}' for a in self.aliases)}")

        if self.examples:
            lines.append("")
            lines.append("**Examples:**")
            for example in self.examples:
                lines.append(f"  `{example}`")

        return "\n".join(lines)


class CommandRegistry:
    """Registry for managing slash commands.

    Handles command registration and lookup.
    """

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Register a command and its aliases.

        Args:
            command: Command instance to register
        """
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command

    def get(self, name: str) -> Command | None:
        """Get a command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            Command if found, None otherwise
        """
        return self._commands.get(name)

    def all_commands(self) -> list[Command]:
        """Get all unique commands (excluding aliases).

        Returns:
            List of all registered commands
        """
        seen: set[str] = set()
        commands: list[Command] = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                commands.append(cmd)
        return commands

    def get_completions(self) -> list[str]:
        """Get all command names and aliases for auto-completion.

        Returns:
            List of all command names and aliases, sorted alphabetically
        """
        return sorted(self._commands.keys())

