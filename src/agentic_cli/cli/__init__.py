"""CLI framework for agentic applications."""

from thinking_prompt import AppInfo

from agentic_cli.cli.commands import (
    Command,
    CommandCategory,
    CommandRegistry,
    ParsedArgs,
    create_simple_command,
)
from agentic_cli.cli.app import BaseCLIApp

__all__ = [
    "AppInfo",
    "BaseCLIApp",
    "Command",
    "CommandCategory",
    "CommandRegistry",
    "ParsedArgs",
    "create_simple_command",
]
