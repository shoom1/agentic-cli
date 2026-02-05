"""Command tokenizer using shlex and regex patterns.

Layer 2: Lexical/Syntactic Analysis
- Parse commands into structured tokens
- Detect pipes, chains, subshells, redirections
- Extract base commands and arguments
"""

import re
import shlex
from dataclasses import dataclass, field

from agentic_cli.tools.shell.models import (
    CommandNode,
    Redirect,
    TokenizeResult,
)


class CommandTokenizer:
    """Tokenizes shell commands into structured CommandNode trees.

    Uses shlex for basic tokenization with regex patterns for detecting
    shell operators like pipes, chains, and redirections.
    """

    # Patterns for shell operators
    PIPE_PATTERN = re.compile(r"(?<![|])\|(?![|])")  # | but not ||
    AND_CHAIN_PATTERN = re.compile(r"&&")
    OR_CHAIN_PATTERN = re.compile(r"\|\|")
    SEMICOLON_PATTERN = re.compile(r";")
    BACKGROUND_PATTERN = re.compile(r"&\s*$")

    # Subshell patterns
    COMMAND_SUB_DOLLAR = re.compile(r"\$\([^)]+\)")  # $(...)
    COMMAND_SUB_BACKTICK = re.compile(r"`[^`]+`")  # `...`
    PROCESS_SUB = re.compile(r"[<>]\([^)]+\)")  # <(...) or >(...)

    # Redirection patterns
    REDIRECT_PATTERNS = [
        re.compile(r"(\d*)>>"),  # Append
        re.compile(r"(\d*)>"),  # Overwrite (must check after >>)
        re.compile(r"(\d*)<"),  # Input
        re.compile(r"2>&1"),  # Stderr to stdout
        re.compile(r"&>"),  # Both to file
    ]

    def tokenize(self, command: str) -> TokenizeResult:
        """Parse a shell command into structured tokens.

        Args:
            command: The shell command string to parse.

        Returns:
            TokenizeResult with parsed command nodes and metadata.
        """
        command = command.strip()
        if not command:
            return TokenizeResult(nodes=[], parse_errors=["Empty command"])

        result = TokenizeResult(nodes=[])

        # Detect features before parsing
        result.has_pipes = bool(self.PIPE_PATTERN.search(command))
        result.has_chains = bool(
            self.AND_CHAIN_PATTERN.search(command)
            or self.OR_CHAIN_PATTERN.search(command)
            or self.SEMICOLON_PATTERN.search(command)
        )
        result.has_subshells = bool(
            self.COMMAND_SUB_DOLLAR.search(command)
            or self.COMMAND_SUB_BACKTICK.search(command)
        )
        result.has_background = bool(self.BACKGROUND_PATTERN.search(command))
        result.has_redirections = any(
            p.search(command) for p in self.REDIRECT_PATTERNS
        )

        # Parse the command
        try:
            nodes = self._parse_command_string(command)
            result.nodes = nodes
        except ValueError as e:
            result.parse_errors.append(str(e))
            # Create a basic node with the raw command
            result.nodes = [
                CommandNode(
                    command=command.split()[0] if command.split() else "",
                    args=command.split()[1:] if len(command.split()) > 1 else [],
                    raw_command=command,
                )
            ]

        return result

    def _parse_command_string(self, command: str) -> list[CommandNode]:
        """Parse a command string into CommandNode objects.

        Handles chained commands (;, &&, ||) and pipes (|).
        """
        nodes = []

        # Split by chain operators first (lowest precedence)
        chain_parts = self._split_by_chains(command)

        for chain_op, part in chain_parts:
            # Each part might contain pipes
            pipe_nodes = self._parse_pipeline(part.strip())

            if pipe_nodes:
                # First node in pipeline gets the chain operator from previous
                if chain_op and nodes:
                    nodes[-1].chained_with.append((chain_op, pipe_nodes[0]))
                else:
                    nodes.extend(pipe_nodes)

        return nodes

    def _split_by_chains(self, command: str) -> list[tuple[str | None, str]]:
        """Split command by chain operators (;, &&, ||).

        Returns list of (operator, command_part) tuples.
        The first tuple has None as operator.
        """
        # This is a simplified approach - we split on operators
        # while respecting quoted strings

        result: list[tuple[str | None, str]] = []
        current = ""
        i = 0
        in_quotes = False
        quote_char = ""

        while i < len(command):
            char = command[i]

            # Handle quotes
            if char in ('"', "'") and (i == 0 or command[i - 1] != "\\"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                current += char
                i += 1
                continue

            # Check for operators only outside quotes
            if not in_quotes:
                # Check for &&
                if command[i : i + 2] == "&&":
                    if current.strip():
                        op = None if not result else "&&"
                        result.append((op, current.strip()) if not result else (None, current.strip()))
                        if result and len(result) == 1:
                            result[0] = (None, current.strip())
                        else:
                            result.append(("&&", ""))
                    current = ""
                    i += 2
                    continue

                # Check for ||
                if command[i : i + 2] == "||":
                    if current.strip():
                        if not result:
                            result.append((None, current.strip()))
                        else:
                            # Update the pending entry
                            pass
                    result.append(("||", ""))
                    current = ""
                    i += 2
                    continue

                # Check for ; (but not inside quotes)
                if char == ";":
                    if current.strip():
                        if not result:
                            result.append((None, current.strip()))
                    result.append((";", ""))
                    current = ""
                    i += 1
                    continue

            current += char
            i += 1

        # Add remaining content
        if current.strip():
            if not result:
                result.append((None, current.strip()))
            elif result[-1][1] == "":
                # Update the last entry with content
                result[-1] = (result[-1][0], current.strip())
            else:
                result.append((None, current.strip()))

        # Simplified: if no chains found, return single entry
        if not result:
            result.append((None, command))

        return result

    def _parse_pipeline(self, command: str) -> list[CommandNode]:
        """Parse a pipeline (commands connected by |)."""
        # Split by pipe, respecting quotes
        parts = self._split_by_pipe(command)
        nodes = []

        for i, part in enumerate(parts):
            node = self._parse_single_command(part.strip())
            if i > 0 and nodes:
                nodes[-1].pipes_to = node
            nodes.append(node)

        return nodes

    def _split_by_pipe(self, command: str) -> list[str]:
        """Split command by pipe operator, respecting quotes."""
        parts = []
        current = ""
        i = 0
        in_quotes = False
        quote_char = ""
        paren_depth = 0

        while i < len(command):
            char = command[i]

            # Track parentheses depth (for subshells)
            if char == "(" and not in_quotes:
                paren_depth += 1
            elif char == ")" and not in_quotes:
                paren_depth -= 1

            # Handle quotes
            if char in ('"', "'") and (i == 0 or command[i - 1] != "\\"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                current += char
                i += 1
                continue

            # Check for | but not || (outside quotes and parens)
            if char == "|" and not in_quotes and paren_depth == 0:
                if i + 1 < len(command) and command[i + 1] == "|":
                    # This is ||, not a pipe
                    current += "||"
                    i += 2
                    continue
                else:
                    # This is a pipe
                    if current.strip():
                        parts.append(current.strip())
                    current = ""
                    i += 1
                    continue

            current += char
            i += 1

        if current.strip():
            parts.append(current.strip())

        return parts if parts else [command]

    def _parse_single_command(self, command: str) -> CommandNode:
        """Parse a single command (no pipes or chains)."""
        node = CommandNode(command="", raw_command=command)

        # Check for background
        if self.BACKGROUND_PATTERN.search(command):
            node.background = True
            command = self.BACKGROUND_PATTERN.sub("", command).strip()

        # Extract subshells
        subshells = self._extract_subshells(command)
        if subshells:
            node.subshells = subshells

        # Extract redirections
        redirections, command = self._extract_redirections(command)
        node.redirections = redirections

        # Parse remaining command with shlex
        try:
            tokens = shlex.split(command)
        except ValueError:
            # Fallback to simple split
            tokens = command.split()

        if tokens:
            node.command = tokens[0]
            node.args = tokens[1:]

        return node

    def _extract_subshells(self, command: str) -> list[CommandNode]:
        """Extract subshell commands from a command string."""
        subshells = []

        # Find $(...) subshells
        for match in self.COMMAND_SUB_DOLLAR.finditer(command):
            inner = match.group()[2:-1]  # Remove $( and )
            inner_nodes = self._parse_command_string(inner)
            subshells.extend(inner_nodes)

        # Find `...` subshells
        for match in self.COMMAND_SUB_BACKTICK.finditer(command):
            inner = match.group()[1:-1]  # Remove backticks
            inner_nodes = self._parse_command_string(inner)
            subshells.extend(inner_nodes)

        return subshells

    def _extract_redirections(self, command: str) -> tuple[list[Redirect], str]:
        """Extract redirections from a command, returning cleaned command."""
        redirections = []
        cleaned = command

        # Pattern for redirections with their targets
        redir_pattern = re.compile(
            r"(\d*)(>>|>|<|2>&1|&>)\s*([^\s;|&]+)?"
        )

        for match in redir_pattern.finditer(command):
            fd = match.group(1) or ""
            op = match.group(2)
            target = match.group(3) or ""

            redirections.append(Redirect(operator=f"{fd}{op}", target=target))
            # Remove from cleaned command
            cleaned = cleaned.replace(match.group(), " ", 1)

        return redirections, cleaned.strip()

    def get_all_commands(self, result: TokenizeResult) -> list[str]:
        """Extract all base commands from a tokenize result.

        Includes commands from subshells and piped commands.
        """
        commands = []

        def collect_from_node(node: CommandNode):
            if node.command:
                commands.append(node.command)
            for subshell in node.subshells:
                collect_from_node(subshell)
            if node.pipes_to:
                collect_from_node(node.pipes_to)
            for _, chained in node.chained_with:
                collect_from_node(chained)

        for node in result.nodes:
            collect_from_node(node)

        return commands
