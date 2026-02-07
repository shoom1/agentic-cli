"""File search tools for LangGraph workflows.

Thin wrappers around the main glob/grep tools that provide
LangChain-compatible @tool decorators for LangGraph agents.

Example:
    from agentic_cli.workflow.langgraph.tools import create_file_search_tools

    tools = create_file_search_tools(workspace_dir="/path/to/workspace")
    llm_with_tools = llm.bind_tools(tools)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_cli.logging import Loggers
from agentic_cli.tools.glob_tool import glob as main_glob
from agentic_cli.tools.grep_tool import grep as main_grep

logger = Loggers.workflow()


def glob_search(
    pattern: str,
    directory: str | None = None,
    max_results: int = 100,
) -> list[str]:
    """Search for files matching a glob pattern.

    Delegates to the main glob tool and returns a flat list of paths.

    Args:
        pattern: Glob pattern to match (e.g., "**/*.py", "src/*.ts").
        directory: Base directory to search in. Defaults to current directory.
        max_results: Maximum number of results to return.

    Returns:
        List of matching file paths relative to the search directory.
    """
    path = directory or "."
    result = main_glob(
        pattern=pattern,
        path=path,
        include_dirs=False,
        max_results=max_results,
    )
    if not result.get("success"):
        logger.warning("glob_search_failed", error=result.get("error"))
        return []
    return result.get("files", [])


def grep_search(
    pattern: str,
    directory: str | None = None,
    file_pattern: str = "*",
    max_results: int = 50,
    context_lines: int = 0,
    case_insensitive: bool = False,
) -> list[dict[str, Any]]:
    """Search for content matching a regex pattern in files.

    Delegates to the main grep tool and reformats output for LangGraph.

    Args:
        pattern: Regex pattern to search for.
        directory: Base directory to search in. Defaults to current directory.
        file_pattern: Glob pattern for files to search (e.g., "*.py").
        max_results: Maximum number of matches to return.
        context_lines: Number of lines of context before/after match.
        case_insensitive: Whether to ignore case in pattern matching.

    Returns:
        List of match dictionaries with keys: file, line, content.
    """
    path = directory or "."
    result = main_grep(
        pattern=pattern,
        path=path,
        ignore_case=case_insensitive,
        file_pattern=file_pattern if file_pattern != "*" else None,
        context_lines=context_lines,
        max_results=max_results,
        output_mode="content",
    )
    if not result.get("success"):
        logger.warning("grep_search_failed", error=result.get("error"))
        return []

    # Reformat matches: main grep uses "line_number", LangGraph uses "line"
    matches = []
    for m in result.get("matches", []):
        entry: dict[str, Any] = {
            "file": m["file"],
            "line": m.get("line_number", 0),
            "content": m.get("content", ""),
        }
        if "context_before" in m:
            entry["context_before"] = m["context_before"]
        if "context_after" in m:
            entry["context_after"] = m["context_after"]
        matches.append(entry)
    return matches


def create_file_search_tools(
    workspace_dir: str | Path | None = None,
) -> list[Any]:
    """Create file search tools for binding to LangGraph agents.

    Creates LangChain-compatible tool instances that can be bound
    to language models via llm.bind_tools().

    Args:
        workspace_dir: Base directory for file operations.
            Defaults to current working directory.

    Returns:
        List of tool instances [glob_tool, grep_tool].
    """
    try:
        from langchain_core.tools import tool
    except ImportError:
        logger.warning("langchain_core_not_installed")
        return []

    base_dir = str(workspace_dir) if workspace_dir else None

    @tool
    def glob_tool(pattern: str, max_results: int = 100) -> list[str]:
        """Search for files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts")
            max_results: Maximum number of results to return

        Returns:
            List of matching file paths
        """
        return glob_search(pattern, directory=base_dir, max_results=max_results)

    @tool
    def grep_tool(
        pattern: str,
        file_pattern: str = "*",
        max_results: int = 50,
        case_insensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for content matching a pattern in files.

        Args:
            pattern: Regex pattern to search for
            file_pattern: Glob pattern for files to search (e.g., "*.py")
            max_results: Maximum number of matches to return
            case_insensitive: Ignore case in matching

        Returns:
            List of matches with file, line number, and content
        """
        return grep_search(
            pattern,
            directory=base_dir,
            file_pattern=file_pattern,
            max_results=max_results,
            case_insensitive=case_insensitive,
        )

    return [glob_tool, grep_tool]
