"""File search tools for LangGraph workflows.

Provides glob and grep search tools that can be bound to LangGraph agents.
These tools wrap filesystem operations with appropriate safety checks.

Example:
    from agentic_cli.workflow.langgraph.tools import create_file_search_tools

    tools = create_file_search_tools(workspace_dir="/path/to/workspace")
    llm_with_tools = llm.bind_tools(tools)
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_cli.logging import Loggers

logger = Loggers.workflow()


def glob_search(
    pattern: str,
    directory: str | None = None,
    max_results: int = 100,
) -> list[str]:
    """Search for files matching a glob pattern.

    Args:
        pattern: Glob pattern to match (e.g., "**/*.py", "src/*.ts").
        directory: Base directory to search in. Defaults to current directory.
        max_results: Maximum number of results to return.

    Returns:
        List of matching file paths relative to the search directory.

    Example:
        >>> glob_search("**/*.py", "/path/to/project")
        ['src/main.py', 'src/utils/helpers.py', 'tests/test_main.py']
    """
    base_dir = Path(directory) if directory else Path.cwd()

    if not base_dir.exists():
        logger.warning("glob_search_dir_not_found", directory=str(base_dir))
        return []

    logger.debug(
        "glob_search_start",
        pattern=pattern,
        directory=str(base_dir),
    )

    results = []
    try:
        for path in base_dir.glob(pattern):
            if path.is_file():
                # Return relative path
                try:
                    rel_path = path.relative_to(base_dir)
                    results.append(str(rel_path))
                except ValueError:
                    results.append(str(path))

                if len(results) >= max_results:
                    logger.debug("glob_search_max_results", count=max_results)
                    break
    except Exception as e:
        logger.error("glob_search_error", error=str(e))

    logger.debug("glob_search_complete", matches=len(results))
    return results


def grep_search(
    pattern: str,
    directory: str | None = None,
    file_pattern: str = "*",
    max_results: int = 50,
    context_lines: int = 0,
    case_insensitive: bool = False,
) -> list[dict[str, Any]]:
    """Search for content matching a regex pattern in files.

    Args:
        pattern: Regex pattern to search for.
        directory: Base directory to search in. Defaults to current directory.
        file_pattern: Glob pattern for files to search (e.g., "*.py").
        max_results: Maximum number of matches to return.
        context_lines: Number of lines of context before/after match.
        case_insensitive: Whether to ignore case in pattern matching.

    Returns:
        List of match dictionaries with keys:
        - file: File path
        - line: Line number (1-indexed)
        - content: Matching line content
        - context_before: Lines before match (if context_lines > 0)
        - context_after: Lines after match (if context_lines > 0)

    Example:
        >>> grep_search("def main", "/path/to/project", "*.py")
        [{'file': 'src/main.py', 'line': 10, 'content': 'def main():'}]
    """
    base_dir = Path(directory) if directory else Path.cwd()

    if not base_dir.exists():
        logger.warning("grep_search_dir_not_found", directory=str(base_dir))
        return []

    logger.debug(
        "grep_search_start",
        pattern=pattern,
        directory=str(base_dir),
        file_pattern=file_pattern,
    )

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        logger.error("grep_search_invalid_pattern", pattern=pattern, error=str(e))
        return []

    results = []
    files_searched = 0

    # Get files matching the file pattern
    for file_path in base_dir.glob(f"**/{file_pattern}"):
        if not file_path.is_file():
            continue

        # Skip binary files and very large files
        try:
            if file_path.stat().st_size > 10_000_000:  # 10MB limit
                continue
        except OSError:
            continue

        files_searched += 1

        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            if regex.search(line):
                match = {
                    "file": str(file_path.relative_to(base_dir)),
                    "line": i + 1,
                    "content": line.strip(),
                }

                if context_lines > 0:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    match["context_before"] = [
                        l.strip() for l in lines[start:i]
                    ]
                    match["context_after"] = [
                        l.strip() for l in lines[i + 1:end]
                    ]

                results.append(match)

                if len(results) >= max_results:
                    logger.debug(
                        "grep_search_max_results",
                        count=max_results,
                        files_searched=files_searched,
                    )
                    return results

    logger.debug(
        "grep_search_complete",
        matches=len(results),
        files_searched=files_searched,
    )
    return results


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

    Example:
        tools = create_file_search_tools("/path/to/project")
        llm_with_tools = llm.bind_tools(tools)
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
