"""Content search tool (grep).

Provides pattern-based content search across files:
- grep: Search for patterns in files (ripgrep-like interface)

This is a read-only tool with PermissionLevel.SAFE.
"""

import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from agentic_cli.tools.registry import (
    ToolCategory,
    PermissionLevel,
    register_tool,
)


@register_tool(
    category=ToolCategory.READ,
    permission_level=PermissionLevel.SAFE,
    description="Search for text patterns inside file contents (regex or literal). Use this to find code references, function definitions, or any text across files. For finding files by name, use glob instead.",
)
def grep(
    pattern: str,
    path: str = ".",
    recursive: bool = True,
    ignore_case: bool = False,
    file_pattern: str | None = None,
    context_lines: int = 0,
    max_results: int = 100,
    use_regex: bool = True,
    output_mode: Literal["content", "files", "count"] = "content",
) -> dict[str, Any]:
    """Search for text patterns inside file contents.

    Use this to find code references, definitions, usage sites, or any
    text across a project. Uses ripgrep when available (faster, respects
    .gitignore), falls back to Python implementation.
    For finding files by name/extension, use glob instead.

    Args:
        pattern: Search pattern (regex by default, literal if use_regex=False).
        path: File or directory to search in (default "." = current directory).
        recursive: Search subdirectories recursively (default True).
        ignore_case: Case-insensitive search (default False).
        file_pattern: Glob pattern to filter files (e.g., "*.py", "*.{js,ts}").
        context_lines: Lines of context around matches (default 0).
        max_results: Maximum number of matches to return (default 100).
        use_regex: Treat pattern as regex (default True).
        output_mode: Output format — "content" (matching lines with context,
            default), "files" (only file paths), or "count" (match counts
            per file).

    Returns:
        dict with:
        - success: True if search completed
        - matches: List of match objects (format depends on output_mode)
        - total_matches: Total number of matches found
        - files_searched: Number of files searched
        - truncated: True if results were truncated due to max_results
    """
    search_path = Path(path).resolve()

    if not search_path.exists():
        return {
            "success": False,
            "error": f"Path not found: {path}",
            "path": str(search_path),
        }

    # Try to use ripgrep if available (faster, respects .gitignore)
    if _ripgrep_available():
        return _grep_with_ripgrep(
            pattern=pattern,
            path=search_path,
            recursive=recursive,
            ignore_case=ignore_case,
            file_pattern=file_pattern,
            context_lines=context_lines,
            max_results=max_results,
            use_regex=use_regex,
            output_mode=output_mode,
        )

    # Fall back to Python implementation
    return _grep_python(
        pattern=pattern,
        path=search_path,
        recursive=recursive,
        ignore_case=ignore_case,
        file_pattern=file_pattern,
        context_lines=context_lines,
        max_results=max_results,
        use_regex=use_regex,
        output_mode=output_mode,
    )


def _ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _grep_with_ripgrep(
    pattern: str,
    path: Path,
    recursive: bool,
    ignore_case: bool,
    file_pattern: str | None,
    context_lines: int,
    max_results: int,
    use_regex: bool,
    output_mode: Literal["content", "files", "count"],
) -> dict[str, Any]:
    """Use ripgrep for fast searching."""
    cmd = ["rg", "--json"]

    if not use_regex:
        cmd.append("--fixed-strings")

    if ignore_case:
        cmd.append("--ignore-case")

    if not recursive:
        cmd.append("--max-depth=1")

    if context_lines > 0:
        cmd.extend(["-C", str(context_lines)])

    if file_pattern:
        cmd.extend(["--glob", file_pattern])

    if output_mode == "files":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")

    cmd.extend(["--max-count", str(max_results * 10)])  # Get more, then trim
    cmd.append(pattern)
    cmd.append(str(path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Search timed out after 30 seconds",
            "matches": [],
            "total_matches": 0,
            "files_searched": 0,
            "truncated": False,
        }
    except FileNotFoundError:
        # Ripgrep not found, fall back to Python
        return _grep_python(
            pattern=pattern,
            path=path,
            recursive=recursive,
            ignore_case=ignore_case,
            file_pattern=file_pattern,
            context_lines=context_lines,
            max_results=max_results,
            use_regex=use_regex,
            output_mode=output_mode,
        )

    # Parse ripgrep JSON output
    import json

    matches = []
    files_searched = set()
    total_matches = 0

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "match":
            match_data = data.get("data", {})
            file_path = match_data.get("path", {}).get("text", "")
            files_searched.add(file_path)

            if len(matches) < max_results:
                if output_mode == "content":
                    matches.append({
                        "file": file_path,
                        "line_number": match_data.get("line_number", 0),
                        "content": match_data.get("lines", {}).get("text", "").rstrip("\n"),
                    })
                total_matches += 1

        elif data.get("type") == "summary":
            stats = data.get("data", {}).get("stats", {})
            total_matches = stats.get("matches", total_matches)

    # Handle files and count modes differently
    if output_mode == "files":
        matches = [{"file": f, "match_count": 1} for f in sorted(files_searched)][:max_results]
    elif output_mode == "count":
        # Re-run with count flag if needed
        pass

    return {
        "success": True,
        "matches": matches,
        "total_matches": total_matches,
        "files_searched": len(files_searched),
        "truncated": len(matches) >= max_results,
    }


def _grep_python(
    pattern: str,
    path: Path,
    recursive: bool,
    ignore_case: bool,
    file_pattern: str | None,
    context_lines: int,
    max_results: int,
    use_regex: bool,
    output_mode: Literal["content", "files", "count"],
) -> dict[str, Any]:
    """Pure Python implementation of grep."""
    # Compile pattern
    flags = re.IGNORECASE if ignore_case else 0
    try:
        if use_regex:
            regex = re.compile(pattern, flags)
        else:
            regex = re.compile(re.escape(pattern), flags)
    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {e}",
            "matches": [],
            "total_matches": 0,
            "files_searched": 0,
            "truncated": False,
        }

    matches = []
    files_searched = 0
    total_matches = 0
    file_counts: dict[str, int] = {}

    # Get files to search
    if path.is_file():
        files = [path]
    else:
        if file_pattern:
            if recursive:
                files = list(path.rglob(file_pattern))
            else:
                files = list(path.glob(file_pattern))
        else:
            if recursive:
                files = [f for f in path.rglob("*") if f.is_file()]
            else:
                files = [f for f in path.iterdir() if f.is_file()]

    for file_path in files:
        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text()
            lines = content.splitlines()
        except (UnicodeDecodeError, PermissionError):
            continue

        files_searched += 1
        file_match_count = 0

        for i, line in enumerate(lines):
            if regex.search(line):
                total_matches += 1
                file_match_count += 1

                if len(matches) < max_results and output_mode == "content":
                    match_obj: dict[str, Any] = {
                        "file": str(file_path),
                        "line_number": i + 1,
                        "content": line,
                    }

                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        match_obj["context_before"] = lines[start:i]
                        match_obj["context_after"] = lines[i + 1:end]

                    matches.append(match_obj)

        if file_match_count > 0:
            file_counts[str(file_path)] = file_match_count

    # Format output based on mode
    if output_mode == "files":
        matches = [
            {"file": f, "match_count": c}
            for f, c in sorted(file_counts.items())
        ][:max_results]
    elif output_mode == "count":
        matches = [
            {"file": f, "count": c}
            for f, c in sorted(file_counts.items())
        ][:max_results]

    return {
        "success": True,
        "matches": matches,
        "total_matches": total_matches,
        "files_searched": files_searched,
        "truncated": total_matches > max_results,
    }
