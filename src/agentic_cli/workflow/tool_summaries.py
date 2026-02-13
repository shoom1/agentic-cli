"""Tool-specific result summary formatters.

Provides meaningful one-line summaries for tool results instead of
generic "Returned N fields" messages. Used by both ADK and LangGraph
event processing paths.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from agentic_cli.constants import format_size, truncate, TOOL_SUMMARY_MAX_LENGTH


def _read_file(r: dict) -> str:
    name = os.path.basename(r["path"])
    if "offset" in r:
        end = r["offset"] + r["lines_read"]
        return f"Read {name} (lines {r['offset']}-{end} of {r['total_lines']})"
    return f"Read {name} ({format_size(r['size'])})"


def _diff_compare(r: dict) -> str:
    s = r["summary"]
    sim = round(r["similarity"] * 100)
    if isinstance(s, dict):
        return f"+{s.get('added', 0)} -{s.get('removed', 0)} ~{s.get('changed', 0)}, {sim}% similar"
    return f"{s}, {sim}% similar"


def _glob(r: dict) -> str:
    return f"{r['count']} files matched"


def _list_dir(r: dict) -> str:
    d = len(r.get("directories", []))
    f = len(r.get("files", []))
    return f"{d} dirs, {f} files"


def _grep(r: dict) -> str:
    matches = r["total_matches"]
    files = len(r["matches"]) if isinstance(r.get("matches"), list) else 0
    return f"{matches} matches in {files} files"


def _write_file(r: dict) -> str:
    name = os.path.basename(r["path"])
    verb = "Created" if r.get("created") else "Wrote"
    return f"{verb} {name} ({format_size(r['size'])})"


def _edit_file(r: dict) -> str:
    name = os.path.basename(r["path"])
    n = r["replacements"]
    return f"{n} replacement{'s' if n != 1 else ''} in {name}"


def _execute_python(r: dict) -> str:
    output = r.get("output", "")
    if output and output.strip():
        first_line = output.strip().splitlines()[0]
        return truncate(first_line, TOOL_SUMMARY_MAX_LENGTH)
    dur = r.get("execution_time_ms", 0)
    return f"No output ({dur:.0f}ms)"


def _shell_executor(r: dict) -> str:
    stdout = r.get("stdout", "")
    if stdout and stdout.strip():
        first_line = stdout.strip().splitlines()[0]
        return truncate(first_line, TOOL_SUMMARY_MAX_LENGTH)
    code = r.get("return_code", 0)
    dur = r.get("duration", 0)
    return f"Exit {code} ({dur:.1f}s)"


def _get_tasks(r: dict) -> str:
    return f"{r['count']} tasks"


def _get_plan(r: dict) -> str:
    if r.get("content"):
        return "Plan retrieved"
    return r.get("message", "No plan")


def _search_memory(r: dict) -> str:
    return f"{r['count']} memories found"


def _web_search(r: dict) -> str:
    results = r.get("results", [])
    query = truncate(r.get("query", ""), 60)
    first_line = f'"{query}" — Found {len(results)} results'
    if not results:
        return first_line
    lines = [first_line]
    show = results[:5]
    for i, res in enumerate(show):
        title = truncate(res["title"], 60)
        prefix = "└─" if i == len(show) - 1 else "├─"
        lines.append(f"  {prefix} {title}")
    return "\n".join(lines)


def _search_arxiv(r: dict) -> str:
    total = r["total_found"]
    query = truncate(r.get("query", ""), 60)
    first_line = f'"{query}" — Found {total} papers'
    papers = r.get("papers", [])
    if not papers:
        return first_line
    lines = [first_line]
    show = papers[:5]
    for i, p in enumerate(show):
        title = truncate(p["title"], 60)
        prefix = "└─" if i == len(show) - 1 else "├─"
        lines.append(f"  {prefix} {title}")
    return "\n".join(lines)


def _fetch_arxiv_paper(r: dict) -> str:
    title = r["paper"]["title"]
    return truncate(title, TOOL_SUMMARY_MAX_LENGTH)


def _ingest_document(r: dict) -> str:
    title = r["title"]
    chunks = r["chunks_created"]
    return f"Ingested '{truncate(title, 40)}' ({chunks} chunks)"


def _read_document(r: dict) -> str:
    title = r["title"]
    trunc = " (truncated)" if r.get("truncated") else ""
    return f"{truncate(title, 70)}{trunc}"


def _list_documents(r: dict) -> str:
    count = r["count"]
    return f"{count} document{'s' if count != 1 else ''}"


def _open_document(r: dict) -> str:
    title = r.get("title", "")
    return f"Opened: {truncate(title, 80)}"


def _request_approval(r: dict) -> str:
    if r.get("approved"):
        return "Approved"
    reason = r.get("reason", "")
    if reason:
        return f"Rejected: {truncate(reason, TOOL_SUMMARY_MAX_LENGTH - 10)}"
    return "Rejected"


_TOOL_FORMATTERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "read_file": _read_file,
    "diff_compare": _diff_compare,
    "glob": _glob,
    "list_dir": _list_dir,
    "grep": _grep,
    "write_file": _write_file,
    "edit_file": _edit_file,
    "execute_python": _execute_python,
    "shell_executor": _shell_executor,
    "get_tasks": _get_tasks,
    "get_plan": _get_plan,
    "search_memory": _search_memory,
    "web_search": _web_search,
    "search_arxiv": _search_arxiv,
    "fetch_arxiv_paper": _fetch_arxiv_paper,
    "ingest_document": _ingest_document,
    "read_document": _read_document,
    "list_documents": _list_documents,
    "open_document": _open_document,
    "request_approval": _request_approval,
}


def format_tool_summary(tool_name: str, result: dict[str, Any]) -> str | None:
    """Format a tool-specific summary from a result dict.

    Args:
        tool_name: Name of the tool that produced the result.
        result: The tool's return dict.

    Returns:
        Human-readable summary string, or None if no formatter exists
        or the formatter fails (missing keys, wrong types, etc.).
    """
    formatter = _TOOL_FORMATTERS.get(tool_name)
    if formatter:
        try:
            return formatter(result)
        except (KeyError, TypeError, IndexError, ValueError, AttributeError):
            return None
    return None
