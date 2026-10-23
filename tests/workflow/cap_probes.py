"""Exempt test tools for the tool result cap, registered once for every test module.

``cap_probe_report`` returns about 120,000 characters, far over the default
limit of 50,000. ``tests/workflow/test_tool_result_cap.py`` (ADK) and
``tests/workflow/deepagents/test_tool_result_cap.py`` both import them; the
registry takes one name once.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agentic_cli.tools.registry import ToolCategory, register_tool
from agentic_cli.workflow.permissions import EXEMPT

#: About 120,000 characters, one numbered line after another.
REPORT = "".join(f"report line {i:06d}\n" for i in range(6_400))

#: The note on a result whose full output was saved, for the default app name.
SAVED_NOTE = re.compile(
    r"^Cut: [\d,]+ characters\. Full output: (?P<path>\./\.agentic_cli/tool-output/"
    r"(?P<name>[A-Za-z0-9_.-]{1,64}-[0-9a-f]{12}\.txt)); read it with read_file using offset "
    r"and limit\.$"
)
#: The note when it could not be saved.
UNSAVED_NOTE = re.compile(r"^Cut: [\d,]+ characters\. The full output could not be saved\.$")
#: The note on a saved output, anywhere in a text.
NOTE_IN_TEXT = re.compile(SAVED_NOTE.pattern.removeprefix("^").removesuffix("$"))
#: Where a string was cut.
CUT = re.compile(r"\n…\[[\d,]+ characters cut\]…\n")


def saved_path(note: str) -> Path:
    """The saved output a note points at."""
    match = SAVED_NOTE.match(note)
    assert match, note
    return Path.cwd() / match["path"]


def saved_outputs(folder: Path) -> list[Path]:
    """The outputs saved in ``folder``, by name."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.name != ".gitignore")


@register_tool(capabilities=EXEMPT, category=ToolCategory.OTHER)
def cap_probe_report() -> dict:
    """Returns a report far over the tool result limit (an exempt test tool)."""
    return {"success": True, "report": REPORT}


@register_tool(capabilities=EXEMPT, category=ToolCategory.OTHER)
def cap_probe_job() -> dict:
    """Returns a finished job's result, its output far over the limit (an exempt test tool)."""
    return {"success": True, "state": "succeeded", "result": {"stdout": REPORT, "exit_code": 0}}


@register_tool(capabilities=EXEMPT, category=ToolCategory.OTHER)
def cap_probe_tasks(tool_context: Any = None) -> dict:
    """Starts a task list, then returns the long report (an exempt ADK test tool)."""
    tool_context.state["tasks"] = [
        {"id": "1", "description": "Read the report", "status": "in_progress"},
    ]
    return {"success": True, "report": REPORT}


@register_tool(capabilities=EXEMPT, category=ToolCategory.OTHER)
def cap_probe_raises() -> dict:
    """Raises, so its call gets the error result (an exempt test tool)."""
    raise RuntimeError("probe failed")
