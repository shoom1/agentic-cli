"""Shared task logic — pure functions, no framework imports.

Operates on plain ``list[dict]`` task data. Each task dict has keys:
id, description, status, priority, tags, created_at, completed_at.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Valid task statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Valid task priorities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---- Validation ----

_VALID_STATUSES = frozenset(s.value for s in TaskStatus)
_VALID_PRIORITIES = frozenset(p.value for p in TaskPriority)


def validate_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Validate a list of task dicts.

    Returns an error dict ``{"success": False, "error": "..."}`` on
    first invalid task, or ``None`` if all are valid.
    """
    for i, task in enumerate(tasks):
        if not task.get("description"):
            return {
                "success": False,
                "error": f"Task at index {i} is missing 'description'",
            }
        status = task.get("status", "pending")
        if status not in _VALID_STATUSES:
            return {
                "success": False,
                "error": (
                    f"Task at index {i} has invalid status '{status}'. "
                    f"Valid: {', '.join(sorted(_VALID_STATUSES))}"
                ),
            }
        priority = task.get("priority", "medium")
        if priority not in _VALID_PRIORITIES:
            return {
                "success": False,
                "error": (
                    f"Task at index {i} has invalid priority '{priority}'. "
                    f"Valid: {', '.join(sorted(_VALID_PRIORITIES))}"
                ),
            }
    return None


# ---- Normalization ----

def _safe_enum_value(valid: frozenset[str], value: str, default: str) -> str:
    return value if value in valid else default


def normalize_tasks(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Assign IDs, defaults, timestamps. Returns ``(normalized_list, ids)``."""
    now = datetime.now().isoformat()
    normalized: list[dict[str, Any]] = []
    ids: list[str] = []
    for t in tasks:
        tid = t.get("id") or str(uuid.uuid4())[:8]
        status = _safe_enum_value(_VALID_STATUSES, t.get("status", "pending"), "pending")
        completed_at = t.get("completed_at", "")
        if status == "completed" and not completed_at:
            completed_at = now
        normalized.append({
            "id": tid,
            "description": t["description"],
            "status": status,
            "priority": _safe_enum_value(_VALID_PRIORITIES, t.get("priority", "medium"), "medium"),
            "tags": t.get("tags", []),
            "created_at": t.get("created_at", now),
            "completed_at": completed_at,
        })
        ids.append(tid)
    return normalized, ids


# ---- Filtering ----

def filter_tasks(
    tasks: list[dict[str, Any]],
    status: str | None = None,
    priority: str | None = None,
    tag: str | None = None,
) -> list[dict[str, Any]]:
    """Filter task dicts by status, priority, and/or tag."""
    result = tasks
    if status:
        result = [t for t in result if t.get("status") == status]
    if priority:
        result = [t for t in result if t.get("priority") == priority]
    if tag:
        result = [t for t in result if tag in t.get("tags", [])]
    return result


# ---- Display ----

_STATUS_ICONS = {
    "completed": "[✓]",
    "in_progress": "[▸]",
    "pending": "[ ]",
    "cancelled": "[-]",
}

_STATUS_ORDER = {
    "in_progress": 0,
    "pending": 1,
    "cancelled": 2,
    "completed": 3,
}


def format_task_checklist(tasks: list[dict[str, Any]]) -> str:
    """Format task dicts as a compact checklist with status icons."""
    if not tasks:
        return ""
    sorted_tasks = sorted(tasks, key=lambda t: _STATUS_ORDER.get(t.get("status", "pending"), 1))
    lines = []
    for t in sorted_tasks:
        icon = _STATUS_ICONS.get(t.get("status", "pending"), "[ ]")
        lines.append(f"{icon} {t['description']}")
    return "\n".join(lines)


# ---- Progress ----

def task_progress_data(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compute progress info from a list of task dicts.

    Returns a dict with keys: progress, display, current_task_id,
    current_task_description, all_done.  Returns ``None`` if the
    list is empty.
    """
    if not tasks:
        return None

    counts = {"total": 0, "pending": 0, "in_progress": 0, "completed": 0, "cancelled": 0}
    current_task = None
    for t in tasks:
        counts["total"] += 1
        s = t.get("status", "pending")
        if s in counts:
            counts[s] += 1
        if s == "in_progress" and current_task is None:
            current_task = t

    terminal = counts["completed"] + counts["cancelled"]
    all_done = counts["total"] > 0 and terminal == counts["total"]

    return {
        "progress": counts,
        "display": format_task_checklist(tasks),
        "current_task_id": current_task["id"] if current_task else None,
        "current_task_description": current_task["description"] if current_task else None,
        "all_done": all_done,
    }
