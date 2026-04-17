"""Bounded tool reflection memory.

Stores heuristics learned from tool failures. Each tool keeps at most
N reflections (FIFO eviction). Reflections can be injected into tool
descriptions to help agents avoid repeating mistakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

from agentic_cli.file_utils import atomic_write_json
from agentic_cli.logging import get_logger
from agentic_cli.tools.registry import PermissionLevel, ToolCategory, register_tool
from agentic_cli.workflow.service_registry import require_service

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings

logger = get_logger(__name__)

REFLECTION_STORE = "reflection_store"


@dataclass
class ToolReflection:
    """A learned heuristic from a tool failure."""

    tool_name: str
    error_summary: str
    heuristic: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "error_summary": self.error_summary,
            "heuristic": self.heuristic,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolReflection:
        return cls(
            tool_name=data["tool_name"],
            error_summary=data["error_summary"],
            heuristic=data["heuristic"],
            created_at=data.get("created_at", ""),
        )


class ReflectionStore:
    """Bounded store for tool-use reflections."""

    def __init__(self, settings: "BaseSettings", max_per_tool: int = 3):
        self._max_per_tool = max_per_tool
        mem_dir = settings.workspace_dir / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        self._path = mem_dir / "reflections.json"
        self._reflections: dict[str, list[ToolReflection]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for tool_name, items in data.items():
                    self._reflections[tool_name] = [
                        ToolReflection.from_dict(item) for item in items
                    ]
            except (json.JSONDecodeError, KeyError):
                logger.warning("corrupted_reflections_file", path=str(self._path))
                self._reflections = {}

    def _save(self) -> None:
        data = {
            tool_name: [r.to_dict() for r in reflections]
            for tool_name, reflections in self._reflections.items()
        }
        atomic_write_json(self._path, data)

    def save(self, tool_name: str, error_summary: str, heuristic: str) -> None:
        """Save a new reflection. Evicts oldest if at capacity."""
        reflection = ToolReflection(
            tool_name=tool_name,
            error_summary=error_summary,
            heuristic=heuristic,
            created_at=datetime.now().isoformat(),
        )
        if tool_name not in self._reflections:
            self._reflections[tool_name] = []
        self._reflections[tool_name].append(reflection)
        if len(self._reflections[tool_name]) > self._max_per_tool:
            self._reflections[tool_name] = self._reflections[tool_name][-self._max_per_tool:]
        self._save()

    def get_for_tool(self, tool_name: str) -> list[ToolReflection]:
        return list(self._reflections.get(tool_name, []))

    def get_all(self) -> dict[str, list[ToolReflection]]:
        return dict(self._reflections)

    def format_for_prompt(self, tool_name: str) -> str:
        """Format reflections for injection into a tool description."""
        reflections = self.get_for_tool(tool_name)
        if not reflections:
            return ""
        lines = [f"Note: Previous experience with {tool_name}:"]
        for r in reflections:
            lines.append(f"- {r.heuristic}")
        return "\n".join(lines)


@register_tool(
    category=ToolCategory.MEMORY,
    permission_level=PermissionLevel.SAFE,
    description="Save a learned heuristic from a tool failure",
)
def save_reflection(
    tool_name: str,
    error_summary: str,
    heuristic: str,
) -> dict[str, Any]:
    """Save a reflection about a tool failure.

    Args:
        tool_name: Name of the tool that failed.
        error_summary: Brief description of what went wrong.
        heuristic: What to do differently next time.

    Returns:
        A dict indicating success.
    """
    store = require_service(REFLECTION_STORE)
    if isinstance(store, dict):
        return store
    store.save(tool_name, error_summary, heuristic)
    return {
        "success": True,
        "message": f"Reflection saved for {tool_name}",
    }
