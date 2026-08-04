"""Load Agent-Skills (SKILL.md folders) into ADK Skill objects.

Skills follow the Agent Skills specification (same SKILL.md format used by
ADK and Anthropic): a folder with frontmatter (name/description), a markdown
body, and optional ``references/`` / ``assets/`` / ``scripts/`` subfolders.

This reuses ADK's native loader (``google.adk.skills.load_skill_from_dir``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SkillStore:
    """Resolve skill references (paths or names) to loaded Skill objects.

    A skill reference is either a path to a skill directory, or a bare name
    looked up as a subdirectory of one of ``skill_dirs``.
    """

    def __init__(self, skill_dirs: list[str | Path] | None = None) -> None:
        self._search_dirs = [Path(d) for d in (skill_dirs or [])]

    def _load_dir(self, path: Path) -> Any:
        from google.adk.skills import load_skill_from_dir

        return load_skill_from_dir(path)

    def _resolve_ref(self, ref: str) -> Path:
        path = Path(ref)
        if path.is_dir():
            return path
        for base in self._search_dirs:
            candidate = base / ref
            if candidate.is_dir():
                return candidate
        searched = [str(d) for d in self._search_dirs]
        raise ValueError(
            f"Skill {ref!r} not found: not a directory and not under "
            f"skills_dirs {searched}."
        )

    def resolve(self, refs: list[str]) -> list[Any]:
        """Load the given skill refs, de-duplicated by skill name (last wins)."""
        loaded = [self._load_dir(self._resolve_ref(ref)) for ref in refs]
        by_name: dict[str, Any] = {}
        for skill in loaded:
            by_name[skill.name] = skill
        return list(by_name.values())
