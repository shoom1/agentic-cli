"""The extraction prompt and its parser: what the extractor asks a model, and
what of its answer becomes candidates (spec §6.8).

The host makes the model call; nothing here talks to a model.
"""

from __future__ import annotations

import json
import re

from agentic_cli.memory.store._files import valid_id
from agentic_cli.memory.store.models import ACTIONS, MEMORY_TYPES, SCOPES, Candidate

MAX_CANDIDATES = 5

_INSTRUCTIONS = """You keep a user's long-term memory for an assistant. Read the conversation below and propose at most 5 memories worth keeping for later conversations.

Propose a memory only when it is:
- lasting: true beyond this conversation (a preference, a correction, a decision, a fact about the user or the project, a pointer to an external resource);
- new: not already in the memory index below (to change one that is there, propose an update of it by its id);
- not readable from the project's own files;
- never a secret (no keys, tokens or passwords).

Types: "user" (who the user is: role, expertise), "feedback" (how the user wants the work done: corrections and confirmed approaches), "project" (ongoing work, goals and decisions not recorded in files), "reference" (where something lives: a URL, a dashboard, a document).
Scopes: "user" only for what holds across all of the user's projects; otherwise "project".

Answer with a JSON array and nothing else; [] when there is nothing worth keeping. Each item is either
{"action": "new", "description": "<one line, at most 150 characters>", "content": "<the memory, at most 4000 characters>", "type": "<type>", "scope": "<scope>"}
or, to change a memory in the index,
{"action": "update", "id": "<its id>", "description": "...", "content": "...", "type": "...", "scope": "<its scope>"}"""

_FENCE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)\n?```", re.DOTALL)


def extraction_prompt(messages: list[dict], index: str, discarded: list[str]) -> str:
    """The prompt: instructions, the index (with ids), the discarded descriptions,
    then the conversation (``role: text`` lines), last."""
    parts = [_INSTRUCTIONS, "Memory index:\n" + (index or "(no memories yet)")]
    if discarded:
        parts.append(
            "The user rejected these proposals; do not propose them again:\n"
            + "\n".join(f"- {description}" for description in discarded)
        )
    conversation = "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)
    parts.append("Conversation:\n" + conversation)
    return "\n\n".join(parts)


def _candidate(item: object) -> Candidate | None:
    if not isinstance(item, dict):
        return None
    action = item.get("action")
    fields = [item.get(key) for key in ("description", "content", "type", "scope")]
    if action not in ACTIONS or not all(isinstance(value, str) for value in fields):
        return None
    description, content, memory_type, scope = fields
    if memory_type not in MEMORY_TYPES or scope not in SCOPES:
        return None
    target = item.get("id")
    if action == "update" and not valid_id(target):
        return None
    if action == "new" and target not in (None, ""):
        return None
    return Candidate(action, target if action == "update" else None, description, content, memory_type, scope)  # type: ignore[arg-type]


def parse_candidates(text: str) -> list[Candidate]:
    """The candidates in a model's answer: a JSON array, bare or in one code fence.

    The first ``MAX_CANDIDATES`` items are read; a malformed item is dropped;
    anything that is not such an array gives ``[]``.
    """
    stripped = text.strip()
    fenced = _FENCE.search(stripped)
    body = fenced.group(1) if fenced else stripped
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return []
    if not isinstance(data, list):
        return []
    found = (_candidate(item) for item in data[:MAX_CANDIDATES])
    return [candidate for candidate in found if candidate is not None]
