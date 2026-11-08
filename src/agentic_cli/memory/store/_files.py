"""One memory file: its frontmatter, its name, its fields' rules (spec §5.2).

A memory is a markdown file. Its frontmatter is a flat subset of YAML that
the package reads and writes itself: ``key: value`` lines between two ``---``
lines, a value bare when that is safe and a JSON string (which YAML also
reads) otherwise. Keys the store does not know are kept, in their order,
whenever it rewrites the file: they are a person's.
"""

from __future__ import annotations

import json
import re
import secrets
import unicodedata
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from agentic_cli.memory.store.models import (
    ACTIONS,
    MEMORY_TYPES,
    SOURCES,
    Memory,
    Proposal,
    Scope,
)

DESCRIPTION_MAX = 150
CONTENT_MAX = 4_000
FILE_MAX_BYTES = 64 * 1024
MEMORY_KEYS: tuple[str, ...] = (
    "id", "description", "type", "source", "conversation", "created", "updated",
)
PROPOSAL_KEYS: tuple[str, ...] = ("action", "target", "outside_content")

_ID = re.compile(r"[A-Za-z0-9-]{1,64}")
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_YAML_WORDS = frozenset({"true", "false", "null", "yes", "no", "on", "off", "~"})
_RADIX_NUMBER = re.compile(r"[-+]?0[xXoObB][0-9a-fA-F_]+")
_LINE_BREAKS = frozenset({"Cc", "Zl", "Zp"})
_COMMENT = re.compile(r"\s#")


class FormatError(ValueError):
    """The text has no frontmatter, or a line of it is not ``key: value``."""


class FieldError(ValueError):
    """A field fails its rule. The message starts with the field's name."""


def valid_id(value: object) -> bool:
    """Whether ``value`` is a memory id: 1-64 letters, digits and dashes."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def new_id(taken: Callable[[str], bool]) -> str:
    """Eight random hex characters that ``taken`` says are free."""
    while True:
        candidate = secrets.token_hex(4)
        if not taken(candidate):
            return candidate


def normalized(text: str) -> str:
    """``text`` as duplicates are compared: NFC, casefolded, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def slug(description: str) -> str:
    """A file-name form of ``description``: at most 40 of ``[a-z0-9-]``, else ``memory``."""
    text = re.sub(r"[^a-z0-9]+", "-", description.casefold()).strip("-")
    if len(text) > 40:
        cut = text[:41]
        dash = cut.rfind("-")
        text = cut[:dash] if dash > 0 else text[:40]
    return text.strip("-") or "memory"


def file_name(description: str, memory_id: str) -> str:
    """``<slug>-<id>.md``."""
    return f"{slug(description)}-{memory_id}.md"


def description_problem(text: str) -> str | None:
    """Why ``text`` (stripped) cannot be a description, or None."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) > DESCRIPTION_MAX:
        return f"longer than {DESCRIPTION_MAX} characters"
    if any(unicodedata.category(ch) in _LINE_BREAKS for ch in stripped):
        return "holds a line break or a control character"
    return None


def content_problem(text: str) -> str | None:
    """Why ``text`` (stripped) cannot be a memory's content, or None."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) > CONTENT_MAX:
        return (
            f"longer than {CONTENT_MAX} characters; split it, or keep long material "
            "in the knowledge base"
        )
    return None


def format_time(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` in UTC."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: str) -> datetime | None:
    """An ISO 8601 date-time or date as an aware UTC time; None when it is neither.

    An offset is converted to UTC, none is taken as UTC, a date is midnight UTC.
    """
    text = value.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return _RADIX_NUMBER.fullmatch(value) is not None
    return True


def _bare_ok(value: str) -> bool:
    return (
        bool(value)
        and value[0].isalnum()
        and value == value.rstrip()
        and ": " not in value
        and " #" not in value
        and "\n" not in value
        and "\r" not in value
        and value.casefold() not in _YAML_WORDS
        and not _is_number(value)
    )


def write_value(value: str) -> str:
    """``value`` as a frontmatter value: bare when safe, else a JSON string."""
    return value if _bare_ok(value) else json.dumps(value, ensure_ascii=False)


def read_value(raw: str) -> str:
    """A frontmatter value: a JSON string, a YAML single-quoted string, or bare text.

    A bare value ends at a ``#`` that follows whitespace, as in YAML: the rest
    of the line is a comment.

    Raises:
        FormatError: a quoted value that does not close.
    """
    text = raw.strip()
    if text.startswith('"'):
        try:
            value = json.loads(text)
        except ValueError as e:
            raise FormatError(f"a quoted value that does not parse: {e}") from None
        if not isinstance(value, str):
            raise FormatError("a quoted value that is not a string")
        return value
    if text.startswith("'"):
        if len(text) < 2 or not text.endswith("'"):
            raise FormatError("a single-quoted value that does not close")
        return text[1:-1].replace("''", "'")
    return _COMMENT.split(raw, maxsplit=1)[0].strip()


def parse(text: str) -> tuple[dict[str, str], str]:
    """The frontmatter's fields and the content after it.

    Raises:
        FormatError: no frontmatter, no closing ``---``, or a line that is not
            ``key: value`` (the message names the line).
    """
    text = text.removeprefix("﻿").replace("\r\n", "\n")
    lines = text.split("\n")
    if lines[0] != "---":
        raise FormatError("no frontmatter: the first line is not ---")
    fields: dict[str, str] = {}
    for number, line in enumerate(lines[1:], start=2):
        if line == "---":
            return fields, "\n".join(lines[number:]).rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, rest = line.partition(":")
        if not sep or line[0].isspace() or not _KEY.fullmatch(key):
            raise FormatError(f"line {number} is not 'key: value'")
        if key in fields:
            raise FormatError(f"line {number} repeats the key {key!r}")
        try:
            fields[key] = read_value(rest)
        except FormatError as e:
            raise FormatError(f"line {number}: {e}") from None
    raise FormatError("the frontmatter has no closing ---")


def render(fields: Mapping[str, str], content: str, order: tuple[str, ...] = MEMORY_KEYS) -> str:
    """A file's text: the keys of ``order`` first, the rest in their order, then ``content``."""
    lines = ["---"]
    lines += [f"{key}: {write_value(fields[key])}" for key in order if key in fields]
    lines += [f"{key}: {write_value(value)}" for key, value in fields.items() if key not in order]
    lines.append("---")
    return "\n".join(lines) + "\n" + content.rstrip() + "\n"


def _time_field(fields: Mapping[str, str], key: str, default: datetime) -> datetime:
    if key not in fields:
        return default
    moment = parse_time(fields[key])
    if moment is None:
        raise FieldError(f"{key}: {fields[key]!r} is not a date")
    return moment


def _common(fields: Mapping[str, str]) -> tuple[str, str, str, str]:
    """The id, description, type and source every file holds, checked."""
    memory_id = fields.get("id", "")
    if not valid_id(memory_id):
        raise FieldError("id: missing, or not 1-64 letters, digits and dashes")
    description = fields.get("description", "")
    reason = description_problem(description)
    if reason:
        raise FieldError(f"description: {reason}")
    memory_type = fields.get("type", "")
    if memory_type not in MEMORY_TYPES:
        raise FieldError(f"type: {memory_type!r} is not one of {', '.join(MEMORY_TYPES)}")
    source = fields.get("source") or "user"
    if source not in SOURCES:
        raise FieldError(f"source: {source!r} is not one of {', '.join(SOURCES)}")
    return memory_id, description.strip(), memory_type, source


def memory_from(
    fields: Mapping[str, str], content: str, *, scope: Scope, path: Path, mtime: datetime
) -> tuple[Memory, dict[str, str]]:
    """The memory a file holds, and the keys the store does not know.

    A file a person wrote may leave out ``source`` (``user``), ``created`` and
    ``updated`` (the file's ``mtime``) and ``conversation``.

    Raises:
        FieldError: a field fails its rule.
    """
    memory_id, description, memory_type, source = _common(fields)
    memory = Memory(
        id=memory_id,
        description=description,
        content=content,
        type=memory_type,  # type: ignore[arg-type]
        scope=scope,
        source=source,  # type: ignore[arg-type]
        conversation=fields.get("conversation") or None,
        created=_time_field(fields, "created", mtime),
        updated=_time_field(fields, "updated", mtime),
        path=path,
    )
    extra = {key: value for key, value in fields.items() if key not in MEMORY_KEYS}
    return memory, extra


def memory_fields(memory: Memory, extra: Mapping[str, str]) -> dict[str, str]:
    """The frontmatter of ``memory``, then ``extra``."""
    fields = {
        "id": memory.id,
        "description": memory.description,
        "type": memory.type,
        "source": memory.source,
    }
    if memory.conversation:
        fields["conversation"] = memory.conversation
    fields["created"] = format_time(memory.created)
    fields["updated"] = format_time(memory.updated)
    fields.update({key: value for key, value in extra.items() if key not in fields})
    return fields


def proposal_from(fields: Mapping[str, str], content: str, *, scope: Scope, path: Path) -> Proposal:
    """The proposal an inbox file holds.

    Raises:
        FieldError: a field fails its rule, or ``action`` and ``target``
            disagree (an update names its memory, a new memory names none).
    """
    proposal_id, description, memory_type, source = _common(fields)
    action = fields.get("action", "")
    if action not in ACTIONS:
        raise FieldError(f"action: {action!r} is not one of {', '.join(ACTIONS)}")
    target = fields.get("target") or None
    if action == "update" and not valid_id(target):
        raise FieldError("target: an update names the memory it changes")
    if action == "new" and target is not None:
        raise FieldError("target: a new memory names none")
    created = parse_time(fields.get("created", ""))
    if created is None:
        raise FieldError("created: missing, or not a date")
    return Proposal(
        id=proposal_id,
        action=action,  # type: ignore[arg-type]
        target=target,
        description=description,
        content=content,
        type=memory_type,  # type: ignore[arg-type]
        scope=scope,
        source=source,  # type: ignore[arg-type]
        conversation=fields.get("conversation") or None,
        outside_content=fields.get("outside_content") or None,
        created=created,
        path=path,
    )


def proposal_fields(proposal: Proposal) -> dict[str, str]:
    """The frontmatter of ``proposal``."""
    fields = {
        "id": proposal.id,
        "description": proposal.description,
        "type": proposal.type,
        "source": proposal.source,
    }
    if proposal.conversation:
        fields["conversation"] = proposal.conversation
    fields["created"] = format_time(proposal.created)
    fields["action"] = proposal.action
    if proposal.target:
        fields["target"] = proposal.target
    if proposal.outside_content:
        fields["outside_content"] = proposal.outside_content
    return fields
