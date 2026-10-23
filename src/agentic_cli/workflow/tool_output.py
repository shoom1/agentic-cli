"""Tool results too big for the context: cut, with the full output saved.

A tool result whose serialized text is longer than ``tool_result_max_chars``
reaches the model cut: its shape, its beginning and its end, and a note saying
how much was cut and where the full output is, a file in
``./.<app_name>/tool-output/`` that ``read_file`` pages through. Both backends
call :func:`cap_tool_result` on every result (ADK's ``ToolResultCapPlugin``,
Deep Agents' ``ToolResultCapMiddleware``). Each tool's own limits stay; this
is the net beneath them.

How a result is cut:

- A dict keeps its keys. Its longest string, wherever it is (nested dicts and
  lists included), is cut first, then the next, until the whole fits; no
  string is cut below :data:`MIN_KEPT` characters. A cut string keeps as much
  as fits, its head (:data:`HEAD_SHARE` of what is kept) and its tail around
  ``…[N characters cut]…``. The note goes under ``truncated``, replacing a
  tool's own flag (it is truthy, as the flag would be). When strings alone
  cannot make it fit (many small values), the result becomes its JSON text,
  cut the same way, under ``result``, with ``success`` and ``error`` kept.
- A string is cut the same way, with the note appended.
- Anything else is wrapped as ``{"result": value}`` and cut as a dict.

The saved file is meant to be read in pages of lines: a string is saved as it
was, and anything else as indented JSON in which each string the cut
shortened stands as a pointer, the string itself following whole, with its
own line breaks, under ``--- <where it was> ---``. A line longer than
:data:`MAX_LINE_CHARS` is split into lines of that length, so a one-line
output (minified JSON, base64) pages too: the file is a copy to read, with
nothing left out, not a byte-exact one.

The folder is inside the project, which a repository controls, so it is
treated like ``web_fetch``'s saved pages (``tools/webfetch/saved.py``): the
location is fixed (the model never chooses it), a symlinked folder is
refused, and cleanup deletes only regular files whose names match the
folder's own pattern. Saving never fails the call: the result is still cut,
and its note says the full output could not be saved.
"""

from __future__ import annotations

import json
import os
import re
import stat
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agentic_cli.file_utils import atomic_write_text
from agentic_cli.logging import Loggers

logger = Loggers.workflow()

FOLDER_NAME = "tool-output"
MAX_AGE_DAYS = 7
MAX_FOLDER_BYTES = 100 * 1024 * 1024
#: The share of a cut string's kept characters taken from its beginning.
HEAD_SHARE = 0.7
#: A cut string keeps at least this many characters.
MIN_KEPT = 200
#: A saved output's lines are at most this long: longer ones are split.
MAX_LINE_CHARS = 2_000

#: An output saved this recently (seconds) is spared by cleanup's size pass:
#: it may be another call's, saved a moment ago, whose note points at it.
_JUST_SAVED = 60

#: A saved output's name: the tool's (made safe), then 12 hex digits.
_OWN_FILE = re.compile(r"^[A-Za-z0-9_.-]{1,64}-[0-9a-f]{12}\.txt$")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")

#: Where a value sits in a result: dict keys and list indexes, outermost first.
_Path = tuple[Any, ...]

#: The folders a save has failed in, each warned about once per process.
_unsaved_folders: set[str] = set()


class _Refused(Exception):
    """The folder is not one outputs are saved in (a symlink, or no folder)."""


def tool_output_dir(app_name: str) -> Path:
    """The project's tool-output folder, ``./.<app_name>/tool-output``."""
    return Path.cwd() / f".{app_name}" / FOLDER_NAME


def cap_tool_result(
    result: Any,
    *,
    tool_name: str,
    max_chars: int,
    folder: Path,
    source: Path | None = None,
) -> Any:
    """``result`` cut to ``max_chars``, with its full output saved in ``folder``.

    Returns ``result`` itself when its serialized text fits (a string as it
    is, anything else as JSON). Otherwise returns a cut copy; ``result`` is
    never changed.

    Args:
        result: What the tool returned.
        tool_name: The tool's name, which the saved file is named after.
        max_chars: The longest serialized result the model reads.
        folder: Where the full output is saved (:func:`tool_output_dir`).
        source: The saved output ``result`` was read from (a ``read_file`` of
            one). The result is then cut without being saved again, and its
            note points at ``source``.
    """
    text = _serialize(result)
    if text is None or len(text) <= max_chars:
        return result

    name = f"{_safe_name(tool_name)}-{uuid.uuid4().hex[:12]}.txt"
    display = _display(source if source is not None else folder / name)
    # The longest note this result can get: the cut is sized to leave room
    # for it, and a shorter one (fewer digits, or "could not be saved") fits.
    longest_note = _saved_note(len(text), display)

    value: Any
    sections: list[tuple[_Path, str]] = []
    if isinstance(result, str):
        value = result
        cut, removed = _cut_text(result, max_chars - len("\n\n") - len(longest_note))
    else:
        value = result if isinstance(result, dict) else {"result": result}
        cut, removed, sections = _cut_dict(value, text, max_chars, longest_note)
    if removed == 0:
        return result

    if source is not None:
        note = _saved_note(removed, display)
    else:
        note = _save(
            folder,
            name,
            _saved_text(value, sections),
            tool_name=tool_name,
            removed=removed,
            display=display,
        )

    if isinstance(cut, str):
        return f"{cut}\n\n{note}"
    return {**cut, "truncated": note}


def is_saved_output(path: Path, folder: Path) -> bool:
    """Whether ``path`` is an output the cap saved in ``folder``.

    It is when it sits directly in ``folder`` (and neither ``folder`` nor its
    parent is a symlink) under a saved output's name.
    """
    try:
        _require_real_dir(folder.parent)
        _require_real_dir(folder)
        path = Path(path).resolve()
        return path.parent == folder.resolve() and _OWN_FILE.match(path.name) is not None
    except (OSError, ValueError, _Refused):
        return False


def cleanup_tool_outputs(
    folder: Path,
    *,
    max_age_days: float = MAX_AGE_DAYS,
    max_bytes: int = MAX_FOLDER_BYTES,
    keep: str | None = None,
) -> None:
    """Delete old saved outputs.

    First every output older than ``max_age_days``, then the oldest until
    the folder holds at most ``max_bytes``, sparing those saved in the last
    minute (another call's, saved a moment ago). The output named ``keep`` is
    never deleted. Only regular files named like the folder's own are touched,
    and nothing in a symlinked folder or under a symlinked parent. Errors are
    logged, never raised.
    """
    try:
        _require_real_dir(folder.parent)
        _require_real_dir(folder)
        outputs: list[tuple[Path, int, float]] = []
        with os.scandir(folder) as entries:
            for entry in entries:
                if _OWN_FILE.match(entry.name) is None:
                    continue
                st = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(st.st_mode):
                    outputs.append((Path(entry.path), st.st_size, st.st_mtime))

        now = time.time()
        oldest_kept = now - max_age_days * 86400
        recent = []
        for output in outputs:
            path, _, mtime = output
            if path.name != keep and mtime < oldest_kept:
                path.unlink(missing_ok=True)
            else:
                recent.append(output)

        total = sum(size for _, size, _ in recent)
        for path, size, mtime in sorted(recent, key=lambda output: output[2]):
            if total <= max_bytes:
                break
            if path.name != keep and now - mtime >= _JUST_SAVED:
                path.unlink(missing_ok=True)
                total -= size
    except FileNotFoundError:
        return  # no folder: nothing was saved
    except Exception as e:
        # A planted entry or a permission error must never stop a tool's
        # result from reaching the model.
        logger.warning("tool_output_cleanup_failed", folder=str(folder), error=str(e))


# --- cutting ---------------------------------------------------------------------


def _serialize(value: Any) -> str | None:
    """``value``'s text as the size is measured: a string as is, else JSON."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError, RecursionError):
        # Not serializable at all (a circular structure, a key JSON cannot
        # hold): nothing to measure it by, so it is left to the backend.
        return None


def _json_len(text: str) -> int:
    """How many characters ``text`` takes inside a JSON document."""
    return len(json.dumps(text, ensure_ascii=False))


def _marker(removed: int) -> str:
    return f"\n…[{removed:,} characters cut]…\n"


def _cut_string(text: str, keep: int) -> str:
    """``text`` with only its first and last ``keep`` characters, around the marker."""
    head = round(keep * HEAD_SHARE)
    tail = text[len(text) - (keep - head):]
    return f"{text[:head]}{_marker(len(text) - keep)}{tail}"


def _cut_text(text: str, room: int) -> tuple[str, int]:
    """``text`` cut to at most ``room`` characters (never below :data:`MIN_KEPT`).

    Returns the cut text and how many characters were cut; 0 when cutting
    cannot shorten it.
    """
    keep = max(MIN_KEPT, room - len(_marker(len(text))))
    if keep + len(_marker(len(text) - keep)) >= len(text):
        return text, 0
    return _cut_string(text, keep), len(text) - keep


def _cut_json_string(text: str, room: int) -> tuple[str, int]:
    """``text`` cut so it takes at most ``room`` characters in JSON.

    It keeps as much as fits, never below :data:`MIN_KEPT`. How much fits
    depends on what is kept (quotes, backslashes and line breaks take two
    characters in JSON), so the kept length is found by bisection: the JSON
    length of a cut grows with it. A cut keeping more than ``room``
    characters cannot fit, which bounds every trial.

    Returns the cut text and how many characters were cut; 0 when cutting
    cannot shorten it.
    """
    full = _json_len(text)
    if full <= room or len(text) <= MIN_KEPT:
        return text, 0
    low, high = MIN_KEPT, max(MIN_KEPT, min(len(text) - 1, room))
    while low < high:
        keep = (low + high + 1) // 2
        if _json_len(_cut_string(text, keep)) <= room:
            low = keep
        else:
            high = keep - 1
    cut = _cut_string(text, low)
    if _json_len(cut) >= full:
        return text, 0
    return cut, len(text) - low


def _cut_dict(
    value: dict, text: str, max_chars: int, note: str
) -> tuple[dict, int, list[tuple[_Path, str]]]:
    """``value`` cut to fit with ``note`` under ``truncated``.

    Returns the cut copy, how many characters were cut, and the strings cut
    (where each was, and the whole string), in the order they appear.
    """
    work = _copy(value)
    total = len(_dumps({**work, "truncated": note}))
    removed = 0
    cut_paths: set[_Path] = set()
    # A tool's own "truncated" gives way to the note, so cutting it saves nothing.
    leaves = [leaf for leaf in _strings(work) if leaf[0][0] != "truncated"]
    # Longest first; equal lengths in the order they appear (sort is stable).
    for path, string in sorted(leaves, key=lambda leaf: -len(leaf[1])):
        if total <= max_chars:
            break
        room = _json_len(string) - (total - max_chars)
        cut, cut_out = _cut_json_string(string, room)
        if cut_out == 0:
            continue
        _set(work, path, cut)
        total += _json_len(cut) - _json_len(string)
        removed += cut_out
        cut_paths.add(path)
    sections = [(path, string) for path, string in leaves if path in cut_paths]

    if total <= max_chars:
        return work, removed, sections

    # Strings alone cannot make it fit: the JSON text, cut, stands for it.
    kept = {key: work[key] for key in ("success", "error") if key in work}
    room = max_chars - len(_dumps({**kept, "result": "", "truncated": note})) + 2
    cut, cut_out = _cut_json_string(text, room)
    return {**kept, "result": cut}, cut_out, sections


def _copy(value: Any) -> Any:
    """``value`` with its dicts and lists copied (tuples become lists), leaves shared."""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy(item) for item in value]
    return value


def _strings(value: Any, path: _Path = ()) -> Iterator[tuple[_Path, str]]:
    """Every string in ``value``, with where it is, in the order they appear."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, index))


def _set(value: Any, path: _Path, new: Any) -> None:
    for step in path[:-1]:
        value = value[step]
    value[path[-1]] = new


def _dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, indent=indent)


# --- saving ------------------------------------------------------------------------


def _saved_text(value: Any, sections: list[tuple[_Path, str]]) -> str:
    """The full output, saved for ``read_file`` to page through by lines.

    A string is saved as it was. Anything else is indented JSON; but in JSON
    a string is one line however many it holds, so each string the cut
    shortened is written whole after the JSON, under ``--- <where it was> ---``,
    with a pointer in its place. Then every line over :data:`MAX_LINE_CHARS`
    is split, so no page is one huge line.
    """
    if isinstance(value, str):
        return _split_long_lines(value)
    if not sections:
        return _split_long_lines(_dumps(value, indent=2))
    skeleton = _copy(value)
    for path, _ in sections:
        _set(skeleton, path, f"[in full below, under --- {_label(path)} ---]")
    return _split_long_lines("\n\n".join(
        [_dumps(skeleton, indent=2)]
        + [f"--- {_label(path)} ---\n{string}" for path, string in sections]
    ))


def _split_long_lines(text: str) -> str:
    """``text`` with each line over :data:`MAX_LINE_CHARS` split into lines of that length."""
    lines = text.split("\n")
    if all(len(line) <= MAX_LINE_CHARS for line in lines):
        return text
    return "\n".join(
        line[start:start + MAX_LINE_CHARS]
        for line in lines
        for start in range(0, max(len(line), 1), MAX_LINE_CHARS)
    )


def _label(path: _Path) -> str:
    """Where a value is in a result: ``content``, ``matches[2].line``."""
    label = ""
    for step in path:
        label += f"[{step}]" if isinstance(step, int) else (f".{step}" if label else str(step))
    return label


def _save(
    folder: Path, name: str, text: str, *, tool_name: str, removed: int, display: str
) -> str:
    """Save ``text`` as ``folder / name`` and clean up; return the note.

    Nothing escapes: a save that fails is logged and the note says so. The
    log is a warning once per folder, then debug: in a project the cap cannot
    write to, every large result fails alike, and the note tells the model.
    """
    try:
        _prepare_folder(folder)
        atomic_write_text(folder / name, text)
    except Exception as e:
        log = logger.debug if str(folder) in _unsaved_folders else logger.warning
        _unsaved_folders.add(str(folder))
        log(
            "tool_output_not_saved",
            tool=tool_name,
            folder=str(folder),
            error_type=type(e).__name__,
            error=str(e),
        )
        return f"Cut: {removed:,} characters. The full output could not be saved."
    cleanup_tool_outputs(folder, keep=name)
    return _saved_note(removed, display)


def _saved_note(removed: int, display: str) -> str:
    return (
        f"Cut: {removed:,} characters. Full output: {display}; "
        "read it with read_file using offset and limit."
    )


def _display(path: Path) -> str:
    """``path`` as the note names it: relative to the cwd, as ``read_file`` reads it."""
    try:
        return f"./{path.relative_to(Path.cwd()).as_posix()}"
    except (OSError, ValueError):
        return str(path)


def _safe_name(tool_name: str) -> str:
    """The tool's name as a file name: ``[A-Za-z0-9_.-]``, at most 64 characters."""
    return _UNSAFE.sub("_", tool_name)[:64] or "tool"


def _require_real_dir(path: Path) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise _Refused(f"{path} is a symlink; tool outputs are not saved through it")
    if not stat.S_ISDIR(st.st_mode):
        raise _Refused(f"{path} is not a directory")


def _prepare_folder(folder: Path) -> None:
    """Create ``folder`` (0700) and its ``.gitignore``; refuse a symlinked one."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    _require_real_dir(folder.parent)
    try:
        folder.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _require_real_dir(folder)
    gitignore = folder / ".gitignore"
    if not os.path.lexists(gitignore):
        atomic_write_text(gitignore, "*\n")
