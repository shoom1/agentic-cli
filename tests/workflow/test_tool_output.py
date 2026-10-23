"""The tool result cap (``workflow/tool_output.py``).

A tool result whose serialized text is longer than ``tool_result_max_chars``
reaches the model cut: its shape, and as much of its beginning and its end as
the limit holds, and a note saying how much was cut and where the full output
was saved, in ``./.<app>/tool-output/``, for ``read_file`` to page through
(lines over 2,000 characters are split). The folder is in the project, which a
repository controls: as for ``web_fetch``'s saved pages, saving refuses a
symlinked folder and cleanup deletes only its own files. Saving never fails
the call.
"""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_cli.config import BaseSettings
from agentic_cli.settings_persistence import PROJECT_SETTABLE_KEYS
from agentic_cli.tools.file_read import read_file
from agentic_cli.workflow import tool_output
from agentic_cli.workflow.tool_output import (
    FOLDER_NAME,
    HEAD_SHARE,
    MAX_AGE_DAYS,
    MAX_FOLDER_BYTES,
    MIN_KEPT,
    cap_tool_result,
    cleanup_tool_outputs,
    is_saved_output,
    tool_output_dir,
)

LIMIT = 20_000
#: The note on a result whose full output was saved.
SAVED_NOTE = re.compile(
    r"^Cut: (?P<removed>[\d,]+) characters\. Full output: (?P<path>\./\.app/tool-output/"
    r"(?P<name>[A-Za-z0-9_.-]{1,64}-[0-9a-f]{12}\.txt)); read it with read_file using offset "
    r"and limit\.$"
)
#: The note when it could not be.
UNSAVED_NOTE = re.compile(
    r"^Cut: (?P<removed>[\d,]+) characters\. The full output could not be saved\.$"
)
#: Where a cut string was cut.
MARKER = re.compile(r"\n…\[(?P<n>[\d,]+) characters cut\]…\n")
#: Records whose JSON is heavy in escapes: quotes around every key and string.
RECORDS = [{"id": i, "name": f"item {i}", "tags": ["a", "b"], "ok": True} for i in range(6_000)]


class _Recorder:
    """Stands in for the module's structlog logger."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def log(event: str, **fields) -> None:
            self.calls.append((level, event, fields))

        return log


@pytest.fixture
def folder(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tool_output_dir("app")


@pytest.fixture
def log(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(tool_output, "logger", recorder)
    return recorder


def lines(count: int, prefix: str = "line") -> str:
    """``count`` numbered lines, each one unique."""
    return "".join(f"{prefix} {i:06d}\n" for i in range(count))


def cap(result, folder, *, max_chars=LIMIT, tool_name="read_file", source=None):
    return cap_tool_result(
        result, tool_name=tool_name, max_chars=max_chars, folder=folder, source=source
    )


def size(result) -> int:
    if isinstance(result, str):
        return len(result)
    return len(json.dumps(result, ensure_ascii=False))


def pieces(cut: str) -> tuple[str, int, str]:
    """A cut string's head, the number of characters cut, and its tail."""
    (match,) = MARKER.finditer(cut)
    return cut[: match.start()], int(match["n"].replace(",", "")), cut[match.end():]


def saved_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.name != ".gitignore") if folder.exists() else []


def note_path(note: str) -> Path:
    return Path.cwd() / SAVED_NOTE.match(note)["path"]


class TestAResultThatFits:
    @pytest.mark.parametrize(
        "result",
        [
            {"success": True, "content": "x" * 100},
            "plain text",
            ["a", "b"],
            None,
            42,
        ],
        ids=["dict", "str", "list", "none", "int"],
    )
    def test_is_returned_as_is(self, folder, result):
        assert cap(result, folder) is result
        assert not folder.exists()

    def test_at_exactly_the_limit_nothing_is_cut(self, folder):
        text = "x" * LIMIT

        assert cap(text, folder) is text

    def test_a_dict_at_exactly_the_limit_nothing_is_cut(self, folder):
        result = {"content": ""}
        result["content"] = "x" * (LIMIT - size(result))

        assert size(result) == LIMIT
        assert cap(result, folder) is result


class TestADict:
    def test_a_long_content_is_cut_and_every_other_key_kept(self, folder):
        content = lines(5_000)
        result = {
            "success": True,
            "content": content,
            "path": "/project/big.log",
            "size": len(content),
            "truncated": False,
        }

        capped = cap(result, folder)

        assert list(capped) == ["success", "content", "path", "size", "truncated"]
        assert capped["success"] is True
        assert capped["path"] == "/project/big.log"
        assert capped["size"] == len(content)
        head, removed, tail = pieces(capped["content"])
        assert content.startswith(head) and content.endswith(tail)
        assert head.startswith("line 000000\n") and tail.endswith("line 004999\n")
        assert removed == len(content) - len(head) - len(tail)
        # The note replaces the tool's own flag, and is truthy as it was not.
        note = capped["truncated"]
        assert SAVED_NOTE.match(note)["removed"] == f"{removed:,}"
        assert size(capped) <= LIMIT

    def test_the_head_is_70_percent_of_what_is_kept(self, folder):
        content = "a" * 30_000 + "b" * 30_000
        capped = cap({"content": content}, folder)

        head, _, tail = pieces(capped["content"])
        assert HEAD_SHARE == 0.7
        assert len(head) == round(HEAD_SHARE * (len(head) + len(tail)))
        assert set(head) == {"a"} and set(tail) == {"b"}

    def test_the_cut_leaves_room_for_the_note_and_no_more_than_needed(self, folder):
        capped = cap({"success": True, "content": "x" * 100_000}, folder)

        # Within a marker's width of the limit: the cut takes what it must.
        assert LIMIT - 40 <= size(capped) <= LIMIT

    def test_the_longest_string_is_cut_first(self, folder):
        stderr = "warning\n" * 500
        capped = cap({"success": True, "stdout": "o" * 60_000, "stderr": stderr}, folder)

        assert capped["stderr"] == stderr
        assert MARKER.search(capped["stdout"])
        assert size(capped) <= LIMIT

    def test_then_the_next_and_none_below_200_characters(self, folder):
        result = {"a": "a" * 40_000, "b": "b" * 30_000, "c": "c" * 100}

        capped = cap(result, folder)

        head, _, tail = pieces(capped["a"])
        assert MIN_KEPT == 200
        assert len(head) + len(tail) == MIN_KEPT
        assert MARKER.search(capped["b"])
        assert capped["c"] == "c" * 100
        assert size(capped) <= LIMIT

    def test_nested_dicts_and_lists_are_reached(self, folder):
        result = {
            "success": True,
            "matches": [
                {"file": "a.py", "line": "x" * 80_000},
                {"file": "b.py", "line": "short"},
            ],
            "query": {"pattern": "x+", "flags": ["i"]},
        }

        capped = cap(result, folder)

        assert capped["matches"][0]["file"] == "a.py"
        assert MARKER.search(capped["matches"][0]["line"])
        assert capped["matches"][1] == {"file": "b.py", "line": "short"}
        assert capped["query"] == {"pattern": "x+", "flags": ["i"]}
        assert SAVED_NOTE.match(capped["truncated"])
        assert size(capped) <= LIMIT

    def test_text_json_escapes_still_fills_the_limit(self, folder):
        # Quotes, backslashes and line breaks take two characters in JSON.
        capped = cap({"success": True, "content": '"\\\n' * 20_000 + "é✓" * 5_000}, folder)

        assert MARKER.search(capped["content"])
        assert LIMIT - 50 <= size(capped) <= LIMIT

    @pytest.mark.parametrize(
        "text",
        [
            json.dumps(RECORDS, indent=2),                         # a JSON data file
            json.dumps(RECORDS * 2),                               # an MCP server's one-line JSON
            "".join(f"line {i:06d}\n" for i in range(20_000)),     # a log of short lines
            "C:\\Users\\me\\src\\file.py\n" * 10_000,              # Windows paths
        ],
        ids=["pretty-json", "one-line-json", "short-lines", "backslashes"],
    )
    def test_a_cut_fills_the_limit_whatever_the_escapes(self, folder, text):
        # The more a text escapes, the less of it fits: it is never cut to
        # less than fits.
        capped = cap({"success": True, "content": text}, folder)

        head, _, tail = pieces(capped["content"])
        assert len(head) + len(tail) > 1_000
        assert LIMIT - 50 <= size(capped) <= LIMIT

    def test_a_result_json_cannot_hold_is_left_to_the_backend(self, folder):
        result: dict = {"content": "x" * 50_000}
        result["self"] = result

        assert cap(result, folder) is result

    def test_the_original_is_left_alone(self, folder):
        result = {"success": True, "items": [{"text": "x" * 50_000}], "truncated": False}
        before = copy.deepcopy(result)

        cap(result, folder)

        assert result == before

    @pytest.mark.parametrize(
        "kept", [{"success": True}, {"success": False, "error": "partial listing"}],
        ids=["success", "error"],
    )
    def test_when_strings_cannot_make_it_fit_the_json_text_is_cut(self, folder, kept):
        result = {**kept, "rows": [{"id": i, "name": f"row {i}"} for i in range(5_000)]}
        text = json.dumps(result, ensure_ascii=False)

        capped = cap(result, folder)

        assert list(capped) == [*kept, "result", "truncated"]
        assert {k: capped[k] for k in kept} == kept
        head, removed, tail = pieces(capped["result"])
        assert text.startswith(head) and text.endswith(tail)
        assert SAVED_NOTE.match(capped["truncated"])["removed"] == f"{removed:,}"
        assert LIMIT - 50 <= size(capped) <= LIMIT


class TestOtherShapes:
    def test_a_string_is_cut_with_the_note_appended(self, folder):
        text = lines(5_000)

        capped = cap(text, folder)

        body, _, note = capped.rpartition("\n\n")
        head, removed, tail = pieces(body)
        assert text.startswith(head) and text.endswith(tail)
        assert SAVED_NOTE.match(note)["removed"] == f"{removed:,}"
        assert len(capped) <= LIMIT

    def test_a_list_is_wrapped_and_cut_as_a_dict(self, folder):
        capped = cap(["x" * 60_000, "short"], folder)

        assert list(capped) == ["result", "truncated"]
        assert MARKER.search(capped["result"][0])
        assert capped["result"][1] == "short"
        assert size(capped) <= LIMIT


class TestTheSavedOutput:
    def test_it_is_private_in_a_private_folder_git_ignores(self, folder):
        capped = cap({"content": "x" * 50_000}, folder)

        (saved,) = saved_files(folder)
        assert note_path(capped["truncated"]) == saved
        assert re.fullmatch(r"read_file-[0-9a-f]{12}\.txt", saved.name)
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        assert (folder / ".gitignore").read_text() == "*\n"

    def test_the_folder_is_in_the_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        assert FOLDER_NAME == "tool-output"
        assert tool_output_dir("app") == tmp_path / ".app" / "tool-output"

    def test_a_string_is_saved_as_it_was(self, folder):
        text = lines(5_000)

        note = cap(text, folder).rpartition("\n\n")[2]

        assert note_path(note).read_text() == text

    def test_every_value_is_saved_whole_and_each_cut_string_on_its_own_lines(self, folder):
        content = lines(5_000)
        result = {"success": True, "content": content, "path": "/p/big.log", "n": 3}

        capped = cap(result, folder)

        saved = note_path(capped["truncated"]).read_text()
        skeleton, _, section = saved.partition("\n\n--- content ---\n")
        assert section == content
        pointer = "[in full below, under --- content ---]"
        assert json.loads(skeleton) == {**result, "content": pointer}

    def test_a_saved_output_pages_with_read_file(self, folder):
        content = lines(5_000)
        capped = cap({"success": True, "content": content}, folder)

        page = read_file(str(note_path(capped["truncated"])), offset=2_500, limit=3)

        assert page["success"] is True
        assert page["lines_read"] == 3
        assert page["content"] in content
        assert page["content"] not in capped["content"]

    def test_lines_over_2000_characters_are_split(self, folder):
        text = "a" * 25_000 + "\nb\n"

        note = cap(text, folder).rpartition("\n\n")[2]

        assert tool_output.MAX_LINE_CHARS == 2_000
        assert note_path(note).read_text() == "\n".join(["a" * 2_000] * 12 + ["a" * 1_000, "b", ""])

    def test_a_one_line_output_pages_with_read_file(self, folder):
        one_line = json.dumps(RECORDS * 2)
        middle = json.dumps(RECORDS[3_000])

        capped = cap(one_line, folder, tool_name="mcp")

        saved = note_path(capped.rpartition("\n\n")[2])
        total = read_file(str(saved), offset=0, limit=1)["total_lines"]
        pages = [read_file(str(saved), offset=i, limit=5)["content"] for i in range(0, total, 5)]
        assert middle not in capped
        assert all(len(page) <= LIMIT for page in pages)
        assert middle in "".join(pages).replace("\n", "")

    def test_without_a_cut_string_it_is_indented_json(self, folder):
        result = {"success": True, "rows": [{"id": i} for i in range(5_000)]}

        capped = cap(result, folder)

        saved = note_path(capped["truncated"]).read_text()
        assert saved == json.dumps(result, ensure_ascii=False, indent=2)

    def test_an_odd_tool_name_is_made_safe(self, folder):
        capped = cap("x" * 30_000, folder, tool_name="mcp/files: read page?" + "z" * 100)

        name = SAVED_NOTE.match(capped.rpartition("\n\n")[2])["name"]
        assert name.startswith("mcp_files__read_page_zzz")
        assert len(name.rpartition("-")[0]) == 64

    def test_every_cut_is_saved_in_a_file_of_its_own(self, folder):
        first = cap("x" * 30_000, folder)
        second = cap("x" * 30_000, folder)

        assert first != second
        assert len(saved_files(folder)) == 2


class TestASaveThatFails:
    @pytest.mark.parametrize("which", ["app", "tool-output"])
    def test_a_symlinked_folder_is_refused(self, tmp_path, monkeypatch, log, which):
        monkeypatch.chdir(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        if which == "app":
            (tmp_path / ".app").symlink_to(elsewhere, target_is_directory=True)
        else:
            (tmp_path / ".app").mkdir()
            (tmp_path / ".app" / "tool-output").symlink_to(elsewhere, target_is_directory=True)

        capped = cap({"success": True, "content": "x" * 50_000}, tool_output_dir("app"))

        assert UNSAVED_NOTE.match(capped["truncated"])
        assert MARKER.search(capped["content"])
        assert size(capped) <= LIMIT
        assert list(elsewhere.iterdir()) == []
        assert [(level, event) for level, event, _ in log.calls] == [
            ("warning", "tool_output_not_saved")
        ]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only folder")
    def test_a_read_only_folder_cuts_anyway(self, folder, log):
        folder.mkdir(parents=True)
        folder.chmod(0o500)
        try:
            capped = cap("x" * 50_000, folder)
        finally:
            folder.chmod(0o700)

        body, _, note = capped.rpartition("\n\n")
        assert UNSAVED_NOTE.match(note)
        assert MARKER.search(body)
        assert saved_files(folder) == []
        ((level, event, fields),) = log.calls
        assert (level, event, fields["tool"]) == ("warning", "tool_output_not_saved", "read_file")

    @pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only folder")
    def test_each_folder_is_warned_about_once(self, folder, log):
        # In a read-only project every large result would print over the UI;
        # the note tells the model each time.
        other = tool_output_dir("other")
        for unwritable in (folder, other):
            unwritable.mkdir(parents=True)
            unwritable.chmod(0o500)
        try:
            for unwritable in (folder, other, folder, other):
                cap("x" * 50_000, unwritable)
        finally:
            for unwritable in (folder, other):
                unwritable.chmod(0o700)

        assert [(level, event, fields["folder"]) for level, event, fields in log.calls] == [
            ("warning", "tool_output_not_saved", str(folder)),
            ("warning", "tool_output_not_saved", str(other)),
            ("debug", "tool_output_not_saved", str(folder)),
            ("debug", "tool_output_not_saved", str(other)),
        ]

    def test_a_file_where_the_folder_should_be(self, tmp_path, monkeypatch, log):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".app").write_text("not a folder")

        capped = cap("x" * 50_000, tool_output_dir("app"))

        assert UNSAVED_NOTE.match(capped.rpartition("\n\n")[2])
        assert (tmp_path / ".app").read_text() == "not a folder"


class TestAResultReadFromASavedOutput:
    def test_is_cut_without_a_second_file_and_points_at_the_first(self, folder):
        capped = cap({"content": lines(5_000)}, folder)
        saved = note_path(capped["truncated"])
        reread = read_file(str(saved))

        recapped = cap(reread, folder, source=saved)

        assert saved_files(folder) == [saved]
        assert note_path(recapped["truncated"]) == saved
        assert MARKER.search(recapped["content"])
        assert size(recapped) <= LIMIT


class TestIsSavedOutput:
    def test_a_file_the_cap_saved(self, folder):
        capped = cap("x" * 30_000, folder)

        assert is_saved_output(note_path(capped.rpartition("\n\n")[2]), folder)

    def test_not_a_file_of_another_name_or_place(self, folder, tmp_path):
        folder.mkdir(parents=True)

        assert not is_saved_output(folder / "notes.txt", folder)
        assert not is_saved_output(tmp_path / "read_file-0123456789ab.txt", folder)

    def test_not_in_a_symlinked_folder(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "read_file-0123456789ab.txt").write_text("planted")
        (tmp_path / ".app").mkdir()
        (tmp_path / ".app" / "tool-output").symlink_to(elsewhere, target_is_directory=True)
        folder = tool_output_dir("app")

        assert not is_saved_output(folder / "read_file-0123456789ab.txt", folder)


def _output(folder: Path, name: str, *, days: float = 0, size: int = 10) -> Path:
    """A saved output ``days`` old."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("x" * size)
    then = time.time() - days * 86400
    os.utime(path, (then, then))
    return path


class TestCleanup:
    def test_the_limits(self):
        assert MAX_AGE_DAYS == 7
        assert MAX_FOLDER_BYTES == 100 * 1024 * 1024

    def test_outputs_older_than_a_week_go(self, folder):
        old = _output(folder, "grep-000000000001.txt", days=8)
        recent = _output(folder, "grep-000000000002.txt", days=6)

        cleanup_tool_outputs(folder)

        assert not old.exists()
        assert recent.exists()

    def test_the_oldest_go_until_the_folder_fits(self, folder):
        outputs = [
            _output(folder, f"grep-00000000000{i}.txt", days=4 - i, size=1_000) for i in range(4)
        ]

        cleanup_tool_outputs(folder, max_bytes=2_000)

        assert [p.exists() for p in outputs] == [False, False, True, True]

    def test_outputs_saved_in_the_last_minute_are_kept_whatever_the_size(self, folder):
        # Another call's output, saved a moment ago: its note must not point
        # at nothing.
        older = _output(folder, "grep-000000000001.txt", days=120 / 86400, size=1_000)
        young = [
            _output(folder, f"grep-00000000000{i}.txt", days=seconds / 86400, size=1_000)
            for i, seconds in ((2, 30), (3, 5))
        ]

        cleanup_tool_outputs(folder, max_bytes=0)

        assert not older.exists()
        assert all(path.exists() for path in young)

    def test_the_output_just_saved_is_kept(self, folder):
        kept = _output(folder, "grep-00000000000a.txt", days=30, size=1_000)
        other = _output(folder, "grep-00000000000b.txt", days=1, size=1_000)

        cleanup_tool_outputs(folder, max_bytes=1, keep=kept.name)

        assert kept.exists()
        assert not other.exists()

    def test_only_its_own_regular_files_are_touched(self, folder, tmp_path):
        _output(folder, ".gitignore", days=60)
        notes = _output(folder, "notes.txt", days=60)
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")
        link = folder / "grep-0000000000aa.txt"
        link.symlink_to(outside)
        long_ago = time.time() - 60 * 86400
        os.utime(link, (long_ago, long_ago), follow_symlinks=False)
        directory = folder / "grep-0000000000bb.txt"
        directory.mkdir()
        os.utime(directory, (long_ago, long_ago))
        own = _output(folder, "grep-0000000000cc.txt", days=60)

        cleanup_tool_outputs(folder, max_bytes=0)

        assert (folder / ".gitignore").exists() and notes.exists()
        assert link.is_symlink() and outside.read_text() == "outside"
        assert directory.is_dir()
        assert not own.exists()

    def test_a_symlinked_folder_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        victim = _output(elsewhere, "grep-000000000001.txt", days=60)
        (tmp_path / ".app").mkdir()
        (tmp_path / ".app" / "tool-output").symlink_to(elsewhere, target_is_directory=True)

        cleanup_tool_outputs(tool_output_dir("app"), max_bytes=0)

        assert victim.exists()

    def test_a_symlinked_parent_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        victim = _output(elsewhere / "tool-output", "grep-000000000001.txt", days=60)
        (tmp_path / ".app").symlink_to(elsewhere, target_is_directory=True)

        cleanup_tool_outputs(tool_output_dir("app"), max_bytes=0)

        assert victim.exists()

    def test_a_missing_folder_is_nothing_to_clean(self, folder, log):
        cleanup_tool_outputs(folder)

        assert log.calls == []

    def test_each_save_cleans_up(self, folder):
        old = _output(folder, "grep-000000000001.txt", days=8)

        cap("x" * 30_000, folder)

        assert not old.exists()
        assert len(saved_files(folder)) == 1


class TestTheSetting:
    def test_the_default_and_its_floor(self):
        assert BaseSettings().tool_result_max_chars == 50_000
        assert BaseSettings(tool_result_max_chars=2_000).tool_result_max_chars == 2_000
        with pytest.raises(ValidationError):
            BaseSettings(tool_result_max_chars=1_999)

    def test_a_project_may_set_it(self):
        assert "tool_result_max_chars" in PROJECT_SETTABLE_KEYS

    def test_it_reads_as_a_setting_in_the_dialog(self):
        field = BaseSettings.model_fields["tool_result_max_chars"]

        assert field.title == "Tool Result Limit"
        assert field.json_schema_extra == {"ui_order": 27}
        assert "tool-output" in field.description
