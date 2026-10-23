"""ripgrep's results reach the agent the way the Python search reports them.

The ripgrep search threw away ripgrep's exit status and its error output, so an
invalid pattern (``grep("[")``) came back as a successful search with no
matches, which an agent can take as evidence that the text is not there. It
also ignored ripgrep's ``context`` events, so ``context_lines`` returned only
the matching lines, where the Python search gives each match the lines around
it (``context_before``, ``context_after``).

Neither CI nor the development environment has ripgrep, so these tests replay
what ``rg --json`` writes (recorded from ripgrep 14.1); the last one runs the
real binary where it is installed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agentic_cli.tools.grep_tool as grep_mod
from agentic_cli.tools.grep_tool import grep

_LINES = ["one", "two needle", "three", "four", "five needle", "six"]

# What ripgrep 14.1 writes on stderr for ``rg -e "["``, exiting with status 2.
_PARSE_ERROR = b"rg: regex parse error:\n    (?:[)\n       ^\nerror: unclosed character class\n"


def _event(kind: str, path: Path, number: int, text: str) -> dict:
    return {"type": kind, "data": {
        "path": {"text": str(path)}, "line_number": number, "lines": {"text": text + "\n"},
    }}


def _summary(matches: int) -> dict:
    return {"type": "summary", "data": {"stats": {"matches": matches}}}


def _ripgrep(monkeypatch, *, events=(), stderr=b"", returncode=0) -> None:
    """Make grep's ripgrep write ``events`` and ``stderr`` and exit ``returncode``."""

    def popen(cmd, **kwargs):
        kwargs["stdout"].write("".join(json.dumps(e) + "\n" for e in events).encode())
        if hasattr(kwargs.get("stderr"), "write"):
            kwargs["stderr"].write(stderr)
        return SimpleNamespace(pid=4321, returncode=returncode, wait=lambda timeout=None: returncode)

    monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: True)
    monkeypatch.setattr(grep_mod.subprocess, "Popen", popen)


@pytest.fixture
def notes(tmp_path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("\n".join(_LINES) + "\n")
    return path


def _python_search(path: Path, **args) -> dict:
    options = dict(recursive=True, ignore_case=False, file_pattern=None, context_lines=0,
                   max_results=100, use_regex=True, output_mode="content")
    return grep_mod._grep_python(path=path, **{**options, **args})


class TestRipgrepErrors:
    def test_an_invalid_pattern_is_an_error_with_ripgrep_s_message(self, monkeypatch, tmp_path):
        _ripgrep(monkeypatch, stderr=_PARSE_ERROR, returncode=2)

        result = grep("[", path=str(tmp_path))

        assert result["success"] is False
        assert "regex parse error" in result["error"]
        assert "unclosed character class" in result["error"]
        assert result["matches"] == [] and result["total_matches"] == 0

    def test_an_error_without_a_message_names_the_exit_status(self, monkeypatch, tmp_path):
        _ripgrep(monkeypatch, returncode=2)

        result = grep("needle", path=str(tmp_path))

        assert result["success"] is False
        assert "2" in result["error"]

    def test_no_match_is_a_successful_empty_search(self, monkeypatch, tmp_path):
        _ripgrep(monkeypatch, events=[_summary(0)], returncode=1)

        result = grep("absent", path=str(tmp_path))

        assert result["success"] is True
        assert result["matches"] == [] and result["total_matches"] == 0

    def test_a_file_it_could_not_read_leaves_the_other_matches(self, monkeypatch, tmp_path):
        # ripgrep exits 2 when it skipped a file, after searching the others.
        found = tmp_path / "ok.txt"
        _ripgrep(
            monkeypatch,
            events=[
                {"type": "begin", "data": {"path": {"text": str(found)}}},
                _event("match", found, 1, "needle"),
                {"type": "end", "data": {"path": {"text": str(found)}}},
                _summary(1),
            ],
            stderr=b"rg: " + str(tmp_path / "locked.txt").encode() + b": Permission denied (os error 13)\n",
            returncode=2,
        )

        result = grep("needle", path=str(tmp_path))

        assert result["success"] is True
        assert [m["content"] for m in result["matches"]] == ["needle"]

    def test_the_error_quoted_is_bounded(self, monkeypatch, tmp_path):
        _ripgrep(monkeypatch, stderr=b"rg: " + b"x" * 100_000, returncode=2)

        result = grep("needle", path=str(tmp_path))

        assert result["success"] is False
        assert len(result["error"]) < 5_000


class TestRipgrepContext:
    def test_each_match_comes_with_its_context_lines(self, monkeypatch, notes):
        # rg --json -C 1: the context lines are events of their own.
        _ripgrep(monkeypatch, events=[
            _event("context", notes, 1, "one"),
            _event("match", notes, 2, "two needle"),
            _event("context", notes, 3, "three"),
            _event("context", notes, 4, "four"),
            _event("match", notes, 5, "five needle"),
            _event("context", notes, 6, "six"),
            _summary(2),
        ])

        result = grep("needle", path=str(notes), context_lines=1)

        assert [(m["context_before"], m["context_after"]) for m in result["matches"]] == [
            (["one"], ["three"]),
            (["four"], ["six"]),
        ]
        assert result["matches"] == _python_search(notes, pattern="needle", context_lines=1)["matches"]

    def test_overlapping_context_is_shared_like_the_python_search(self, monkeypatch, notes):
        # With -C 2, ripgrep prints lines 3 and 4 once, between the two matches.
        _ripgrep(monkeypatch, events=[
            _event("context", notes, 1, "one"),
            _event("match", notes, 2, "two needle"),
            _event("context", notes, 3, "three"),
            _event("context", notes, 4, "four"),
            _event("match", notes, 5, "five needle"),
            _event("context", notes, 6, "six"),
            _summary(2),
        ])

        result = grep("needle", path=str(notes), context_lines=2)

        assert result["matches"] == _python_search(notes, pattern="needle", context_lines=2)["matches"]
        assert result["matches"][0]["context_after"] == ["three", "four"]
        assert result["matches"][1]["context_before"] == ["three", "four"]

    def test_a_match_within_another_s_context_is_part_of_it(self, monkeypatch, notes):
        _ripgrep(monkeypatch, events=[
            _event("match", notes, 2, "two needle"),
            _event("context", notes, 3, "three"),
            _event("context", notes, 4, "four"),
            _event("match", notes, 5, "five needle"),
            _summary(2),
        ])

        result = grep("needle", path=str(notes), context_lines=3)

        assert result["matches"][0]["context_after"] == ["three", "four", "five needle"]
        assert result["matches"][1]["context_before"] == ["two needle", "three", "four"]

    def test_without_context_lines_a_match_has_no_context_keys(self, monkeypatch, notes):
        _ripgrep(monkeypatch, events=[_event("match", notes, 2, "two needle"), _summary(1)])

        (match,) = grep("needle", path=str(notes))["matches"]

        assert "context_before" not in match and "context_after" not in match


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
class TestRealRipgrep:
    @pytest.fixture(autouse=True)
    def _ripgrep_on(self, monkeypatch):
        monkeypatch.setattr(grep_mod, "_ripgrep_available", lambda: True)

    def test_an_invalid_pattern_is_an_error(self, tmp_path):
        result = grep("[", path=str(tmp_path))
        assert result["success"] is False
        assert "regex parse error" in result["error"]

    @pytest.mark.parametrize("context_lines", [1, 2, 3])
    def test_context_is_what_the_python_search_gives(self, notes, context_lines):
        def lines(result):
            return [{k: v for k, v in m.items() if k != "file"} for m in result["matches"]]

        result = grep("needle", path=str(notes), context_lines=context_lines)

        python = _python_search(notes, pattern="needle", context_lines=context_lines)
        assert lines(result) == lines(python)
