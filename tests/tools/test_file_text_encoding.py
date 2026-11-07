"""The file tools read and write UTF-8, and an edit keeps the file's line endings.

``edit_file`` read a file with the locale's encoding and universal newlines
and wrote it back as UTF-8: every CRLF in the file became LF after a one-word
edit, and where the locale is not UTF-8 (Windows' cp1252) a UTF-8 file came
back double-encoded. ``read_file`` and ``diff_compare`` read with the locale
too. Now they read UTF-8 whatever the locale, a file that is not UTF-8 is
refused with that reason (never converted), and an edit changes only the text
it replaces.

The locale is simulated by ``io.text_encoding``, which is how ``open`` and
``Path.read_text`` pick the encoding when none is given.
"""

from __future__ import annotations

import io

import pytest

from agentic_cli.tools.file_read import diff_compare, read_file
from agentic_cli.tools.file_write import edit_file


@pytest.fixture
def cp1252_locale(monkeypatch):
    """A process whose locale encoding is Windows' cp1252."""
    real = io.text_encoding

    def text_encoding(encoding, stacklevel=2):
        return real(encoding, stacklevel) if encoding is not None else "cp1252"

    monkeypatch.setattr(io, "text_encoding", text_encoding)


def test_a_crlf_file_keeps_its_line_endings(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

    result = edit_file(str(target), "two", "TWO")

    assert result["success"] is True
    assert target.read_bytes() == b"one\r\nTWO\r\nthree\r\n"


def test_lines_in_old_and_new_text_match_and_take_the_crlf_ending(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")

    result = edit_file(str(target), "one\ntwo", "uno\ndos\nmas")

    assert result["success"] is True
    assert target.read_bytes() == b"uno\r\ndos\r\nmas\r\nthree\r\n"


def test_mixed_line_endings_are_left_as_they_are(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"a\r\nb\nc\r\n")

    result = edit_file(str(target), "b", "B")

    assert result["success"] is True
    assert target.read_bytes() == b"a\r\nB\nc\r\n"


def test_a_utf8_file_is_edited_as_utf8_whatever_the_locale(tmp_path, cp1252_locale):
    target = tmp_path / "notes.txt"
    target.write_bytes("naïve\nvalue = 1\n".encode())

    result = edit_file(str(target), "value = 1", "value = 2")

    assert result["success"] is True
    assert target.read_bytes() == "naïve\nvalue = 2\n".encode()


def test_a_utf8_bom_is_kept(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"\xef\xbb\xbfx = 1\n")

    result = edit_file(str(target), "x = 1", "x = 2")

    assert result["success"] is True
    assert target.read_bytes() == b"\xef\xbb\xbfx = 2\n"


def test_a_file_that_is_not_utf8_is_not_edited(tmp_path):
    target = tmp_path / "latin1.txt"
    target.write_bytes("café = 1\n".encode("latin-1"))

    result = edit_file(str(target), "= 1", "= 2")

    assert result["success"] is False
    assert "UTF-8" in result["error"]
    assert target.read_bytes() == "café = 1\n".encode("latin-1")


def test_read_file_reads_utf8_whatever_the_locale(tmp_path, cp1252_locale):
    target = tmp_path / "notes.txt"
    target.write_bytes("naïve\n".encode())

    result = read_file(str(target))

    assert result["success"] is True
    assert result["content"] == "naïve\n"


def test_read_file_names_utf8_when_it_refuses_a_file(tmp_path):
    target = tmp_path / "latin1.txt"
    target.write_bytes("café\n".encode("latin-1"))

    result = read_file(str(target))

    assert result["success"] is False
    assert "UTF-8" in result["error"]


def test_diff_compare_reads_utf8_whatever_the_locale(tmp_path, cp1252_locale):
    old = tmp_path / "old.txt"
    new = tmp_path / "new.txt"
    old.write_bytes("naïve\n".encode())
    new.write_bytes("naïf\n".encode())

    result = diff_compare(str(old), str(new))

    assert result["success"] is True
    assert "+naïf" in result["diff"]
    assert "-naïve" in result["diff"]


def test_a_multi_line_edit_matches_in_a_file_with_mixed_endings(tmp_path):
    """read_file shows every line ending as LF, so the model's old_text has
    LF; each line keeps the ending it had, and the new lines take the ending
    of the text they replace."""
    target = tmp_path / "config.ini"
    target.write_bytes(b"[a]\r\nk=1\r\n[b]\nk=2\n")

    result = edit_file(str(target), "[a]\nk=1", "[a]\nk=10\nj=0")

    assert result["success"] is True
    assert target.read_bytes() == b"[a]\r\nk=10\r\nj=0\r\n[b]\nk=2\n"


def test_a_multi_line_edit_matches_in_a_file_with_cr_endings(tmp_path):
    target = tmp_path / "old_mac.txt"
    target.write_bytes(b"x\ry\rz\r")

    result = edit_file(str(target), "x\ny", "X\nY")

    assert result["success"] is True
    assert target.read_bytes() == b"X\rY\rz\r"


def test_a_regex_template_adding_a_crlf_line_does_not_double_the_cr(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"a\r\nb\r\n")

    result = edit_file(str(target), "a", r"a\r\nz", use_regex=True)

    assert result["success"] is True
    assert target.read_bytes() == b"a\r\nz\r\nb\r\n"


def test_python_grep_reads_utf8_whatever_the_locale(tmp_path, cp1252_locale):
    from unittest.mock import patch

    from agentic_cli.tools.grep_tool import grep

    (tmp_path / "notes.txt").write_bytes("naïve\n".encode())

    with patch("agentic_cli.tools.grep_tool._ripgrep_available", return_value=False):
        result = grep("naïve", path=str(tmp_path))

    assert result["success"] is True
    assert result["total_matches"] == 1


def test_a_crlf_line_break_is_never_read_as_two(tmp_path):
    """old_text with a blank line between x and y is not in a file where y
    follows x directly."""
    target = tmp_path / "notes.txt"
    target.write_bytes(b"x\r\ny\r\n")

    result = edit_file(str(target), "x\n\ny", "z")

    assert result["success"] is False
    assert target.read_bytes() == b"x\r\ny\r\n"
