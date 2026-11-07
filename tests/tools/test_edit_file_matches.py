"""``edit_file`` changes the one place the model named, or nothing.

Without ``replace_all`` it replaced the first match and reported success, so
when the text was in two places the model got ``replacements: 1`` for an edit
of a place it never meant (``return None`` in ``a()`` changed, ``b()`` left
as it was). An empty ``old_text`` was accepted: with ``replace_all`` it put
the new text between every two characters. Now more than one match without
``replace_all`` is an error naming the count, and an empty ``old_text`` is
refused; the file is left as it was.
"""

from __future__ import annotations

import pytest

from agentic_cli.tools.file_write import edit_file

TWO_FUNCTIONS = "def a():\n    return None\n\n\ndef b():\n    return None\n"


def test_text_found_twice_is_not_edited_without_replace_all(tmp_path):
    target = tmp_path / "funcs.py"
    target.write_text(TWO_FUNCTIONS)

    result = edit_file(str(target), "    return None", "    return 1")

    assert result["success"] is False
    assert "2 times" in result["error"]
    assert "replace_all" in result["error"]
    assert target.read_text() == TWO_FUNCTIONS


def test_more_context_makes_the_edit_land_where_meant(tmp_path):
    target = tmp_path / "funcs.py"
    target.write_text(TWO_FUNCTIONS)

    result = edit_file(str(target), "def b():\n    return None", "def b():\n    return 1")

    assert result["success"] is True
    assert result["replacements"] == 1
    assert target.read_text() == "def a():\n    return None\n\n\ndef b():\n    return 1\n"


def test_replace_all_still_replaces_every_match(tmp_path):
    target = tmp_path / "funcs.py"
    target.write_text(TWO_FUNCTIONS)

    result = edit_file(str(target), "return None", "return 1", replace_all=True)

    assert result["success"] is True
    assert result["replacements"] == 2
    assert target.read_text().count("return 1") == 2


def test_a_regex_matching_twice_is_not_applied_without_replace_all(tmp_path):
    target = tmp_path / "ids.txt"
    target.write_text("id1 id2\n")

    result = edit_file(str(target), r"id\d", "ID", use_regex=True)

    assert result["success"] is False
    assert "2 times" in result["error"]
    assert target.read_text() == "id1 id2\n"


def test_a_regex_matching_once_is_applied(tmp_path):
    target = tmp_path / "ids.txt"
    target.write_text("id1 name\n")

    result = edit_file(str(target), r"id\d", "ID", use_regex=True)

    assert result["success"] is True
    assert target.read_text() == "ID name\n"


@pytest.mark.parametrize("replace_all", [False, True])
@pytest.mark.parametrize("use_regex", [False, True])
def test_an_empty_old_text_is_refused(tmp_path, replace_all, use_regex):
    target = tmp_path / "abc.txt"
    target.write_text("abc")

    result = edit_file(str(target), "", "X", replace_all=replace_all, use_regex=use_regex)

    assert result["success"] is False
    assert "empty" in result["error"]
    assert target.read_text() == "abc"
