"""A session list's row (``SessionInfo``) and a conversation's title (``session_title``).

Every backend lists its stored conversations as ``SessionInfo`` rows, newest
first: the id, the title (the first prompt on one line), the number of turns
the user ran, and when it last changed. ``session_title`` makes the title the
same way on every backend: control characters removed, whitespace collapsed
to single spaces, at most 80 characters, a longer one cut with "…".
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from agentic_cli.workflow.sessions import TITLE_MAX, SessionInfo, session_title


class TestSessionTitle:
    def test_lines_and_runs_of_whitespace_become_one_line(self):
        assert session_title("  Fix the\nlogin   redirect\t please \n") == (
            "Fix the login redirect please"
        )

    def test_control_characters_are_removed(self):
        title = session_title("\x1b[31mred\x07 alert\x00")
        assert title == "[31mred alert"
        assert all(ch.isprintable() for ch in title)

    def test_invisible_format_characters_are_removed_not_spaced(self):
        assert session_title("co\xadop") == "coop"  # a soft hyphen
        assert session_title("trip \u200emarks") == "trip marks"  # a left-to-right mark

    def test_a_long_prompt_is_cut_with_an_ellipsis(self):
        title = session_title("word " * 40)
        assert len(title) == TITLE_MAX == 80
        assert title.endswith("…")
        assert not title[:-1].endswith(" ")

    def test_a_prompt_of_the_limit_is_kept_whole(self):
        assert session_title("x" * 80) == "x" * 80

    def test_nothing_but_whitespace_is_no_title(self):
        assert session_title(" \n\t ") == ""


class TestSessionInfo:
    def test_only_the_id_is_required(self):
        assert SessionInfo("abc") == SessionInfo("abc", "", None, None)

    def test_it_is_frozen(self):
        info = SessionInfo("abc", "Hello", 2, datetime(2026, 10, 24, tzinfo=timezone.utc))
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.turns = 3  # type: ignore[misc]
