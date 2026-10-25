"""The session flags every app's command line takes (``cli/session_args.py``).

``-c``/``--continue`` opens the newest stored conversation, ``-r``/``--resume
<id>`` the one named, and ``--resume`` alone a picker; they exclude each
other. ``parse_session_start`` turns what was given into a ``SessionStart``.
"""

from __future__ import annotations

import argparse

import pytest

from agentic_cli.cli.session_args import (
    SessionStart,
    add_session_arguments,
    parse_session_start,
)


def parsed(*argv: str) -> SessionStart:
    parser = argparse.ArgumentParser(prog="app")
    add_session_arguments(parser)
    return parse_session_start(parser.parse_args(list(argv)))


@pytest.mark.parametrize("argv, start", [
    ((), SessionStart("new")),
    (("--continue",), SessionStart("continue")),
    (("-c",), SessionStart("continue")),
    (("--resume", "3f2a9c1d0b7e"), SessionStart("resume", "3f2a9c1d0b7e")),
    (("-r", "3f2a9c1d0b7e"), SessionStart("resume", "3f2a9c1d0b7e")),
    (("--resume=3f2a9c1d0b7e",), SessionStart("resume", "3f2a9c1d0b7e")),
    (("--resume",), SessionStart("pick")),
])
def test_each_flag(argv, start):
    assert parsed(*argv) == start


def test_continue_and_resume_exclude_each_other(capsys):
    with pytest.raises(SystemExit) as exited:
        parsed("--continue", "--resume", "abc")

    assert exited.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_the_help_names_both(capsys):
    with pytest.raises(SystemExit):
        parsed("--help")

    text = capsys.readouterr().out
    assert "--continue" in text and "--resume [ID]" in text


@pytest.mark.parametrize("mode, session_id", [
    ("resume", None), ("new", "abc"), ("continue", "abc"), ("pick", "abc"), ("other", None),
])
def test_a_start_that_cannot_be_is_refused(mode, session_id):
    with pytest.raises(ValueError):
        SessionStart(mode, session_id)  # type: ignore[arg-type]


def test_session_start_is_exported_with_the_app():
    """An app overriding from_args builds its own start: it imports it from agentic_cli.cli."""
    from agentic_cli.cli import SessionStart as exported
    from agentic_cli.cli.session_args import SessionStart

    assert exported is SessionStart
