"""The session flags every app's command line takes: ``--continue`` and ``--resume [ID]``.

``BaseCLIApp.main()`` adds them to its parser (``add_session_arguments``) and
turns what was given into a ``SessionStart`` (``parse_session_start``), how the app
starts: a new conversation, the newest stored one (``-c``/``--continue``), one
by id (``-r``/``--resume <id>``), or a picker (``--resume`` alone). An app
with an argument parser of its own can add them to it the same way.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

StartMode = Literal["new", "continue", "resume", "pick"]

#: What ``--resume`` without an id stands for.
_PICK = object()


@dataclass(frozen=True, slots=True)
class SessionStart:
    """How an app starts.

    Attributes:
        mode: ``"new"``, a new conversation; ``"continue"``, the newest
            stored one; ``"resume"``, ``session_id``; ``"pick"``, the one
            picked once the prompt is up.
        session_id: The conversation ``"resume"`` opens (the only mode that
            takes one).
    """

    mode: StartMode = "new"
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("new", "continue", "resume", "pick"):
            raise ValueError(f"Unknown session start {self.mode!r}")
        if (self.mode == "resume") != bool(self.session_id):
            raise ValueError(
                f"A {self.mode!r} start {'needs' if self.mode == 'resume' else 'takes no'} "
                "session id"
            )


def add_session_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``-c``/``--continue`` and ``-r``/``--resume [ID]`` to ``parser``; they exclude each other."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c", "--continue", dest="continue_session", action="store_true",
        help="continue the newest saved conversation",
    )
    group.add_argument(
        "-r", "--resume", nargs="?", const=_PICK, default=None, metavar="ID",
        help="resume the saved conversation ID, or pick one (without an ID)",
    )


def parse_session_start(args: argparse.Namespace) -> SessionStart:
    """The ``SessionStart`` the flags ``add_session_arguments`` added ask for."""
    if getattr(args, "continue_session", False):
        return SessionStart("continue")
    resume = getattr(args, "resume", None)
    if resume is _PICK or resume == "":
        return SessionStart("pick")
    if resume is not None:
        return SessionStart("resume", resume)
    return SessionStart()
