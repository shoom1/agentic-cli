"""The session flags every app's command line takes: ``--continue`` and ``--resume <id>``.

``BaseCLIApp.main()`` adds them to its parser (``add_session_arguments``) and
turns what was given into a ``SessionStart`` (``parse_session_start``), how the app
starts: a new conversation, the newest stored one no other process has open
(``-c``/``--continue``), or one by id (``-r``/``--resume <id>``). ``--resume``
needs its id: the picker is ``/resume``, in the app. An app with an argument
parser of its own can add them to it the same way.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, NoReturn

StartMode = Literal["new", "continue", "resume"]

#: What ``--resume`` given without its id says.
_NEEDS_AN_ID = (
    "--resume needs a conversation id: --continue opens the newest, "
    "/resume in the app picks one"
)


@dataclass(frozen=True, slots=True)
class SessionStart:
    """How an app starts.

    Attributes:
        mode: ``"new"``, a new conversation; ``"continue"``, the newest
            stored one no other process has open; ``"resume"``, ``session_id``.
        session_id: The conversation ``"resume"`` opens (the only mode that
            takes one).
    """

    mode: StartMode = "new"
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("new", "continue", "resume"):
            raise ValueError(f"Unknown session start {self.mode!r}")
        if (self.mode == "resume") != bool(self.session_id):
            raise ValueError(
                f"A {self.mode!r} start {'needs' if self.mode == 'resume' else 'takes no'} "
                "session id"
            )


def add_session_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``-c``/``--continue`` and ``-r``/``--resume <id>`` to ``parser``; they exclude each other."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c", "--continue", dest="continue_session", action="store_true",
        help="continue the newest saved conversation not open elsewhere",
    )
    group.add_argument(
        "-r", "--resume", metavar="ID", type=_conversation_id,
        help="resume the saved conversation ID (/resume in the app picks one)",
    )


def _conversation_id(value: str) -> str:
    if not value:  # --resume=
        raise argparse.ArgumentTypeError("expected one argument")
    return value


class SessionArgumentParser(argparse.ArgumentParser):
    """The parser ``BaseCLIApp.main()`` builds: ``--resume`` given without its
    id is refused with what to use instead, which argparse's own "expected one
    argument" does not say."""

    def error(self, message: str) -> NoReturn:
        if message.startswith("argument -r/--resume: expected one argument"):
            message = _NEEDS_AN_ID
        super().error(message)


def parse_session_start(args: argparse.Namespace) -> SessionStart:
    """The ``SessionStart`` the flags ``add_session_arguments`` added ask for."""
    if getattr(args, "continue_session", False):
        return SessionStart("continue")
    resume = getattr(args, "resume", None)
    if resume is not None:
        return SessionStart("resume", resume)
    return SessionStart()
