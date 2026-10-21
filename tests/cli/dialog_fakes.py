"""Session and dialog fakes made from thinking-prompt's own classes.

A hand-written session fake can grow a method thinking-prompt lacks (an
``input_dialog``), and the code calling it then passes its tests and fails on
every real session. These fakes are specced from the real classes instead:

- ``autospec_session()`` is a ``ThinkingPromptSession`` whose methods record
  their calls, and a call to a method it lacks, or with an argument it does
  not take, fails. ``opening_boxes(session)`` makes each of its
  ``start_thinking`` calls open a new ``autospec_box()``.
- ``edited_dialog(edits)`` is a ``Dialog`` subclass: the real constructor
  runs, so a bad argument fails, and its ``values`` are what the fields would
  hold had the user changed the ``edits`` fields before pressing a button.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock, create_autospec

from thinking_prompt import Dialog, ThinkingBox, ThinkingPromptSession


def autospec_session() -> MagicMock:
    """A ``ThinkingPromptSession`` fake with the real class's methods only.

    Its async methods are ``AsyncMock``s: set ``show_dialog.return_value`` to
    ``True`` for Save (or OK) and leave it ``None`` for Cancel or Escape.
    """
    return create_autospec(ThinkingPromptSession, instance=True)


def autospec_box(*, finished: bool = False) -> MagicMock:
    """A ``ThinkingBox`` fake: ``finish()`` ends it (``is_finished``), as it
    ends the real one."""
    box = create_autospec(ThinkingBox, instance=True)
    box.is_finished = finished

    def finish(add_to_history: bool = True, echo_to_console: bool | None = None) -> str:
        box.is_finished = True
        return ""

    box.finish.side_effect = finish
    return box


def opening_boxes(session: MagicMock) -> MagicMock:
    """``session``, each of whose ``start_thinking`` calls opens a new box,
    kept in order in ``session.boxes``."""
    session.boxes = []

    def start_thinking(*args: Any, **kwargs: Any) -> MagicMock:
        box = autospec_box()
        session.boxes.append(box)
        return box

    session.start_thinking.side_effect = start_thinking
    return session


def edited_dialog(edits: dict[str, Any]) -> type[Dialog]:
    """A ``Dialog`` class whose fields hold ``edits`` over their defaults.

    Patch the name the code under test builds its dialog with, e.g.
    ``agentic_cli.cli.settings_command.Dialog``. Every dialog built is kept in
    the class's ``built`` list. ``edits`` is read each time ``values`` is, so a
    test can set new edits for the next dialog; an edit to a key the dialog
    has no field for fails, since no user could make it.
    """

    class EditedDialog(Dialog):
        built: ClassVar[list[Dialog]] = []

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            EditedDialog.built.append(self)

        @property
        def values(self) -> dict[str, Any]:
            values = super().values
            unknown = sorted(set(edits) - set(values))
            assert not unknown, f"the dialog has no field for {unknown}"
            return {**values, **edits}

    return EditedDialog
