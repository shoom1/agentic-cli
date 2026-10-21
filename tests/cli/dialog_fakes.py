"""Session and dialog fakes made from thinking-prompt's own classes.

A hand-written session fake can grow a method thinking-prompt lacks (an
``input_dialog``), and the code calling it then passes its tests and fails on
every real session. These fakes are specced from the real classes instead:

- ``autospec_session()`` is a ``ThinkingPromptSession`` whose methods record
  their calls, and a call to a method it lacks, or with an argument it does
  not take, fails.
- ``edited_dialog(edits)`` is a ``Dialog`` subclass: the real constructor
  runs, so a bad argument fails, and its ``values`` are what the fields would
  hold had the user changed the ``edits`` fields before pressing a button.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock, create_autospec

from thinking_prompt import Dialog, ThinkingPromptSession


def autospec_session() -> MagicMock:
    """A ``ThinkingPromptSession`` fake with the real class's methods only.

    Its async methods are ``AsyncMock``s: set ``show_dialog.return_value`` to
    ``True`` for Save (or OK) and leave it ``None`` for Cancel or Escape.
    """
    return create_autospec(ThinkingPromptSession, instance=True)


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
