"""The single interpretation of a filesystem path argument.

A tool's path argument is interpreted twice: once by the permission engine
(to decide) and once by the tool (to act). If the two interpretations ever
differ, the engine authorizes one file while the tool touches another. Both
therefore call :func:`resolve_path`, and nothing else.

The interpretation:

* ``~`` expands to the user's home directory.
* A relative path is anchored to the process's current directory, which is
  what the tool's own filesystem calls use.
* ``..`` and symlinks are resolved (``Path.resolve(strict=False)``), so the
  result is the real location, whether or not it exists yet.
* ``${workdir}``-style placeholders are **not** expanded. They belong to rule
  patterns (see ``workflow.permissions``), never to arguments; in an argument
  they are ordinary characters in a directory name.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["resolve_path"]


def resolve_path(raw: str | os.PathLike[str]) -> Path:
    """Resolve a path argument to the absolute location a tool will act on.

    Raises:
        ValueError: if ``raw`` cannot name a path (for example it contains a
            NUL byte).
    """
    return Path(raw).expanduser().resolve(strict=False)
