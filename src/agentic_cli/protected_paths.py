"""The locations an agent may not reach, whatever else it is allowed.

One set of lists for both enforcers: the permission engine's builtin deny rules
(``workflow/permissions/store.py::BUILTIN_RULES``) and the OS sandbox's policy
(``tools/shell/os_sandbox/policy.py``) are built from them, so a file the
sandbox hides from code is one the engine refuses a tool, and the reverse.

A location is spelled ``~/…`` (under the home directory), ``/…`` (absolute) or
relative (under the working directory), and ``{app_name}`` stands for the
application's name. A directory covers itself and everything below it; a file
covers only itself.

Stdlib only, importing nothing else from agentic_cli: the permission engine
loads it at startup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtectedPath:
    """A protected location, spelled as the module docstring says."""

    location: str
    is_dir: bool


def _dir(location: str) -> ProtectedPath:
    return ProtectedPath(location, is_dir=True)


def _file(location: str) -> ProtectedPath:
    return ProtectedPath(location, is_dir=False)


#: Credential stores: neither readable nor writable.
CREDENTIAL_STORES: tuple[ProtectedPath, ...] = (
    _dir("~/.ssh"),
    _dir("~/.gnupg"),
    _dir("~/.aws"),
    _dir("~/.azure"),
    _dir("~/.kube"),
    _dir("~/.config/gcloud"),
    _dir("~/.config/gh"),
    _file("~/.docker/config.json"),
    _file("~/.netrc"),
    _file("~/.git-credentials"),
    _file("~/.pypirc"),
    _file("~/.npmrc"),
)

#: Never writable: what configures or runs programs outside any sandbox.
NEVER_WRITABLE: tuple[ProtectedPath, ...] = (
    # Shell start-up files
    _file("~/.bashrc"),
    _file("~/.bash_profile"),
    _file("~/.zshrc"),
    _file("~/.zprofile"),
    _file("~/.profile"),
    # Git configuration
    _file("~/.gitconfig"),
    _file("~/.git-credentials"),
    # SSH and GPG
    _dir("~/.ssh"),
    _dir("~/.gnupg"),
    # System directories
    _dir("/etc"),
    _dir("/usr"),
    _dir("/bin"),
    _dir("/sbin"),
    _dir("/boot"),
    # macOS system
    _dir("/System"),
    _dir("/Library"),
    # Agent config
    _dir("~/.claude"),
)

#: In the working directory, never writable: ``.git`` holds the hooks and the
#: config git runs outside any sandbox (the whole directory, or it could be
#: moved aside and replaced), ``.{app_name}`` the project's settings.
WORKDIR_NEVER_WRITABLE: tuple[ProtectedPath, ...] = (
    _dir(".git"),
    _dir(".{app_name}"),
)

#: In the working directory, never readable: the keys in ``.env``.
WORKDIR_NEVER_READABLE: tuple[ProtectedPath, ...] = (
    _file(".env"),
)

#: The application's own config directory (its ``.env`` with the keys, its
#: settings, the permission grants): neither readable nor writable.
APP_CONFIG_DIR: ProtectedPath = _dir("~/.{app_name}")
