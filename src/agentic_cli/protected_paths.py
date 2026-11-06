"""The locations an agent may not reach, whatever else it is allowed.

One set of lists for both enforcers: the permission engine's builtin deny rules
(``workflow/permissions/store.py::BUILTIN_RULES``) and the OS sandbox's policy
(``tools/os_sandbox/policy.py``) are built from them, so a file the
sandbox protects from code is one the engine refuses a tool, and the reverse,
with two differences. Of ``~/.{app_name}`` the sandbox hides everything, the
engine denies reading only the keys and the settings (see ``APP_CONFIG_DIR``).
The engine also denies the env files anywhere in the working directory
(``WORKDIR_ENV_FILE_GLOBS``), where the sandbox, which cannot express a glob,
hides only the root ``.env``.

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
    _file("~/.zshenv"),
    _file("~/.zlogin"),
    _file("~/.profile"),
    _dir("~/.config/fish"),
    # Git configuration (``~/.config/git/config`` is git's XDG config: its
    # core.hooksPath and core.fsmonitor run at the next git command)
    _file("~/.gitconfig"),
    _file("~/.git-credentials"),
    _dir("~/.config/git"),
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

#: Anywhere in the working directory, never readable by a tool: env files
#: beside the code they configure (``packages/api/.env``, ``.env.local``,
#: direnv's ``.envrc``). These are glob patterns (``**`` is any depth, the
#: root included), so only the permission engine uses them; the OS sandbox
#: cannot express a glob and hides the root ``.env`` only.
WORKDIR_ENV_FILE_GLOBS: tuple[ProtectedPath, ...] = (
    _file("**/.env"),
    _file("**/.env.*"),
    _file("**/.envrc"),
)

#: The application's own config directory: its keys (``.env``), settings,
#: permission grants, jobs and sessions. Never writable (an agent that wrote
#: it could rewrite its own permissions). The OS sandbox also hides all of it
#: from the code it runs; the permission engine denies reading only
#: ``APP_NEVER_READABLE`` (and every env file the settings read), because an
#: app may keep files there for its agent to read (a ``workspace_dir``, the
#: sandbox's artifacts) and, started from home, ``./.{app_name}`` is this
#: directory, holding the tool outputs and pages the framework saves for
#: ``read_file``.
APP_CONFIG_DIR: ProtectedPath = _dir("~/.{app_name}")

#: In the application's config directory, never readable by a tool: the keys
#: (``.env``) and the user settings file, a trusted source the settings read
#: unfiltered, so a key written there by hand is honoured.
APP_NEVER_READABLE: tuple[ProtectedPath, ...] = (
    _file("~/.{app_name}/.env"),
    _file("~/.{app_name}/settings.json"),
)
