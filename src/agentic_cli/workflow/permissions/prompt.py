"""UserInputRequest construction + response parsing for the permission engine."""

from __future__ import annotations

import uuid
from pathlib import Path

from agentic_cli.constants import truncate
from agentic_cli.workflow.events import InputType, UserInputRequest
from agentic_cli.workflow.permissions.capabilities import ResolvedCapability
from agentic_cli.workflow.permissions.engine import (
    broaden_target_for_grant,
    is_storable_grant,
)
from agentic_cli.workflow.permissions.rules import AskScope

# Args that carry executable payloads, shown in the prompt for *.exec grants.
_CODE_ARGS = ("code", "command", "script")
_CODE_PREVIEW_MAX = 1000

# Strings kept module-level so the UI and parser stay in sync.
ALLOW_ONCE_CHOICE = "Allow once"
ALLOW_SESSION_CHOICE = "Allow for this session"
ALLOW_ALWAYS_CHOICE = "Allow always for this project"
DENY_CHOICE = "Deny"

# The label this choice used to display. It said "save to project", which
# described the *scope* but implied the grant is written into the repository —
# it is not: grants live in ``~/.{app}/project_grants.json``, keyed by the
# resolved project path (see ``permissions/store.py``). Private: an internal
# parser-compatibility detail, not API. It is still accepted so a response
# captured or queued under the old wording keeps its meaning, but it is never
# offered — callers should use ``ALLOW_ALWAYS_CHOICE``.
_LEGACY_ALLOW_ALWAYS_CHOICE = "Allow always (save to project)"

_CHOICE_TO_SCOPE = {
    ALLOW_ONCE_CHOICE: AskScope.ONCE,
    ALLOW_SESSION_CHOICE: AskScope.SESSION,
    ALLOW_ALWAYS_CHOICE: AskScope.PROJECT,
    _LEGACY_ALLOW_ALWAYS_CHOICE: AskScope.PROJECT,
    DENY_CHOICE: AskScope.DENY,
}


def build_request(
    tool_name: str,
    capabilities: list[ResolvedCapability],
    args: dict | None = None,
    *,
    home: Path | None = None,
) -> UserInputRequest:
    """Construct a ``UserInputRequest`` (CHOICE) describing the pending grant.

    The displayed target is the **effective grant scope** — i.e. what will be
    stored as a rule if the user picks Session or Always. For ``filesystem.*``
    that's a whole directory (``/foo/**``) rather than the exact file, so one
    grant covers the files there (see ``broaden_target_for_grant``).

    When nothing in the request can be remembered (a resource capability whose
    tool named no target), only "Allow once" and "Deny" are offered.

    For code-execution capabilities (``*.exec``) the pending ``code``/``command``
    payload is shown: the capability target is ``*`` (allow-any-code), so the
    payload — not the target — is what the user is actually approving.
    """
    lines = [f"Tool `{tool_name}` wants:"]
    has_broadened_filesystem = False
    for cap in capabilities:
        display_target = broaden_target_for_grant(cap, home=home)
        if not display_target:
            display_target = "*"
        lines.append(f"  • {cap.name} → {display_target}")
        if cap.name.startswith("filesystem.") and display_target != cap.target:
            has_broadened_filesystem = True
    lines.append("")
    if has_broadened_filesystem:
        lines.append("(Grant scope widened to cover the whole directory.)")
    rememberable = all(is_storable_grant(cap) for cap in capabilities)
    if not rememberable:
        lines.append("(This tool names no specific target, so approval applies to this call only.)")
    code_preview = _code_preview(capabilities, args)
    if code_preview:
        lines.append("Code to execute:")
        lines.append(code_preview)
        lines.append("")
    lines.append("Allow?")
    prompt = "\n".join(lines)

    return UserInputRequest(
        request_id=f"perm-{uuid.uuid4().hex[:8]}",
        tool_name=tool_name,
        prompt=prompt,
        input_type=InputType.CHOICE,
        choices=(
            [ALLOW_ONCE_CHOICE, ALLOW_SESSION_CHOICE, ALLOW_ALWAYS_CHOICE, DENY_CHOICE]
            if rememberable
            else [ALLOW_ONCE_CHOICE, DENY_CHOICE]
        ),
        default=DENY_CHOICE,
    )


def _code_preview(
    capabilities: list[ResolvedCapability], args: dict | None
) -> str:
    """Truncated preview of the executable payload for ``*.exec`` grants, else ''."""
    if not args:
        return ""
    # Match the whole exec namespace: "python.exec" AND sub-capabilities like
    # "python.exec.stateful" (sandbox_execute). Keying only on the ".exec"
    # suffix would skip the stateful kernel — the more dangerous tool.
    if not any(cap.name.endswith(".exec") or ".exec." in cap.name
               for cap in capabilities):
        return ""
    for key in _CODE_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return truncate(value, _CODE_PREVIEW_MAX)
    return ""


def parse_response(text: str) -> AskScope:
    """Parse a choice string into an ``AskScope``. Unknown values deny.

    Besides the four labels this module offers, one superseded "allow always"
    wording is still accepted, so renaming the displayed choice cannot turn a
    user's "always" answer into a denial. That compatibility string is internal
    and is never offered as a choice. Anything unrecognised denies.
    """
    return _CHOICE_TO_SCOPE.get((text or "").strip(), AskScope.DENY)
