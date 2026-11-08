"""UserInputRequest construction + response parsing for the permission engine."""

from __future__ import annotations

import uuid
from collections.abc import Collection, Mapping
from pathlib import Path

from agentic_cli.workflow.events import InputType, UserInputRequest
from agentic_cli.workflow.permissions.capabilities import ResolvedCapability
from agentic_cli.workflow.permissions.engine import (
    Unstorable,
    broaden_target_for_grant,
    unstorable_reason,
)
from agentic_cli.workflow.permissions.rules import AskScope

# Args that carry executable payloads, shown in the prompt for *.exec grants.
_CODE_ARGS = ("code", "command", "script")

#: Why a prompt offers only "Allow once" and "Deny", by the reason a grant
#: may not be remembered (``unstorable_reason``), in the order shown.
_ONCE_ONLY_NOTES = {
    Unstorable.OUTSIDE_CONTENT: (
        "(This conversation has read outside content, so approval applies to this call only.)"
    ),
    Unstorable.UNCONTAINED: (
        "(This code runs on your machine without a sandbox, "
        "so approval applies to this call only.)"
    ),
    Unstorable.TARGETLESS: (
        "(This tool names no specific target, so approval applies to this call only.)"
    ),
    Unstorable.WILDCARD: (
        "(This target contains wildcard characters, so approval applies to this call only.)"
    ),
}

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
    uncontained: Collection[str] = (),
    per_call: Mapping[str, str] | None = None,
) -> UserInputRequest:
    """Construct a ``UserInputRequest`` (CHOICE) describing the pending grant.

    Each capability line names the **exact target** the call acts on, the
    file or URL "Allow once" approves. When a session or "always" grant would
    cover more (``broaden_target_for_grant``: the directory a file is in, a
    URL's site), the next line says what: ``(for this session / always: all
    of /foo/**)``. A file directly in home, at the root or above home is
    granted exactly, so its line stands alone.

    When nothing in the request can be remembered (see ``unstorable_reason``:
    code its owner says runs on the host without a sandbox, named in
    ``uncontained``; a capability approved for each call, named in
    ``per_call`` with its note (a memory write in a conversation that read
    outside content); a resource capability whose tool named no target; a
    target whose rule would hold a wildcard character), only "Allow once" and
    "Deny" are offered, with a note saying why, and no capability promises a
    wider scope: there is none to approve. The per-call reason shows the
    call's arguments in full, as ``*.exec`` shows the code: the arguments
    are what is being approved.

    For code-execution capabilities (``*.exec``) the pending ``code``/``command``
    payload is shown, all of it (the dialog body scrolls): the capability
    target is ``*`` (allow-any-code), so the payload — not the target — is
    what the user is actually approving.
    """
    lines = [f"Tool `{tool_name}` wants:"]
    per_call = per_call or {}
    reasons = {
        unstorable_reason(cap, home=home, uncontained=uncontained, per_call=per_call)
        for cap in capabilities
    } - {None}
    rememberable = not reasons
    for cap in capabilities:
        # The exact target is what this call acts on (a file, or a URL and
        # any data it carries), whatever a grant would cover.
        lines.append(f"  • {cap.name} → {cap.target or '*'}")
        scope = broaden_target_for_grant(cap, home=home)
        if rememberable and scope != cap.target:
            lines.append(f"    (for this session / always: all of {scope})")
    lines.append("")
    for reason, note in _ONCE_ONLY_NOTES.items():
        if reason not in reasons:
            continue
        if reason is Unstorable.OUTSIDE_CONTENT:
            notes = [per_call[cap.name] for cap in capabilities if per_call.get(cap.name)]
            lines += list(dict.fromkeys(notes)) or [note]
        else:
            lines.append(note)
    if Unstorable.OUTSIDE_CONTENT in reasons and args:
        lines.append("Arguments:")
        lines += [
            f"{key}: {value}" for key, value in args.items()
            if value is not None and not str(key).startswith("__")
        ]
        lines.append("")
    code = _code_to_approve(capabilities, args)
    if code:
        lines.append("Code to execute:")
        lines.append(code)
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


def _code_to_approve(
    capabilities: list[ResolvedCapability], args: dict | None
) -> str:
    """The whole executable payload for ``*.exec`` grants, else ''.

    Never cut: a preview that stopped at 1,000 characters let code placed
    after them be approved unseen.
    """
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
            return value
    return ""


def parse_response(text: str) -> AskScope:
    """Parse a choice string into an ``AskScope``. Unknown values deny.

    Besides the four labels this module offers, one superseded "allow always"
    wording is still accepted, so renaming the displayed choice cannot turn a
    user's "always" answer into a denial. That compatibility string is internal
    and is never offered as a choice. Anything unrecognised denies.
    """
    return _CHOICE_TO_SCOPE.get((text or "").strip(), AskScope.DENY)
