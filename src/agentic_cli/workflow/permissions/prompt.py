"""UserInputRequest construction + response parsing for the permission engine."""

from __future__ import annotations

import uuid

from agentic_cli.workflow.events import InputType, UserInputRequest
from agentic_cli.workflow.permissions.capabilities import ResolvedCapability
from agentic_cli.workflow.permissions.rules import AskScope

# Strings kept module-level so the UI and parser stay in sync.
ALLOW_ONCE_CHOICE = "Allow once"
ALLOW_SESSION_CHOICE = "Allow for this session"
ALLOW_ALWAYS_CHOICE = "Allow always (save to project)"
DENY_CHOICE = "Deny"

_CHOICE_TO_SCOPE = {
    ALLOW_ONCE_CHOICE: AskScope.ONCE,
    ALLOW_SESSION_CHOICE: AskScope.SESSION,
    ALLOW_ALWAYS_CHOICE: AskScope.PROJECT,
    DENY_CHOICE: AskScope.DENY,
}


def build_request(tool_name: str, capabilities: list[ResolvedCapability]) -> UserInputRequest:
    """Construct a ``UserInputRequest`` (CHOICE) describing the pending grant."""
    lines = [f"Tool `{tool_name}` wants:"]
    has_filesystem = False
    for cap in capabilities:
        target = cap.target if cap.target else "*"
        lines.append(f"  • {cap.name} → {target}")
        if cap.name.startswith("filesystem.") and target not in ("*", ""):
            has_filesystem = True
    lines.append("")
    if has_filesystem:
        lines.append("(Session/Always grants apply to the parent directory.)")
    lines.append("Allow?")
    prompt = "\n".join(lines)

    return UserInputRequest(
        request_id=f"perm-{uuid.uuid4().hex[:8]}",
        tool_name=tool_name,
        prompt=prompt,
        input_type=InputType.CHOICE,
        choices=[
            ALLOW_ONCE_CHOICE,
            ALLOW_SESSION_CHOICE,
            ALLOW_ALWAYS_CHOICE,
            DENY_CHOICE,
        ],
        default=DENY_CHOICE,
    )


def parse_response(text: str) -> AskScope:
    """Parse a choice string into an ``AskScope``. Unknown values deny."""
    return _CHOICE_TO_SCOPE.get((text or "").strip(), AskScope.DENY)
