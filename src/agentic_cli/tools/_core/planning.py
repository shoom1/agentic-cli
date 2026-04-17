"""Shared planning logic — pure functions, no framework imports."""


def summarize_checkboxes(content: str) -> str:
    """Parse checkboxes and return a summary like '5 tasks: 2 done, 3 pending'.

    Args:
        content: Markdown plan string.

    Returns:
        Summary string, or empty string if no checkboxes found.
    """
    done = 0
    pending = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            done += 1
        elif stripped.startswith("- [ ]"):
            pending += 1

    total = done + pending
    if total == 0:
        return ""

    parts: list[str] = []
    if done:
        parts.append(f"{done} done")
    if pending:
        parts.append(f"{pending} pending")
    return f"{total} tasks: {', '.join(parts)}"
