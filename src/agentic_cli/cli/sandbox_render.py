"""Rich rendering helpers for the stateful executor (``sandbox_execute``).

Kept as pure functions returning Rich renderables (or ``None``) so the
layout/truncation logic is unit-testable without a live
``ThinkingPromptSession``. The CLI's ``MessageProcessor`` calls these to show
the code being run (on the tool call) and a single combined result message
(on the result)::

    - Running Python in the stateful executor      # on call  (green "-")
        <code, indented 4>

    + Stateful Python executor output:             # on result (green "+", red "x")
        <output, indented 4>

Content is rendered at normal brightness; only the ``… (+N more lines)``
truncation hint is dim.
"""
from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from rich.syntax import Syntax
from rich.text import Text

# Display caps (lines). The code block shows the first N lines of the submitted
# source; the result shows the first M lines of output (or the error).
CODE_MAX_LINES = 25
OUTPUT_MAX_LINES = 10

_HEADER = "Running Python in the stateful executor"
_INDENT = "    "  # 4 spaces


def _append_indented(
    out: Text, lines: list[Any], dropped: int, *, body_style: str | None = None
) -> None:
    """Append body ``lines`` each indented 4 spaces; plain strings or Rich
    ``Text`` (already-highlighted code). A dim ``… (+N more lines)`` hint is
    appended when ``dropped`` > 0."""
    for line in lines:
        out.append("\n")
        out.append(_INDENT)
        if isinstance(line, Text):
            out.append_text(line)
        else:
            out.append(line, style=body_style)
    if dropped:
        out.append("\n")
        out.append(f"{_INDENT}… (+{dropped} more lines)", style="dim")


def render_sandbox_code(code: str, max_lines: int = CODE_MAX_LINES) -> RenderableType | None:
    """Header + syntax-highlighted code for a starting ``sandbox_execute`` call.

    Returns ``None`` when there is nothing to show (empty/blank code). The
    source is capped at ``max_lines`` with a dim truncation hint.
    """
    if not code or not code.strip():
        return None
    lines = code.splitlines()
    dropped = max(0, len(lines) - max_lines)
    shown_lines = lines[:max_lines]
    shown = "\n".join(shown_lines)
    # Highlight the whole block once (preserves multi-line context), then split
    # into per-line Text so we can indent each line ourselves.
    highlighted = Syntax(
        shown, "python", theme="ansi_dark", background_color="default"
    ).highlight(shown)
    code_lines = list(highlighted.split("\n"))[: len(shown_lines)]

    out = Text()
    out.append("- ", style="green")      # mirrors the "+" shown on completion
    out.append(_HEADER)                   # normal brightness
    _append_indented(out, code_lines, dropped)
    return out


def _select_output(result: dict[str, Any]) -> tuple[str, str | None] | None:
    """The (text, style) to show for a finished run, or ``None`` if nothing.

    stdout wins when present (even on failure — stdout is the output); otherwise
    a failed run falls back to error then stderr so it is not silent.
    """
    stdout = result.get("stdout") or ""
    if stdout.strip():
        return stdout, None  # normal brightness
    if not result.get("success", True):
        body = result.get("error") or result.get("stderr") or ""
        if body.strip():
            return body, "red"
    return None


def render_sandbox_result(
    result: dict[str, Any],
    *,
    success: bool = True,
    max_lines: int = OUTPUT_MAX_LINES,
) -> RenderableType:
    """Single combined result message: a ``+``/``x`` icon and a fixed header
    label, with the run's output indented below.

    Header is ``Stateful Python executor output:`` for captured stdout and
    ``… error:`` for a failure's error/stderr. With nothing to show the header
    stands alone: ``+ Stateful Python executor: no output`` (success) or
    ``x Stateful Python executor failed``."""
    icon, icon_style = ("+", "green") if success else ("x", "red")
    out = Text()
    out.append(f"{icon} ", style=icon_style)
    selected = _select_output(result)
    if not selected:
        out.append(
            "Stateful Python executor: no output" if success
            else "Stateful Python executor failed"
        )
        return out
    body, style = selected
    label = "error" if style == "red" else "output"  # style "red" == error/stderr
    out.append(f"Stateful Python executor {label}:")
    lines = body.splitlines()
    dropped = max(0, len(lines) - max_lines)
    _append_indented(out, lines[:max_lines], dropped, body_style=style)
    return out
