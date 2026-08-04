"""Tests for sandbox_render — Rich rendering helpers for sandbox_execute.

Pure functions returning Rich renderables (or None). Content is asserted by
rendering to plain text via a captured Console; styling (brightness/color) is
asserted by inspecting the Text spans.
"""
from __future__ import annotations

from rich.console import Console
from rich.text import Text

from agentic_cli.cli.sandbox_render import (
    CODE_MAX_LINES,
    OUTPUT_MAX_LINES,
    render_sandbox_code,
    render_sandbox_result,
)


def _to_text(renderable) -> str:
    if renderable is None:
        return ""
    console = Console(width=200, no_color=True)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def _styles_at(text: Text, index: int) -> list[str]:
    """String forms of every span style covering a character index."""
    return [str(s.style) for s in text.spans if s.start <= index < s.end]


class TestRenderCode:
    def test_none_for_empty_or_blank_code(self):
        assert render_sandbox_code("") is None
        assert render_sandbox_code("   \n  \n") is None

    def test_header_is_first_line_with_green_dash(self):
        block = render_sandbox_code("print('hi')")
        assert block.plain.splitlines()[0] == "- Running Python in the stateful executor"
        assert any("green" in s for s in _styles_at(block, 0)), block.spans

    def test_short_block_has_no_dim_styling(self):
        block = render_sandbox_code("import os\nprint(os.getcwd())")
        assert not any("dim" in str(sp.style).lower() for sp in block.spans), block.spans

    def test_code_lines_indented_four_no_marker(self):
        out = _to_text(render_sandbox_code("a = 1\nb = 2"))
        assert "╰" not in out
        lines = out.splitlines()
        assert lines[1].startswith("    ") and lines[1].strip() == "a = 1"
        assert lines[1].index("a") == 4
        assert lines[2].startswith("    ") and lines[2].strip() == "b = 2"

    def test_short_code_no_more_lines_marker(self):
        out = _to_text(render_sandbox_code("import os\nprint(os.getcwd())"))
        assert "import os" in out and "print(os.getcwd())" in out
        assert "more lines" not in out

    def test_long_code_truncated_to_cap_with_marker(self):
        code = "\n".join(f"line_{i} = {i}" for i in range(40))
        out = _to_text(render_sandbox_code(code, max_lines=25))
        assert "line_0 = 0" in out
        assert "line_24 = 24" in out
        assert "line_25 = 25" not in out
        assert "+15 more lines" in out
        assert "╰" not in out

    def test_default_cap_is_25(self):
        assert CODE_MAX_LINES == 25


class TestRenderResult:
    def test_success_fixed_header_then_indented_output(self):
        block = render_sandbox_result(
            {"success": True, "stdout": "line one\nline two\n"}, success=True
        )
        lines = block.plain.splitlines()
        assert lines[0] == "+ Stateful Python executor output:"
        assert any("green" in s for s in _styles_at(block, 0)), block.spans  # "+" green
        assert lines[1] == "    line one"
        assert lines[2] == "    line two"
        assert "╰" not in block.plain

    def test_no_stdout_says_no_output(self):
        out = _to_text(render_sandbox_result({"success": True, "stdout": ""}, success=True))
        assert out.strip() == "+ Stateful Python executor: no output"

    def test_single_line_output_indented_under_header(self):
        out = _to_text(render_sandbox_result({"success": True, "stdout": "only line\n"}, success=True))
        assert out.splitlines() == ["+ Stateful Python executor output:", "    only line"]

    def test_failure_uses_red_x_and_error_label(self):
        block = render_sandbox_result(
            {"success": False, "error": "NameError: x", "stdout": ""}, success=False
        )
        lines = block.plain.splitlines()
        assert lines[0] == "x Stateful Python executor error:"
        assert lines[1] == "    NameError: x"
        assert any("red" in s for s in _styles_at(block, 0)), block.spans

    def test_failure_falls_back_to_stderr(self):
        out = _to_text(
            render_sandbox_result(
                {"success": False, "error": "", "stderr": "Trace boom", "stdout": ""},
                success=False,
            )
        )
        assert out.splitlines()[:2] == ["x Stateful Python executor error:", "    Trace boom"]

    def test_stdout_on_failure_uses_output_label(self):
        out = _to_text(
            render_sandbox_result(
                {"success": False, "stdout": "partial\n", "error": "later"}, success=False
            )
        )
        # body is stdout -> "output" label even though the run failed (red x)
        assert out.splitlines()[:2] == ["x Stateful Python executor output:", "    partial"]

    def test_failure_no_body_says_failed(self):
        out = _to_text(
            render_sandbox_result(
                {"success": False, "stdout": "", "error": "", "stderr": ""}, success=False
            )
        )
        assert out.strip() == "x Stateful Python executor failed"

    def test_output_truncated_to_cap_with_marker(self):
        stdout = "\n".join(f"out{i}" for i in range(30))
        out = _to_text(
            render_sandbox_result({"success": True, "stdout": stdout}, success=True, max_lines=10)
        )
        assert out.splitlines()[0] == "+ Stateful Python executor output:"
        assert "    out0" in out
        assert "    out9" in out       # 10 output lines shown
        assert "out10" not in out
        assert "+20 more lines" in out
        assert "╰" not in out

    def test_default_cap_is_10(self):
        assert OUTPUT_MAX_LINES == 10
