"""Security tests for the grep tool.

The pattern argument is attacker-influenceable (LLM-supplied). It must never
be parseable by ripgrep as an option, or flags like ``--pre=<cmd>`` would run
an arbitrary program per searched file (RCE with only filesystem.read).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_cli.tools.grep_tool import grep


def _run_grep_capturing_argv(tmp_path: Path, pattern: str, rg_stdout: str = ""):
    """Call grep() forcing the ripgrep path and capture the argv built."""
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        result = MagicMock()
        result.stdout = rg_stdout
        result.returncode = 0
        return result

    with patch("agentic_cli.tools.grep_tool._ripgrep_available", return_value=True), \
            patch("agentic_cli.tools.grep_tool.subprocess.run", side_effect=fake_run):
        out = grep(pattern=pattern, path=str(tmp_path))
    return captured["cmd"], out


def test_flag_like_pattern_is_not_passed_as_option(tmp_path):
    """A pattern starting with '-' must be bound to -e, never a bare positional."""
    cmd, _ = _run_grep_capturing_argv(tmp_path, "--pre=sh")

    # The malicious value appears only as the value of -e ...
    assert "-e" in cmd
    e_idx = cmd.index("-e")
    assert cmd[e_idx + 1] == "--pre=sh"
    # ... and never standing alone in option position.
    assert cmd.count("--pre=sh") == 1

    # Option parsing is terminated before the path positional.
    assert "--" in cmd
    resolved = str(Path(tmp_path).resolve())
    assert cmd[-1] == resolved
    assert cmd[-2] == "--"
    assert cmd.index("--") < cmd.index(resolved)


def test_pattern_with_file_flag_not_treated_as_option(tmp_path):
    """`-f FILE` (arbitrary-file read primitive) must also be neutralized."""
    cmd, _ = _run_grep_capturing_argv(tmp_path, "-f/etc/passwd")
    e_idx = cmd.index("-e")
    assert cmd[e_idx + 1] == "-f/etc/passwd"
    assert "--" in cmd and cmd.index("--") < len(cmd) - 1


def test_normal_pattern_still_searches(tmp_path):
    """A benign pattern is still passed via -e and produces results."""
    match_line = (
        '{"type":"match","data":{"path":{"text":"a.py"},'
        '"line_number":1,"lines":{"text":"needle here\\n"}}}'
    )
    cmd, out = _run_grep_capturing_argv(tmp_path, "needle", rg_stdout=match_line)
    assert cmd[cmd.index("-e") + 1] == "needle"
    assert out["success"] is True
    assert out["total_matches"] == 1
