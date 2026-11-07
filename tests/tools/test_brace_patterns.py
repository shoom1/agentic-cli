"""``glob`` and ``grep`` understand brace patterns such as ``*.{js,ts}``.

Python's pattern matching has no braces, so ``glob("**/*.{js,ts}")`` (the
example in ``glob``'s own docstring) looked for files literally named
``….{js,ts}`` and reported success with no files; ``grep``'s Python search
did the same with ``file_pattern``, while ripgrep understood it. A pattern is
now expanded into its alternatives first. Every alternative is checked
against the search root, so ``{x,../*}`` cannot reach above it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agentic_cli.tools.glob_tool import glob
from agentic_cli.tools.grep_tool import grep


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.js").write_text("needle\n")
    (src / "b.ts").write_text("needle\n")
    (src / "c.py").write_text("needle\n")
    return tmp_path


@pytest.fixture
def python_grep():
    with patch("agentic_cli.tools.grep_tool._ripgrep_available", return_value=False):
        yield


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("*.py", ["*.py"]),
        ("*.{js,ts}", ["*.js", "*.ts"]),
        ("{a,b{1,2}}.txt", ["a.txt", "b1.txt", "b2.txt"]),
        ("{x}.txt", ["{x}.txt"]),
        ("{x.txt", ["{x.txt"]),
        ("[{]a,b}", ["[{]a,b}"]),
        ("{a,a}.txt", ["a.txt"]),
    ],
)
def test_expand_braces(pattern, expected):
    from agentic_cli.file_utils import expand_braces

    assert expand_braces(pattern) == expected


def test_a_pattern_with_too_many_alternatives_is_refused():
    from agentic_cli.file_utils import expand_braces

    with pytest.raises(ValueError, match="alternatives"):
        expand_braces("{a,b}" * 8)


def test_glob_finds_every_alternative(project):
    result = glob("**/*.{js,ts}", path=str(project))

    assert result["success"] is True
    assert sorted(result["files"]) == ["src/a.js", "src/b.ts"]


def test_glob_lists_a_file_matching_two_alternatives_once(project):
    result = glob("src/*.{js,j*}", path=str(project))

    assert result["files"] == ["src/a.js"]


def test_glob_refuses_an_alternative_that_leaves_the_root(project):
    result = glob("{x,../*}", path=str(project / "src"))

    assert result["success"] is False
    assert "escapes" in result["error"]


def test_glob_reports_too_many_alternatives_as_an_error(project):
    result = glob("{a,b}" * 8, path=str(project))

    assert result["success"] is False
    assert "alternatives" in result["error"]


def test_python_grep_searches_every_alternative(project, python_grep):
    result = grep("needle", path=str(project), file_pattern="*.{js,ts}", output_mode="files")

    assert result["success"] is True
    assert sorted(m["file"].rsplit("/", 1)[-1] for m in result["matches"]) == ["a.js", "b.ts"]


def test_grep_refuses_an_alternative_that_leaves_the_root(project, python_grep):
    result = grep("needle", path=str(project / "src"), file_pattern="{x,../*}")

    assert result["success"] is False
    assert "escapes" in result["error"]


def test_ripgrep_gets_one_glob_per_alternative(project):
    captured = {}

    class Done:
        pid = 4321
        returncode = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return Done()

    with patch("agentic_cli.tools.grep_tool._ripgrep_available", return_value=True), \
            patch("agentic_cli.tools.grep_tool.subprocess.Popen", side_effect=fake_popen):
        grep("needle", path=str(project), file_pattern="*.{js,ts}")

    cmd = captured["cmd"]
    globs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--glob"]
    assert globs == ["*.js", "*.ts", "!.*"]
