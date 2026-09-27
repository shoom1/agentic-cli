"""``resolve_path``: the one interpretation of a path argument.

The permission engine checks a path and the tool then opens one. Both call
``resolve_path``, so these tests pin down what that single interpretation is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_cli.paths import resolve_path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home.resolve()


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd.resolve()


def test_relative_path_is_anchored_to_the_current_directory(cwd):
    assert resolve_path("notes/a.txt") == cwd / "notes" / "a.txt"


def test_tilde_expands_to_home(home, cwd):
    assert resolve_path("~/notes.txt") == home / "notes.txt"


def test_dotdot_is_collapsed(cwd):
    assert resolve_path("a/b/../c") == cwd / "a" / "c"


def test_placeholder_text_is_an_ordinary_directory_name(cwd):
    """``${workdir}`` in an argument is literal text, never a variable."""
    assert resolve_path("${workdir}/y") == cwd / "${workdir}" / "y"
    assert resolve_path("d/${workdir}/../x") == cwd / "d" / "x"


def test_symlinks_are_followed(cwd):
    real = cwd / "real"
    real.mkdir()
    (cwd / "link").symlink_to(real)
    assert resolve_path("link/f.txt") == real / "f.txt"


def test_absolute_path_is_kept(cwd, tmp_path):
    target = tmp_path / "elsewhere" / "f.txt"
    assert resolve_path(str(target)) == target.resolve()


def test_accepts_path_objects(cwd):
    assert resolve_path(Path("x")) == cwd / "x"


def test_null_byte_raises_value_error(cwd):
    with pytest.raises(ValueError):
        resolve_path("a\x00b")
