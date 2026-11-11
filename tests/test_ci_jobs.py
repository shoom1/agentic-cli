"""Which CI jobs a pull request runs (``.github/scripts/ci-jobs.sh``).

The offline suite runs on every change. The four jobs that test one part
each (the kb extra, the built wheel, the LaTeX engine, Docker) run on a pull
request only when it touches that part, and on every push to develop and
main. These tests run the script against a throwaway git repository.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "ci-jobs.sh"
JOBS = ("kb", "wheel", "latex", "docker")
NONE = dict.fromkeys(JOBS, False)
EVERY = dict.fromkeys(JOBS, True)


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` with no user or system configuration (no hooks, no signing)."""
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
    ).stdout.strip()


def commit(repo: Path, files: dict[str, str | None], message: str = "change") -> str:
    """Write (None: delete) ``files`` in ``repo``, commit them, and return the commit."""
    for name, text in files.items():
        path = repo / name
        if text is None:
            git(repo, "rm", "-q", name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


#: One file of each part, and the files the tests change: a change below
#: modifies a file, since adding or removing one under ``src/`` runs the wheel.
PARTS = {
    "pyproject.toml": "[project]\n",
    "environment.yml": "name: x\n",
    ".github/workflows/ci.yml": "name: CI\n",
    "src/agentic_cli/cli/app.py": "a = 1\n",
    "tests/cli/test_app.py": "a = 1\n",
    "README.md": "x\n",
    **dict.fromkeys(
        (
            "src/agentic_cli/memory/kb/convert.py",
            "src/agentic_cli/memory/_core/embeddings.py",
            "src/agentic_cli/tools/webfetch/converter.py",
            "src/agentic_cli/tools/knowledge_tools.py",
            "src/agentic_cli/workflow/memory_services.py",
            "src/agentic_cli/logging.py",
            "tests/memory/test_kb_extra.py",
            "tests/tools/test_webfetch_summary_text.py",
            "src/agentic_cli/tools/sandbox/manager.py",
            "src/agentic_cli/tools/sandbox/backends/jupyter_docker.py",
            "src/agentic_cli/file_utils.py",
            "tests/tools/test_sandbox_docker_live.py",
            "src/agentic_cli/tools/document/compile.py",
            "src/agentic_cli/paths.py",
            "tests/tools/test_document_compile_latex.py",
            "examples/research_demo/app.py",
            "tests/examples/console_smoke.py",
            "tests/examples/test_research_demo_wheel.py",
            "src/agentic_cli/models/data/table.json",
        ),
        "a = 1\n",
    ),
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose first commit holds ``PARTS``."""
    git(tmp_path, "init", "-q", "-b", "develop")
    commit(tmp_path, PARTS, "base")
    return tmp_path


def jobs(repo: Path, *args: str) -> dict[str, bool]:
    """The script's answer, as each job's flag."""
    out = subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=repo, check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    ).stdout
    flags = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    assert set(flags) == set(JOBS), out
    return {job: flags[job] == "true" for job in JOBS}


def changed(repo: Path, files: dict[str, str | None]) -> dict[str, bool]:
    """The jobs a pull request changing ``files`` on top of the base runs."""
    base = git(repo, "rev-parse", "HEAD")
    head = commit(repo, files)
    return jobs(repo, base, head)


def test_a_change_elsewhere_runs_none_of_them(repo):
    assert changed(repo, {"src/agentic_cli/cli/app.py": "a = 2\n"}) == NONE


def test_a_test_or_doc_change_elsewhere_runs_none_of_them(repo):
    assert changed(repo, {"tests/cli/test_app.py": "a = 2\n", "README.md": "y\n"}) == NONE


@pytest.mark.parametrize(
    ("path", "job"),
    [
        ("src/agentic_cli/memory/kb/convert.py", "kb"),
        ("src/agentic_cli/memory/_core/embeddings.py", "kb"),
        ("src/agentic_cli/tools/webfetch/converter.py", "kb"),
        ("src/agentic_cli/tools/knowledge_tools.py", "kb"),
        ("src/agentic_cli/workflow/memory_services.py", "kb"),
        ("src/agentic_cli/logging.py", "kb"),
        ("tests/memory/test_kb_extra.py", "kb"),
        ("tests/tools/test_webfetch_summary_text.py", "kb"),
        ("src/agentic_cli/tools/sandbox/manager.py", "docker"),
        ("src/agentic_cli/tools/sandbox/backends/jupyter_docker.py", "docker"),
        ("src/agentic_cli/file_utils.py", "docker"),
        ("tests/tools/test_sandbox_docker_live.py", "docker"),
        ("src/agentic_cli/tools/document/compile.py", "latex"),
        ("src/agentic_cli/paths.py", "latex"),
        ("tests/tools/test_document_compile_latex.py", "latex"),
        ("examples/research_demo/app.py", "wheel"),
        ("tests/examples/console_smoke.py", "wheel"),
        ("tests/examples/test_research_demo_wheel.py", "wheel"),
        ("src/agentic_cli/models/data/table.json", "wheel"),
    ],
)
def test_a_change_to_one_part_runs_its_job(repo, path, job):
    assert changed(repo, {path: "a = 2\n"}) == {**NONE, job: True}


@pytest.mark.parametrize("path", ["pyproject.toml", "environment.yml", ".github/workflows/ci.yml"])
def test_a_change_to_what_every_job_builds_from_runs_them_all(repo, path):
    assert changed(repo, {path: "changed\n"}) == EVERY


@pytest.mark.parametrize(
    "files",
    [
        {"src/agentic_cli/cli/new_module.py": "a = 1\n"},  # added
        {"src/agentic_cli/cli/app.py": None},  # deleted
    ],
    ids=["added", "deleted"],
)
def test_a_module_added_or_removed_runs_the_wheel(repo, files):
    assert changed(repo, files) == {**NONE, "wheel": True}


def test_a_moved_module_runs_the_wheel(repo):
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "src/agentic_cli/cli/app.py", "src/agentic_cli/cli/main.py")
    git(repo, "commit", "-q", "-m", "move")
    assert jobs(repo, base, git(repo, "rev-parse", "HEAD")) == {**NONE, "wheel": True}


def test_only_the_pull_requests_own_changes_count(repo):
    """The diff is from where the branch left its base: a change the base
    branch made since is not the pull request's."""
    fork = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "-q", "-c", "feature")
    head = commit(repo, {"src/agentic_cli/cli/app.py": "a = 2\n"})
    git(repo, "switch", "-q", "develop")
    base = commit(repo, {"src/agentic_cli/tools/sandbox/manager.py": "a = 2\n"})
    assert fork != base
    assert jobs(repo, base, head) == NONE


def test_a_push_runs_them_all(repo):
    assert jobs(repo) == EVERY


def test_a_diff_that_cannot_be_read_runs_them_all(repo):
    assert jobs(repo, "0" * 40, git(repo, "rev-parse", "HEAD")) == EVERY
