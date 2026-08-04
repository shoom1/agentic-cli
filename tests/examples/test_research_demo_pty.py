"""Real-terminal smoke test for the research-demo console process.

Everything else in the suite drives Python objects — ``process_input()``,
``MessageProcessor``, the workflow manager. None of it proves the thing a user
actually runs starts up, draws a prompt, answers a command and exits.

This launches the **actual console process** over a pty and interacts with it:
startup → prompt → ``/help`` → prompt → ``/exit``. It needs no API key, no
network, no Docker and no LLM — the workflow's background initialization is
allowed to fail (there are no credentials), which is itself part of what is
being asserted: a demo with no keys must still come up, answer ``/help`` and
exit 0 rather than crash.

This module covers the **source checkout** (via ``python -m research_demo`` on
the editable install). ``test_research_demo_wheel.py`` runs the identical
session against the installed console script from a built wheel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.examples.console_smoke import (
    ALLOWED_PARENT_VARS,
    EXPLICIT_CHILD_VARS,
    assert_clean_session,
    child_env,
    platform_skip_reason,
    probe_child_imports,
    project_version,
    run_console_smoke,
)

_SKIP = platform_skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """An isolated (home, cwd) pair for one console run."""
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    return home, cwd


def _argv() -> list[str]:
    """How the source checkout is launched."""
    return [sys.executable, "-m", "research_demo"]


class TestChildImportOrigin:
    """Pin *which* installation the spawned child resolves.

    The parent pytest process is not evidence: it has the editable install on
    ``sys.path`` already. Only a child launched exactly the way the smoke
    launches one — same interpreter, cwd, HOME and stripped environment — can
    say where the console under test comes from. Without this, a wheel-only
    failure would look like a checkout failure, and vice versa.
    """

    def test_child_resolves_the_checkout(self, sandbox) -> None:
        home, cwd = sandbox

        imports = probe_child_imports(sys.executable, cwd=cwd, home=home)

        assert imports.demo.is_relative_to(_REPO_ROOT), (
            f"the child resolved research_demo to {imports.demo}, "
            "which is not the checkout this module is meant to cover"
        )
        assert imports.framework.is_relative_to(_REPO_ROOT), (
            f"the child resolved agentic_cli to {imports.framework}"
        )
        assert imports.version == project_version()

    def test_the_probe_uses_the_same_interpreter_as_the_smoke(self, sandbox) -> None:
        home, cwd = sandbox
        imports = probe_child_imports(sys.executable, cwd=cwd, home=home)
        assert imports.executable == Path(sys.executable).resolve()
        assert _argv()[0] == sys.executable


class TestChildEnvironmentIsHermetic:
    """Nothing from the developer's shell may change what the smoke measures."""

    def test_only_allowlisted_variables_reach_the_child(self, tmp_path) -> None:
        env = child_env(tmp_path)
        unexpected = set(env) - set(ALLOWED_PARENT_VARS) - set(EXPLICIT_CHILD_VARS)
        assert unexpected == set(), f"unexpected variables in the child env: {unexpected}"

    def test_explicit_variables_are_set(self, tmp_path) -> None:
        env = child_env(tmp_path)
        assert env["HOME"] == str(tmp_path)
        for name in EXPLICIT_CHILD_VARS:
            assert name in env, f"{name} was not set on the child"

    @pytest.mark.parametrize(
        "name, value",
        [
            # Provider credentials
            ("GOOGLE_API_KEY", "sk-should-not-leak"),
            ("ANTHROPIC_API_KEY", "sk-should-not-leak"),
            ("TAVILY_API_KEY", "sk-should-not-leak"),
            # Cloud authentication
            ("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/creds.json"),
            ("GOOGLE_CLOUD_PROJECT", "some-project"),
            ("GOOGLE_GENAI_USE_VERTEXAI", "true"),
            ("VERTEX_LOCATION", "us-central1"),
            # Application settings overrides
            ("RESEARCH_DEMO_DEFAULT_MODEL", "some-other-model"),
            ("RESEARCH_DEMO_WORKSPACE_DIR", "/tmp/elsewhere"),
            ("RESEARCH_DEMO_PERMISSIONS_ENABLED", "false"),
            ("AGENTIC_ORCHESTRATOR", "langgraph"),
            ("AGENTIC_CLI_LOG_LEVEL", "debug"),
            # Import path
            ("PYTHONPATH", "/somewhere/else"),
            ("PYTHONSTARTUP", "/tmp/startup.py"),
        ],
    )
    def test_hazardous_variables_never_reach_the_child(
        self, tmp_path, monkeypatch, name, value
    ) -> None:
        """Set it in the parent; it must still be absent from the child env."""
        monkeypatch.setenv(name, value)

        env = child_env(tmp_path)

        assert name not in env, (
            f"{name} leaked into the console child and could change the smoke"
        )

    def test_a_leaked_setting_would_actually_change_the_demo(
        self, tmp_path, monkeypatch
    ) -> None:
        """The pin above is only worth having if such a variable would matter.

        Proves ``RESEARCH_DEMO_DEFAULT_MODEL`` really is load-bearing, so the
        allowlist is protecting against something real rather than asserting a
        vacuous property.
        """
        # `tests.demo_isolation` already puts the repo root on sys.path at
        # import; scoped here too so this test does not depend on that and does
        # not leave an entry behind if it ever becomes the first importer.
        monkeypatch.syspath_prepend(str(_REPO_ROOT))
        from examples.research_demo.settings import ResearchDemoSettings

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RESEARCH_DEMO_DEFAULT_MODEL", "sentinel-model")

        settings = ResearchDemoSettings(
            workspace_dir=tmp_path / "ws", _env_file=None
        )
        assert settings.default_model == "sentinel-model"


class TestConsoleSmoke:
    """The shipped console entry point, driven as a user would drive it."""

    def test_startup_help_and_exit(self, sandbox) -> None:
        home, cwd = sandbox

        result = run_console_smoke(_argv(), cwd=cwd, home=home)

        assert_clean_session(result)

    def test_a_help_and_exit_session_is_side_effect_free(self, sandbox) -> None:
        """Starting up, asking for /help and exiting must write nothing at all.

        The demo resolves its project KB and permission workdir from ``cwd``
        and its user config, user KB and session store from ``HOME``. A session
        that never runs a turn should touch none of them — so this asserts both
        that the repository is untouched (a leaked path would litter the
        checkout) *and* that the temp HOME/cwd are still empty, which is the
        stronger statement and the one that would catch a stray write landing
        somewhere the repo check does not look.
        """
        home, cwd = sandbox

        before = {p.name for p in _REPO_ROOT.iterdir()}

        result = run_console_smoke(_argv(), cwd=cwd, home=home)
        assert_clean_session(result)

        assert {p.name for p in _REPO_ROOT.iterdir()} == before, (
            "the console run created entries in the repository root"
        )
        assert list(home.rglob("*")) == [], (
            f"the run wrote into HOME: {[str(p) for p in home.rglob('*')][:10]}"
        )
        assert list(cwd.rglob("*")) == [], (
            f"the run wrote into cwd: {[str(p) for p in cwd.rglob('*')][:10]}"
        )

    def test_runs_without_any_credentials_in_the_environment(
        self, sandbox, monkeypatch
    ) -> None:
        """Even with keys exported in the parent, the child runs without them.

        ``monkeypatch.setenv`` rather than writing ``os.environ`` directly: a
        developer running this with a real ``GOOGLE_API_KEY`` exported would
        otherwise have it deleted by the cleanup, because ``os.environ.pop``
        cannot restore a value it never saved.
        """
        home, cwd = sandbox
        monkeypatch.setenv("GOOGLE_API_KEY", "sk-not-a-real-key")

        env = child_env(home)
        assert "GOOGLE_API_KEY" not in env

        result = run_console_smoke(_argv(), cwd=cwd, home=home)

        assert_clean_session(result)
