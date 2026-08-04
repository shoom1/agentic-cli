"""Acceptance smoke against the **built wheel**, not the checkout.

Every other test in the suite imports the source tree. That cannot catch a
packaging defect: a module or data file left out of
``[tool.hatch.build.targets.wheel]`` is invisible until someone installs the
artifact. This builds the wheel, installs it into an isolated virtualenv, and
runs the same console session ``test_research_demo_pty.py`` runs — through the
**installed ``research-demo`` console script**, which is what a user actually
types, not ``python -m``.

Building a wheel and installing its dependencies takes minutes and needs the
network, which is exactly what the offline suite is not. So this is opt-in on
two axes: it carries the ``wheel`` marker *and* requires
``AGENTIC_WHEEL_ACCEPTANCE=1``. That keeps the offline selector
(``-m 'not llm and not docker'``) fast and network-free without redefining it,
and CI runs the acceptance as its own step:

    AGENTIC_WHEEL_ACCEPTANCE=1 conda run -n agenticcli python -m pytest \
        tests/examples/test_research_demo_wheel.py -m wheel -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from tests.examples.console_smoke import (
    assert_clean_session,
    platform_skip_reason,
    probe_child_imports,
    project_version,
    run_console_smoke,
)

_PLATFORM_SKIP = platform_skip_reason()
_OPT_IN = os.environ.get("AGENTIC_WHEEL_ACCEPTANCE") == "1"

pytestmark = [
    pytest.mark.wheel,
    pytest.mark.skipif(_PLATFORM_SKIP is not None, reason=_PLATFORM_SKIP or ""),
    pytest.mark.skipif(
        not _OPT_IN,
        reason=(
            "wheel acceptance builds a wheel and installs its dependencies "
            "(slow + needs network); set AGENTIC_WHEEL_ACCEPTANCE=1 to run it"
        ),
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Non-Python files the demo cannot work without. They live under
#: ``examples/research_demo`` and are only in the wheel because hatchling
#: packages that directory — exactly the kind of thing that silently vanishes.
REQUIRED_PACKAGE_DATA = (
    Path("data") / "benchmarks.csv",
    Path("skills") / "report-writer" / "SKILL.md",
    Path("skills") / "report-writer" / "assets" / "report_template.tex",
)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(cmd[:4])}…\n"
            f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
        )
    return result


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build the wheel and install it into a throwaway virtualenv.

    Module-scoped: building and installing once is the expensive part, and the
    tests below only read from the result.
    """
    root = tmp_path_factory.mktemp("wheel-acceptance")
    dist = root / "dist"
    env_dir = root / "venv"

    _run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist),
         str(_REPO_ROOT)],
        cwd=str(root),  # outside the repository
    )
    wheels = sorted(dist.glob("agentic_cli-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    venv.EnvBuilder(with_pip=True, symlinks=True).create(env_dir)
    python = env_dir / "bin" / "python"
    console = env_dir / "bin" / "research-demo"
    assert python.exists(), f"virtualenv has no interpreter at {python}"

    _run([str(python), "-m", "pip", "install", "--quiet", str(wheels[0])],
         cwd=str(root))

    return {
        "python": python,
        "console": console,
        "wheel": wheels[0],
        "root": root,
        "env_dir": env_dir,
    }


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    return home, cwd


class TestWheelIsSelfContained:
    def test_child_resolves_from_the_isolated_install(self, installed_wheel, sandbox):
        """A child launched as the smoke launches one must not see the checkout.

        Probed in a subprocess with the same interpreter, cwd, HOME and
        stripped environment the console gets — the parent pytest process has
        the editable install on ``sys.path``, so its own imports prove nothing
        about the child.
        """
        home, cwd = sandbox

        imports = probe_child_imports(
            str(installed_wheel["python"]), cwd=cwd, home=home
        )

        env_dir = installed_wheel["env_dir"].resolve()
        for label, path in (("demo", imports.demo), ("framework", imports.framework)):
            assert path.is_relative_to(env_dir), (
                f"{label} resolved to {path}, outside the isolated venv"
            )
            assert not path.is_relative_to(_REPO_ROOT), (
                f"{label} resolved into the repository checkout: {path}"
            )
        assert imports.version == project_version()

    def test_package_data_is_present(self, installed_wheel, sandbox):
        home, cwd = sandbox
        imports = probe_child_imports(
            str(installed_wheel["python"]), cwd=cwd, home=home
        )
        pkg_dir = imports.demo.parent

        missing = [str(rel) for rel in REQUIRED_PACKAGE_DATA
                   if not (pkg_dir / rel).is_file()]
        assert missing == [], (
            f"the installed wheel is missing package data: {missing}\n"
            f"(looked under {pkg_dir})"
        )

    def test_console_script_is_installed_and_executable(self, installed_wheel):
        console = installed_wheel["console"]
        assert console.exists(), (
            "the `research-demo` console script was not installed by the wheel"
        )
        assert os.access(console, os.X_OK), f"{console} is not executable"

    def test_console_script_runs_the_venv_interpreter(self, installed_wheel):
        """Its shebang must point into the venv, not at the checkout's python.

        This is what makes running the script equivalent to running the venv's
        interpreter — and therefore what lets the import probe above stand for
        the console process.

        The shebang path is compared **unresolved**: ``venv/bin/python`` is a
        symlink to the base interpreter (``EnvBuilder(symlinks=True)``), so
        resolving it lands on the conda binary and says nothing. What makes an
        interpreter a venv interpreter is its ``sys.prefix``, which is asserted
        separately below.
        """
        env_dir = installed_wheel["env_dir"]
        shebang = installed_wheel["console"].read_text(errors="replace").splitlines()[0]
        assert shebang.startswith("#!"), f"no shebang in the console script: {shebang!r}"

        interpreter = Path(shebang[2:].strip().split()[0])
        assert interpreter.is_relative_to(env_dir), (
            f"console script runs {interpreter}, outside the venv {env_dir}"
        )

        # The property that actually confers isolation: this interpreter's
        # site-packages is the venv's, not the base environment's.
        prefix = subprocess.run(
            [str(interpreter), "-c", "import sys; print(sys.prefix)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert Path(prefix) == env_dir, (
            f"the console script's interpreter has sys.prefix={prefix}, "
            f"expected the venv at {env_dir}"
        )


class TestWheelConsoleSmoke:
    def test_installed_console_script_startup_help_and_exit(
        self, installed_wheel, sandbox
    ):
        """The user-facing path: run `research-demo`, not `python -m`."""
        home, cwd = sandbox

        result = run_console_smoke(
            [str(installed_wheel["console"])], cwd=cwd, home=home
        )

        assert_clean_session(result)
        assert result.argv == [str(installed_wheel["console"])], (
            "the smoke did not execute the installed console script"
        )

    def test_module_invocation_also_works(self, installed_wheel, sandbox):
        """`python -m research_demo` from the installed wheel, for parity."""
        home, cwd = sandbox

        result = run_console_smoke(
            [str(installed_wheel["python"]), "-m", "research_demo"],
            cwd=cwd,
            home=home,
        )

        assert_clean_session(result)


class TestArtifactCarriesTheDeclaredVersion:
    """The built and installed artifact must match ``pyproject.toml``.

    All three surfaces are checked because they can disagree: the wheel
    filename comes from the build, the distribution metadata from the install,
    and ``__version__`` is a separate literal in the package that a bump can
    forget.
    """

    def test_wheel_filename_carries_the_declared_version(self, installed_wheel):
        expected = project_version()
        assert installed_wheel["wheel"].name.startswith(f"agentic_cli-{expected}-"), (
            f"wheel {installed_wheel['wheel'].name} does not carry the declared "
            f"version {expected}"
        )

    def test_installed_metadata_and_module_version_agree(self, installed_wheel):
        """Distribution metadata and ``agentic_cli.__version__``, from the venv."""
        expected = project_version()
        probe = (
            "import json, importlib.metadata as md, agentic_cli; "
            "print(json.dumps({"
            "'distribution': md.version('agentic-cli'), "
            "'module': agentic_cli.__version__}))"
        )
        result = _run(
            [str(installed_wheel["python"]), "-c", probe],
            cwd=str(installed_wheel["root"]),
        )
        found = json.loads(result.stdout.strip().splitlines()[-1])

        assert found["distribution"] == expected, (
            f"installed distribution metadata says {found['distribution']}, "
            f"pyproject.toml declares {expected}"
        )
        assert found["module"] == expected, (
            f"agentic_cli.__version__ is {found['module']}, "
            f"pyproject.toml declares {expected}"
        )
