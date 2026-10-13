"""``agentic_cli.memory`` loads without the rest of agentic-cli.

``test_boundary.py`` checks the package's own imports. This checks what Python
loads: importing a submodule first runs every parent package's ``__init__``,
so an eager ``agentic_cli/__init__.py`` would load the CLI and the workflow
along with the knowledge base. Each import runs in a fresh interpreter, since
this one has loaded everything already.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_PROBE = (
    "import importlib, json, sys; importlib.import_module(sys.argv[1]); "
    "print(json.dumps(sorted(m for m in sys.modules if m.startswith('agentic_cli'))))"
)


def _outside_the_package(loaded: list[str]) -> list[str]:
    return [
        m for m in loaded
        if m not in ("agentic_cli", "agentic_cli.memory") and not m.startswith("agentic_cli.memory.")
    ]


@pytest.mark.parametrize("module", ["agentic_cli.memory", "agentic_cli.memory.kb"])
def test_importing_it_loads_no_other_agentic_cli_module(module):
    run = subprocess.run([sys.executable, "-c", _PROBE, module], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    assert _outside_the_package(json.loads(run.stdout)) == []
