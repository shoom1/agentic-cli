"""``agentic_cli.models`` loads without the rest of agentic-cli and without provider SDKs.

``test_boundary.py`` checks the package's own imports. This checks what Python
loads: importing a submodule first runs every parent package's ``__init__``,
and a library the package uses could pull a provider SDK in. Each import runs
in a fresh interpreter, since this one has loaded everything already.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_PROVIDER_SDKS = ("google", "anthropic", "openai", "langchain", "langgraph")

# Only what the import adds counts: a ``*-nspkg.pth`` file (google-cloud's)
# puts the empty ``google`` namespace package in ``sys.modules`` at startup.
_PROBE = (
    "import importlib, json, sys; before = set(sys.modules); "
    "importlib.import_module(sys.argv[1]); "
    "print(json.dumps(sorted(set(sys.modules) - before)))"
)


def _loaded(module: str) -> list[str]:
    """The modules importing ``module`` loads, in a fresh interpreter."""
    run = subprocess.run([sys.executable, "-c", _PROBE, module], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.parametrize("module", ["agentic_cli.models"])
def test_importing_it_loads_no_other_agentic_cli_module(module):
    outside = [
        m
        for m in _loaded(module)
        if m.startswith("agentic_cli")
        and m not in ("agentic_cli", "agentic_cli.models")
        and not m.startswith("agentic_cli.models.")
    ]

    assert outside == []


@pytest.mark.parametrize("module", ["agentic_cli.models"])
def test_importing_it_loads_no_provider_sdk(module):
    sdks = [m for m in _loaded(module) if m.split(".")[0].startswith(_PROVIDER_SDKS)]

    assert sdks == []
