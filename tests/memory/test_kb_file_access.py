"""The knowledge base opens its files only through ``_core.io``.

A project can ship its knowledge base (``./.{app}/knowledge_base``), and git
keeps symlinks, so any name in its folder may lead to a file elsewhere, to a
FIFO or to a device such as ``/dev/zero``. ``_core.io`` reads and appends a
file only when it is a regular file, never through a symlink, and its atomic
writes replace a link rather than write through it. This scan fails when code
in ``memory/kb/`` opens a path some other way: ``Path.read_text`` and its
kin, the ``open`` builtin, ``os.open``, ``json.load`` on an open file, or a
library that opens the path itself (``faiss.read_index`` given a path rather
than a reader over a file ``_core.io`` opened).
"""

from __future__ import annotations

import ast
from pathlib import Path

KB = Path(__file__).resolve().parents[2] / "src" / "agentic_cli" / "memory" / "kb"

# Methods that open the path they are called on (``Path`` and its kin).
PATH_METHODS = {"open", "read_text", "read_bytes", "write_text", "write_bytes"}
# module.function calls that open a path they are given.
OPENING_CALLS = {("os", "open"), ("io", "open"), ("json", "load"), ("faiss", "read_index")}


def _reads_a_stream(call: ast.Call) -> bool:
    """``faiss.read_index(faiss.PyCallbackIOReader(...))``: no path is opened."""
    first = call.args[0] if call.args else None
    return (
        isinstance(first, ast.Call)
        and isinstance(first.func, ast.Attribute)
        and first.func.attr == "PyCallbackIOReader"
    )


def _openings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        where = f"{path.name}:{node.lineno}"
        if isinstance(func, ast.Name) and func.id == "open":
            found.append(f"{where} open()")
        elif isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            if (owner, func.attr) == ("faiss", "read_index") and _reads_a_stream(node):
                continue
            if (owner, func.attr) in OPENING_CALLS:
                found.append(f"{where} {owner}.{func.attr}()")
            elif func.attr in PATH_METHODS and owner not in ("self", "cls"):
                found.append(f"{where} .{func.attr}()")
    return found


def test_the_knowledge_base_opens_no_path_itself():
    found = [hit for path in sorted(KB.glob("*.py")) for hit in _openings(path)]

    assert found == []


def test_the_scan_sees_each_way_of_opening(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import faiss, json, os\n"
        "path.read_text()\n"
        "Path(p).write_bytes(b'')\n"
        "open(p)\n"
        "os.open(p, 0)\n"
        "json.load(f)\n"
        "faiss.read_index(str(p))\n"
        "self.read_text(doc)\n"
        "faiss.read_index(faiss.PyCallbackIOReader(f.read))\n"
    )

    assert [hit.split(" ", 1)[1] for hit in _openings(sample)] == [
        ".read_text()", ".write_bytes()", "open()", "os.open()", "json.load()", "faiss.read_index()",
    ]
