"""Two processes share one scope folder: the scope's lock orders their changes (spec §5.4).

Review Focus 1, across processes. Two threads of one process are kept apart by
the store's own lock, and saves with different descriptions never contend, so
neither shows whether the scope's file lock works: this does. Both processes
save the *same* descriptions, so a check that does not see the other's file yet
saves a duplicate.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

from agentic_cli.memory.store._files import normalized
from agentic_cli.memory.store.store import MemoryStore

COUNT = 30
TIMEOUT_S = 120

_WORKER = textwrap.dedent(
    """
    import json, sys
    from agentic_cli.memory.store.models import DuplicateMemory
    from agentic_cli.memory.store.store import MemoryStore

    user, project, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    store = MemoryStore(user, project)
    print("ready", flush=True)
    sys.stdin.readline()  # the parent says go once both processes are ready
    saved = duplicates = 0
    other = []
    for n in range(count):
        try:
            store.save(f"shared fact {n}", f"saved from {project}", "user", scope="user")
            saved += 1
        except DuplicateMemory:
            duplicates += 1
        except Exception as e:
            other.append(f"{type(e).__name__}: {e}")
    print(json.dumps({"saved": saved, "duplicates": duplicates, "other": other}), flush=True)
    """
)


def _start(user: Path, project: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _WORKER, str(user), str(project), str(COUNT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_ready(process: subprocess.Popen) -> None:
    for line in process.stdout:
        if line.strip() == "ready":
            return
    raise AssertionError("a worker ended before it was ready")


def test_two_processes_saving_the_same_descriptions_keep_each_once(tmp_path):
    user = tmp_path / "user"
    processes = [_start(user, tmp_path / f"project-{n}" / "memory") for n in (1, 2)]
    watchdogs = [threading.Timer(TIMEOUT_S, process.kill) for process in processes]
    for watchdog in watchdogs:
        watchdog.start()
    try:
        for process in processes:
            _wait_ready(process)
        for process in processes:
            process.stdin.write("go\n")
            process.stdin.flush()
        results = []
        for process in processes:
            output, _ = process.communicate(timeout=TIMEOUT_S)
            assert process.returncode == 0, output
            results.append(json.loads(output.strip().splitlines()[-1]))
    finally:
        for watchdog in watchdogs:
            watchdog.cancel()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()

    assert [result["other"] for result in results] == [[], []]
    assert sum(result["saved"] for result in results) == COUNT
    assert sum(result["duplicates"] for result in results) == COUNT
    store = MemoryStore(user)
    memories = store.memories("user")
    assert len(memories) == COUNT
    assert len({normalized(memory.description) for memory in memories}) == COUNT
    assert store.problems() == []
