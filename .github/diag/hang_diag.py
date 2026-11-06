"""Hang diagnostics for the CI hang hunt (diag branch only).

Loaded with ``-p hang_diag``. Writes into $HANG_DIAG_DIR:

- tests.log: each test's start and end, and after each test the child
  processes (any depth) and threads that appeared or went away.
- faulthandler.txt: every thread's C-level stack on SIGUSR1.
- signal-dump.txt: on SIGUSR2, every thread's Python stack and the stack of
  every task of every running asyncio loop (a stuck await shows only the
  loop in a thread dump).
"""

from __future__ import annotations

import asyncio
import faulthandler
import gc
import os
import signal
import sys
import threading
import time
import traceback

OUT = os.environ.get("HANG_DIAG_DIR", "/tmp/hang-diag")
os.makedirs(OUT, exist_ok=True)
_log = open(os.path.join(OUT, "tests.log"), "a", buffering=1)
_fh = open(os.path.join(OUT, "faulthandler.txt"), "a")
_DUMP = os.path.join(OUT, "signal-dump.txt")
_ME = os.getpid()


def _descendants() -> dict[int, str]:
    parent: dict[int, int] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        parent[int(name)] = int(stat.rsplit(")", 1)[1].split()[1])
    children: dict[int, list[int]] = {}
    for pid, ppid in parent.items():
        children.setdefault(ppid, []).append(pid)
    found: dict[int, str] = {}
    todo = list(children.get(_ME, []))
    while todo:
        pid = todo.pop()
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = "?"
        found[pid] = cmd[:300]
        todo.extend(children.get(pid, []))
    return found


def _threads() -> set[str]:
    return {f"{t.name} daemon={t.daemon}" for t in threading.enumerate()}


def _os_threads() -> int:
    try:
        return len(os.listdir(f"/proc/{_ME}/task"))
    except OSError:
        return -1


_procs: dict[int, str] = {}
_thread_names: set[str] = set()


def _write(line: str) -> None:
    _log.write(f"{time.time():.1f} {line}\n")


def _dump(signum, frame) -> None:
    with open(_DUMP, "a") as f:
        f.write(f"=== {time.strftime('%H:%M:%S')} signal {signum}\n")
        names = {t.ident: t.name for t in threading.enumerate()}
        for ident, fr in sys._current_frames().items():
            f.write(f"--- thread {names.get(ident, '?')} ({ident})\n")
            traceback.print_stack(fr, file=f)
        loops = []
        for obj in gc.get_objects():
            try:
                if isinstance(obj, asyncio.AbstractEventLoop) and obj.is_running():
                    loops.append(obj)
            except Exception:
                pass
        for loop in loops:
            f.write(f"--- loop {loop!r}\n")
            for _ in range(3):
                try:
                    tasks = list(asyncio.all_tasks(loop))
                    break
                except RuntimeError:
                    tasks = []
            for task in tasks:
                f.write(f"task {task!r}\n")
                try:
                    task.print_stack(file=f)
                except Exception as e:  # diagnostics only
                    f.write(f"  (no stack: {e!r})\n")
        f.flush()


def pytest_configure(config) -> None:
    faulthandler.register(signal.SIGUSR1, file=_fh, all_threads=True)
    signal.signal(signal.SIGUSR2, _dump)
    _write(f"CONFIGURE pid={_ME}")


def pytest_runtest_logstart(nodeid, location) -> None:
    _write(f"START {nodeid}")


def pytest_runtest_logfinish(nodeid, location) -> None:
    global _procs, _thread_names
    procs = _descendants()
    names = _threads()
    _write(f"END {nodeid} procs={len(procs)} pythreads={len(names)} osthreads={_os_threads()}")
    for pid in procs.keys() - _procs.keys():
        _log.write(f"   + proc {pid} {procs[pid]}\n")
    for pid in _procs.keys() - procs.keys():
        _log.write(f"   - proc {pid} {_procs[pid]}\n")
    for name in sorted(names - _thread_names):
        _log.write(f"   + thread {name}\n")
    for name in sorted(_thread_names - names):
        _log.write(f"   - thread {name}\n")
    _procs, _thread_names = procs, names


def pytest_unconfigure(config) -> None:
    procs = _descendants()
    _write(f"UNCONFIGURE procs={len(procs)} osthreads={_os_threads()}")
    for pid, cmd in procs.items():
        _log.write(f"   left proc {pid} {cmd}\n")
    for name in sorted(_threads()):
        _log.write(f"   left thread {name}\n")
