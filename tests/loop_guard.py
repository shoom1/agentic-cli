"""A guard for tests: no blocking call on a thread that runs an event loop.

An async tool runs on the event loop itself, so a blocking call inside it
freezes the whole app (the spinner, Ctrl+C, other tool calls, job notices)
for as long as the call takes; slow work goes through ``asyncio.to_thread``.
``no_blocking_calls()`` checks that by thread, not by timing, so it holds on
a loaded CI runner:

    with no_blocking_calls():
        result = await tool(...)

While it is active, each known blocking primitive (:data:`PRIMITIVES`) called
on a thread whose event loop is running (``asyncio.get_running_loop()``
succeeds there) is refused: it raises :class:`BlockingCallOnLoop` and is
recorded. On any other thread, an ``asyncio.to_thread`` worker say, it runs
unchanged. When the block ends, the test fails with every refused call, the
primitive and the first agentic_cli frame of the stack that made it, also
when the code under test caught the error.

A function is replaced wherever a loaded module binds it (``from time import
sleep`` included), a method on its class. Whatever is bound when the guard
starts is what it wraps, so a test's fake (a ``socket.getaddrinfo`` stub)
must be in place before the ``with``.

This is plumbing shared by tests; it defines no test classes, so pytest does
not collect it.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import os
import sys
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import agentic_cli

#: How frames name agentic_cli's files (as imported, and with links resolved).
_PACKAGE_DIRS = tuple({
    os.path.dirname(os.path.abspath(agentic_cli.__file__)) + os.sep,
    os.path.dirname(os.path.realpath(agentic_cli.__file__)) + os.sep,
})

_MISSING = object()


class BlockingCallOnLoop(AssertionError):
    """A blocking primitive was called on a thread that runs an event loop."""


def _always(*args: Any, **kwargs: Any) -> bool:
    return True


def _sleeps(seconds: float = 0, *args: Any, **kwargs: Any) -> bool:
    return seconds > 0


def _waits_for_lock(fd: Any, operation: int, *args: Any, **kwargs: Any) -> bool:
    """``flock``/``lockf`` without ``LOCK_NB`` waits for the lock; unlocking never does."""
    import fcntl

    return not operation & (fcntl.LOCK_NB | fcntl.LOCK_UN)


def _blocking_socket(sock: Any, *args: Any, **kwargs: Any) -> bool:
    """A socket in blocking or timeout mode; the loop's own are non-blocking (timeout 0)."""
    return sock.gettimeout() != 0.0


def _not_done(future: Any, timeout: float | None = None, *args: Any, **kwargs: Any) -> bool:
    """``Future.result`` waits only while the future runs; asyncio reads a done one."""
    return not future.done() and timeout != 0


@dataclass(frozen=True)
class Primitive:
    """One blocking call: ``module.name``, or ``module.owner.name`` for a method.

    ``blocks`` gets the call's arguments (the instance first, for a method)
    and says whether this call would block. An ``optional`` module (an extra,
    or POSIX-only) that cannot be imported is skipped.
    """

    module: str
    name: str
    owner: str | None = None
    blocks: Callable[..., bool] = _always
    optional: bool = False

    @property
    def label(self) -> str:
        return ".".join(part for part in (self.module, self.owner, self.name) if part)


#: The blocking calls the guard refuses on a loop's thread.
PRIMITIVES: tuple[Primitive, ...] = (
    # DNS
    Primitive("socket", "getaddrinfo"),
    Primitive("socket", "gethostbyname"),
    Primitive("socket", "gethostbyname_ex"),
    Primitive("socket", "gethostbyaddr"),
    # Synchronous connects and HTTP
    Primitive("socket", "create_connection"),
    Primitive("socket", "connect", owner="socket", blocks=_blocking_socket),
    Primitive("httpx", "send", owner="Client"),
    # Blocking file locks
    Primitive("fcntl", "flock", blocks=_waits_for_lock, optional=True),
    Primitive("fcntl", "lockf", blocks=_waits_for_lock, optional=True),
    # Sleeping, and waiting for a process or a thread
    Primitive("time", "sleep", blocks=_sleeps),
    Primitive("subprocess", "run"),
    Primitive("subprocess", "wait", owner="Popen"),
    Primitive("subprocess", "communicate", owner="Popen"),
    Primitive("concurrent.futures", "result", owner="Future", blocks=_not_done),
    # PDF parsing and page conversion
    Primitive("pypdf", "__init__", owner="PdfReader"),
    Primitive("agentic_cli.memory.kb.convert", "convert_document"),
    Primitive("trafilatura", "extract", optional=True),
    Primitive("html2text", "handle", owner="HTML2Text"),
    # Feed parsing (an arXiv reply)
    Primitive("feedparser", "parse"),
    # Embedding
    Primitive("agentic_cli.memory._core.mock_embeddings", "embed_text", owner="MockEmbeddingService"),
    Primitive("agentic_cli.memory._core.mock_embeddings", "embed_batch", owner="MockEmbeddingService"),
    Primitive("agentic_cli.memory._core.embeddings", "embed_text", owner="EmbeddingService"),
    Primitive("agentic_cli.memory._core.embeddings", "embed_batch", owner="EmbeddingService"),
)

#: The guard in force, if any; a wrapper left bound after it ended passes through.
_active: "_Guard | None" = None


def _on_loop_thread() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _first_agentic_cli_frame(frame: types.FrameType | None) -> str:
    """The nearest agentic_cli frame at or above ``frame``, as ``path:line in function``."""
    innermost = frame
    while frame is not None:
        filename = frame.f_code.co_filename
        for package in _PACKAGE_DIRS:
            if filename.startswith(package):
                where = os.path.join("agentic_cli", filename[len(package):])
                return f"{where}:{frame.f_lineno} in {frame.f_code.co_name}"
        frame = frame.f_back
    if innermost is None:
        return "no agentic_cli frame"
    code = innermost.f_code
    return f"no agentic_cli frame (called from {code.co_filename}:{innermost.f_lineno} in {code.co_name})"


def _guarded(primitive: Primitive, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        guard = _active
        if guard is not None and _on_loop_thread() and _would_block(primitive, args, kwargs):
            guard.refuse(primitive.label, sys._getframe(1))
        return original(*args, **kwargs)

    guarded.__loop_guard__ = primitive.label  # type: ignore[attr-defined]
    return guarded


def _would_block(primitive: Primitive, args: tuple, kwargs: dict) -> bool:
    try:
        return bool(primitive.blocks(*args, **kwargs))
    except Exception:
        return False  # arguments the primitive itself refuses: it raises, as it would


def _module(primitive: Primitive) -> types.ModuleType | None:
    try:
        return importlib.import_module(primitive.module)
    except ImportError:
        if primitive.optional:
            return None
        raise


class _Guard:
    """The context manager ``no_blocking_calls()`` returns."""

    def __init__(self) -> None:
        self.refused: list[str] = []
        self._lock = threading.Lock()
        self._undo: list[Callable[[], None]] = []

    def refuse(self, label: str, caller: types.FrameType | None) -> NoReturn:
        where = _first_agentic_cli_frame(caller)
        with self._lock:
            self.refused.append(f"{label}, from {where}")
        raise BlockingCallOnLoop(
            f"{label} was called on the event loop's thread, from {where}: "
            "run it with asyncio.to_thread"
        )

    def __enter__(self) -> "_Guard":
        global _active
        if _active is not None:
            raise RuntimeError("no_blocking_calls() is already active")
        try:
            self._install()
        except BaseException:
            self._uninstall()
            raise
        _active = self
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        global _active
        _active = None
        self._uninstall()
        if self.refused:
            calls = "\n".join(f"  - {call}" for call in self.refused)
            raise AssertionError(
                "Blocking calls on the event loop's thread (an async tool must run them "
                f"with asyncio.to_thread):\n{calls}"
            ) from exc
        return False

    def _install(self) -> None:
        functions: dict[int, tuple[Any, Callable[..., Any]]] = {}
        for primitive in PRIMITIVES:
            module = _module(primitive)
            if module is None:
                continue
            if primitive.owner is None:
                original = getattr(module, primitive.name)
                functions[id(original)] = (original, _guarded(primitive, original))
            else:
                owner = module
                for part in primitive.owner.split("."):
                    owner = getattr(owner, part)
                self._replace_method(owner, primitive)
        # Every module that binds a guarded function, by any name.
        for module in list(sys.modules.values()):
            if not isinstance(module, types.ModuleType):
                continue
            namespace = module.__dict__
            for key, value in list(namespace.items()):
                entry = functions.get(id(value))
                if entry is not None and entry[0] is value:
                    namespace[key] = entry[1]
                    self._undo.append(functools.partial(_put_back, namespace, key, entry[1], value))

    def _replace_method(self, owner: type, primitive: Primitive) -> None:
        own = owner.__dict__.get(primitive.name, _MISSING)
        wrapper = _guarded(primitive, getattr(owner, primitive.name))
        setattr(owner, primitive.name, wrapper)

        def undo() -> None:
            if owner.__dict__.get(primitive.name) is not wrapper:
                return  # replaced again meanwhile: not ours to put back
            if own is _MISSING:
                delattr(owner, primitive.name)
            else:
                setattr(owner, primitive.name, own)

        self._undo.append(undo)

    def _uninstall(self) -> None:
        while self._undo:
            self._undo.pop()()


def _put_back(namespace: dict[str, Any], key: str, wrapper: Any, original: Any) -> None:
    if namespace.get(key) is wrapper:
        namespace[key] = original


def no_blocking_calls() -> _Guard:
    """A context manager that fails the test on a blocking call on a loop's thread."""
    return _Guard()
