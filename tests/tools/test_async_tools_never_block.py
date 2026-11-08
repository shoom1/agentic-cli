"""An async tool never makes a blocking call on the event loop.

A synchronous tool runs on a worker thread on both backends (ADK's tool
thread pool, LangChain's executor); an async tool runs on the event loop
itself, so a blocking call inside it froze the whole app (the spinner, Ctrl+C,
the other tool calls of a turn, job notices) for as long as it took.
``web_fetch`` resolved DNS there, PDF text was extracted there, and the
knowledge base embedded and waited for its folder lock there (fixed in #261
and #262). The rule: an async tool runs its slow work with
``asyncio.to_thread``.

Every async tool the framework builds is listed here (the registry's, the
service-bound variants ``tools/factories.py`` makes, Deep Agents' skill
tools), so a new one fails until it is added; and each is called through
``tests/loop_guard.py``'s ``no_blocking_calls``, offline, on the paths that
reach the slow work: DNS, an HTML and a PDF page, a PDF download, a file
ingest, a PDF's text, a concept page. The guard's own behaviour comes first.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import importlib
import inspect
import io
import json
import os
import pkgutil
import socket
import subprocess
import sys
import textwrap
import time
import types
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from agentic_cli.config import BaseSettings, SettingsContext
from agentic_cli.memory import EmbeddingService
from agentic_cli.memory.kb import KnowledgeBase, KnowledgeBaseManager, SourceType
from agentic_cli.tools.arxiv_source import ArxivSearchSource
from agentic_cli.tools.pdf_utils import extract_pdf_text
from agentic_cli.workflow.service_registry import (
    ARXIV_SOURCE,
    KNOWLEDGE_BASE,
    LLM_SUMMARIZER,
    WORKFLOW,
    set_service_registry,
)
from tests.loop_guard import PRIMITIVES, BlockingCallOnLoop, no_blocking_calls
from tests.memory.fake_embeddings import FakeEmbeddingService
from tests.memory.kb_extra import require_kb_module
from tests.memory.pdf_bytes import pdf_with_text
from tests.tools.arxiv_fakes import ArxivAPI, Paper

fcntl = pytest.importorskip("fcntl")


# --- the guard itself ----------------------------------------------------------------


def _refused(label: str) -> Any:
    """The failure the guard ends a block with when ``label`` was called on the loop."""
    return pytest.raises(
        AssertionError, match=rf"Blocking calls on the event loop's thread.*\n  - {label}, from "
    )


async def test_a_blocking_call_on_the_loop_fails_the_test():
    with _refused(r"time\.sleep") as failed:
        with no_blocking_calls():
            time.sleep(0.01)

    assert isinstance(failed.value.__cause__, BlockingCallOnLoop)


async def test_a_refusal_the_code_caught_still_fails_naming_its_agentic_cli_frame():
    """``extract_pdf_text`` catches every error and returns ""; the test fails all the same."""
    with _refused(r"pypdf\.PdfReader\.__init__") as failed:
        with no_blocking_calls():
            assert extract_pdf_text(pdf_with_text("Magma")) == ""

    assert "from agentic_cli/tools/pdf_utils.py:" in str(failed.value)
    assert "in extract_pdf_text" in str(failed.value)


async def test_a_worker_thread_may_block():
    with no_blocking_calls():
        await asyncio.to_thread(time.sleep, 0.01)
        text = await asyncio.to_thread(extract_pdf_text, pdf_with_text("Magma"))

    assert text == "Magma"


async def test_a_function_bound_under_another_name_is_guarded(monkeypatch):
    module = types.ModuleType("binds_sleep")
    module.nap = time.sleep  # from time import sleep as nap
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with _refused(r"time\.sleep"):
        with no_blocking_calls():
            module.nap(0.01)

    assert module.nap is time.sleep


async def test_a_call_that_does_not_block_passes(tmp_path):
    fd = os.open(tmp_path / "lock", os.O_RDWR | os.O_CREAT)
    done: concurrent.futures.Future[int] = concurrent.futures.Future()
    done.set_result(1)
    try:
        with no_blocking_calls():
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            time.sleep(0)
            assert done.result() == 1
            await asyncio.sleep(0.01)
    finally:
        os.close(fd)


async def test_an_asyncio_connection_is_not_refused():
    """asyncio connects a non-blocking socket: the guard lets that through."""

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.read(5))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        with no_blocking_calls():
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"Magma")
            await writer.drain()
            assert await reader.read(5) == b"Magma"
            writer.close()
            await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


def _pdf_reader() -> None:
    import pypdf

    pypdf.PdfReader(io.BytesIO(pdf_with_text("Magma")))


def _html2text() -> None:
    import html2text

    html2text.HTML2Text().handle("<p>Magma</p>")


def _trafilatura() -> None:
    import trafilatura

    trafilatura.extract("<html><body><p>Magma</p></body></html>")


def _feedparser() -> None:
    import feedparser

    feedparser.parse(b"<feed xmlns='http://www.w3.org/2005/Atom'><title>Magma</title></feed>")


def _convert_document() -> None:
    from agentic_cli.memory.kb import convert_document

    convert_document(b"Magma", content_type=None, extension=".txt")


def _embed(mock: bool, method: str) -> Callable[[_Held], None]:
    def call(held: _Held) -> None:
        from agentic_cli.memory import EmbeddingService, MockEmbeddingService

        # Refused before it runs: the real service never loads its model.
        service = MockEmbeddingService() if mock else EmbeddingService(device="cpu")
        getattr(service, method)("Magma" if method == "embed_text" else ["Magma"])

    return call


@dataclass
class _Held:
    """What a primitive's call needs released after the guard: a socket, a file, a process."""

    close: list[Callable[[], Any]]


def _socket_connect(held: _Held) -> None:
    sock = socket.socket()
    held.close.append(sock.close)
    sock.connect(("127.0.0.1", 9))


def _flock(held: _Held, function: str) -> Callable[[], None]:
    def call() -> None:
        fd = os.open(os.devnull, os.O_RDWR)
        held.close.append(lambda: os.close(fd))
        getattr(fcntl, function)(fd, fcntl.LOCK_EX)

    return call


def _popen(held: _Held, method: str) -> Callable[[], None]:
    def call() -> None:
        process = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)
        held.close.append(process.communicate)
        getattr(process, method)()

    return call


def _httpx_get(held: _Held) -> None:
    client = httpx.Client()
    held.close.append(client.close)
    client.get("http://127.0.0.1:9/")


def _future_result() -> None:
    concurrent.futures.Future().result()


#: A call of each primitive; one that blocks is refused before it runs (no
#: network, no wait, no model loaded).
PRIMITIVE_CALLS: dict[str, Callable[[_Held], Any]] = {
    "socket.getaddrinfo": lambda held: socket.getaddrinfo("localhost", 80),
    "socket.gethostbyname": lambda held: socket.gethostbyname("localhost"),
    "socket.gethostbyname_ex": lambda held: socket.gethostbyname_ex("localhost"),
    "socket.gethostbyaddr": lambda held: socket.gethostbyaddr("127.0.0.1"),
    "socket.create_connection": lambda held: socket.create_connection(("127.0.0.1", 9)),
    "socket.socket.connect": _socket_connect,
    "httpx.Client.send": lambda held: _httpx_get(held),
    "fcntl.flock": lambda held: _flock(held, "flock")(),
    "fcntl.lockf": lambda held: _flock(held, "lockf")(),
    "time.sleep": lambda held: time.sleep(0.01),
    "subprocess.run": lambda held: subprocess.run([sys.executable, "-c", "pass"]),
    "subprocess.Popen.wait": lambda held: _popen(held, "wait")(),
    "subprocess.Popen.communicate": lambda held: _popen(held, "communicate")(),
    "concurrent.futures.Future.result": lambda held: _future_result(),
    "pypdf.PdfReader.__init__": lambda held: _pdf_reader(),
    "agentic_cli.memory.kb.convert.convert_document": lambda held: _convert_document(),
    "trafilatura.extract": lambda held: _trafilatura(),
    "html2text.HTML2Text.handle": lambda held: _html2text(),
    "feedparser.parse": lambda held: _feedparser(),
    "agentic_cli.memory._core.mock_embeddings.MockEmbeddingService.embed_text": _embed(True, "embed_text"),
    "agentic_cli.memory._core.mock_embeddings.MockEmbeddingService.embed_batch": _embed(True, "embed_batch"),
    "agentic_cli.memory._core.embeddings.EmbeddingService.embed_text": _embed(False, "embed_text"),
    "agentic_cli.memory._core.embeddings.EmbeddingService.embed_batch": _embed(False, "embed_batch"),
}


def test_each_primitive_has_a_call():
    assert set(PRIMITIVE_CALLS) == {p.label for p in PRIMITIVES}


@pytest.mark.parametrize("label", sorted(PRIMITIVE_CALLS))
async def test_each_primitive_is_refused_on_the_loop(label):
    if label == "trafilatura.extract":
        require_kb_module("trafilatura")
    held = _Held([])
    try:
        with pytest.raises(AssertionError, match=rf"\n  - {label.replace('.', '[.]')}, from "):
            with no_blocking_calls():
                PRIMITIVE_CALLS[label](held)
    finally:
        for close in held.close:
            close()


def _bindings() -> dict[str, Any]:
    """What the guard replaces: each module's function, or each class's own attribute
    (none for an inherited method), for the primitives whose module imports."""
    bound = {}
    for primitive in PRIMITIVES:
        try:
            owner: Any = importlib.import_module(primitive.module)
        except ImportError:
            continue  # an optional module (trafilatura) that is not installed
        if primitive.owner is None:
            bound[primitive.label] = getattr(owner, primitive.name)
            continue
        for part in primitive.owner.split("."):
            owner = getattr(owner, part)
        bound[primitive.label] = owner.__dict__.get(primitive.name)
    return bound


def test_the_primitives_are_guarded_while_it_is_active_and_put_back_after():
    before = _bindings()

    with no_blocking_calls():
        during = _bindings()

    assert _bindings() == before
    assert {label: getattr(f, "__loop_guard__", None) for label, f in during.items()} == {
        label: label for label in during
    }


# --- the async tools -----------------------------------------------------------------

#: The async tools the framework registers (``@register_tool``).
REGISTERED = frozenset({
    "ask_clarification",
    "fetch_arxiv_paper",
    "ingest_arxiv_paper",
    "kb_ingest_file",
    "kb_ingest_text",
    "kb_read",
    "kb_search_concepts",
    "kb_write_concept",
    "search_arxiv",
    "web_fetch",
    "web_search",
})
#: The async tools ``tools/factories.py`` builds, bound to their services.
BOUND = frozenset({
    "ask_clarification",
    "fetch_arxiv_paper",
    "ingest_arxiv_paper",
    "kb_ingest_file",
    "kb_ingest_text",
    "kb_read",
    "kb_search_concepts",
    "kb_write_concept",
    "search_arxiv",
    "web_fetch",
})
#: Deep Agents' skill tools (``tools/skills/langchain_tools.py``), coroutines.
SKILL_TOOLS = frozenset({"list_skills", "load_skill", "load_skill_resource"})

_ADD_IT = (
    "An async tool runs on the event loop, where a blocking call freezes the "
    "app: its slow work (network, files, locks, parsing) goes through "
    "asyncio.to_thread. List the tool in {where} and add a call of it to "
    "{calls} in tests/tools/test_async_tools_never_block.py, which runs it "
    "through the loop guard (tests/loop_guard.py)."
)


def _register_every_framework_tool() -> None:
    """Import every module under ``agentic_cli.tools``, so each registers its tools."""
    import agentic_cli.tools

    for module in pkgutil.walk_packages(agentic_cli.tools.__path__, "agentic_cli.tools."):
        try:
            importlib.import_module(module.name)
        except ImportError:
            pass  # a backend of an extra that is not installed


def _defined_in_agentic_cli(func: Callable[..., Any]) -> bool:
    """Whether ``func`` (or what a ``functools.partial`` of it wraps) is the framework's."""
    while isinstance(func, functools.partial):
        func = func.func
    return getattr(inspect.unwrap(func), "__module__", "").startswith("agentic_cli.")


def test_the_registry_s_async_tools_are_the_guarded_ones():
    from agentic_cli.tools.registry import get_registry

    _register_every_framework_tool()
    found = set()
    for definition in get_registry().list_tools():
        for func in (definition.func, *definition.variants):
            if func is not None and _defined_in_agentic_cli(func) and inspect.iscoroutinefunction(func):
                found.add(definition.name)

    assert found == REGISTERED, _ADD_IT.format(where="REGISTERED", calls="CALLS")


@pytest.fixture
def knowledge_base(tmp_path) -> KnowledgeBase:
    return KnowledgeBase({"project": KnowledgeBaseManager(tmp_path / "kb", use_mock=True)})


def test_the_factories_async_tools_are_the_guarded_ones(knowledge_base):
    from agentic_cli.tools import factories

    service = object()
    arguments: dict[str, tuple] = {
        "make_memory_tools": (service,),
        "make_kb_tools": (knowledge_base,),
        "make_webfetch_tool": (service,),
        "make_sandbox_tool": (service,),
        "make_arxiv_tools": (service,),
        "make_ingest_arxiv_tool": (service, knowledge_base),
        "make_interaction_tools": (service,),
    }
    makers = {
        name for name, func in inspect.getmembers(factories, inspect.isfunction)
        if name.startswith("make_") and func.__module__ == factories.__name__
    }
    assert makers == set(arguments), (
        "A factory was added or removed: give its arguments here. "
        + _ADD_IT.format(where="BOUND", calls="CALLS")
    )

    found = set()
    for name, args in arguments.items():
        made = getattr(factories, name)(*args)
        if isinstance(made, dict):
            made = list(made.values())
        for tool in made if isinstance(made, (list, tuple)) else [made]:
            if inspect.iscoroutinefunction(tool):
                found.add(tool.__name__)

    assert found == BOUND, _ADD_IT.format(where="BOUND", calls="CALLS")


def _skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skills" / "bird-guide"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        textwrap.dedent(
            """\
            ---
            name: bird-guide
            description: Name the birds of a marsh.
            ---
            # Bird guide
            Look at the legs first.
            """
        ),
        encoding="utf-8",
    )
    (skill / "references" / "herons.md").write_text("Herons wade at dawn.\n", encoding="utf-8")
    return skill.parent


def _skill_tools(tmp_path: Path) -> dict[str, Any]:
    pytest.importorskip("langchain_core")
    from agentic_cli.models import ProviderKind
    from agentic_cli.tools.skills import SkillStore
    from agentic_cli.tools.skills.langchain_tools import make_skill_tools

    skills = SkillStore([str(_skill(tmp_path))]).resolve(["bird-guide"])
    tools, _ = make_skill_tools(skills, provider=ProviderKind.ANTHROPIC)
    return {tool.name: tool for tool in tools}


def test_the_skill_tools_are_the_guarded_ones(tmp_path):
    tools = _skill_tools(tmp_path)

    found = {name for name, tool in tools.items() if inspect.iscoroutinefunction(tool.coroutine)}

    assert found == SKILL_TOOLS, _ADD_IT.format(where="SKILL_TOOLS", calls="SKILL_CALLS")


# --- calling them through the guard ----------------------------------------------------

PAPER_ID = "1706.03762"
SITE = "https://example.com"
#: What the fake DNS answers for every host: a global address, so the pinned
#: transport lets the request through to the fake web.
ADDRESS = "93.184.216.34"
ARTICLE = "Herons wade in shallow water at dawn, waiting for fish."
HTML_PAGE = (
    "<html><head><title>Herons</title></head><body><nav><a href='/'>Home</a></nav>"
    f"<article><h1>Herons</h1><p>{ARTICLE}</p><p>{ARTICLE}</p></article></body></html>"
).encode()
SIDECAR = "SUMMARY:\nA summary.\n\nCLAIMS:\n- Herons wade.\n\nENTITIES:\nBirds: heron\n"


class _Summarizer:
    """The LLM, faked: a sidecar's three sections, and what it was given to read."""

    def __init__(self) -> None:
        self.contents: list[str] = []

    async def summarize(self, content: str, prompt: str) -> str:
        self.contents.append(content)
        return SIDECAR


class _Web:
    """The web under the pinned transport, faked: pages by host and path; anything else
    (robots.txt included) is a 404."""

    def __init__(self) -> None:
        self.pages: dict[tuple[str, str], tuple[bytes, str]] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        # The pinned transport sends to the address; the Host header names the site.
        found = self.pages.get((request.headers["host"], request.url.path))
        if found is None:
            return httpx.Response(404, text="not found")
        body, content_type = found
        return httpx.Response(200, content=body, headers={"content-type": content_type})


def _search_api(request: httpx.Request) -> httpx.Response:
    """Tavily, faked."""
    return httpx.Response(200, json={"results": [
        {"title": "Herons", "url": f"{SITE}/herons", "content": ARTICLE, "score": 0.9},
    ]})


class _SearchClient(httpx.AsyncClient):
    """httpx's client; one made without a transport (the search backends') reaches the fake."""

    def __init__(self, *args: Any, transport: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, transport=transport or httpx.MockTransport(_search_api), **kwargs)


def _workflow():
    """A workflow manager with nothing built but its input callback (``request_user_input``)."""
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

    with patch.object(BaseWorkflowManager, "__abstractmethods__", set()):
        manager = BaseWorkflowManager.__new__(BaseWorkflowManager)
    manager._user_input_callback = ContextVar("loop_guard_input_callback", default=None)
    return manager


async def _answer(request) -> str:
    return request.choices[0]


@dataclass
class Env:
    """The services and files the calls use, all offline."""

    project: Path
    summarizer: _Summarizer
    kb: KnowledgeBase
    source: ArxivSearchSource
    workflow: Any
    settings: BaseSettings
    herons: str  # a text document's id
    ledger: str  # a PDF stored without its text (as before text was kept) and without a sidecar

    @property
    def services(self) -> dict[str, Any]:
        return {
            KNOWLEDGE_BASE: self.kb,
            ARXIV_SOURCE: self.source,
            LLM_SUMMARIZER: self.summarizer,
            WORKFLOW: self.workflow,
        }

    def text_of(self, doc_id: str) -> str:
        return self.kb.get_document(doc_id).document.content


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    from agentic_cli.tools import arxiv_source, webfetch_tool
    from agentic_cli.tools.webfetch.transport import PinnedTransport

    monkeypatch.chdir(tmp_path)  # web_fetch saves its pages under the working directory

    def resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ADDRESS, port))]

    web = _Web()
    web.pages[("arxiv.org", f"/pdf/{PAPER_ID}v1")] = (
        pdf_with_text("Attention replaces recurrence"), "application/pdf"
    )
    web.pages[("example.com", "/article")] = (HTML_PAGE, "text/html; charset=utf-8")
    web.pages[("example.com", "/paper.pdf")] = (pdf_with_text("Magma rises as lava"), "application/pdf")
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(
        webfetch_tool, "PinnedTransport",
        lambda validator: PinnedTransport(validator, inner=httpx.MockTransport(web.handle)),
    )
    monkeypatch.setattr(webfetch_tool, "_fetcher", None)
    monkeypatch.setattr(webfetch_tool, "_fetcher_settings_snapshot", None)
    monkeypatch.setattr(arxiv_source.ArxivSearchSource, "rate_limit", property(lambda self: 0.0))
    monkeypatch.setattr(httpx, "AsyncClient", _SearchClient)

    api = ArxivAPI(papers=[Paper(PAPER_ID, title="Attention Is All You Need", abstract="Attention only.")])
    summarizer = _Summarizer()
    manager = KnowledgeBaseManager(tmp_path / "kb", embedding_service=_embedding_service(), summarizer=summarizer)
    kb = KnowledgeBase({"project": manager})
    herons = manager.ingest_document(
        content="Herons wade at dawn.", title="Herons", source_type=SourceType.USER
    )
    ledger = manager.ingest_document(
        content="", title="Ledger", source_type=SourceType.LOCAL,
        file_bytes=pdf_with_text("Magma ledger entries"), file_extension=".pdf",
    )
    manager.sidecar_path(ledger.id).unlink()
    kb.write_concept("Wading birds", "Herons wade at dawn.", [herons.id])

    (tmp_path / "notes.md").write_text("# Otters\n\nOtters nest in burrows.\n", encoding="utf-8")
    (tmp_path / "notes.pdf").write_bytes(pdf_with_text("Otters nest in burrows"))
    (tmp_path / "page.html").write_bytes(HTML_PAGE)

    return Env(
        project=tmp_path,
        summarizer=summarizer,
        kb=kb,
        source=ArxivSearchSource(transport=api.transport),
        workflow=_workflow(),
        settings=BaseSettings(search_backend="tavily", tavily_api_key="test-key"),
        herons=herons.id,
        ledger=ledger.id,
    )


class _Encoder:
    """Stands in for a sentence-transformers model: fixed vectors, nothing loaded."""

    def encode(self, texts, **kwargs):
        import numpy as np

        vectors = np.asarray([FakeEmbeddingService().vector(t) for t in ([texts] if isinstance(texts, str) else texts)])
        return vectors[0] if isinstance(texts, str) else vectors


def _embedding_service() -> EmbeddingService:
    """An ``EmbeddingService`` whose model is ``_Encoder``: the knowledge base
    embeds through the guarded ``EmbeddingService.embed_batch`` (a keyword-only
    knowledge base would embed nothing, and so reach no guarded call)."""
    service = EmbeddingService(model_name="offline-encoder", device="cpu")
    service._model, service._embedding_dim = _Encoder(), 8
    return service


Tool = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Call:
    """One call of an async tool through the guard, and what its result must show."""

    tool: str
    id: str
    run: Callable[[Tool, Env], Awaitable[dict[str, Any]]]
    check: Callable[[dict[str, Any], Env], bool]


CALLS: tuple[Call, ...] = (
    Call(
        "ask_clarification", "a-question",
        lambda tool, env: tool("Which bird wades at dawn?", options=["Heron", "Egret"]),
        lambda result, env: result["response"] == "Heron",
    ),
    Call(
        "search_arxiv", "a-search",
        lambda tool, env: tool("attention"),
        lambda result, env: [p["arxiv_id"] for p in result["papers"]] == [PAPER_ID],
    ),
    Call(
        "fetch_arxiv_paper", "a-paper",
        lambda tool, env: tool(PAPER_ID),
        lambda result, env: result["paper"]["title"] == "Attention Is All You Need",
    ),
    Call(
        "ingest_arxiv_paper", "a-pdf-download",
        lambda tool, env: tool(PAPER_ID),
        lambda result, env: (
            result["pdf_downloaded"] is True
            and env.text_of(result["document_id"]) == "Attention replaces recurrence"
        ),
    ),
    Call(
        "kb_ingest_text", "a-text",
        lambda tool, env: tool("Egrets fish in shallow water.", title="Egrets"),
        lambda result, env: env.text_of(result["document_id"]) == "Egrets fish in shallow water.",
    ),
    Call(
        "kb_ingest_file", "a-text-file",
        lambda tool, env: tool(str(env.project / "notes.md")),
        lambda result, env: "Otters nest in burrows." in env.text_of(result["document_id"]),
    ),
    Call(
        "kb_ingest_file", "a-pdf-file",
        lambda tool, env: tool(str(env.project / "notes.pdf")),
        lambda result, env: env.text_of(result["document_id"]) == "Otters nest in burrows",
    ),
    Call(
        "kb_ingest_file", "an-html-file",
        lambda tool, env: tool(str(env.project / "page.html")),
        lambda result, env: ARTICLE in env.text_of(result["document_id"]),
    ),
    Call(
        "kb_read", "a-pdf-s-text",
        lambda tool, env: tool("Ledger", full=True),
        lambda result, env: result["content"] == "Magma ledger entries",
    ),
    Call(
        "kb_read", "a-sidecar-written-from-a-pdf",
        lambda tool, env: tool("Ledger"),
        lambda result, env: (
            "A summary." in result["sidecar"] and env.summarizer.contents == ["Magma ledger entries"]
        ),
    ),
    Call(
        "kb_write_concept", "a-concept-page",
        lambda tool, env: tool("Herons", "Herons wade at dawn.", [env.herons]),
        lambda result, env: result["slug"] == "herons",
    ),
    Call(
        "kb_search_concepts", "a-concept-search",
        lambda tool, env: tool("wading"),
        lambda result, env: result["count"] == 1,
    ),
    Call(
        "web_fetch", "an-html-page",
        lambda tool, env: tool(f"{SITE}/article", "What do herons do?"),
        lambda result, env: "saved_path" in result and ARTICLE in env.summarizer.contents[-1],
    ),
    Call(
        "web_fetch", "a-pdf-page",
        lambda tool, env: tool(f"{SITE}/paper.pdf", "What rises?"),
        lambda result, env: "saved_path" in result and "Magma rises as lava" in env.summarizer.contents[-1],
    ),
    Call(
        "web_search", "a-search",
        lambda tool, env: tool("herons"),
        lambda result, env: [r["title"] for r in result["results"]] == ["Herons"],
    ),
)


#: A call of each skill tool: its arguments, and what its answer holds.
SKILL_CALLS: dict[str, tuple[dict[str, str], str]] = {
    "list_skills": ({}, "bird-guide"),
    "load_skill": ({"skill_name": "bird-guide"}, "Look at the legs first."),
    "load_skill_resource": (
        {"skill_name": "bird-guide", "file_path": "references/herons.md"}, "Herons wade at dawn."
    ),
}


def test_every_async_tool_has_a_call():
    assert {call.tool for call in CALLS} == REGISTERED | BOUND, "a listed tool has no call in CALLS"
    assert set(SKILL_CALLS) == SKILL_TOOLS, "a skill tool has no call in SKILL_CALLS"


def _registered(name: str, env: Env) -> Tool:
    from agentic_cli.tools.registry import get_registry

    _register_every_framework_tool()
    return get_registry().get(name).func


def _bound(name: str, env: Env) -> Tool:
    from agentic_cli.tools.factories import (
        make_arxiv_tools,
        make_ingest_arxiv_tool,
        make_interaction_tools,
        make_kb_tools,
        make_webfetch_tool,
    )

    tools = [
        *make_kb_tools(env.kb),
        make_webfetch_tool(env.summarizer),
        *make_arxiv_tools(env.source),
        make_ingest_arxiv_tool(env.source, env.kb),
        *make_interaction_tools(env.workflow),
    ]
    return {tool.__name__: tool for tool in tools}[name]


FLAVORS = {"registered": _registered, "bound": _bound}


@pytest.mark.parametrize(
    ("call", "flavor"),
    [
        pytest.param(call, flavor, id=f"{call.tool}-{call.id}-{flavor}")
        for call in CALLS
        for flavor in FLAVORS
        if flavor == "registered" or call.tool in BOUND
    ],
)
async def test_an_async_tool_makes_no_blocking_call_on_the_loop(call, flavor, env):
    tool = FLAVORS[flavor](call.tool, env)
    token = set_service_registry(env.services if flavor == "registered" else {})
    env.workflow.set_input_callback(_answer)
    try:
        with SettingsContext(env.settings):
            with no_blocking_calls():
                result = await call.run(tool, env)
    finally:
        token.var.reset(token)

    assert result["success"] is True, result
    assert call.check(result, env), result


@pytest.mark.parametrize("name", sorted(SKILL_CALLS))
async def test_a_skill_tool_makes_no_blocking_call_on_the_loop(name, tmp_path):
    tool = _skill_tools(tmp_path)[name]
    arguments, expected = SKILL_CALLS[name]

    with no_blocking_calls():
        result = await tool.ainvoke(arguments)

    assert expected in (result if isinstance(result, str) else json.dumps(result))
