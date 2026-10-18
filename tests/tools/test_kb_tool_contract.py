"""What the knowledge-base tools return, pinned before 0.6.4 moves their logic.

0.6.4 moves the logic behind the ``kb_*`` tools, ``ingest_arxiv_paper`` and
``/papers`` into ``agentic_cli.memory``. Their names, arguments and results
must not change, so each scenario here calls a tool through its public entry
points, the module-level tool (with the service registry) and the
factory-bound tool, and compares the full result with
``kb_tool_contract.json``, recorded from 0.6.3. The two entry points must
return the same result.

Each scenario runs with one knowledge base (``one``), with a project and a
user knowledge base (``two``), and with both names pointing at one knowledge
base (``same``). Document IDs, timestamps, timings and the temporary
directory are replaced by placeholders.

The JSON changes once: when arXiv papers go through the ingestion pipeline,
the ``summary`` of the ``arxiv_*`` scenarios becomes the synthesis other
documents already get. Any other change to it is a broken contract.

The JSON was written from 0.6.3 by ``python -m tests.tools.test_kb_tool_contract``.
Never rewrite it to make a failing scenario pass.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import inspect
import json
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

import pytest

from tests.kb_registry import kb_services

GOLDEN = Path(__file__).with_name("kb_tool_contract.json")
LAYOUTS = ("one", "two", "same")
ENTRIES = ("registry", "factory")
EPOCH = datetime(2026, 1, 1, 9, 0, 0)

KB_TOOLS = {
    "kb_search": "sync",
    "kb_ingest_text": "async",
    "kb_ingest_file": "async",
    "kb_ingest_url": "async",
    "kb_read": "async",
    "kb_list": "sync",
    "kb_write_concept": "async",
    "kb_search_concepts": "async",
}

ARTICLE = (
    "<html><head><title>Volcano notes</title>"
    '<meta name="author" content="Ada Lovelace">'
    "</head><body><nav>Home</nav><article><h1>Volcanoes</h1>"
    "<p>Magma rises through the crust and erupts as lava.</p>"
    "</article></body></html>"
)
PAGE_URL = "https://example.com/articles/volcanoes"
CRASH_CODEC = "kb-contract-crash"


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------


class _Summarizer:
    """Answers the knowledge base's two prompts with text built from the title."""

    async def summarize(self, content: str, prompt: str) -> str:
        title = prompt.split("Title: ", 1)[1].split("\n", 1)[0] if "Title: " in prompt else "?"
        if "SUMMARY:" in prompt:
            return (
                f"SUMMARY:\nSynthesis of {title}.\n\n"
                f"CLAIMS:\n- {title} makes one claim.\n\n"
                "ENTITIES:\nTopics: geology"
            )
        return f"Short summary of {title}."


def _pdf_with_text(text: str) -> bytes:
    """A one-page PDF whose text pypdf extracts."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


class _ArxivSource:
    """An arXiv source that answers from memory."""

    PAPER = {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "abstract": "We propose the Transformer, built on attention alone.",
        "abs_url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "src_url": "https://arxiv.org/e-print/1706.03762",
        "categories": ["cs.CL"],
        "primary_category": "cs.CL",
        "published_date": "2017-06-12",
        "updated_date": "2017-12-06",
    }

    def __init__(self, *, pdf: bytes | Exception | None = None, missing: bool = False) -> None:
        self._pdf = _pdf_with_text("Attention replaces recurrence") if pdf is None else pdf
        self._missing = missing

    async def fetch_by_id(self, arxiv_id: str) -> dict[str, Any]:
        if self._missing:
            raise LookupError(f"Paper not found: {arxiv_id}")
        return dict(self.PAPER)

    async def download_pdf(self, url: str) -> bytes:
        if isinstance(self._pdf, Exception):
            raise self._pdf
        return self._pdf


class _Fetcher:
    """The web fetcher ``kb_ingest_url`` uses, answering with ARTICLE."""

    async def fetch(self, url: str, timeout: int = 30):
        from agentic_cli.tools.webfetch.fetcher import FetchResult

        body = ARTICLE.encode()
        return FetchResult(
            success=True,
            content=ARTICLE,
            content_type="text/html; charset=utf-8",
            raw=body,
            charset="utf-8",
            final_url=url,
        )


def _crash_codec(name: str):
    """A codec whose decoder fails the way a converter bug would."""
    if name != CRASH_CODEC.replace("-", "_"):
        return None

    def decode(data, errors="strict"):
        raise RuntimeError("decoder exploded")

    return codecs.CodecInfo(
        name=CRASH_CODEC, encode=lambda s, errors="strict": (s.encode(), len(s)), decode=decode
    )


# ---------------------------------------------------------------------------
# One scenario's world
# ---------------------------------------------------------------------------


@dataclass
class _Env:
    root: Path
    layout: str
    entry: str
    project: Any
    user: Any  # None in "one"; the project manager in "same"
    docs: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)

    @property
    def second(self):
        """Where the user scope's documents go: the user KB, else the project KB."""
        return self.user if self.user is not None else self.project

    async def call(self, name: str, **kwargs) -> Any:
        result = self.tools[name](**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result


def _seed(env: _Env) -> None:
    from agentic_cli.memory.kb import SourceType

    def add(key, kb, minute, **kwargs):
        doc = kb.ingest_document(**kwargs)
        doc.updated_at = EPOCH + timedelta(minutes=minute)
        env.docs[key] = doc

    add(
        "volcano", env.project, 1,
        content="Basalt lava cools into columns near the volcano.",
        title="Volcano field notes",
        source_type=SourceType.USER,
        metadata={"authors": ["Ada Lovelace"], "tags": ["field"]},
    )
    add(
        "basalt", env.project, 2,
        content="Columns of basalt form as thick lava flows cool slowly.",
        title="Basalt columns",
        source_type=SourceType.WEB,
        source_url="https://example.org/basalt",
    )
    add(
        "glacier", env.second, 3,
        content="The glacier retreated two kilometres and meltwater carved new channels.",
        title="Glacier survey",
        source_type=SourceType.ARXIV,
        metadata={"arxiv_id": "2401.00001", "authors": ["Mary Somerville"]},
    )
    add(
        "ledger", env.second, 4,
        content="",
        title="Scanned ledger",
        source_type=SourceType.LOCAL,
        file_bytes=_pdf_with_text("Magma ledger entries"),
        file_extension=".pdf",
    )


@contextlib.contextmanager
def _patched(target: Any, name: str, value: Any) -> Iterator[None]:
    old = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, old)


def registry_entry(env: _Env, arxiv: Any) -> tuple[dict[str, Any], dict[str, Callable[..., Any]]]:
    """The module-level tools, finding the knowledge bases in the service registry."""
    from agentic_cli.tools import arxiv_tools, knowledge_tools

    registry = kb_services(env.project, env.user)
    if arxiv is not None:
        registry["arxiv_source"] = arxiv
    tools = {name: getattr(knowledge_tools, name) for name in KB_TOOLS}
    tools["ingest_arxiv_paper"] = arxiv_tools.ingest_arxiv_paper
    return registry, tools


def factory_entry(env: _Env, arxiv: Any) -> tuple[dict[str, Any], dict[str, Callable[..., Any]]]:
    """The factory-bound tools, with an empty service registry."""
    from agentic_cli.tools.factories import make_ingest_arxiv_tool, make_kb_tools

    tools = {t.__name__: t for t in make_kb_tools(env.project, env.user)}
    if arxiv is not None:
        tools["ingest_arxiv_paper"] = make_ingest_arxiv_tool(arxiv, env.project)
    return {}, tools


# Later tests add entry points of their own and compare with the same JSON.
ENTRY_POINTS = {"registry": registry_entry, "factory": factory_entry}


@contextlib.contextmanager
def _world(layout: str, entry: str, entry_points: dict[str, Any], *, arxiv: Any = None) -> Iterator[_Env]:
    """Temp directories, settings, managers, the registry and the tools."""
    from agentic_cli.config import BaseSettings, set_settings
    from agentic_cli.memory.kb import KnowledgeBaseManager
    from agentic_cli.workflow.service_registry import set_service_registry

    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        root = Path(tmp)
        (root / "cwd").mkdir()
        old_cwd = os.getcwd()
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(root / "home")
        os.chdir(root / "cwd")
        # The converter's main-text extractor is optional; the contract is
        # recorded without it so it holds with and without the kb extra.
        old_trafilatura = sys.modules.get("trafilatura", None)
        had_trafilatura = "trafilatura" in sys.modules
        sys.modules["trafilatura"] = None  # type: ignore[assignment]
        token = None
        try:
            set_settings(BaseSettings(workspace_dir=root / "workspace", knowledge_base_use_mock=True))
            project = KnowledgeBaseManager(root / "project_kb", summarizer=_Summarizer(), use_mock=True)
            if layout == "one":
                user = None
            elif layout == "same":
                user = project
            else:
                user = KnowledgeBaseManager(root / "user_kb", summarizer=_Summarizer(), use_mock=True)
            env = _Env(root=root, layout=layout, entry=entry, project=project, user=user)
            _seed(env)
            registry, env.tools = entry_points[entry](env, arxiv)
            token = set_service_registry(registry)
            yield env
        finally:
            if token is not None:
                token.var.reset(token)
            set_settings(None)  # type: ignore[arg-type]
            if had_trafilatura:
                sys.modules["trafilatura"] = old_trafilatura  # type: ignore[assignment]
            else:
                sys.modules.pop("trafilatura", None)
            os.chdir(old_cwd)
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

Scenario = Callable[[_Env], Awaitable[Any]]
SCENARIOS: dict[str, tuple[Scenario, dict[str, Any]]] = {}


def scenario(name: str, **world: Any):
    def register(fn: Scenario) -> Scenario:
        SCENARIOS[name] = (fn, world)
        return fn

    return register


@scenario("search_all")
async def _(env):
    return await env.call("kb_search", query="basalt glacier volcano")


@scenario("search_filtered")
async def _(env):
    return await env.call("kb_search", query="basalt glacier volcano", filters='{"source_type": "arxiv"}')


@scenario("search_top_k")
async def _(env):
    return await env.call("kb_search", query="basalt glacier volcano", top_k=1)


@scenario("search_bad_filters")
async def _(env):
    return await env.call("kb_search", query="basalt", filters="not json")


@scenario("search_bad_filter_value")
async def _(env):
    return await env.call("kb_search", query="basalt", filters='{"source_type": "blog"}')


@scenario("ingest_text")
async def _(env):
    ingested = await env.call(
        "kb_ingest_text",
        content="Pumice floats on water because of trapped gas.",
        title="Pumice",
        source_type="internal",
        source_url="https://example.org/pumice",
        authors=["Mary Anning"],
        abstract="Why pumice floats.",
        tags=["rocks"],
    )
    listed = await env.call("kb_list", query="pumice")
    found = await env.call("kb_search", query="pumice trapped gas", top_k=1)
    return {"ingest": ingested, "listed": listed, "found": found}


@scenario("ingest_text_defaults")
async def _(env):
    return await env.call(
        "kb_ingest_text",
        content="Obsidian is volcanic glass that forms when lava cools too fast to crystallise.",
    )


@scenario("ingest_text_bad_source_type")
async def _(env):
    return await env.call("kb_ingest_text", content="Some text.", source_type="blog")


@scenario("ingest_text_empty")
async def _(env):
    return await env.call("kb_ingest_text", content="")


@scenario("ingest_file_markdown")
async def _(env):
    path = env.root / "notes.md"
    path.write_text("# Lava tubes\n\nLava tubes form when the surface of a flow hardens.\n")
    ingested = await env.call("kb_ingest_file", path=str(path), tags=["caves"])
    read = await env.call("kb_read", doc_id_or_title=ingested.get("document_id", ""), full=True)
    return {"ingest": ingested, "read": read}


@scenario("ingest_file_pdf")
async def _(env):
    path = env.root / "scan.pdf"
    path.write_bytes(_pdf_with_text("Tephra layers record eruptions"))
    ingested = await env.call("kb_ingest_file", path=str(path))
    read = await env.call("kb_read", doc_id_or_title=ingested.get("document_id", ""))
    return {"ingest": ingested, "read": read}


@scenario("ingest_file_html")
async def _(env):
    path = env.root / "page.html"
    path.write_text(ARTICLE)
    return await env.call("kb_ingest_file", path=str(path), title="", source_url="https://example.org/page")


@scenario("ingest_file_saved_page")
async def _(env):
    from agentic_cli.config import get_settings
    from agentic_cli.tools.webfetch.saved import save_page, saved_pages_dir

    saved = save_page(
        saved_pages_dir(get_settings().app_name), url=PAGE_URL, final_url=PAGE_URL,
        content_type="text/html; charset=utf-8", charset="utf-8",
        data=ARTICLE.encode(), truncated=False,
    )
    ingested = await env.call("kb_ingest_file", path=str(saved))
    found = await env.call("kb_search", query="magma crust", top_k=1)
    return {"ingest": ingested, "found": found}


@scenario("ingest_file_saved_page_truncated")
async def _(env):
    from agentic_cli.config import get_settings
    from agentic_cli.tools.webfetch.saved import save_page, saved_pages_dir

    saved = save_page(
        saved_pages_dir(get_settings().app_name), url=PAGE_URL, final_url=PAGE_URL,
        content_type="text/html", charset=None, data=ARTICLE.encode(), truncated=True,
    )
    return await env.call("kb_ingest_file", path=str(saved), title="Cut page", source_type="internal")


@scenario("ingest_file_saved_page_metadata")
async def _(env):
    from agentic_cli.config import get_settings
    from agentic_cli.tools.webfetch.saved import save_page, saved_pages_dir

    saved = save_page(
        saved_pages_dir(get_settings().app_name), url=PAGE_URL, final_url=PAGE_URL,
        content_type="text/html", charset="utf-8", data=ARTICLE.encode(), truncated=False,
    )
    meta = saved.with_name(saved.stem + ".meta.json")
    return await env.call("kb_ingest_file", path=str(meta))


@scenario("ingest_file_undecodable")
async def _(env):
    path = env.root / "blob.bin"
    path.write_bytes(b"\xff\xfe\x00\x01binary")
    return await env.call("kb_ingest_file", path=str(path))


@scenario("ingest_file_converter_crash")
async def _(env):
    from agentic_cli.config import get_settings
    from agentic_cli.tools.webfetch.saved import save_page, saved_pages_dir

    saved = save_page(
        saved_pages_dir(get_settings().app_name), url=PAGE_URL, final_url=PAGE_URL,
        content_type="text/plain", charset=CRASH_CODEC, data=b"plain words", truncated=False,
    )
    codecs.register(_crash_codec)
    try:
        return await env.call("kb_ingest_file", path=str(saved))
    finally:
        codecs.unregister(_crash_codec)


@scenario("ingest_file_missing")
async def _(env):
    return await env.call("kb_ingest_file", path=str(env.root / "absent.txt"))


@scenario("ingest_url")
async def _(env):
    import agentic_cli.tools.webfetch_tool as webfetch_tool

    with _patched(webfetch_tool, "get_or_create_fetcher", lambda: _Fetcher()):
        return await env.call("kb_ingest_url", url=PAGE_URL, tags=["web"])


@scenario("ingest_url_bad_scheme")
async def _(env):
    return await env.call("kb_ingest_url", url="file:///etc/passwd")


@scenario("read_sidecar")
async def _(env):
    return await env.call("kb_read", doc_id_or_title="Glacier survey")


@scenario("read_full")
async def _(env):
    return await env.call("kb_read", doc_id_or_title=env.docs["volcano"].id, full=True)


@scenario("read_full_truncated")
async def _(env):
    return await env.call("kb_read", doc_id_or_title="basalt", full=True, max_chars=12)


@scenario("read_by_id_prefix")
async def _(env):
    return await env.call("kb_read", doc_id_or_title=env.docs["glacier"].id[:8], full=True)


@scenario("read_pdf_without_text")
async def _(env):
    return await env.call("kb_read", doc_id_or_title="Scanned ledger", full=True)


@scenario("read_missing_sidecar_is_written")
async def _(env):
    doc = env.docs["ledger"]
    env.second.sidecar_path(doc.id).unlink()
    first = await env.call("kb_read", doc_id_or_title=doc.id)
    return {"read": first, "written": env.second.sidecar_path(doc.id).exists()}


@scenario("read_unknown")
async def _(env):
    return await env.call("kb_read", doc_id_or_title="no such document")


@scenario("list_all")
async def _(env):
    return await env.call("kb_list")


@scenario("list_by_author")
async def _(env):
    return await env.call("kb_list", query="somerville")


@scenario("list_by_source_type")
async def _(env):
    return await env.call("kb_list", source_type="web")


@scenario("list_bad_source_type")
async def _(env):
    return await env.call("kb_list", source_type="blog")


@scenario("list_limit")
async def _(env):
    return await env.call("kb_list", limit=2)


@scenario("concept_write_cites_both_scopes")
async def _(env):
    return await env.call(
        "kb_write_concept",
        title="Volcanic rocks",
        body="Basalt and pumice both come from lava.",
        sources=[env.docs["volcano"].id, env.docs["glacier"].id, "missing-id"],
    )


@scenario("concept_write_no_valid_source")
async def _(env):
    return await env.call("kb_write_concept", title="Nothing", body="No sources.", sources=["missing-id"])


@scenario("concept_search")
async def _(env):
    await env.call(
        "kb_write_concept",
        title="Volcanic rocks",
        body="Basalt and pumice both come from lava.",
        sources=[env.docs["volcano"].id],
        slug="volcanic-rocks",
    )
    return await env.call("kb_search_concepts", query="pumice")


@scenario("arxiv_ingest", arxiv=lambda: _ArxivSource())
async def _(env):
    ingested = await env.call("ingest_arxiv_paper", arxiv_id="https://arxiv.org/abs/1706.03762v2", tags=["nlp"])
    found = await env.call("kb_search", query="attention recurrence", top_k=1)
    return {"ingest": ingested, "found": found}


@scenario("arxiv_download_fails", arxiv=lambda: _ArxivSource(pdf=RuntimeError("network down")))
async def _(env):
    ingested = await env.call("ingest_arxiv_paper", arxiv_id="1706.03762")
    found = await env.call("kb_search", query="transformer attention", top_k=1)
    return {"ingest": ingested, "found": found}


@scenario("arxiv_not_found", arxiv=lambda: _ArxivSource(missing=True))
async def _(env):
    return await env.call("ingest_arxiv_paper", arxiv_id="9999.99999")


async def run_scenario(
    name: str, layout: str, entry: str, entry_points: dict[str, Any] = ENTRY_POINTS
) -> Any:
    fn, world = SCENARIOS[name]
    arxiv = world["arxiv"]() if "arxiv" in world else None
    with _world(layout, entry, entry_points, arxiv=arxiv) as env:
        return _normalize(await fn(env), env.root)


def tools_of(entry: str, entry_points: dict[str, Any] = ENTRY_POINTS) -> dict[str, Callable[..., Any]]:
    """The tools an entry point hands out, for checking their signatures."""
    with _world("two", entry, entry_points, arxiv=_ArxivSource()) as env:
        return dict(env.tools)


def call_styles(tools: dict[str, Callable[..., Any]]) -> dict[str, str]:
    return {name: "async" if inspect.iscoroutinefunction(fn) else "sync" for name, fn in tools.items()}


def arguments(tools: dict[str, Callable[..., Any]]) -> dict[str, list[list[Any]]]:
    return {
        name: [
            [p.name, None if p.default is inspect.Parameter.empty else p.default]
            for p in inspect.signature(fn).parameters.values()
        ]
        for name, fn in tools.items()
    }


# ---------------------------------------------------------------------------
# No knowledge base, and /papers
# ---------------------------------------------------------------------------

NO_KB_CALLS = {
    "kb_search": {"query": "basalt"},
    "kb_ingest_text": {"content": "Some text."},
    "kb_ingest_file": {"path": "notes.md"},
    "kb_ingest_url": {"url": PAGE_URL},
    "kb_read": {"doc_id_or_title": "Basalt columns"},
    "kb_list": {},
    "kb_write_concept": {"title": "T", "body": "B", "sources": ["x"]},
    "kb_search_concepts": {"query": "basalt"},
    "ingest_arxiv_paper": {"arxiv_id": "1706.03762"},
}


async def run_without_knowledge_base(tool: str) -> Any:
    """The module-level tool with an empty registry (arXiv gets its source)."""
    from agentic_cli.tools import arxiv_tools, knowledge_tools
    from agentic_cli.workflow.service_registry import set_service_registry

    registry = {"arxiv_source": _ArxivSource()} if tool == "ingest_arxiv_paper" else {}
    fn = arxiv_tools.ingest_arxiv_paper if tool == "ingest_arxiv_paper" else getattr(knowledge_tools, tool)
    token = set_service_registry(registry)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = fn(**NO_KB_CALLS[tool])
            if inspect.isawaitable(result):
                result = await result
        return result
    finally:
        token.var.reset(token)


class _Session:
    def __init__(self) -> None:
        self.out: list[Any] = []

    def add_rich(self, table) -> None:
        columns = [str(c.header) for c in table.columns]
        rows = [list(r) for r in zip(*[[str(cell) for cell in c._cells] for c in table.columns])]
        for row in rows:
            row[columns.index("ID")] = "<id>"
            row[columns.index("Created")] = "<date>"
        self.out.append({"table": str(table.title), "columns": columns, "rows": rows})

    def add_message(self, role: str, content: str) -> None:
        self.out.append({"message": content})

    def add_warning(self, content: str) -> None:
        self.out.append({"warning": content})

    def add_error(self, content: str) -> None:
        self.out.append({"error": content})


PAPERS_CALLS = {
    "papers_project": "",
    "papers_global": "--global",
    "papers_query": "volcano",
    "papers_query_author": "somerville --global",
    "papers_source": "--source=web",
    "papers_bad_source": "--source=blog",
}


async def run_papers(name: str, layout: str, with_workflow: bool) -> Any:
    """``/papers`` over knowledge bases on disk, with or without a ready workflow."""
    from agentic_cli.cli.builtin_commands import PapersCommand
    from agentic_cli.config import BaseSettings
    from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType

    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        root = Path(tmp)
        (root / "cwd").mkdir()
        old_cwd = os.getcwd()
        os.chdir(root / "cwd")
        try:
            # The user knowledge base is workspace_dir/knowledge_base, so a
            # workspace of ./.agentic_cli puts both scopes in one directory.
            workspace = root / "cwd" / ".agentic_cli" if layout == "same" else root / "workspace"
            settings = BaseSettings(workspace_dir=workspace, knowledge_base_use_mock=True)
            project_dir = root / "cwd" / f".{settings.app_name}" / "knowledge_base"
            project = KnowledgeBaseManager(project_dir, use_mock=True)
            if layout == "same":
                assert settings.knowledge_base_dir.resolve() == project_dir.resolve()
                user = project
            else:
                user = KnowledgeBaseManager(settings.knowledge_base_dir, use_mock=True)
            project.ingest_document(
                content="Basalt lava cools into columns.", title="Volcano field notes",
                source_type=SourceType.USER, metadata={"authors": ["Ada Lovelace"]},
            )
            project.ingest_document(
                content="Columns of basalt.", title="Basalt columns", source_type=SourceType.WEB,
            )
            user.ingest_document(
                content="The glacier retreated.", title="Glacier survey",
                source_type=SourceType.ARXIV, metadata={"authors": ["Mary Somerville"]},
            )
            workflow = _workflow(settings) if with_workflow else None
            app = _App(settings=settings, session=_Session(), workflow=workflow)
            await PapersCommand().execute(PAPERS_CALLS[name], app)
            return app.session.out
        finally:
            os.chdir(old_cwd)


class _App:
    def __init__(self, settings, session, workflow) -> None:
        self.settings = settings
        self.session = session
        self._workflow = workflow

    @property
    def workflow(self):
        if self._workflow is None:
            raise RuntimeError("workflow not ready")
        return self._workflow


def _workflow(settings):
    """A workflow that built its knowledge-base services from ``settings``."""
    from agentic_cli.workflow.base_manager import BaseWorkflowManager

    class _Manager(BaseWorkflowManager):
        @property
        def backend_type(self) -> str:
            return "test"

        async def _do_initialize(self) -> None:
            pass

        async def process(self, message, user_id, session_id=None):
            if False:
                yield

        async def reinitialize(self, model=None, preserve_sessions=True) -> None:
            pass

        async def cleanup(self) -> None:
            pass

        def _get_state_tools(self) -> list:
            return []

    manager = _Manager(agent_configs=[], settings=settings)
    manager._required_managers = {"knowledge_base"}
    manager._ensure_managers_initialized()
    return manager


# ---------------------------------------------------------------------------
# Normalizing and the golden file
# ---------------------------------------------------------------------------

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_TIMINGS = {"query_embedding_time_ms", "search_time_ms", "total_time_ms"}


def _without_timings(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("<ms>" if k in _TIMINGS else _without_timings(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_without_timings(v) for v in value]
    return value


def _normalize(value: Any, root: Path) -> Any:
    text = json.dumps(_without_timings(value), sort_keys=True, default=str)
    for prefix in sorted({str(root.resolve()), str(root)}, key=len, reverse=True):
        text = text.replace(json.dumps(prefix)[1:-1], "<tmp>")
    ids: dict[str, str] = {}
    text = _UUID.sub(lambda m: ids.setdefault(m.group(0), f"<id{len(ids) + 1}>"), text)
    text = _TIME.sub("<time>", text)
    return json.loads(text)


def _cases() -> list[tuple[str, str]]:
    return [(name, layout) for name in SCENARIOS for layout in LAYOUTS]


async def record() -> dict[str, Any]:
    golden: dict[str, Any] = {}
    for name, layout in _cases():
        golden[f"{name}/{layout}"] = await run_scenario(name, layout, "registry")
    for tool in NO_KB_CALLS:
        golden[f"no_knowledge_base/{tool}"] = await run_without_knowledge_base(tool)
    for name in PAPERS_CALLS:
        for layout in ("two", "same"):
            for with_workflow in (False, True):
                key = f"{name}/{layout}/{'workflow' if with_workflow else 'no_workflow'}"
                golden[key] = await run_papers(name, layout, with_workflow)
    return golden


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(("name", "layout"), _cases())
async def test_the_tool_returns_what_0_6_3_returned(name, layout, entry):
    assert await run_scenario(name, layout, entry) == _golden()[f"{name}/{layout}"]


@pytest.mark.parametrize("tool", list(NO_KB_CALLS))
async def test_without_a_knowledge_base_the_error_is_unchanged(tool):
    assert await run_without_knowledge_base(tool) == _golden()[f"no_knowledge_base/{tool}"]


@pytest.mark.parametrize("with_workflow", [False, True], ids=["no_workflow", "workflow"])
@pytest.mark.parametrize("layout", ["two", "same"])
@pytest.mark.parametrize("name", list(PAPERS_CALLS))
async def test_papers_lists_what_0_6_3_listed(name, layout, with_workflow):
    key = f"{name}/{layout}/{'workflow' if with_workflow else 'no_workflow'}"
    assert await run_papers(name, layout, with_workflow) == _golden()[key]


@pytest.mark.parametrize("entry", ENTRIES)
def test_each_tool_keeps_its_call_style(entry):
    assert call_styles(tools_of(entry)) == {**KB_TOOLS, "ingest_arxiv_paper": "async"}


@pytest.mark.parametrize("entry", ENTRIES)
def test_each_tool_keeps_its_arguments(entry):
    assert arguments(tools_of(entry)) == _golden()["arguments"]


if __name__ == "__main__":
    golden = asyncio.run(record())
    golden["arguments"] = arguments(tools_of("registry"))
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(golden)} entries to {GOLDEN}")
