"""``kb_read`` reads a document's stored file only from inside the knowledge base.

A document with no text of its own is read from its stored PDF, found through
``file_path`` in ``metadata.json``. The project knowledge base lives in the
project (``./.{app}/knowledge_base``), so a cloned repository can ship that
file, and ``file_path`` was joined to the files directory unchecked: a ``..``
path, a ``files`` directory that is a symlink, or a stored file that is a
symlink put the text of a PDF elsewhere on disk into the model's context, both
through ``kb_read(full=True)`` and through the summary built when a document
has none. The stored file must now be named directly inside the files
directory and resolve to a file inside the knowledge base.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentic_cli.memory._core.mock_embeddings import MockEmbeddingService

from agentic_cli.memory.kb._mock_vector_store import MockVectorStore
from agentic_cli.memory.kb.manager import KnowledgeBaseManager
from agentic_cli.memory.kb.models import SourceType
from agentic_cli.tools.knowledge_tools import kb_read
from agentic_cli.workflow.service_registry import set_service_registry

INSIDE = "inside ledger"
OUTSIDE = "outside ledger"  # only in a PDF outside the knowledge base


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


@pytest.fixture
def outside(tmp_path) -> Path:
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / "ledger.pdf").write_bytes(_pdf_with_text(OUTSIDE))
    return folder


@pytest.fixture
def kb_dir(tmp_path) -> Path:
    return tmp_path / "project" / ".app" / "knowledge_base"


def _open_kb(kb_dir: Path) -> KnowledgeBaseManager:
    return KnowledgeBaseManager(
        base_dir=kb_dir,
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(index_path=kb_dir / "embeddings" / "index.mock"),
    )


def _ingest_pdf(kb_dir: Path) -> str:
    """A document with no text of its own, only a stored PDF."""
    doc = _open_kb(kb_dir).ingest_document(
        content="",
        title="Ledger",
        source_type=SourceType.USER,
        file_bytes=_pdf_with_text(INSIDE),
        file_extension=".pdf",
    )
    return doc.id


def _set_file_path(kb_dir: Path, doc_id: str, file_path: str) -> None:
    """Rewrite metadata.json as a repository could ship it."""
    metadata_path = kb_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entries = metadata["documents"]
    entry = entries[doc_id] if isinstance(entries, dict) else next(
        e for e in entries if e["id"] == doc_id
    )
    entry["file_path"] = file_path
    metadata_path.write_text(json.dumps(metadata))


async def _read(kb_dir: Path, doc_id: str, **kwargs) -> str:
    """What kb_read hands the model, as one string."""
    token = set_service_registry({"kb_manager": _open_kb(kb_dir)})
    try:
        return json.dumps(await kb_read(doc_id, **kwargs))
    finally:
        token.var.reset(token)


async def test_a_stored_pdf_inside_the_knowledge_base_is_read(kb_dir):
    doc_id = _ingest_pdf(kb_dir)

    assert INSIDE in await _read(kb_dir, doc_id, full=True)


async def test_a_dotdot_file_path_is_not_read(kb_dir, outside):
    doc_id = _ingest_pdf(kb_dir)
    _set_file_path(kb_dir, doc_id, "../../../../elsewhere/ledger.pdf")

    assert OUTSIDE not in await _read(kb_dir, doc_id, full=True)


async def test_an_absolute_file_path_is_not_read(kb_dir, outside):
    doc_id = _ingest_pdf(kb_dir)
    _set_file_path(kb_dir, doc_id, str(outside / "ledger.pdf"))

    assert OUTSIDE not in await _read(kb_dir, doc_id, full=True)


async def test_a_files_directory_that_links_elsewhere_is_not_followed(kb_dir, outside):
    doc_id = _ingest_pdf(kb_dir)
    _set_file_path(kb_dir, doc_id, "ledger.pdf")
    shutil.rmtree(kb_dir / "files")
    (kb_dir / "files").symlink_to(outside, target_is_directory=True)

    assert OUTSIDE not in await _read(kb_dir, doc_id, full=True)


async def test_a_stored_file_that_links_elsewhere_is_not_followed(kb_dir, outside):
    doc_id = _ingest_pdf(kb_dir)
    stored = kb_dir / "files" / f"{doc_id}.pdf"
    stored.unlink()
    stored.symlink_to(outside / "ledger.pdf")

    assert OUTSIDE not in await _read(kb_dir, doc_id, full=True)


async def test_the_summary_is_not_built_from_a_file_outside(kb_dir, outside):
    """A document without a summary file gets one built on first read."""
    doc_id = _ingest_pdf(kb_dir)
    _set_file_path(kb_dir, doc_id, "../../../../elsewhere/ledger.pdf")
    (kb_dir / "documents" / f"{doc_id}.md").unlink()

    assert OUTSIDE not in await _read(kb_dir, doc_id)
