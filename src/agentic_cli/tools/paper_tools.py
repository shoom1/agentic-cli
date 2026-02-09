"""Paper management tools for agentic workflows.

Provides tools for saving, listing, and opening research papers:
- save_paper: Download/copy PDF and store with metadata
- list_papers: List saved papers with optional filters
- get_paper_info: Get detailed paper metadata
- open_paper: Open PDF in system viewer

The PaperStore is auto-created by the workflow manager when
these tools are used (via @requires("paper_store")).

Example:
    from agentic_cli.tools import paper_tools

    AgentConfig(
        tools=[
            paper_tools.save_paper,
            paper_tools.list_papers,
            paper_tools.get_paper_info,
            paper_tools.open_paper,
        ],
    )
"""

import json
import platform
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agentic_cli.config import BaseSettings
from agentic_cli.logging import get_logger
from agentic_cli.persistence._utils import atomic_write_json
from agentic_cli.tools import requires, require_context
from agentic_cli.tools.registry import (
    register_tool,
    ToolCategory,
    PermissionLevel,
)
from agentic_cli.workflow.context import get_context_paper_store

logger = get_logger("agentic_cli.tools.papers")


# ---------------------------------------------------------------------------
# PaperSourceType enum
# ---------------------------------------------------------------------------


class PaperSourceType(str, Enum):
    """Source type for a saved paper."""

    ARXIV = "arxiv"
    WEB = "web"
    LOCAL = "local"


# ---------------------------------------------------------------------------
# PaperMetadata dataclass
# ---------------------------------------------------------------------------


@dataclass
class PaperMetadata:
    """Metadata for a saved paper."""

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    source_type: str = PaperSourceType.LOCAL
    source_url: str = ""
    pdf_url: str = ""
    arxiv_id: str = ""
    file_path: str = ""  # relative within papers/pdfs/
    added_at: str = ""
    file_size_bytes: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id,
            "file_path": self.file_path,
            "added_at": self.added_at,
            "file_size_bytes": self.file_size_bytes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperMetadata":
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            authors=data.get("authors", []),
            abstract=data.get("abstract", ""),
            source_type=data.get("source_type", PaperSourceType.LOCAL),
            source_url=data.get("source_url", ""),
            pdf_url=data.get("pdf_url", ""),
            arxiv_id=data.get("arxiv_id", ""),
            file_path=data.get("file_path", ""),
            added_at=data.get("added_at", ""),
            file_size_bytes=data.get("file_size_bytes", 0),
            tags=data.get("tags", []),
        )


# ---------------------------------------------------------------------------
# PaperStore – persistent file-backed store
# ---------------------------------------------------------------------------


class PaperStore:
    """Persistent paper store with metadata index and PDF files.

    Storage layout:
        {workspace_dir}/papers/pdfs/       - PDF files
        {workspace_dir}/papers/papers_index.json - metadata index

    Index format: {"papers": {"paper_id": {...}}}

    Example:
        >>> store = PaperStore(settings)
        >>> paper_id = store.add(title="Attention Is All You Need", ...)
        >>> info = store.get(paper_id)
        >>> papers = store.list_papers(source_type="arxiv")
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._settings = settings
        self._papers_dir = settings.workspace_dir / "papers"
        self._pdfs_dir = self._papers_dir / "pdfs"
        self._index_path = self._papers_dir / "papers_index.json"
        self._papers: dict[str, PaperMetadata] = {}
        self._load()

    @property
    def papers_dir(self) -> Path:
        """Base papers directory."""
        return self._papers_dir

    @property
    def pdfs_dir(self) -> Path:
        """Directory for PDF files."""
        return self._pdfs_dir

    def _load(self) -> None:
        """Load paper index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                for paper_id, paper_data in data.get("papers", {}).items():
                    self._papers[paper_id] = PaperMetadata.from_dict(paper_data)
            except (json.JSONDecodeError, KeyError):
                logger.warning("paper_index_load_failed", path=str(self._index_path))
                self._papers = {}

    def _save(self) -> None:
        """Save paper index to disk atomically."""
        self._papers_dir.mkdir(parents=True, exist_ok=True)
        data = {"papers": {pid: p.to_dict() for pid, p in self._papers.items()}}
        atomic_write_json(self._index_path, data)

    def add(self, metadata: PaperMetadata) -> str:
        """Add a paper to the store.

        Args:
            metadata: Paper metadata (id should be pre-assigned).

        Returns:
            The paper ID.
        """
        self._papers[metadata.id] = metadata
        self._save()
        return metadata.id

    def get(self, paper_id: str) -> PaperMetadata | None:
        """Get paper metadata by ID."""
        return self._papers.get(paper_id)

    def find(self, id_or_title: str) -> PaperMetadata | None:
        """Find a paper by ID (prefix) or title (substring, case-insensitive).

        Args:
            id_or_title: Paper ID (or prefix) or title substring.

        Returns:
            Matching PaperMetadata, or None if not found.
        """
        # Try exact ID match first
        if id_or_title in self._papers:
            return self._papers[id_or_title]

        # Try ID prefix match
        for pid, paper in self._papers.items():
            if pid.startswith(id_or_title):
                return paper

        # Try title substring match (case-insensitive)
        query_lower = id_or_title.lower()
        for paper in self._papers.values():
            if query_lower in paper.title.lower():
                return paper

        return None

    def list_papers(
        self,
        query: str | None = None,
        source_type: str | None = None,
    ) -> list[PaperMetadata]:
        """List papers with optional filters.

        Args:
            query: Filter by title substring (case-insensitive).
            source_type: Filter by source type (arxiv, web, local).

        Returns:
            List of matching PaperMetadata objects.
        """
        results = list(self._papers.values())

        if source_type:
            results = [p for p in results if p.source_type == source_type]

        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.title.lower()
                or any(query_lower in a.lower() for a in p.authors)
            ]

        return results

    def get_pdf_path(self, paper_id: str) -> Path | None:
        """Get the absolute PDF file path for a paper.

        Args:
            paper_id: The paper ID.

        Returns:
            Absolute Path to the PDF file, or None if paper not found.
        """
        paper = self._papers.get(paper_id)
        if paper and paper.file_path:
            return self._pdfs_dir / paper.file_path
        return None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _detect_source_type(url_or_path: str) -> PaperSourceType:
    """Detect source type from URL or file path."""
    if "arxiv.org" in url_or_path:
        return PaperSourceType.ARXIV
    if url_or_path.startswith(("http://", "https://")):
        return PaperSourceType.WEB
    return PaperSourceType.LOCAL


def _extract_arxiv_id(url: str) -> str:
    """Extract arXiv ID from a URL."""
    match = re.search(r"(\d{4}\.\d{4,5})", url)
    return match.group(1) if match else ""


def _ensure_arxiv_pdf_url(url: str) -> str:
    """Convert arXiv URL to PDF URL if needed."""
    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return url


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.CAUTION,
    description="Save a research paper (PDF) from URL or local file. For ArXiv papers, metadata (title, authors, abstract) is fetched automatically. Use this to build a local paper library.",
)
@requires("paper_store")
@require_context("Paper store", get_context_paper_store)
async def save_paper(
    url_or_path: str,
    title: str = "",
    authors: list[str] | None = None,
    abstract: str = "",
    source_type: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Save a research paper from URL or local file.

    Downloads the PDF and stores it with metadata. For ArXiv papers,
    metadata is auto-populated if not provided.

    Args:
        url_or_path: URL to download PDF from, or local file path.
        title: Paper title (auto-fetched for ArXiv if empty).
        authors: List of author names (auto-fetched for ArXiv if empty).
        abstract: Paper abstract (auto-fetched for ArXiv if empty).
        source_type: Override source type (arxiv, web, local). Auto-detected if empty.
        tags: Optional tags for categorization.

    Returns:
        A dict with the saved paper info.
    """
    store: PaperStore = get_context_paper_store()

    # Detect source type
    detected_type = PaperSourceType(source_type) if source_type else _detect_source_type(url_or_path)

    # Generate paper ID
    paper_id = str(uuid.uuid4())[:8]

    # For ArXiv: fetch metadata separately, download PDF bytes directly
    arxiv_id = ""
    pdf_url = url_or_path
    arxiv_pdf_bytes: bytes | None = None
    if detected_type == PaperSourceType.ARXIV:
        arxiv_id = _extract_arxiv_id(url_or_path)
        pdf_url = _ensure_arxiv_pdf_url(url_or_path)

        if arxiv_id:
            # Fetch metadata only (no download)
            try:
                from agentic_cli.tools.arxiv_tools import fetch_arxiv_paper
                metadata_result = await fetch_arxiv_paper(arxiv_id)
                if metadata_result.get("success") and "paper" in metadata_result:
                    paper_info = metadata_result["paper"]
                    title = title or paper_info.get("title", "")
                    authors = authors or paper_info.get("authors", [])
                    abstract = abstract or paper_info.get("abstract", "")
            except Exception as e:
                logger.warning("arxiv_metadata_fetch_failed", arxiv_id=arxiv_id, error=str(e))

            # Download raw PDF bytes directly (bypasses LLM context)
            try:
                from agentic_cli.tools.arxiv_tools import _download_arxiv_pdf, _get_arxiv_source
                pdf_result = await _download_arxiv_pdf(arxiv_id, _get_arxiv_source())
                if not pdf_result.get("error"):
                    import base64
                    arxiv_pdf_bytes = base64.b64decode(pdf_result["pdf_bytes"])
            except Exception as e:
                logger.warning("arxiv_pdf_download_failed", arxiv_id=arxiv_id, error=str(e))

    # Ensure PDFs directory exists
    store.pdfs_dir.mkdir(parents=True, exist_ok=True)

    # Download or copy the PDF
    filename = f"{paper_id}.pdf"
    dest_path = store.pdfs_dir / filename

    if detected_type == PaperSourceType.LOCAL:
        # Copy local file
        source_path = Path(url_or_path).expanduser().resolve()
        if not source_path.exists():
            return {"success": False, "error": f"File not found: {url_or_path}"}
        if not source_path.suffix.lower() == ".pdf":
            return {"success": False, "error": f"Not a PDF file: {url_or_path}"}
        try:
            shutil.copy2(str(source_path), str(dest_path))
        except OSError as e:
            return {"success": False, "error": f"Failed to copy file: {e}"}
    elif detected_type == PaperSourceType.ARXIV and arxiv_pdf_bytes is not None:
        # Use PDF bytes from direct _download_arxiv_pdf call
        max_bytes = getattr(store._settings, "webfetch_max_pdf_bytes", 5242880)
        if len(arxiv_pdf_bytes) > max_bytes:
            return {
                "success": False,
                "error": f"PDF too large ({len(arxiv_pdf_bytes)} bytes, max {max_bytes})",
            }
        dest_path.write_bytes(arxiv_pdf_bytes)
    else:
        # Download from URL (non-ArXiv, or ArXiv PDF download failed)
        try:
            import httpx
        except ImportError:
            return {"success": False, "error": "httpx not installed, cannot download PDFs"}

        download_url = pdf_url if detected_type == PaperSourceType.ARXIV else url_or_path

        # Respect ArXiv 3-second rate limit before PDF download
        if detected_type == PaperSourceType.ARXIV:
            try:
                from agentic_cli.tools.arxiv_tools import _get_arxiv_source
                _get_arxiv_source().wait_for_rate_limit()
            except Exception:
                pass  # Don't fail the download if rate limiter unavailable

        max_bytes = getattr(store._settings, "webfetch_max_pdf_bytes", 5242880)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                response = await client.get(download_url)
                response.raise_for_status()

                content = response.content
                if len(content) > max_bytes:
                    return {
                        "success": False,
                        "error": f"PDF too large ({len(content)} bytes, max {max_bytes})",
                    }

                dest_path.write_bytes(content)
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"HTTP error downloading PDF: {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"success": False, "error": f"Failed to download PDF: {e}"}

    # Get file size
    file_size = dest_path.stat().st_size

    # Use filename as fallback title
    if not title:
        title = Path(url_or_path).stem if detected_type == PaperSourceType.LOCAL else url_or_path

    # Create metadata
    metadata = PaperMetadata(
        id=paper_id,
        title=title,
        authors=authors or [],
        abstract=abstract,
        source_type=detected_type,
        source_url=url_or_path,
        pdf_url=pdf_url if detected_type != PaperSourceType.LOCAL else "",
        arxiv_id=arxiv_id,
        file_path=filename,
        added_at=datetime.now().isoformat(),
        file_size_bytes=file_size,
        tags=tags or [],
    )

    store.add(metadata)

    return {
        "success": True,
        "paper_id": paper_id,
        "title": title,
        "file_size_bytes": file_size,
        "message": f"Paper saved: {title}",
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="List saved research papers with optional filters by title/author query or source type (arxiv, web, local).",
)
@requires("paper_store")
@require_context("Paper store", get_context_paper_store)
def list_papers(
    query: str = "",
    source_type: str = "",
) -> dict[str, Any]:
    """List saved papers with optional filters.

    Args:
        query: Filter by title or author substring (case-insensitive).
        source_type: Filter by source type (arxiv, web, local).

    Returns:
        A dict with matching papers.
    """
    store: PaperStore = get_context_paper_store()
    papers = store.list_papers(
        query=query or None,
        source_type=source_type or None,
    )

    items = [
        {
            "id": p.id,
            "title": p.title,
            "authors": p.authors,
            "source_type": p.source_type,
            "arxiv_id": p.arxiv_id,
            "added_at": p.added_at,
            "file_size_bytes": p.file_size_bytes,
            "tags": p.tags,
        }
        for p in papers
    ]

    return {
        "success": True,
        "papers": items,
        "count": len(items),
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.SAFE,
    description="Get detailed metadata for a specific saved paper by ID or title. Returns title, authors, abstract, source info, and file details.",
)
@requires("paper_store")
@require_context("Paper store", get_context_paper_store)
def get_paper_info(
    paper_id_or_title: str,
) -> dict[str, Any]:
    """Get detailed paper metadata.

    Args:
        paper_id_or_title: Paper ID (or prefix) or title substring.

    Returns:
        A dict with paper metadata or error.
    """
    store: PaperStore = get_context_paper_store()
    paper = store.find(paper_id_or_title)

    if paper is None:
        return {
            "success": False,
            "error": f"Paper not found: {paper_id_or_title}",
        }

    return {
        "success": True,
        "paper": paper.to_dict(),
    }


@register_tool(
    category=ToolCategory.KNOWLEDGE,
    permission_level=PermissionLevel.CAUTION,
    description="Open a saved paper's PDF in the system default viewer. Provide a paper ID or title to identify the paper.",
)
@requires("paper_store")
@require_context("Paper store", get_context_paper_store)
def open_paper(
    paper_id_or_title: str,
) -> dict[str, Any]:
    """Open a saved paper's PDF in the system viewer.

    Args:
        paper_id_or_title: Paper ID (or prefix) or title substring.

    Returns:
        A dict with the result.
    """
    store: PaperStore = get_context_paper_store()
    paper = store.find(paper_id_or_title)

    if paper is None:
        return {
            "success": False,
            "error": f"Paper not found: {paper_id_or_title}",
        }

    pdf_path = store.get_pdf_path(paper.id)
    if pdf_path is None or not pdf_path.exists():
        return {
            "success": False,
            "error": f"PDF file not found for paper: {paper.title}",
        }

    # Open in system viewer
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(pdf_path)])
        elif system == "Linux":
            subprocess.Popen(["xdg-open", str(pdf_path)])
        elif system == "Windows":
            subprocess.Popen(["start", "", str(pdf_path)], shell=True)
        else:
            return {
                "success": False,
                "error": f"Unsupported platform: {system}",
            }
    except OSError as e:
        return {"success": False, "error": f"Failed to open PDF: {e}"}

    return {
        "success": True,
        "title": paper.title,
        "pdf_path": str(pdf_path),
        "message": f"Opened: {paper.title}",
    }
