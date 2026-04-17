"""Embedding service for the knowledge base.

Provides text embedding generation using sentence-transformers
and document chunking for efficient retrieval.
"""

from __future__ import annotations

import platform
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def resolve_embedding_device(preference: str = "auto") -> str:
    """Resolve the device string for sentence-transformers.

    PyTorch's own autodetect trusts ``torch.backends.mps.is_available()``,
    which returns True on Intel Macs with discrete AMD GPUs where MPS
    command buffers routinely fail with ``Internal Error (e00002bd)``.
    This helper restricts MPS to Apple Silicon (``arm64``) where it's
    stable and falls back to CPU on Intel Macs.

    Args:
        preference: One of ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``.
            ``"auto"`` picks the best stable device; explicit values are
            honored as-is (no fallback).

    Returns:
        Resolved device string to pass to ``SentenceTransformer(..., device=...)``.
    """
    if preference != "auto":
        return preference

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if platform.machine() == "arm64" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingService:
    """Generates embeddings for document chunks.

    Uses sentence-transformers models for semantic embedding generation.
    Supports lazy loading to avoid loading the model until needed.
    """

    @staticmethod
    def is_available() -> bool:
        """Check whether sentence-transformers can be imported."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Recommended models:
                - "all-MiniLM-L6-v2" (fast, 384 dims, good quality)
                - "all-mpnet-base-v2" (slower, 768 dims, best quality)
            batch_size: Batch size for embedding generation.
            device: Device preference — ``"auto"`` (default) resolves via
                :func:`resolve_embedding_device`. Explicit values
                ``"cpu"``, ``"mps"``, ``"cuda"`` are passed through.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = resolve_embedding_device(device)
        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Get the sentence transformer model (lazy loading)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Force model loading to get dimension
            _ = self.model
        return self._embedding_dim  # type: ignore[return-value]

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def chunk_document(
        self,
        content: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Split content into chunks, respecting structure.

        Splits on markdown structure (headings, code fences) first,
        then splits prose sections into sentences. Code blocks are
        never split.

        Args:
            content: The text content to chunk.
            chunk_size: Target size per chunk in characters.
            overlap: Overlap between adjacent prose chunks in characters.

        Returns:
            List of text chunks.
        """
        if not content or not content.strip():
            return []

        blocks = self._split_structural_blocks(content)

        chunks = []
        for block_type, block_text in blocks:
            if block_type == "code":
                chunks.append(block_text.strip())
            else:
                sentences = self._split_sentences(block_text)
                prose_chunks = self._merge_sentences(sentences, chunk_size, overlap)
                chunks.extend(prose_chunks)

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _split_structural_blocks(content: str) -> list[tuple[str, str]]:
        """Split content into structural blocks: code vs prose.

        Returns list of (type, text) tuples where type is 'code' or 'prose'.
        """
        blocks: list[tuple[str, str]] = []
        lines = content.split("\n")
        current_lines: list[str] = []
        in_code_fence = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") and not in_code_fence:
                # Start of code block — flush prose
                if current_lines:
                    blocks.append(("prose", "\n".join(current_lines)))
                    current_lines = []
                in_code_fence = True
                current_lines.append(line)
            elif stripped.startswith("```") and in_code_fence:
                # End of code block
                current_lines.append(line)
                blocks.append(("code", "\n".join(current_lines)))
                current_lines = []
                in_code_fence = False
            elif not in_code_fence and re.match(r"^#{1,6}\s", stripped):
                # Markdown heading — natural boundary
                if current_lines:
                    blocks.append(("prose", "\n".join(current_lines)))
                    current_lines = []
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            block_type = "code" if in_code_fence else "prose"
            blocks.append((block_type, "\n".join(current_lines)))

        return blocks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Uses nltk.sent_tokenize if available, falls back to regex.
        """
        if not text.strip():
            return []
        try:
            import nltk
            return nltk.sent_tokenize(text)
        except ImportError:
            pass
        # Regex fallback
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _merge_sentences(
        sentences: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Merge sentences into chunks respecting size and overlap."""
        if not sentences:
            return []
        chunks = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if current_len + len(sentence) > chunk_size and current:
                chunks.append(" ".join(current))
                # Keep overlap from end of current chunk
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sentences
                current_len = overlap_len
            current.append(sentence)
            current_len += len(sentence)

        if current:
            chunks.append(" ".join(current))

        return chunks
