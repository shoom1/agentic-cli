"""Embedding service for the knowledge base.

Provides text embedding generation using sentence-transformers
and document chunking for efficient retrieval.
"""

from __future__ import annotations

import platform
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_cli.memory._core.chunking import chunk_text

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


@dataclass(frozen=True)
class EmbeddingConfig:
    """Which sentence-transformers model to load, and how."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "auto"


class EmbeddingService:
    """Generates embeddings for document chunks.

    Uses sentence-transformers models for semantic embedding generation.
    Supports lazy loading to avoid loading the model until needed.

    A model that fails to load is not tried again: every later use raises
    that error at once. One service is shared by the knowledge base and the
    memory store, and each probes it; an offline first run would otherwise
    wait on the network for each of them.
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
        self._load_error: Exception | None = None
        # One service is shared by both knowledge-base scopes and the memory
        # store, and called from worker threads; a SentenceTransformer is not
        # safe to call from two threads at once. Reentrant: encoding loads
        # the model on first use.
        self._lock = threading.RLock()

    @property
    def model(self) -> SentenceTransformer:
        """Get the sentence transformer model (loaded once, on first use).

        Raises:
            Exception: the model did not load, now or on an earlier use.
        """
        with self._lock:
            if self._model is None:
                if self._load_error is not None:
                    raise self._load_error.with_traceback(None)
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.model_name, device=self.device)
                    dimension = model.get_sentence_embedding_dimension()
                except Exception as e:
                    self._load_error = e
                    raise
                self._model, self._embedding_dim = model, dimension
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
        with self._lock:
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

        with self._lock:
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
        """``content`` in chunks of at most ``chunk_size + overlap`` characters
        (``agentic_cli.memory._core.chunking``). Loads no model."""
        return chunk_text(content, chunk_size, overlap)
