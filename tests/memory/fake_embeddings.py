"""An embedding service for tests: real to the knowledge base, with no model."""

from __future__ import annotations

import hashlib

from agentic_cli.memory import MockEmbeddingService


class FakeEmbeddingService:
    """A deterministic embedding service with a fixed dimension.

    The knowledge base treats it as a real service (it is not a
    ``MockEmbeddingService``): it builds a FAISS index for it and searches
    semantically. ``calls`` counts the texts embedded. With ``fail_after``,
    ``embed_batch`` raises once embedding a batch would pass that many texts.
    """

    def __init__(
        self, model_name: str = "fake-model", embedding_dim: int = 8, *, fail_after: int | None = None
    ) -> None:
        self.model_name = model_name
        self.batch_size = 32
        self.calls = 0
        self.fail_after = fail_after
        self._dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        while len(digest) < self._dim:
            digest += hashlib.sha256(digest).digest()
        return [byte / 255 - 0.5 for byte in digest[: self._dim]]

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        return self.vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.fail_after is not None and self.calls + len(texts) > self.fail_after:
            raise RuntimeError("the embedding model failed")
        self.calls += len(texts)
        return [self.vector(text) for text in texts]

    def chunk_document(self, content: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        return MockEmbeddingService().chunk_document(content, chunk_size, overlap)
