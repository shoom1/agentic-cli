"""Knowledge-base search returns what matches, and keyword matching ignores punctuation.

Without the ``kb`` extra (or with ``knowledge_base_use_mock``) the knowledge
base embeds text with MockEmbeddingService, whose vectors are hashes of the
text: they carry no meaning. Search fused those "semantic" hits with the
keyword hits, so every result list was padded with arbitrary documents. And
every keyword index split text on whitespace only, so "orchestras." never
matched the query "orchestras".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_cli.memory import MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, MockVectorStore, SourceType
from agentic_cli.memory.kb._bm25_backends import BM25sIndex, RankBM25Index
from agentic_cli.memory.kb._mock_bm25 import MockBM25Index

TOPICS = {
    "Otters": "Otters are semiaquatic mammals that eat fish and live near rivers.",
    "Volcanoes": "Volcanoes erupt magma, ash and gas from beneath the earth's crust.",
    "Violins": "Violins are string instruments played with a bow in orchestras.",
    "Glaciers": "Glaciers are slow rivers of ice that carve valleys over centuries.",
    "Bees": "Bees pollinate flowers and produce honey in hives.",
    "Comets": "Comets are icy bodies that grow tails near the sun.",
    "Sourdough": "Sourdough bread rises with wild yeast and lactic bacteria.",
    "Chess": "Chess is a strategy board game played on sixty-four squares.",
}


def _mock_kb(base_dir: Path) -> KnowledgeBaseManager:
    kb = KnowledgeBaseManager(base_dir, use_mock=True)
    for title, text in TOPICS.items():
        kb.ingest_document(content=text, title=title, source_type=SourceType.USER)
    return kb


def _titles(kb: KnowledgeBaseManager, query: str) -> list[str]:
    return [hit["document_title"] for hit in kb.search(query, top_k=5)["results"]]


@pytest.mark.parametrize("query, title", [("volcanoes", "Volcanoes"), ("honey", "Bees")])
def test_search_without_the_kb_extra_returns_only_matching_documents(tmp_path, query, title):
    kb = _mock_kb(tmp_path / "kb")

    assert _titles(kb, query) == [title]


def test_injected_hash_embeddings_are_not_searched_either(tmp_path):
    kb = KnowledgeBaseManager(
        tmp_path / "kb",
        embedding_service=MockEmbeddingService(),
        vector_store=MockVectorStore(index_path=tmp_path / "kb" / "embeddings" / "index.mock"),
    )
    for title, text in TOPICS.items():
        kb.ingest_document(content=text, title=title, source_type=SourceType.USER)

    assert _titles(kb, "volcanoes") == ["Volcanoes"]


class _ConceptEmbedder:
    """An embedder whose vectors mean something: one dimension per concept."""

    model_name = "concept-embedder"
    batch_size = 8
    embedding_dim = 2
    _CONCEPTS = (("lava", "magma", "volcan"), ("insect", "bee", "pollinat"))

    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if any(word in lowered for word in words) else 0.0
            for words in self._CONCEPTS
        ]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def chunk_document(self, content: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
        return [content]


def test_a_meaningful_embedder_still_finds_documents_without_a_keyword_match(tmp_path):
    kb = KnowledgeBaseManager(
        tmp_path / "kb",
        embedding_service=_ConceptEmbedder(),
        vector_store=MockVectorStore(index_path=tmp_path / "kb" / "embeddings" / "index.mock", embedding_dim=2),
    )
    kb.ingest_document(content="Hot lava flows downhill.", title="Lava", source_type=SourceType.USER)
    kb.ingest_document(content="A busy insect visits flowers.", title="Insect", source_type=SourceType.USER)

    # No word of "magma" appears in either document: only the embedding links it to "lava".
    assert _titles(kb, "magma")[0] == "Lava"


@pytest.mark.parametrize("reopen", [False, True])
def test_punctuation_next_to_a_word_does_not_stop_a_match(tmp_path, reopen):
    kb = _mock_kb(tmp_path / "kb")
    if reopen:
        kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    assert _titles(kb, "orchestras") == ["Violins"]
    assert _titles(kb, "squares") == ["Chess"]
    assert _titles(kb, "Sixty-Four") == ["Chess"]


def test_an_index_saved_by_the_old_tokenizer_is_rebuilt_on_open(tmp_path):
    _mock_kb(tmp_path / "kb")
    index_file = tmp_path / "kb" / "embeddings" / "bm25_index.json"
    saved = json.loads(index_file.read_text())
    # What 0.6.2 wrote: whitespace-split tokens and no tokenizer name.
    chunk_texts = {
        chunk_id: chunk.content
        for chunk_id, chunk in KnowledgeBaseManager(tmp_path / "kb", use_mock=True)._chunks.items()
    }
    index_file.write_text(json.dumps({
        "chunk_ids": saved["chunk_ids"],
        "documents": [chunk_texts[cid].lower().split() for cid in saved["chunk_ids"]],
    }))

    reopened = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)

    assert _titles(reopened, "orchestras") == ["Violins"]
    assert "tokenizer" in json.loads(index_file.read_text())


@pytest.mark.parametrize("index_cls", [MockBM25Index, RankBM25Index, BM25sIndex])
def test_every_keyword_index_splits_words_the_same_way(tmp_path, index_cls):
    index = index_cls()
    index.add_documents(["c1"], ["Played with a bow, in ORCHESTRAS."])
    index.save(tmp_path)

    reloaded = index_cls()
    reloaded.load(tmp_path)

    assert reloaded.size == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    tokens = saved.get("documents", saved.get("tokenized"))
    assert tokens == [["played", "with", "a", "bow", "in", "orchestras"]]


@pytest.mark.parametrize(
    "index_cls, file_name, key",
    [
        (MockBM25Index, "bm25_index.json", "documents"),
        (RankBM25Index, "bm25_rank.json", "tokenized"),
        (BM25sIndex, "bm25s_sidecar.json", "tokenized"),
    ],
)
def test_an_index_saved_by_the_old_tokenizer_loads_empty(tmp_path, index_cls, file_name, key):
    (tmp_path / file_name).write_text(json.dumps({"chunk_ids": ["c1"], key: [["orchestras."]]}))

    index = index_cls()
    index.load(tmp_path)

    assert index.size == 0
