"""Knowledge-base search returns what matches, and keyword matching ignores punctuation.

Without the ``kb`` extra (or with ``knowledge_base_use_mock``) the knowledge
base embeds text with MockEmbeddingService, whose vectors are hashes of the
text: they carry no meaning. Search fused those "semantic" hits with the
keyword hits, so every result list was padded with arbitrary documents. And
every keyword index split text on whitespace only, so "orchestras." never
matched the query "orchestras".
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from agentic_cli.memory import MockEmbeddingService
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType
from agentic_cli.memory.kb._tokenize import tokenize
from tests.memory.kb_storage import raw_write

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
    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=MockEmbeddingService())
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
    kb = KnowledgeBaseManager(tmp_path / "kb", embedding_service=_ConceptEmbedder())
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


def _nfc(words: list[str]) -> list[str]:
    return [unicodedata.normalize("NFC", word).casefold() for word in words]


@pytest.mark.parametrize(
    "text, words",
    [
        ("मुझे किताब पसंद है", ["मुझे", "किताब", "पसंद", "है"]),  # Devanagari vowel signs
        ("தமிழ் மொழி", ["தமிழ்", "மொழி"]),  # Tamil virama
        ("مُحَمَّد", ["مُحَمَّد"]),  # Arabic harakat
        ("می‌خواهم", ["می‌خواهم"]),  # Persian zero-width non-joiner
    ],
)
def test_words_with_combining_marks_stay_whole(text, words):
    """Whitespace splitting kept these words whole; splitting at \\w must too."""
    assert tokenize(text) == _nfc(words)


def test_composed_and_decomposed_accents_are_the_same_word():
    composed = "Résumé"
    decomposed = unicodedata.normalize("NFD", composed)

    assert tokenize(decomposed) == tokenize(composed) == ["résumé"]


def test_a_word_with_combining_marks_does_not_match_its_fragments(tmp_path):
    kb = KnowledgeBaseManager(tmp_path / "kb", use_mock=True)
    kb.ingest_document(content="मुझे किताब पसंद है", title="Book", source_type=SourceType.USER)
    kb.ingest_document(content="सब ठीक है", title="Fine", source_type=SourceType.USER)

    assert _titles(kb, "किताब") == ["Book"]


def test_a_tokenizer_change_recomputes_every_chunk_s_terms(tmp_path, monkeypatch):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)
    kb.ingest_document(content="水獭住在河边。", title="otters", source_type=SourceType.USER)
    raw_write(tmp_path, "UPDATE chunks SET terms = 'stale'", "UPDATE info SET value = 'words-v2' WHERE key = 'tokenizer'",
              "INSERT INTO chunk_terms (chunk_terms) VALUES ('rebuild')")

    reopened = KnowledgeBaseManager(tmp_path, use_mock=True)

    assert [h["document_title"] for h in reopened.search("水獭")["results"]] == ["otters"]
