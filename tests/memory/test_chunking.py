"""Chunks stay within their size in every script (review K6, memory-kb F7).

A sentence ended only at ., ! or ? followed by a space and a capital A-Z, and
nothing cut a sentence that ran long, so a Russian, Greek or Chinese text, or
any text without such sentence ends, became one chunk: the embedding model
read only its first ~1,000 characters, and every keyword hit returned the
whole document.
"""

from __future__ import annotations

import sys
import types

from agentic_cli.memory import EmbeddingService, MockEmbeddingService

LIMIT, OVERLAP = 512, 50


def chunks(text: str, chunk_size: int = LIMIT, overlap: int = OVERLAP) -> list[str]:
    return MockEmbeddingService().chunk_document(text, chunk_size, overlap)


def test_russian_text_is_cut_into_chunks_within_the_limit():
    result = chunks("Выдры живут у рек и строят норы в берегах. " * 250)

    assert len(result) > 15
    assert max(map(len, result)) <= LIMIT + OVERLAP


def test_greek_text_is_cut_into_chunks_within_the_limit():
    result = chunks("Οι ερωδιοί περπατούν στα ρηχά νερά την αυγή. " * 200)

    assert len(result) > 10
    assert max(map(len, result)) <= LIMIT + OVERLAP


def test_a_sentence_may_start_with_a_lower_case_letter():
    assert chunks("Otters swim. herons wade.", chunk_size=15, overlap=0) == ["Otters swim.", "herons wade."]


def test_chinese_text_is_cut_at_its_full_stops():
    result = chunks("水獭住在河边的洞穴里。" * 400)

    assert len(result) >= 8
    assert max(map(len, result)) <= LIMIT + OVERLAP
    assert all(chunk.endswith("。") for chunk in result)


def test_text_without_sentence_ends_or_spaces_is_cut_at_the_limit():
    result = chunks("水獭住在河边的洞穴里" * 400)  # 4,000 characters

    assert len(result) == 8
    assert max(map(len, result)) <= LIMIT


def test_a_long_sentence_is_cut_at_whitespace_and_loses_nothing():
    words = [f"word{i}" for i in range(400)]

    result = chunks(" ".join(words), chunk_size=200, overlap=20)

    assert max(map(len, result)) <= 220
    assert " ".join(result).split() == words


def test_a_long_code_block_is_cut_between_lines():
    lines = [f"x_{i} = {i}" for i in range(300)]

    result = chunks("```python\n" + "\n".join(lines) + "\n```", chunk_size=200, overlap=20)

    assert len(result) > 1
    assert max(map(len, result)) <= 200
    kept = [line for chunk in result for line in chunk.split("\n") if line not in ("```python", "```")]
    assert kept == lines


def test_nltk_is_never_used(monkeypatch):
    def sent_tokenize(text):
        raise AssertionError("chunking called nltk")

    monkeypatch.setitem(sys.modules, "nltk", types.SimpleNamespace(sent_tokenize=sent_tokenize))

    assert chunks("Otters nest in burrows. Herons wade at dawn.", chunk_size=25, overlap=0) == [
        "Otters nest in burrows.",
        "Herons wade at dawn.",
    ]


def test_short_text_is_one_chunk_and_blank_text_none():
    assert chunks("Hello world.") == ["Hello world."]
    assert chunks("   \n\t ") == []


def test_both_embedding_services_chunk_alike():
    service = EmbeddingService.__new__(EmbeddingService)  # chunking loads no model
    text = "Выдры живут у рек. " * 60 + "```\ncode = 1\n```\n" + "水獭住在河边。" * 50

    assert service.chunk_document(text, 100, 10) == MockEmbeddingService().chunk_document(text, 100, 10)
