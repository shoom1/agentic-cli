"""Chunks stay within their size in every script (review K6, memory-kb F7).

A sentence ended only at ., ! or ? followed by a space and a capital A-Z, and
nothing cut a sentence that ran long, so a Russian, Greek or Chinese text, or
any text without such sentence ends, became one chunk: the embedding model
read only its first ~1,000 characters, and every keyword hit returned the
whole document.
"""

from __future__ import annotations

import random
import sys
import types

import pytest

from agentic_cli.memory import EmbeddingService, MockEmbeddingService
from agentic_cli.memory._core.chunking import _cut_lines, _merge, chunk_text
from agentic_cli.memory.kb import KnowledgeBaseManager, SourceType

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


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-5, 0), (0, 50), (512, -1)],
)
def test_a_chunk_size_below_one_or_a_negative_overlap_is_refused_at_once(chunk_size, overlap):
    # Zero cut nothing off a sentence and looped forever; "" is refused too.
    services = (
        MockEmbeddingService(),
        EmbeddingService.__new__(EmbeddingService),  # chunking loads no model
    )
    for text in ("Otters swim. Herons wade.", ""):
        with pytest.raises(ValueError):
            chunk_text(text, chunk_size, overlap)
        for service in services:
            with pytest.raises(ValueError):
                service.chunk_document(text, chunk_size, overlap)


def test_ingesting_with_a_chunk_size_below_one_is_refused(tmp_path):
    kb = KnowledgeBaseManager(tmp_path, use_mock=True)

    with pytest.raises(ValueError):
        kb.ingest_document(content="Otters swim.", title="otters", source_type=SourceType.USER, chunk_size=0)


# (first sentence, second sentence, the mark that ends both)
ENDS_BEFORE_A_SPACE = {
    "devanagari danda": ("भारत एक देश है", "यहाँ कई भाषाएँ हैं", "\u0964"),
    "devanagari double danda": ("राम वन गए", "सीता उनके साथ गईं", "\u0965"),
    "arabic question mark": ("هل تسبح الأنهار", "نعم تسبح الأنهار", "\u061f"),
    "urdu full stop": ("یہ ایک جملہ ہے", "یہ دوسرا جملہ ہے", "\u06d4"),
    "armenian full stop": ("Ջրասամույրները լողում են", "Ձկները ապրում են գետում", "\u0589"),
    "ethiopic full stop": ("ውሃ ጥሩ ነው", "ወንዙ ሰፊ ነው", "\u1362"),
    "burmese sentence mark": ("ရေခွေးများ ရေကူးကြသည်", "ငါးများ မြစ်တွင် နေကြသည်", "\u104b"),
}
ENDS_WITHOUT_A_SPACE = {
    "fullwidth full stop": ("本研究では新しい手法を提案する", "実験の結果精度が向上した", "\uff0e"),
    "halfwidth ideographic full stop": ("ｶﾜｳｿｶﾞｽﾑ", "ｻｶﾅｶﾞｵﾖｸﾞ", "\uff61"),
}


@pytest.mark.parametrize("script", ENDS_BEFORE_A_SPACE)
def test_a_sentence_ends_at_its_own_scripts_mark_before_a_space(script):
    first, second, mark = ENDS_BEFORE_A_SPACE[script]

    # One character short of the whole text, so that it must be cut somewhere.
    result = chunks(f"{first}{mark} {second}{mark}", chunk_size=len(first) + len(second) + 2, overlap=0)

    assert result == [first + mark, second + mark]


@pytest.mark.parametrize("script", ENDS_WITHOUT_A_SPACE)
def test_a_sentence_ends_at_a_fullwidth_stop_needing_no_space(script):
    first, second, mark = ENDS_WITHOUT_A_SPACE[script]

    result = chunks(f"{first}{mark}{second}{mark}", chunk_size=len(first) + len(second) + 1, overlap=0)

    assert result == [first + mark, second + mark]


def test_hindi_and_japanese_text_is_cut_at_sentence_ends():
    hindi = chunks("भारत एक विशाल देश है। यहाँ कई भाषाएँ बोली जाती हैं। " * 20, chunk_size=100, overlap=0)
    japanese = chunks("本研究では，新しい手法を提案する．実験の結果，精度が向上した．" * 30, chunk_size=100, overlap=0)

    assert len(hindi) > 5
    assert all(chunk.endswith("।") for chunk in hindi)
    assert len(japanese) > 5
    assert all(chunk.endswith("．") for chunk in japanese)


def _plain_merge(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """What ``_merge`` computes, by joining the chunk again for every sentence."""
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > chunk_size:
            chunks.append(" ".join(current))
            kept: list[str] = []
            for previous in reversed(current):
                if len(" ".join([previous, *kept])) + 1 > overlap:
                    break
                kept.insert(0, previous)
            current = kept
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _plain_cut_lines(lines: list[str], limit: int) -> list[str]:
    """What ``_cut_lines`` computes for lines no longer than ``limit``."""
    pieces: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and len("\n".join([*current, line])) > limit:
            pieces.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(current))
    return [piece.strip("\n") for piece in pieces if piece.strip()]


@pytest.mark.parametrize("seed", range(8))
def test_merging_sentences_gives_what_joining_them_again_gives(seed):
    rng = random.Random(seed)
    chunk_size, overlap = rng.choice([(40, 0), (40, 10), (60, 25), (30, 100), (200, 50), (12, 5)])
    sentences = ["x" * rng.randint(1, chunk_size) for _ in range(300)]

    assert _merge(sentences, chunk_size, overlap) == _plain_merge(sentences, chunk_size, overlap)


@pytest.mark.parametrize("seed", range(8))
def test_cutting_code_between_lines_gives_what_joining_it_again_gives(seed):
    rng = random.Random(seed)
    limit = rng.choice([10, 40, 120])
    lines = ["y" * rng.randint(0, limit) for _ in range(300)]

    assert _cut_lines("\n".join(lines), limit) == _plain_cut_lines(lines, limit)
