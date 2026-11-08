"""How the keyword indexes split text into terms.

Every keyword index (MockBM25Index and the bm25s / rank_bm25 backends) uses
:func:`tokenize`, so a query and the indexed text are split the same way.
Indexes save their tokenized text, so each saved index records ``TOKENIZER``;
one saved with a different tokenizer loads empty, and the knowledge base then
re-indexes its chunks. A run of Chinese, Japanese or Korean characters, which
no spaces divide into words, is indexed as overlapping two-character terms.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterator
from functools import cache

# Bump when tokenize() changes, so indexes saved by the old version are rebuilt.
TOKENIZER = "words-v3"


@cache
def _word() -> re.Pattern[str]:
    """Word characters, combining marks and the zero-width (non-)joiners.

    ``\\w`` alone does not match combining marks (Unicode categories Mn, Mc,
    Me), so it would cut words of Devanagari, Tamil, Arabic with vowel marks
    or decomposed Latin accents into fragments. Built on first use.
    """
    marks = "".join(
        chr(code)
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)).startswith("M")
    )
    return re.compile("[\\w" + re.escape(marks) + "\u200c\u200d]+")


# Scripts written without spaces between words, whose runs become overlapping
# two-character terms: Han (with its extensions and compatibility ideographs),
# Hiragana, Katakana (half-width too) and Hangul.
_CJK = re.compile(
    "[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf\u4e00-\u9fff"
    "\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f\U00020000-\U0003134f]+"
)


def _terms(word: str) -> Iterator[str]:
    """``word`` as terms: itself, except that each CJK run in it becomes its
    overlapping two-character terms (a one-character run stays one term)."""
    start = 0
    for run in _CJK.finditer(word):
        if run.start() > start:
            yield word[start : run.start()]
        text = run.group()
        if len(text) == 1:
            yield text
        else:
            yield from (text[i : i + 2] for i in range(len(text) - 1))
        start = run.end()
    if start < len(word):
        yield word[start:]


def tokenize(text: str) -> list[str]:
    """Case-folded words, punctuation dropped; a run of Chinese, Japanese or
    Korean characters as its overlapping two-character terms.

    Those scripts put no spaces between words, so a run is not one word: as
    search engines do (Elasticsearch's cjk analyzer), every two adjacent
    characters are a term, in the indexed text and the query alike. A
    one-character query then matches only a character that stands alone.
    """
    words = _word().findall(unicodedata.normalize("NFC", text).casefold())
    return [term for word in words for term in _terms(word)]
