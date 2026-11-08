"""How the knowledge base splits text into keyword terms.

Each chunk's terms are stored with it (``chunks.terms`` in ``kb.sqlite``),
and the keyword index (FTS5) indexes them, so a query split by
:func:`tokenize` and the indexed text are split the same way. The database
records ``TOKENIZER``; one recorded with a different tokenizer has every
chunk's terms recomputed when it opens. A run of Chinese, Japanese or Korean
characters, which spaces do not divide into words, is indexed as overlapping
two-character terms.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterator
from functools import cache

# Bump when tokenize() changes, so terms stored by the old version are recomputed.
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


# Scripts whose runs become overlapping two-character terms: Han (with its
# iteration and closing marks 々〆〇, the Hangzhou numerals and 〻, the
# extensions and the compatibility ideographs, up to Extension H), Hiragana,
# Katakana (the phonetic extensions and half-width too) and Hangul. Chinese and
# Japanese put no spaces between words; a Korean word carries its particles.
_CJK = re.compile(
    "[\u1100-\u11ff\u3005-\u3007\u3021-\u3029\u3038-\u303b\u3040-\u30ff"
    "\u3130-\u318f\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    "\uf900-\ufaff\uff66-\uff9f\U00020000-\U000323af]+"
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

    Chinese and Japanese put no spaces between words, and a Korean word
    carries its particles (수달이), so a run is not one word: as search
    engines do (Elasticsearch's cjk analyzer), every two adjacent characters
    are a term, in the indexed text and the query alike. A one-character query
    then matches only a character that stands alone. Thai, Lao, Khmer and
    Burmese, written without spaces too, are not split into words.
    """
    words = _word().findall(unicodedata.normalize("NFC", text).casefold())
    return [term for word in words for term in _terms(word)]
