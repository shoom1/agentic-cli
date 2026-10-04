"""How the keyword indexes split text into terms.

Every keyword index (MockBM25Index and the bm25s / rank_bm25 backends) uses
:func:`tokenize`, so a query and the indexed text are split the same way.
Indexes save their tokenized text, so each saved index records ``TOKENIZER``;
one saved with a different tokenizer loads empty, and the knowledge base then
re-indexes its chunks.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from functools import cache

# Bump when tokenize() changes, so indexes saved by the old version are rebuilt.
TOKENIZER = "words-v2"


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


def tokenize(text: str) -> list[str]:
    """Case-folded words: punctuation separates terms and is dropped."""
    return _word().findall(unicodedata.normalize("NFC", text).casefold())
