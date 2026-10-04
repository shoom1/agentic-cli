"""How the keyword indexes split text into terms.

Every keyword index (MockBM25Index and the bm25s / rank_bm25 backends) uses
:func:`tokenize`, so a query and the indexed text are split the same way.
Indexes save their tokenized text, so each saved index records ``TOKENIZER``;
one saved with a different tokenizer loads empty, and the knowledge base then
re-indexes its chunks.
"""

from __future__ import annotations

import re

# Bump when tokenize() changes, so indexes saved by the old version are rebuilt.
TOKENIZER = "words-v2"

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase words: punctuation separates terms and is dropped."""
    return _WORD.findall(text.lower())
