"""Keyword search over memories: BM25, computed when asked (spec §6.5).

A store holds hundreds of short notes, so the terms are counted per call and
nothing is kept on disk. A description's terms count twice: it says what the
memory is about.
"""

from __future__ import annotations

import math
from collections import Counter

from agentic_cli.memory._core.tokenize import tokenize
from agentic_cli.memory.store.models import Memory, newest_first

K1 = 1.2
B = 0.75
LIMIT_MAX = 50


def ranked(memories: list[Memory], query: str, limit: int = 10) -> list[Memory]:
    """``memories`` that match ``query``, best first; the newest for an empty query."""
    limit = max(1, min(LIMIT_MAX, int(limit)))
    if not query.strip():
        return sorted(memories, key=newest_first)[:limit]
    terms = set(tokenize(query))
    if not terms or not memories:
        return []
    documents = [tokenize(m.description) * 2 + tokenize(m.content) for m in memories]
    count = len(documents)
    average = sum(len(d) for d in documents) / count or 1.0
    frequency = Counter(term for document in documents for term in set(document))
    scored: list[tuple[float, Memory]] = []
    for memory, document in zip(memories, documents):
        counts = Counter(document)
        length = len(document)
        score = 0.0
        for term in terms:
            found = counts.get(term, 0)
            if not found:
                continue
            idf = math.log(1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * found * (K1 + 1) / (found + K1 * (1 - B + B * length / average))
        if score > 0:
            scored.append((score, memory))
    scored.sort(key=lambda pair: (-pair[0], *newest_first(pair[1])))
    return [memory for _, memory in scored[:limit]]
