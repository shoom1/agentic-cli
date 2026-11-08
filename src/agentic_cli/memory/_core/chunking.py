"""How text is cut into chunks: what a knowledge base embeds and searches.

:func:`chunk_text` is the one chunker, for the knowledge base and both
embedding services. Markdown headings and code fences split the text into
blocks; prose is split into sentences, which are merged into chunks of up to
``chunk_size`` characters, the last sentences of a chunk (up to ``overlap``
characters) starting the next. A sentence longer than ``chunk_size`` is cut
first, at its last whitespace before the limit or, in text without spaces,
at the limit; a code block is cut between lines. So no chunk is longer than
``chunk_size + overlap``.

A sentence ends at ``.``, ``!``, ``?`` or ``…`` followed by whitespace,
whatever comes next (a Cyrillic or Greek capital, a lower-case letter), and
likewise at the full stops and question marks of Devanagari, Arabic and Urdu,
Armenian, Ethiopic and Burmese; at the Chinese and Japanese full stops
``。！？``, ``．`` and half-width ``｡``, which need no space after them. The
rule is the same on every install: nltk, which chunked differently wherever
it happened to be installed, is not used.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"#{1,6}\s")
# Marks that end a sentence when whitespace follows: Latin, Devanagari danda
# and double danda (U+0964-0965), Urdu full stop (U+06D4), Arabic question mark
# (U+061F), Armenian full stop (U+0589), Ethiopic full stop (U+1362) and Burmese
# sentence mark (U+104B); then those that need no whitespace: the Chinese and
# Japanese ones, full-width full stop (U+FF0E) and half-width ideographic full
# stop (U+FF61).
_SENTENCE_END = re.compile(
    r"(?<=[.!?…\u0964\u0965\u06d4\u061f\u0589\u1362\u104b])\s+"
    r"|(?<=[。！？\uff0e\uff61])\s*"
)


def chunk_text(content: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """``content`` in chunks of at most ``chunk_size + overlap`` characters.

    Raises ``ValueError`` for a ``chunk_size`` under 1 (nothing could be cut
    to that size) or a negative ``overlap``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, not {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, not {overlap}")
    if not content or not content.strip():
        return []
    chunks: list[str] = []
    for kind, block in split_blocks(content):
        if kind == "code":
            chunks.extend(_cut_lines(block.strip(), chunk_size))
        else:
            pieces = [piece for sentence in split_sentences(block) for piece in _cut(sentence, chunk_size)]
            chunks.extend(_merge(pieces, chunk_size, overlap))
    return [chunk for chunk in chunks if chunk.strip()]


def split_blocks(content: str) -> list[tuple[str, str]]:
    """``content`` as ``("prose" | "code", text)`` blocks: a code fence is one
    block, and a markdown heading starts a new prose block."""
    blocks: list[tuple[str, str]] = []
    current: list[str] = []
    in_code = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") and not in_code:
            if current:
                blocks.append(("prose", "\n".join(current)))
                current = []
            in_code = True
            current.append(line)
        elif stripped.startswith("```") and in_code:
            current.append(line)
            blocks.append(("code", "\n".join(current)))
            current = []
            in_code = False
        elif not in_code and _HEADING.match(stripped):
            if current:
                blocks.append(("prose", "\n".join(current)))
                current = []
            current.append(line)
        else:
            current.append(line)
    if current:
        blocks.append(("code" if in_code else "prose", "\n".join(current)))
    return blocks


def split_sentences(text: str) -> list[str]:
    """``text``'s sentences, stripped (the module docstring says where one ends)."""
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


def _cut(text: str, limit: int) -> list[str]:
    """``text`` in pieces of at most ``limit`` characters, each cut at its last
    whitespace before the limit, or at the limit where there is none."""
    pieces: list[str] = []
    while len(text) > limit:
        cut = next((i for i in range(limit, 0, -1) if text[i].isspace()), 0)
        if cut:
            pieces.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
        else:
            pieces.append(text[:limit])
            text = text[limit:]
    if text:
        pieces.append(text)
    return [piece for piece in pieces if piece]


def _cut_lines(block: str, limit: int) -> list[str]:
    """A code block in pieces of at most ``limit`` characters, cut between
    lines; a line longer than that is cut on its own (see :func:`_cut`)."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0  # len("\n".join(current)), kept so a long block is not joined again and again
    for line in block.split("\n"):
        for part in _cut(line, limit) if len(line) > limit else [line]:
            if current and size + 1 + len(part) > limit:
                pieces.append("\n".join(current))
                current = []
            size = size + 1 + len(part) if current else len(part)
            current.append(part)
    if current:
        pieces.append("\n".join(current))
    return [piece.strip("\n") for piece in pieces if piece.strip()]


def _merge(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Sentences joined into chunks of up to ``chunk_size`` characters; each
    chunk after the first starts with the previous one's last sentences, at
    most ``overlap`` characters with the space that follows them."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0  # len(" ".join(current)), kept so a long chunk is not joined again and again
    for sentence in sentences:
        if current and size + 1 + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            start, size = len(current), 0
            while start:
                joined = len(current[start - 1]) + (size + 1 if start < len(current) else 0)
                if joined + 1 > overlap:
                    break
                start, size = start - 1, joined
            current = current[start:]
        size = size + 1 + len(sentence) if current else len(sentence)
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks
