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
at the Chinese and Japanese full stops ``。！？``, which need no space after
them. The rule is the same on every install: nltk, which chunked differently
wherever it happened to be installed, is not used.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"#{1,6}\s")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])\s*")


def chunk_text(content: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """``content`` in chunks of at most ``chunk_size + overlap`` characters."""
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
    for line in block.split("\n"):
        for part in _cut(line, limit) if len(line) > limit else [line]:
            if current and len("\n".join([*current, part])) > limit:
                pieces.append("\n".join(current))
                current = []
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
