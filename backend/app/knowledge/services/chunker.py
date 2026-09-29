"""Deterministic text chunking for knowledge ingestion.

Pure functions with no I/O, so chunk boundaries are unit-testable without an
embedding model, a database, or a Qdrant server.

Strategy: paragraphs are the preferred boundaries, then sentence endings, then
word boundaries, and only as a last resort a hard character cut. Each chunk
after the first carries a short tail of its predecessor so a fact that straddles
a boundary is still findable from both sides.
"""

import re

from app.knowledge.config import PARAGRAPH_SEPARATOR

SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?")
SENTENCE_GAP = " "
LINE_BREAKS: tuple[str, str] = ("\r\n", "\n")


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks no longer than ``max_chars`` characters."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    bounded_overlap = max(0, min(overlap, max_chars - 1))
    pieces = _split_into_pieces(_normalise(text), max_chars)
    return _pack(pieces, max_chars, bounded_overlap)


def _normalise(text: str) -> str:
    """Canonicalise line breaks and collapse runs of blank lines."""
    normalised = text.replace(LINE_BREAKS[0], LINE_BREAKS[1])
    normalised = re.sub(r"[ \t]+\n", "\n", normalised.replace("\r", "\n"))
    return re.sub(r"\n{3,}", PARAGRAPH_SEPARATOR, normalised).strip()


def _split_into_pieces(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for paragraph in text.split(PARAGRAPH_SEPARATOR):
        stripped = paragraph.strip()
        if stripped:
            pieces.extend(_split_oversized(stripped, max_chars))
    return pieces


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    remaining = paragraph
    while len(remaining) > max_chars:
        cut = _find_cut(remaining, max_chars)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining.strip():
        pieces.append(remaining.strip())
    return pieces


def _find_cut(text: str, max_chars: int) -> int:
    """Best split point at or before ``max_chars``: sentence, then word, then hard."""
    window = text[: max(1, max_chars - len(SENTENCE_GAP))]
    terminator_at = max(window.rfind(f"{mark}{SENTENCE_GAP}") for mark in SENTENCE_TERMINATORS)
    if terminator_at > 0:
        return terminator_at + len(SENTENCE_TERMINATORS[0]) + len(SENTENCE_GAP)
    space_at = window.rfind(SENTENCE_GAP)
    return space_at if space_at > 0 else max_chars


def _pack(pieces: list[str], max_chars: int, overlap: int) -> list[str]:
    """Join pieces up to the limit, prefixing each chunk with a tail of the last."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = _join(current, piece)
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        carried = _tail(current, overlap)
        current = piece if len(_join(carried, piece)) > max_chars else _join(carried, piece)
    if current:
        chunks.append(current)
    return chunks


def _join(left: str, right: str) -> str:
    return right if not left else f"{left}{PARAGRAPH_SEPARATOR}{right}"


def _tail(text: str, overlap: int) -> str:
    """The last ``overlap`` characters of ``text``, snapped to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    space_at = tail.find(SENTENCE_GAP)
    return tail[space_at + 1 :] if space_at > 0 else tail
