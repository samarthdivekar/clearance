"""Email-aware chunking.

Each chunk is prefixed with a compact header (sender, date, subject). Without it, a chunk like
"Approved - go ahead." is unretrievable: the *who/when/what* lives in the headers, not the body.
Bodies are packed paragraph-by-paragraph up to `max_words`, with a small word overlap between
consecutive chunks of the same email.
"""

from __future__ import annotations

import re

from clearance.models import Email


def header_line(em: Email) -> str:
    date = em.date[:10] if em.date else "unknown date"
    to = ", ".join(em.to[:4]) + (" ..." if len(em.to) > 4 else "")
    return f"From: {em.sender} | To: {to} | Date: {date} | Subject: {em.subject or '(no subject)'}"


def _split_long(paragraph: str, max_words: int) -> list[str]:
    words = paragraph.split()
    if len(words) <= max_words:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], []
    for s in sentences:
        sw = s.split()
        if len(sw) > max_words:  # pathological run-on text: hard split
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.extend(" ".join(sw[i : i + max_words]) for i in range(0, len(sw), max_words))
        elif len(cur) + len(sw) > max_words:
            out.append(" ".join(cur))
            cur = sw
        else:
            cur.extend(sw)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_email(em: Email, max_words: int = 180, overlap_words: int = 25) -> list[str]:
    header = header_line(em)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", em.body) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(re.sub(r"\s+", " ", p), max_words))
    if not pieces:
        return [header]

    chunks: list[str] = []
    cur: list[str] = []
    for piece in pieces:
        n = len(piece.split())
        if cur and sum(len(c.split()) for c in cur) + n > max_words:
            chunks.append(" ".join(cur))
            tail = " ".join(cur).split()[-overlap_words:] if overlap_words else []
            cur = [" ".join(tail)] if tail else []
        cur.append(piece)
    if cur:
        chunks.append(" ".join(cur))
    return [f"{header}\n{c}" for c in chunks]
