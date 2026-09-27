"""Which context chunks does a generated answer actually depend on?

The semantic cache authorizes a hit by checking the requester can read every chunk the cached
answer depends on. Using the *whole* context (8 chunks) is safe but on real data almost never
shareable: the first user's context nearly always contains some email the second user wasn't
sent, even though the answer only used one or two of them.

Citations alone are too optimistic: a model can state a fact from an uncited chunk. So the
dependency set is:

    cited chunks
  ∪ uncited chunks that contain a *distinctive term* of the answer (number, amount, proper noun,
    acronym) that doesn't already appear in the question or the cited chunks
  ∪ uncited chunks sharing a 4-word run of content words with the answer that isn't in the
    cited chunks

Anything the answer says that could only have come from chunk X therefore pulls X into the ACL
check. This is conservative for verbatim and lightly paraphrased reuse; a heavily reworded fact
with no names or numbers can still slip through, which is why the red-team suite runs with a
real LLM too (see docs/SECURITY.md).
"""

from __future__ import annotations

import re

from clearance.graph.extract import COMMON_WORDS
from clearance.models import Chunk

_TOKEN_RE = re.compile(r"\$?[A-Za-z0-9][A-Za-z0-9.,&'-]*")
_WORD_RE = re.compile(r"[a-z0-9]+")
_CITATION_RE = re.compile(r"\[\d+\]")
SHINGLE = 4

_EXTRA_STOP = frozenset(
    """i couldn't find that in the emails you have access to according email emails wrote said
    sent mentioned regarding""".split()
)


def _norm(token: str) -> str:
    return token.strip(".,'-").lower().replace(",", "")


def distinctive_terms(text: str) -> set[str]:
    """Numbers/amounts, acronyms and capitalized words that aren't sentence-initial."""
    text = _CITATION_RE.sub(" ", text)
    out: set[str] = set()
    for sentence in re.split(r"(?<=[.!?:\n])\s+", text):
        tokens = _TOKEN_RE.findall(sentence)
        for i, tok in enumerate(tokens):
            norm = _norm(tok)
            if not norm or norm in COMMON_WORDS or norm in _EXTRA_STOP:
                continue
            has_digit = any(ch.isdigit() for ch in tok)
            is_acronym = len(tok) >= 2 and tok.isupper() and tok.isalpha()
            is_proper = i > 0 and tok[:1].isupper()
            if has_digit or is_acronym or is_proper:
                out.add(norm)
    return out


def _words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in COMMON_WORDS]


def _shingles(text: str) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i : i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


def _vocab(text: str) -> set[str]:
    return {_norm(t) for t in _TOKEN_RE.findall(text)} | set(_WORD_RE.findall(text.lower()))


def answer_dependencies(answer: str, question: str, context: list[Chunk], cited_ids: set[int]) -> list[int]:
    cited = [c for c in context if c.id in cited_ids]
    cited_text = "\n".join(c.text for c in cited)
    known = _vocab(question) | _vocab(cited_text)
    novel_terms = {t for t in distinctive_terms(answer) if t not in known}
    novel_shingles = _shingles(_CITATION_RE.sub(" ", answer)) - _shingles(cited_text)

    deps = set(cited_ids)
    for c in context:
        if c.id in deps:
            continue
        if novel_terms & _vocab(c.text) or novel_shingles & _shingles(c.text):
            deps.add(c.id)
    return [c.id for c in context if c.id in deps]
