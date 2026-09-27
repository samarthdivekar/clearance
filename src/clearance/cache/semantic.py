"""Semantic answer cache — and why the naive version is a data leak.

A semantic cache returns a stored answer when a new question is *similar enough* to an old one.
In a multi-user system the obvious implementation is a security bug:

    1. The CFO asks  "What did we decide about the Raptor hedges?"  -> answer built from CFO-only emails, cached.
    2. An analyst asks "What was decided on the Raptor hedges?"    -> cosine 0.95 -> cache HIT -> CFO's answer.

The analyst never retrieved a forbidden chunk, yet they read its contents. Modes:

* ``global``    - the naive cache above. INSECURE; kept so the red-team suite can demonstrate the leak.
* ``per_user``  - key entries by principal. Safe, but users never share hits, so hit rate collapses.
* ``acl_aware`` - entries record the chunk ids their answer was built from (*all* context sent to
                  the LLM, not only cited ones: the model can paraphrase uncited context). A hit is
                  served only if (a) the requester can read every source chunk, and (b) the
                  requester's own retrieval overlaps the entry's sources (Jaccard >= min_overlap).
                  (a) prevents leaks; (b) prevents serving an answer that is *incomplete* for a
                  user who can see more. Retrieval runs before the lookup - it costs milliseconds,
                  the LLM call it saves costs seconds and cents.
* ``off``

Answers with no sources ("I couldn't find that") are never shared across users: "not found" for
one principal may be wrong for another.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from clearance.security.acl import Principal
from clearance.store.db import Database

MODES = ("off", "global", "per_user", "acl_aware")


@dataclass
class CacheHit:
    entry_id: int
    similarity: float
    payload: dict
    source_chunks: list[int]
    owner: str


@dataclass
class LookupResult:
    hit: CacheHit | None
    blocked_for_acl: int = 0  # similar entries rejected because the requester can't read a source
    blocked_for_scope: int = 0  # similar entries rejected because the requester's context differs


class SemanticCache:
    def __init__(
        self,
        db: Database,
        mode: str = "acl_aware",
        threshold: float = 0.92,
        ttl_seconds: int = 86400,
        min_overlap: float = 0.5,
    ):
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}")
        self.db = db
        self.mode = mode
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.min_overlap = min_overlap
        self._load()

    def _load(self) -> None:
        rows = self.db.conn.execute(
            "SELECT id, created_at, owner, embedding, source_chunks FROM cache_entries ORDER BY id"
        ).fetchall()
        self._ids = [r["id"] for r in rows]
        self._owners = [r["owner"] for r in rows]
        self._created = [r["created_at"] for r in rows]
        self._sources = [json.loads(r["source_chunks"]) for r in rows]
        self._matrix = (
            np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM cache_entries")
        self.db.commit()
        self._load()

    def __len__(self) -> int:
        return len(self._ids)

    def lookup(self, qvec: np.ndarray, principal: Principal, retrieved_ids: list[int]) -> LookupResult:
        result = LookupResult(hit=None)
        if self.mode == "off" or not self._ids:
            return result
        sims = self._matrix @ qvec.astype(np.float32)
        now = time.time()
        for i in np.argsort(-sims):
            sim = float(sims[i])
            if sim < self.threshold:
                break
            if now - self._created[i] > self.ttl_seconds:
                continue
            sources = self._sources[i]
            owner = self._owners[i]
            if self.mode == "per_user" and owner != principal.id:
                continue
            if self.mode == "acl_aware":
                if not sources and owner != principal.id:
                    continue
                if self.db.unreadable_among(principal, sources):
                    result.blocked_for_acl += 1
                    continue
                if sources and _jaccard(sources, retrieved_ids) < self.min_overlap:
                    result.blocked_for_scope += 1
                    continue
            result.hit = self._hit(i, sim)
            return result
        return result

    def _hit(self, i: int, sim: float) -> CacheHit:
        entry_id = self._ids[i]
        row = self.db.conn.execute("SELECT answer FROM cache_entries WHERE id = ?", (entry_id,)).fetchone()
        self.db.conn.execute("UPDATE cache_entries SET hits = hits + 1 WHERE id = ?", (entry_id,))
        self.db.commit()
        return CacheHit(entry_id, sim, json.loads(row["answer"]), self._sources[i], self._owners[i])

    def store(self, question: str, qvec: np.ndarray, principal: Principal, payload: dict, source_chunks: list[int]) -> None:
        if self.mode == "off":
            return
        vec = qvec.astype(np.float32)
        created = time.time()
        cur = self.db.conn.execute(
            """INSERT INTO cache_entries (created_at, owner, question, embedding, answer, source_chunks)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (created, principal.id, question, vec.tobytes(), json.dumps(payload), json.dumps(source_chunks)),
        )
        self.db.commit()
        self._ids.append(cur.lastrowid)
        self._owners.append(principal.id)
        self._created.append(created)
        self._sources.append(list(source_chunks))
        self._matrix = vec[None, :] if self._matrix.size == 0 else np.vstack([self._matrix, vec[None, :]])


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
