"""Permission-filtered graph traversal for multi-hop questions.

Plain vector search fails on "Who approved the deal that Vince warned about?" because no single
chunk contains the whole answer. The graph path is:

    query ──match──▶ seed entities ──1-2 hops over readable edges──▶ bridge entities
                                         │
                                         └──▶ chunks that mention seeds *and* bridges

Chunks are scored by how many (idf-weighted) seed/bridge entities they mention, discounted by
hop distance. The resulting ranked list is fused with BM25 and vector results via RRF.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from clearance.graph.extract import COMMON_WORDS
from clearance.store.db import Database


class GraphIndex:
    def __init__(self, db: Database):
        self.db = db
        self.reload()

    def reload(self) -> None:
        conn = self.db.conn
        self.labels: dict[int, str] = {}
        self.types: dict[int, str] = {}
        self.name_to_id: dict[str, int] = {}
        for r in conn.execute("SELECT id, name, label, type FROM entities"):
            self.labels[r["id"]] = r["label"]
            self.types[r["id"]] = r["type"]
            for alias in self._aliases(r["name"], r["label"], r["type"]):
                self.name_to_id.setdefault(alias, r["id"])

        self.adj: dict[int, list[tuple[int, int]]] = defaultdict(list)  # entity -> [(neighbor, chunk)]
        for r in conn.execute("SELECT src, dst, chunk_id FROM edges"):
            self.adj[r["src"]].append((r["dst"], r["chunk_id"]))
            self.adj[r["dst"]].append((r["src"], r["chunk_id"]))

        self.mentions: dict[int, list[int]] = defaultdict(list)  # entity -> chunks
        for r in conn.execute("SELECT entity_id, chunk_id FROM mentions"):
            self.mentions[r["entity_id"]].append(r["chunk_id"])
        n = max(1, self.db.chunk_count())
        self.idf = {e: math.log(1 + n / (1 + len(c))) for e, c in self.mentions.items()}
        self.max_ngram = max((len(a.split()) for a in self.name_to_id), default=1)

    @staticmethod
    def _aliases(name: str, label: str, type_: str) -> set[str]:
        aliases = {name, label.lower()}
        if type_ == "person" and "@" in name:
            parts = label.lower().split()
            if len(parts) >= 2:
                aliases.add(f"{parts[0]} {parts[-1]}")
                aliases.add(parts[-1])  # surname: "skilling"
        return {a for a in aliases if len(a) >= 3 and a not in COMMON_WORDS}

    def seed_entities(self, query: str) -> list[int]:
        tokens = re.findall(r"[a-z0-9&.@\-]+", query.lower())
        seeds: list[int] = []
        i = 0
        while i < len(tokens):
            for size in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                cand = " ".join(tokens[i : i + size]).strip(".")
                eid = self.name_to_id.get(cand)
                if eid is not None:
                    if eid not in seeds:
                        seeds.append(eid)
                    i += size
                    break
            else:
                i += 1
        return seeds

    def search(self, query: str, readable: set[int] | None, limit: int, hops: int = 2) -> list[tuple[int, float]]:
        seeds = self.seed_entities(query)
        if not seeds:
            return []

        def ok(chunk_id: int) -> bool:
            return readable is None or chunk_id in readable

        # BFS over edges the principal can read. weight = 1 / (1 + hop)
        weight: dict[int, float] = {s: 1.0 for s in seeds}
        frontier = list(seeds)
        bridge_chunks: dict[int, float] = defaultdict(float)
        for hop in range(1, hops + 1):
            nxt: dict[int, int] = defaultdict(int)
            for e in frontier:
                for nb, cid in self.adj.get(e, ()):
                    if not ok(cid):
                        continue
                    bridge_chunks[cid] += 1.0 / hop
                    if nb not in weight:
                        nxt[nb] += 1
            # Keep the best-connected neighbors only; hubs would otherwise flood the result.
            ranked = sorted(nxt, key=lambda n: -nxt[n] * self.idf.get(n, 1.0))[:25]
            for nb in ranked:
                weight[nb] = 1.0 / (1 + hop)
            frontier = ranked

        scores: dict[int, float] = defaultdict(float)
        for e, w in weight.items():
            idf = self.idf.get(e, 1.0)
            for cid in self.mentions.get(e, ()):
                if ok(cid):
                    scores[cid] += w * idf
        for cid, s in bridge_chunks.items():
            scores[cid] += 0.5 * s
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]

    def describe(self, query: str) -> list[str]:
        return [self.labels[e] for e in self.seed_entities(query)]
