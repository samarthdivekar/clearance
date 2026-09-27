"""Hybrid, permission-aware retrieval.

    BM25 (FTS5, ACL in SQL) ─┐
    Vector (ACL pre-filter) ─┼─▶ Reciprocal Rank Fusion ─▶ (optional) cross-encoder rerank ─▶ top-k
    Graph (ACL on edges)    ─┘

Each retriever applies the ACL itself; nothing unauthorized is ever scored. The pipeline then
re-checks the final set against the database as defense in depth.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from clearance.embeddings import Embedder, embed_query
from clearance.graph.retriever import GraphIndex
from clearance.models import ScoredChunk
from clearance.security.acl import Principal
from clearance.store.db import Database
from clearance.store.vectors import VectorIndex

RRF_K = 60
# Graph evidence is broader and noisier than lexical/semantic matches: it votes at half weight.
RRF_WEIGHTS = {"bm25": 1.0, "vector": 1.0, "graph": 0.5}


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"  # bm25 | vector | hybrid
    use_graph: bool = True
    use_reranker: bool = False
    top_k: int = 8
    candidate_k: int = 50

    @property
    def label(self) -> str:
        parts = [self.mode]
        if self.use_graph:
            parts.append("graph")
        if self.use_reranker:
            parts.append("rerank")
        return "+".join(parts)


def rrf(ranked_lists: dict[str, list[tuple[int, float]]], k: int = RRF_K) -> list[tuple[int, float, dict]]:
    fused: dict[int, float] = defaultdict(float)
    provenance: dict[int, dict] = defaultdict(dict)
    for name, items in ranked_lists.items():
        for rank, (cid, _score) in enumerate(items, start=1):
            fused[cid] += RRF_WEIGHTS.get(name, 1.0) / (k + rank)
            provenance[cid][name] = rank
    return [(cid, s, provenance[cid]) for cid, s in sorted(fused.items(), key=lambda kv: -kv[1])]


class Reranker:
    """Cross-encoder reranker (query, passage) -> relevance logit."""

    def __init__(self, model_name: str, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                batch = self.tokenizer([query] * len(chunk), chunk, padding=True, truncation=True,
                                       max_length=512, return_tensors="pt")
                scores.extend(self.model(**batch).logits[:, 0].tolist())
        return scores


class Retriever:
    def __init__(
        self,
        db: Database,
        vectors: VectorIndex | None,
        embedder: Embedder | None,
        graph: GraphIndex | None = None,
        reranker: Reranker | None = None,
    ):
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.graph = graph
        self.reranker = reranker
        self._readable_cache: OrderedDict[Principal, set[int]] = OrderedDict()

    def readable(self, principal: Principal) -> set[int] | None:
        """Chunk ids the principal may read (None = everything, for auditors). LRU-cached."""
        if principal.is_auditor:
            return None
        if principal in self._readable_cache:
            self._readable_cache.move_to_end(principal)
            return self._readable_cache[principal]
        ids = self.db.readable_chunk_ids(principal)
        self._readable_cache[principal] = ids
        if len(self._readable_cache) > 256:
            self._readable_cache.popitem(last=False)
        return ids

    def invalidate(self) -> None:
        self._readable_cache.clear()

    def retrieve(self, query: str, principal: Principal, cfg: RetrievalConfig) -> list[ScoredChunk]:
        lists: dict[str, list[tuple[int, float]]] = {}
        readable = self.readable(principal)

        if cfg.mode in ("bm25", "hybrid"):
            lists["bm25"] = self.db.bm25_search(query, principal, cfg.candidate_k)
        if cfg.mode in ("vector", "hybrid") and self.vectors is not None and self.embedder is not None:
            qvec = embed_query(self.embedder, query)
            lists["vector"] = self.vectors.search(qvec, readable, cfg.candidate_k)
        if cfg.use_graph and self.graph is not None:
            lists["graph"] = self.graph.search(query, readable, cfg.candidate_k)

        fused = rrf(lists)
        if not fused:
            return []
        pool = fused[: max(cfg.top_k * 3, 20)] if cfg.use_reranker else fused[: cfg.top_k]
        chunks = self.db.get_chunks([cid for cid, _, _ in pool])
        results = [ScoredChunk(chunks[cid], score, prov) for cid, score, prov in pool if cid in chunks]

        if cfg.use_reranker and self.reranker is not None and results:
            scores = self.reranker.score(query, [r.chunk.text for r in results])
            for r, s in zip(results, scores, strict=True):
                r.sources["rerank"] = s
                r.score = s
            results.sort(key=lambda r: -r.score)
        return results[: cfg.top_k]
