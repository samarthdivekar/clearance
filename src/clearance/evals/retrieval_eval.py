"""Retrieval ablation: which retrieval stack finds the right emails?

Runs every configuration over the gold set, as each question's principal (so ACL filtering is
part of what's measured), and reports recall@k / MRR / nDCG split by single- vs multi-hop.
"""

from __future__ import annotations

import time

from clearance.evals import report
from clearance.evals.dataset import GoldQuestion
from clearance.evals.metrics import dedup, hit_at_k, mean, mrr, ndcg_at_k, recall_at_k
from clearance.pipeline import RAGService
from clearance.retrieval.retriever import RetrievalConfig

ABLATIONS = [
    RetrievalConfig(mode="bm25", use_graph=False),
    RetrievalConfig(mode="vector", use_graph=False),
    RetrievalConfig(mode="hybrid", use_graph=False),
    RetrievalConfig(mode="hybrid", use_graph=True),
    RetrievalConfig(mode="hybrid", use_graph=True, use_reranker=True),
]


def evaluate_config(service: RAGService, questions: list[GoldQuestion], cfg: RetrievalConfig, k: int = 5) -> dict:
    per_hop: dict[str, dict[str, list[float]]] = {}
    latencies = []
    cfg = RetrievalConfig(cfg.mode, cfg.use_graph, cfg.use_reranker, top_k=max(k, 10), candidate_k=50)
    for q in questions:
        t0 = time.perf_counter()
        results = service.retriever.retrieve(q.question, service.principal(q.principal), cfg)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = dedup([r.chunk.message_id for r in results])
        relevant = set(q.relevant_message_ids)
        bucket = "multi-hop" if q.hops > 1 else "single-hop"
        for key in (bucket, "all"):
            m = per_hop.setdefault(key, {"recall": [], "hit": [], "mrr": [], "ndcg": []})
            m["recall"].append(recall_at_k(ranked, relevant, k))
            m["hit"].append(hit_at_k(ranked, relevant, k))
            m["mrr"].append(mrr(ranked, relevant))
            m["ndcg"].append(ndcg_at_k(ranked, relevant, k))
    row = {"config": cfg.label}
    for bucket in ("all", "single-hop", "multi-hop"):
        if bucket in per_hop:
            m = per_hop[bucket]
            tag = {"all": "", "single-hop": "1hop_", "multi-hop": "multihop_"}[bucket]
            row[f"{tag}recall@{k}"] = mean(m["recall"])
            if bucket == "all":
                row[f"hit@{k}"] = mean(m["hit"])
                row["mrr"] = mean(m["mrr"])
                row[f"ndcg@{k}"] = mean(m["ndcg"])
    row["p50_ms"] = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    return row


def run(service: RAGService, questions: list[GoldQuestion], k: int = 5, configs: list[RetrievalConfig] | None = None) -> list[dict]:
    rows = []
    for cfg in configs or ABLATIONS:
        if cfg.use_reranker and service.retriever.reranker is None:
            try:
                from clearance.retrieval.retriever import Reranker

                service.retriever.reranker = Reranker(service.settings.reranker_model)
            except Exception as e:  # optional dependency / offline
                print(f"skipping {cfg.label}: reranker unavailable ({e})")
                continue
        if cfg.use_graph and service.retriever.graph is None:
            from clearance.graph.retriever import GraphIndex

            service.retriever.graph = GraphIndex(service.db)
        rows.append(evaluate_config(service, questions, cfg, k))
        print(f"  {cfg.label:<24} recall@{k}={rows[-1][f'recall@{k}']:.3f}")
    n_multi = sum(1 for q in questions if q.hops > 1)
    report.write(
        "retrieval_ablation",
        "Retrieval ablation",
        rows,
        notes=f"{len(questions)} questions ({n_multi} multi-hop). Relevance at email level; each query runs as its principal.",
    )
    return rows
