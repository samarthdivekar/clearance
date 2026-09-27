"""Cache + routing benchmark: what do the cache modes and the router save, and what do they cost?

Builds a realistic workload from the corpus: for shared emails (several authorized readers),
each reader asks a paraphrase of the same question, as happens when a thread circulates in a
team. The workload is replayed once per configuration on a fresh cache.
"""

from __future__ import annotations

import random
import statistics

from clearance.cache.semantic import SemanticCache
from clearance.evals import report
from clearance.graph.extract import person_label
from clearance.pipeline import RAGService
from clearance.security.acl import Principal

TEMPLATES = [
    "What did {who} say about {subject}?",
    "What did {who} say regarding {subject}?",
    "what did {who} say about {subject}",
    "What has {who} said about {subject}?",
]


def build_workload(service: RAGService, n_emails: int = 60, seed: int = 11) -> list[tuple[str, str]]:
    db = service.db
    owners = set(db.mailbox_owners())
    rng = random.Random(seed)
    rows = db.conn.execute(
        """SELECT e.id, e.subject, e.sender FROM emails e JOIN email_acl a ON a.email_id = e.id
           WHERE e.subject != '' GROUP BY e.id HAVING COUNT(a.principal) BETWEEN 3 AND 12"""
    ).fetchall()
    rng.shuffle(rows)
    workload: list[tuple[str, str]] = []
    used = 0
    for r in rows:
        readers = [p[5:] for p in db.email_acl(r["id"]) if p.startswith("user:") and p[5:] in owners]
        readers = readers or [p[5:] for p in db.email_acl(r["id"]) if p.startswith("user:")][:3]
        if len(readers) < 2:
            continue
        for i, reader in enumerate(readers[: len(TEMPLATES)]):
            q = TEMPLATES[i % len(TEMPLATES)].format(who=person_label(r["sender"]), subject=r["subject"])
            workload.append((reader, q))
        used += 1
        if used >= n_emails:
            break
    rng.shuffle(workload)
    return workload


def _replay(service: RAGService, workload: list[tuple[str, str]]) -> dict:
    costs, lats, hits, leaks, large = [], [], 0, 0, 0
    for user, q in workload:
        p = Principal.from_email(user)
        ans = service.ask(q, p)
        costs.append(ans.cost_usd)
        lats.append(ans.latency_ms)
        hits += ans.cache_status == "hit"
        large += ans.model == service.settings.large_model and ans.cache_status != "hit"
        leaks += bool(service.db.unreadable_among(p, ans.retrieved_chunk_ids))
    n = len(workload)
    return {
        "queries": n,
        "cache_hit_rate": hits / n,
        "large_model_share": large / n,
        "total_cost_usd": sum(costs),
        "latency_p50_ms": statistics.median(lats),
        "latency_p95_ms": sorted(lats)[int(0.95 * (n - 1))],
        "leaks": leaks,
    }


def run(service: RAGService, n_emails: int = 60) -> list[dict]:
    workload = build_workload(service, n_emails)
    if not workload:
        raise RuntimeError("Could not build a workload: need emails shared by 2+ mailbox owners.")
    print(f"cache bench: {len(workload)} queries over {n_emails} shared emails")
    original_cache, original_router, original_audit = service.cache, service.router, service.audit_enabled
    service.audit_enabled = False
    s = service.settings
    configs = [
        ("no cache, always large", "off", s.large_model),
        ("no cache, routed", "off", None),
        ("per_user cache, routed", "per_user", None),
        ("acl_aware cache, routed", "acl_aware", None),
        ("global cache (INSECURE), routed", "global", None),
    ]
    rows = []
    try:
        for label, mode, force in configs:
            service.cache = SemanticCache(service.db, mode=mode, threshold=s.cache_threshold)
            service.cache.clear()
            if force:
                service.router = _ForcedRouter(force)
            else:
                service.router = original_router
            rows.append({"config": label, **_replay(service, workload)})
            print(f"  {label:<34} hit={rows[-1]['cache_hit_rate']:.2f} cost=${rows[-1]['total_cost_usd']:.4f} leaks={rows[-1]['leaks']}")
    finally:
        service.cache.clear()
        service.cache, service.router, service.audit_enabled = original_cache, original_router, original_audit
    baseline = rows[0]["total_cost_usd"] or 1e-9
    for r in rows:
        r["cost_vs_baseline"] = f"{100 * (1 - r['total_cost_usd'] / baseline):+.0f}% saved" if r is not rows[0] else "baseline"
    provider_note = (
        "Costs are **estimated** (offline FakeLLM: tokens ~ chars/4, priced at real model rates). "
        if s.llm_provider == "fake" else "Costs are from real Claude API usage. "
    )
    report.write(
        "cache_bench",
        "Cost, latency and safety by cache mode and routing",
        rows,
        notes=provider_note + "Quality at fixed cost is measured separately by `clearance eval answers`.",
    )
    return rows


class _ForcedRouter:
    def __init__(self, model: str):
        self.model = model

    def route(self, question, context):
        from clearance.llm.router import RouteDecision

        return RouteDecision(self.model, 1.0, "forced (baseline)", "large")
