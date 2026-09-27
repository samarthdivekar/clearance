"""Append-only audit log: who asked what, which chunks reached the LLM, what it cost."""

from __future__ import annotations

import json
import statistics

from clearance.models import Answer
from clearance.security.acl import Principal
from clearance.store.db import Database


def record(db: Database, principal: Principal, answer: Answer) -> None:
    db.conn.execute(
        """INSERT INTO audit_log (principal, principal_groups, question, retrieved, cited, cache_status, model, route_reason,
                                  input_tokens, output_tokens, cost_usd, latency_ms, security_events)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            principal.id,
            json.dumps(sorted(principal.groups)),
            answer.question,
            json.dumps(answer.retrieved_chunk_ids),
            json.dumps([c.chunk_id for c in answer.citations]),
            answer.cache_status,
            answer.model,
            answer.route_reason,
            answer.input_tokens,
            answer.output_tokens,
            answer.cost_usd,
            answer.latency_ms,
            json.dumps(answer.security_events),
        ),
    )
    db.commit()


def recent(db: Database, limit: int = 100, principal: str | None = None) -> list[dict]:
    q = "SELECT * FROM audit_log"
    args: list = []
    if principal:
        q += " WHERE principal = ?"
        args.append(principal)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in db.conn.execute(q, args):
        d = dict(r)
        for key in ("principal_groups", "retrieved", "cited", "security_events"):
            d[key] = json.loads(d[key] or "[]")
        out.append(d)
    return out


def who_accessed(db: Database, chunk_id: int) -> list[dict]:
    """Every principal whose query sent `chunk_id` to the LLM (breach-investigation query)."""
    rows = db.conn.execute(
        """SELECT a.ts, a.principal, a.question FROM audit_log a, json_each(a.retrieved) j
           WHERE j.value = ? ORDER BY a.id""",
        (chunk_id,),
    )
    return [dict(r) for r in rows]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def metrics(db: Database) -> dict:
    rows = db.conn.execute(
        "SELECT cache_status, model, cost_usd, latency_ms, security_events FROM audit_log"
    ).fetchall()
    n = len(rows)
    if not n:
        return {"queries": 0}
    lat = [r["latency_ms"] or 0.0 for r in rows]
    hits = sum(1 for r in rows if r["cache_status"] == "hit")
    by_model: dict[str, int] = {}
    for r in rows:
        by_model[r["model"] or "none"] = by_model.get(r["model"] or "none", 0) + 1
    return {
        "queries": n,
        "cache_hit_rate": round(hits / n, 3),
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
        "avg_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows) / n, 6),
        "latency_p50_ms": round(_percentile(lat, 50), 1),
        "latency_p95_ms": round(_percentile(lat, 95), 1),
        "model_mix": by_model,
        "security_events": sum(len(json.loads(r["security_events"] or "[]")) for r in rows),
    }
