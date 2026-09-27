import os

from clearance.evals import cache_bench, redteam, report
from clearance.evals.dataset import GoldQuestion, load, save
from clearance.evals.metrics import mrr, ndcg_at_k, recall_at_k
from clearance.evals.retrieval_eval import evaluate_config
from clearance.retrieval.retriever import RetrievalConfig

from .conftest import KAMINSKI


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert mrr(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["b"], {"b"}, 5) == 1.0


def test_dataset_roundtrip(tmp_path):
    q = GoldQuestion("q1", "What?", KAMINSKI, ["m1"], hops=2)
    save([q], tmp_path / "g.jsonl")
    assert load(tmp_path / "g.jsonl") == [q]


def test_retrieval_eval_runs(service):
    msg_id = service.db.conn.execute("SELECT message_id FROM emails WHERE subject = 'Model review'").fetchone()[0]
    qs = [GoldQuestion("q1", "gas forward curve recalibration", KAMINSKI, [msg_id])]
    row = evaluate_config(service, qs, RetrievalConfig(mode="hybrid", use_graph=True), k=5)
    assert row["recall@5"] == 1.0


def test_redteam_finds_global_leak_and_no_acl_aware_leak(service, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    rows = {r["cache_mode"]: r for r in redteam.run(service, n_targets=10)}
    assert rows["acl_aware"]["context_leaks"] == rows["acl_aware"]["cache_leaks"] == rows["acl_aware"]["content_leaks"] == 0
    assert rows["per_user"]["cache_leaks"] == 0
    assert rows["global"]["cache_leaks"] > 0
    assert rows["acl_aware (context)"]["cache_leaks"] == rows["acl_aware (context)"]["content_leaks"] == 0
    assert rows["acl_aware"]["authorized_cache_hit_rate"] >= rows["acl_aware (context)"]["authorized_cache_hit_rate"]
    assert rows["acl_aware"]["authorized_cache_hit_rate"] >= rows["per_user"]["authorized_cache_hit_rate"]
    assert os.path.exists(tmp_path / "reports" / "redteam.md")


def test_cache_bench_runs(service, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    rows = cache_bench.run(service, n_emails=5)
    by = {r["config"]: r for r in rows}
    assert by["acl_aware cache, routed"]["leaks"] == 0
