"""The properties that matter most: nothing unauthorized reaches a user, by any path."""

from clearance.cache.semantic import SemanticCache
from clearance.retrieval.retriever import RetrievalConfig

RAPTOR_Q = "Are the Raptor vehicles underwater before the earnings call?"


def _texts(service, principal, query, cfg):
    return [r.chunk.text for r in service.retriever.retrieve(query, principal, cfg)]


def test_bm25_respects_acl(service, people):
    cfg = RetrievalConfig(mode="bm25", use_graph=False, top_k=10)
    assert any("underwater" in t for t in _texts(service, people["fastow"], RAPTOR_Q, cfg))
    assert not any("underwater" in t for t in _texts(service, people["analyst"], RAPTOR_Q, cfg))


def test_vector_respects_acl(service, people):
    cfg = RetrievalConfig(mode="vector", use_graph=False, top_k=10)
    assert any("underwater" in t for t in _texts(service, people["lay"], RAPTOR_Q, cfg))
    assert not any("underwater" in t for t in _texts(service, people["analyst"], RAPTOR_Q, cfg))


def test_graph_respects_acl(service, people):
    cfg = RetrievalConfig(mode="bm25", use_graph=True, top_k=10)
    for t in _texts(service, people["analyst"], "What did Andrew Fastow say about Raptor and LJM?", cfg):
        assert "underwater" not in t and "LJM" not in t


def test_auditor_sees_everything(service, people):
    cfg = RetrievalConfig(mode="hybrid", use_graph=True, top_k=10)
    assert any("underwater" in t for t in _texts(service, people["auditor"], RAPTOR_Q, cfg))


def test_forward_recipient_can_read_forwarded_content(service, people):
    cfg = RetrievalConfig(mode="bm25", use_graph=False, top_k=5)
    assert any("weather derivatives" in t for t in _texts(service, people["analyst"], "weather derivatives desk Houston", cfg))


def test_answer_never_contains_unreadable_text(service, people):
    ans = service.ask(RAPTOR_Q, people["analyst"])
    assert "underwater" not in ans.text
    assert not ans.security_events


def test_naive_global_cache_leaks(service, people):
    """Documents the vulnerability: a shared semantic cache serves the CFO's answer to an analyst."""
    service.cache = SemanticCache(service.db, mode="global", threshold=0.8)
    first = service.ask(RAPTOR_Q, people["fastow"])
    assert "underwater" in first.text
    leaked = service.ask(RAPTOR_Q, people["analyst"])
    assert leaked.cache_status == "hit"
    assert "underwater" in leaked.text
    assert any(e.startswith("CACHE_LEAK") for e in leaked.security_events)


def test_acl_aware_cache_blocks_leak(service, people):
    service.cache = SemanticCache(service.db, mode="acl_aware", threshold=0.8)
    service.ask(RAPTOR_Q, people["fastow"])
    safe = service.ask(RAPTOR_Q, people["analyst"])
    assert safe.cache_status != "hit"
    assert "underwater" not in safe.text
    assert any(e.startswith("CACHE_ACL_BLOCKED") for e in safe.security_events)


def test_acl_aware_cache_shares_between_authorized_users(service, people):
    service.cache = SemanticCache(service.db, mode="acl_aware", threshold=0.8)
    miss = service.ask(RAPTOR_Q, people["fastow"])
    assert miss.cache_status == "miss"
    hit = service.ask(RAPTOR_Q, people["lay"])  # Lay received the same email
    assert hit.cache_status == "hit"
    assert hit.cost_usd == 0


def test_per_user_cache_never_shares(service, people):
    service.cache = SemanticCache(service.db, mode="per_user", threshold=0.8)
    service.ask(RAPTOR_Q, people["fastow"])
    assert service.ask(RAPTOR_Q, people["lay"]).cache_status == "miss"
    assert service.ask(RAPTOR_Q, people["fastow"]).cache_status == "hit"


def test_not_found_answers_are_not_shared(service, people):
    service.cache = SemanticCache(service.db, mode="acl_aware", threshold=0.8)
    q = "What is the Raptor vehicles restructuring plan before earnings?"
    first = service.ask(q, people["analyst"])  # analyst can't see Raptor emails
    assert first.citations == []
    assert service.ask(q, people["fastow"]).cache_status != "hit"


def test_audit_log_records_every_query(service, people):
    from clearance.security import audit

    service.ask(RAPTOR_Q, people["fastow"])
    service.ask("gas curve model", people["analyst"])
    rows = audit.recent(service.db)
    assert [r["principal"] for r in rows] == [people["analyst"].id, people["fastow"].id]
    retrieved = rows[1]["retrieved"]
    assert retrieved and audit.who_accessed(service.db, retrieved[0])[0]["principal"] == people["fastow"].id
