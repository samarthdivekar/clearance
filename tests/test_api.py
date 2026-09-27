from fastapi.testclient import TestClient

from clearance.api import app as app_module

from .conftest import ANALYST, FASTOW


def _client(service):
    app_module.get_service.cache_clear()
    app_module.app.dependency_overrides[app_module.get_service] = lambda: service
    return TestClient(app_module.app)


def test_ask_requires_identity(service):
    r = _client(service).post("/api/ask", json={"question": "Raptor"})
    assert r.status_code == 422


def test_ask_as_two_users(service):
    c = _client(service)
    q = {"question": "Are the Raptor vehicles underwater?"}
    cfo = c.post("/api/ask", json=q, headers={"X-Principal": FASTOW}).json()
    analyst = c.post("/api/ask", json=q, headers={"X-Principal": ANALYST}).json()
    assert "underwater" in cfo["answer"]
    assert "underwater" not in analyst["answer"]


def test_audit_log_scoped_to_caller_unless_auditor(service):
    c = _client(service)
    c.post("/api/ask", json={"question": "Raptor hedges"}, headers={"X-Principal": FASTOW})
    c.post("/api/ask", json={"question": "gas curve"}, headers={"X-Principal": ANALYST})
    mine = c.get("/api/audit", headers={"X-Principal": ANALYST}).json()["rows"]
    assert {r["principal"] for r in mine} == {ANALYST}
    everything = c.get("/api/audit", headers={"X-Principal": "auditor"}).json()["rows"]
    assert {r["principal"] for r in everything} == {ANALYST, FASTOW}


def test_metrics_and_principals(service):
    c = _client(service)
    c.post("/api/ask", json={"question": "Raptor hedges"}, headers={"X-Principal": FASTOW})
    assert c.get("/api/metrics").json()["queries"] == 1
    assert FASTOW in c.get("/api/principals").json()["principals"]
    assert c.get("/").status_code == 200
