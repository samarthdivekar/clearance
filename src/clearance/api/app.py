"""HTTP API + demo UI.

Identity: the demo takes the caller's identity from the ``X-Principal`` header so you can switch
users in the UI. In production this comes from a verified session/JWT - never from a
client-controlled header. Everything downstream only ever sees a `Principal`, so swapping the
auth layer touches `current_principal` alone.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from clearance.config import get_settings
from clearance.llm.client import LLMError
from clearance.pipeline import RAGService
from clearance.security import audit
from clearance.security.acl import Principal

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Clearance", description="Permission-aware GraphRAG over Enron email", version="0.1.0")


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    settings = get_settings()
    if not settings.db_path.exists():
        raise HTTPException(503, "No index found. Run `clearance download` and `clearance ingest` first.")
    return RAGService.from_settings(settings)


def current_principal(
    x_principal: str = Header(..., description="Demo identity; use real auth in production"),
    svc: RAGService = Depends(get_service),
) -> Principal:
    if not x_principal.strip():
        raise HTTPException(401, "Missing identity")
    return svc.principal(x_principal)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "indexed": s.db_path.exists(), "llm": s.llm_provider, "cache_mode": s.cache_mode}


@app.get("/api/principals")
def principals(svc: RAGService = Depends(get_service)) -> dict:
    owners = svc.db.mailbox_owners()
    members = sorted({m for g in svc.directory.groups.values() for m in g.members} - set(owners))
    return {"principals": owners + members, "special": ["auditor"]}


@app.post("/api/ask")
def ask(req: AskRequest, principal: Principal = Depends(current_principal), svc: RAGService = Depends(get_service)) -> dict:
    try:
        return svc.ask(req.question, principal).to_dict()
    except LLMError as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/metrics")
def metrics(svc: RAGService = Depends(get_service)) -> dict:
    return {**audit.metrics(svc.db), "cache_entries": len(svc.cache), "cache_mode": svc.cache.mode}


@app.get("/api/audit")
def audit_log(
    limit: int = Query(50, le=500),
    principal: Principal = Depends(current_principal),
    svc: RAGService = Depends(get_service),
) -> dict:
    # Only auditors may read the full log; everyone else sees their own history.
    rows = audit.recent(svc.db, limit, principal=None if principal.is_auditor else principal.id)
    return {"rows": rows}
