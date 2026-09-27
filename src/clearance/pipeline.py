"""The query path, end to end.

    question + principal
        │
        ├─ retrieve (BM25 + vector + graph, each ACL-filtered)      ~10-50 ms
        ├─ guard: re-check every chunk against the DB ACL            defense in depth
        ├─ semantic cache lookup (ACL + scope validated)             hit → return, $0
        ├─ route: small vs large model by difficulty
        ├─ generate with citations
        ├─ cache store (sources = all context chunks)
        └─ audit log
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from clearance.cache.provenance import answer_dependencies
from clearance.cache.semantic import SemanticCache
from clearance.config import Settings
from clearance.embeddings import Embedder, embed_query, load_embedder
from clearance.graph.retriever import GraphIndex
from clearance.llm.client import LLM, load_llm
from clearance.llm.prompts import ANSWER_SYSTEM, NOT_FOUND, answer_user_prompt
from clearance.llm.router import Router
from clearance.models import Answer, Citation, ScoredChunk
from clearance.retrieval.retriever import Reranker, RetrievalConfig, Retriever
from clearance.security import audit
from clearance.security.acl import Principal
from clearance.security.groups import Directory
from clearance.store.db import Database
from clearance.store.vectors import VectorIndex

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class RAGService:
    settings: Settings
    db: Database
    retriever: Retriever
    embedder: Embedder
    llm: LLM
    router: Router
    cache: SemanticCache
    directory: Directory
    audit_enabled: bool = True

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        db: Database | None = None,
        directory: Directory | None = None,
    ) -> RAGService:
        db = db or Database(settings.db_path)
        embedder = embedder or load_embedder(settings.embedder, settings.embed_model)
        vectors = None
        if settings.vectors_path.exists():
            vectors = VectorIndex.load(settings.vectors_path)
            if vectors.model_name != embedder.name:
                raise RuntimeError(
                    f"Vector index was built with {vectors.model_name!r} but the configured embedder is "
                    f"{embedder.name!r}. Re-run `clearance ingest` or change CLEARANCE_EMBEDDER."
                )
        graph = GraphIndex(db) if settings.use_graph else None
        reranker = Reranker(settings.reranker_model) if settings.use_reranker else None
        retriever = Retriever(db, vectors, embedder, graph, reranker)
        return cls(
            settings=settings,
            db=db,
            retriever=retriever,
            embedder=embedder,
            llm=llm or load_llm(settings.llm_provider),
            router=Router(settings.small_model, settings.large_model, settings.router_threshold),
            cache=SemanticCache(db, settings.cache_mode, settings.cache_threshold, settings.cache_ttl_seconds),
            directory=directory or Directory(settings.groups_path),
        )

    def principal(self, email: str) -> Principal:
        """Resolve an authenticated identity to a Principal with its current group memberships."""
        return self.directory.principal(email)

    def default_retrieval_config(self) -> RetrievalConfig:
        s = self.settings
        return RetrievalConfig(
            mode=s.retrieval_mode,
            use_graph=s.use_graph and self.retriever.graph is not None,
            use_reranker=s.use_reranker and self.retriever.reranker is not None,
            top_k=s.top_k,
            candidate_k=s.candidate_k,
        )

    # ------------------------------------------------------------------ query
    def ask(
        self,
        question: str,
        principal: Principal,
        cfg: RetrievalConfig | None = None,
        force_model: str | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        cfg = cfg or self.default_retrieval_config()
        events: list[str] = []

        context = self.retriever.retrieve(question, principal, cfg)

        # Defense in depth: every retriever filters by ACL, but verify against the source of truth
        # before anything reaches the LLM. A non-empty set here is a bug, and gets logged loudly.
        leaked = self.db.unreadable_among(principal, [c.chunk.id for c in context])
        if leaked:
            events.append(f"ACL_GUARD_BLOCKED chunks={sorted(leaked)}")
            context = [c for c in context if c.chunk.id not in leaked]
        context_ids = [c.chunk.id for c in context]

        qvec = embed_query(self.embedder, question)
        lookup = self.cache.lookup(qvec, principal, context_ids)
        if lookup.blocked_for_acl:
            events.append(f"CACHE_ACL_BLOCKED entries={lookup.blocked_for_acl}")

        if lookup.hit is not None:
            hit = lookup.hit
            unreadable = self.db.unreadable_among(principal, hit.source_chunks)
            if unreadable:  # only possible in the insecure `global` mode
                events.append(f"CACHE_LEAK served_unreadable_chunks={sorted(unreadable)} owner={hit.owner}")
            payload = hit.payload
            answer = Answer(
                question=question,
                text=payload["answer"],
                citations=[Citation(**c) for c in payload["citations"]],
                model=payload.get("model", ""),
                route_reason=f"cache hit (similarity {hit.similarity:.3f}, entry {hit.entry_id})",
                cache_status="hit",
                retrieved_chunk_ids=hit.source_chunks,
                security_events=events,
            )
        elif not context:
            answer = Answer(
                question=question, text=NOT_FOUND, citations=[], model="none",
                route_reason="no readable context", cache_status="bypass", security_events=events,
            )
        else:
            answer = self._generate(question, context, force_model)
            answer.security_events = events
            # "Not found" answers get no sources and are never shared (see SemanticCache).
            sources = self._cache_sources(question, answer, context) if answer.citations else []
            self.cache.store(
                question,
                qvec,
                principal,
                {"answer": answer.text, "citations": [c.__dict__ for c in answer.citations], "model": answer.model},
                sources,
                context_ids,
            )

        answer.latency_ms = (time.perf_counter() - t0) * 1000
        if self.audit_enabled:
            audit.record(self.db, principal, answer)
        return answer

    def _cache_sources(self, question: str, answer: Answer, context: list[ScoredChunk]) -> list[int]:
        chunks = [c.chunk for c in context]
        if self.settings.cache_provenance == "context":
            return [c.id for c in chunks]
        return answer_dependencies(answer.text, question, chunks, {c.chunk_id for c in answer.citations})

    def _generate(self, question: str, context: list[ScoredChunk], force_model: str | None) -> Answer:
        if force_model:
            model, reason = force_model, "forced"
        else:
            decision = self.router.route(question, context)
            model, reason = decision.model, decision.reason
        effort = self.settings.large_effort if model == self.settings.large_model else None
        chunks = [c.chunk for c in context]
        result = self.llm.complete(ANSWER_SYSTEM, answer_user_prompt(question, chunks), model, effort=effort)

        citations: list[Citation] = []
        seen: set[int] = set()
        for m in _CITE_RE.finditer(result.text):
            n = int(m.group(1))
            if 1 <= n <= len(chunks) and n not in seen:
                seen.add(n)
                c = chunks[n - 1]
                body = c.text.split("\n", 1)[-1]
                citations.append(
                    Citation(n=n, chunk_id=c.id, message_id=c.message_id, subject=c.subject,
                             sender=c.sender, date=c.date[:10], snippet=body[:280])
                )
        return Answer(
            question=question,
            text=result.text,
            citations=citations,
            model=result.model,
            route_reason=reason,
            cache_status="miss",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            retrieved_chunk_ids=[c.id for c in chunks],
        )
