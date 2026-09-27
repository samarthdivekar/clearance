# Architecture

```
                 ┌──────────────────────── ingest (offline) ───────────────────────┐
 CMU maildir ──▶ │ parse + dedup (content hash) ─▶ ACL per email ─▶ email-aware    │
 (517k files)    │ strip quoted replies, keep forwards               chunking      │
                 │        │                                            │           │
                 │        ▼                                            ▼           │
                 │  SQLite: emails, email_acl, chunks, FTS5      embeddings (.npz) │
                 │        │                                                        │
                 │        ▼                                                        │
                 │  knowledge graph: entities, mentions, edges (each → source chunk)│
                 └─────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────── query (online) ─────────────────────────┐
 question ─────▶ │ BM25 (ACL in SQL) ─┐                                             │
 + principal     │ vector (ACL mask) ─┼─ RRF ─▶ [rerank] ─▶ ACL guard ─▶ cache? ──┐ │
                 │ graph (ACL edges) ─┘                                  hit │    │ │
                 │                                                          ▼    │ │
                 │                                   router ─▶ Claude ─▶ cite ─▶ │ │
                 │                                   (small/large)   store cache  │ │
                 │                                                          ▼    ▼ │
                 │                                                 audit log ─▶ answer
                 └─────────────────────────────────────────────────────────────────┘
```

## Modules

| Path | Responsibility |
|---|---|
| `ingest/parser.py` | Maildir → `Email`; dedup by content hash (Enron stores each email many times with different Message-IDs); owner address resolution; reply stripping |
| `ingest/chunker.py` | Header-prefixed chunks (who/when/subject live in headers, not bodies), paragraph packing with overlap |
| `store/db.py` | SQLite schema, FTS5 BM25 with ACL JOIN, readable-set queries |
| `store/vectors.py` | Exact dot-product index with ACL pre-filter |
| `graph/extract.py` | People from headers, orgs/acronyms/code names from text (df-filtered), co-mention edges |
| `graph/retriever.py` | Seed-entity matching (incl. surnames), ACL-filtered 2-hop BFS, idf-weighted chunk scoring |
| `retrieval/retriever.py` | Runs retrievers, fuses with RRF, optional cross-encoder rerank |
| `cache/semantic.py` | `off` / `global` / `per_user` / `acl_aware` semantic cache |
| `llm/router.py` | Feature-based difficulty score → small or large model, with logged reason |
| `llm/client.py` | Claude client (adaptive thinking + effort on the large model, server-side refusal fallback); offline `FakeLLM` |
| `pipeline.py` | The query path; defense-in-depth ACL guard; citation parsing |
| `security/audit.py` | Audit log, metrics (p50/p95, cost, hit rate), "who accessed chunk N" |
| `evals/` | Retrieval ablation, red-team, cache benchmark, LLM-judged answer quality, question generation |
| `api/` | FastAPI + single-file two-user comparison UI |

## Key decisions

**SQLite + exact vectors instead of a vector DB.** At ≤200k chunks brute-force search is a few
milliseconds, and pre-filtering is exact. ANN indexes that filter after the graph walk can return
fewer than k results for low-access users, which quietly skews both quality and eval numbers.
The `VectorIndex` interface is small enough to swap for Qdrant/pgvector with native payload
filtering when the corpus outgrows RAM.

**ACL at the email level, inherited everywhere.** One source of truth. Chunks, edges and cache
entries never carry their own permissions, so they can't drift.

**RRF over score blending.** BM25, cosine and graph scores are on incomparable scales; rank
fusion needs no tuning and is robust when one retriever returns nothing (common for graph).

**Heuristic graph extraction by default, LLM extraction optional.** Headers give an exact
people graph for free. Pattern-based org/code-name extraction with a document-frequency floor is
noisy but cheap enough to run on the whole corpus; the eval harness measures whether the graph
actually helps multi-hop recall rather than assuming it does.

**Router is transparent, not learned (yet).** A weighted feature score with a logged reason is
debuggable and good enough to show the cost curve. The roadmap's next step is to fit the weights
on judge labels from `eval answers --compare-routing`.

**Everything runs offline.** `CLEARANCE_LLM_PROVIDER=fake` + `CLEARANCE_EMBEDDER=hash` gives a
deterministic, free pipeline for tests and CI. The security evals use it because the properties
under test (retrieval, ACL, cache) don't depend on the model.
