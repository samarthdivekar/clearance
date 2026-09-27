# Clearance

**Permission-aware GraphRAG over the Enron email corpus**, with a semantic cache that shares
answers between users without leaking data, cost-aware model routing, and an eval harness that
backs every claim with a number.

> Ask *"What were the concerns about the Raptor structures?"* as the CFO and as a junior analyst,
> side by side. Each gets an answer built only from emails they were actually sent.

---

## Why this project exists

Most RAG demos have one user. Real internal-knowledge assistants have thousands, each with
different access. That breaks three things the demos take for granted:

1. **Retrieval must be permission-filtered.** Every retriever (keyword, vector, and graph) has to
   enforce it, or one of them becomes a side channel.
2. **Semantic caching becomes a data leak.** A shared cache keyed on question similarity will
   happily serve the CFO's answer to the analyst. The analyst's own retrieval was filtered
   correctly, and they still read the CFO's data. [How Clearance fixes it ↓](#the-cache-leak)
3. **Multi-hop questions span emails from different people.** Graph traversal connects them,
   but only along edges the asker is allowed to see.

Enron is a good testbed: about 500k real corporate emails where the permissions come built in
(who sent what to whom) and the interesting questions span many threads.

## Features

| | |
|---|---|
| 🔐 **Permission-aware retrieval** | ACL per email (sender, recipients, mailbox owners), inherited by chunks, graph edges and cache entries. Enforced inside the BM25 SQL query, as an exact pre-filter on vectors, and on every graph edge. A final guard re-checks the context before the LLM call. |
| 🧠 **Hybrid + GraphRAG** | BM25 (SQLite FTS5) + dense vectors (BGE) + knowledge-graph traversal, fused with weighted reciprocal-rank fusion; optional cross-encoder reranker |
| 💾 **Leak-proof semantic cache** | 4 modes (`off`, `global`, `per_user`, `acl_aware`). `acl_aware` shares hits across users only when the requester can read every source chunk behind the cached answer |
| 💸 **Cost-aware routing** | Transparent difficulty score sends easy questions to Claude Haiku 4.5 and hard, multi-email ones to Claude Opus 5. Every decision is logged with its reason |
| 📜 **Audit log** | Every query: principal, chunk ids sent to the LLM, citations, cache status, model, tokens, cost, latency. `clearance audit --chunk N` answers "who had this in their context?" |
| 🧪 **Eval harness** | Retrieval ablations (recall@k, MRR, nDCG, split by single-hop and multi-hop), an automated red-team suite, a cache/cost benchmark, and LLM-judged faithfulness |
| 🧰 **Runs offline** | `FakeLLM` + hash embeddings: the whole pipeline, all 42 tests and the security evals run free and deterministically in CI |

## Architecture

```
 question + principal
   │
   ├─ BM25 (ACL in SQL) ─┐
   ├─ vector (ACL mask) ─┼─ weighted RRF ─▶ [rerank] ─▶ ACL guard ─▶ semantic cache ── hit ──▶ answer ($0)
   └─ graph (ACL edges) ─┘                                               │ miss
                                                                        ▼
                                      router ─▶ Claude Haiku 4.5 / Opus 5 ─▶ cited answer ─▶ cache + audit log
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Threat model: [docs/SECURITY.md](docs/SECURITY.md) · Plan: [docs/ROADMAP.md](docs/ROADMAP.md)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml]"
cp .env.example .env                                # add ANTHROPIC_API_KEY

clearance download          # ~423 MB from CMU, extracts 20 key mailboxes (~180k files)
clearance ingest            # parse, dedup, ACL, chunk, embed (BGE-small), build graph
clearance users             # list demo identities
clearance serve             # http://127.0.0.1:8000: two-user comparison UI
```

Ask from the CLI:

```bash
clearance ask "What were Vince Kaminski's concerns about LJM?" --as vince.kaminski@enron.com
clearance ask "What were Vince Kaminski's concerns about LJM?" --as jeff.dasovich@enron.com
```

No API key? Everything runs offline with `CLEARANCE_LLM_PROVIDER=fake CLEARANCE_EMBEDDER=hash`.

## Evals

```bash
clearance eval redteam --n 200                      # cross-user leak attacks, per cache mode (offline, free)
clearance eval cache --n 80                         # cost / latency / hit rate / leaks per cache mode
clearance gen-questions --method llm                # candidate gold questions (review before use)
clearance eval retrieval --gold eval_data/gold.jsonl  # bm25 vs vector vs hybrid vs +graph vs +rerank
clearance eval answers --gold eval_data/gold.jsonl --compare-routing   # LLM judge: small vs large vs routed
```

Reports are written to `reports/*.md` and `reports/*.json`.

## Results

<!-- RESULTS:START -->
**Red-team** (3 real mailboxes: Kaminski, Lay, Skilling; 17,616 unique emails; 100 restricted
emails × 3 probes each: exact, paraphrase, prompt-injection; offline extractive LLM):

| cache configuration | cache leaks | content leaks | legit 2nd reader served from cache |
|---|---|---|---|
| global (naive) | **223 / 300** | **153 / 300** | 100% |
| per_user | 0 | 0 | 0% |
| acl_aware, v1 (context provenance) | 0 | 0 | 7% |
| **acl_aware, v2 (dependency provenance)** | **0** | **0** | **37%** |

The naive cache leaks in about 3 of 4 attacks. The first permission-aware version was leak-free
but shared almost nothing; checking only the chunks an answer depends on raised sharing about 5×
with zero leaks ([iteration log](docs/SECURITY.md#iteration-log-getting-it-to-share-on-real-data)).
Still to do: repeat with the real Claude model (paraphrased leaks can't occur with the offline
model), and retrieval/answer-quality numbers once the hand-reviewed gold set exists.
<!-- RESULTS:END -->

## The cache leak

```
1. CFO:     "What did we decide about the Raptor hedges?"   → answer built from CFO-only emails → cached
2. Analyst: "What was decided on the Raptor hedges?"        → cosine 0.95 → HIT → CFO's answer 💥
```

| Design | Leaks? | Shares hits across users? |
|---|---|---|
| Global cache (tutorial default) | **yes** | yes |
| Per-user cache | no | **no**: hit rate collapses |
| **Source-authorized cache (Clearance)** | **no** | **yes, whenever safe** |

Each cache entry records which chunks its answer **depends on**: the cited ones, plus any
uncited chunk the answer lifted a distinctive term (name, number, acronym) or phrase from, since
models don't always cite what they use. A hit is served only if:

1. the requester can read every chunk the answer depends on, **and**
2. the requester's own best-matching email was in the context the answer was generated from,
   so a user who can see *more* doesn't get a less complete answer.

The first version checked *every* chunk the model was shown. It was safe but shared almost
nothing on real data; see the [iteration log](docs/SECURITY.md#iteration-log-getting-it-to-share-on-real-data).

Retrieval runs before the lookup. It costs milliseconds; the LLM call it saves costs seconds
and cents. "Not found" answers are never shared across users.

The red-team suite demonstrates the leak in `global` mode and verifies zero leaks in `acl_aware`
mode. CI runs it on every push.

## Data-engineering notes (things the real corpus taught me)

- **Each email is stored many times** (sender's sent folder, every recipient's inbox,
  `all_documents`...) with *different* Message-IDs. Dedup is by content hash, and the ACL is the
  union of every mailbox that held a copy.
- **Executives didn't send their own email.** The most frequent sender in Ken Lay's sent folder
  is his assistant. Mailboxes are resolved to owners by surname matching, and aliases
  (`vince.kaminski@enron.com`, `j.kaminski@enron.com`, `vkaminski@aol.com`) are canonicalized.
- **Legal disclaimers swamp retrieval.** Paragraphs repeated in 20+ emails are kept only at their
  first occurrence (about 4k removed from 3 mailboxes alone).
- **Replies quote the whole thread.** Quoted history is stripped so each fact is indexed once,
  under the ACL of the email that carried it. Forwarded content is kept, because forwarding is
  how the information legitimately reached new people.
- **Shouted words look like acronyms** ("PLEASE", "CONCERNS"). Graph entities are dropped when
  the lowercase form is a common word in the corpus.

## Project layout

```
src/clearance/
  ingest/      parser (dedup, identity, boilerplate), chunker, pipeline
  store/       SQLite (FTS5, ACL, graph, audit, cache tables), vector index
  graph/       entity extraction, ACL-filtered traversal
  retrieval/   hybrid retriever, RRF, reranker
  cache/       semantic cache (4 modes)
  llm/         Claude client + offline FakeLLM, router, prompts, pricing
  security/    principals/ACL, audit log + metrics
  evals/       retrieval ablation, red-team, cache bench, LLM judge, question generation
  api/         FastAPI + single-page two-user UI
tests/         42 tests on a synthetic corpus with deliberately restricted emails
```

## Stack

Python 3.12 · SQLite FTS5 · NumPy · PyTorch + Hugging Face `transformers` (BGE-small,
MiniLM cross-encoder) · Claude API (Haiku 4.5, Opus 5, Sonnet 5 as judge) · FastAPI · pytest ·
GitHub Actions

## License

MIT. The Enron corpus is public (released by FERC; distributed by CMU).
