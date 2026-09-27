# Roadmap

The 5-week plan. Every milestone ends with a number you can put on a resume.

## Week 1: Baseline RAG with permissions ✅ scaffolded
- [x] Maildir parser: dedup, owner resolution, reply stripping
- [x] Email-aware chunking, SQLite + FTS5, embeddings
- [x] Permission-filtered BM25 + vector + RRF
- [x] CLI `ask`, API, two-user UI
- [ ] Ingest the default 20-mailbox subset with `bge-small` embeddings
- [ ] **Hand-write 50 gold questions** (`eval_data/gold.jsonl`), 15+ of them multi-hop
- [ ] Record baseline `recall@5` for bm25 / vector / hybrid

## Week 2: Security hardening
- [x] Audit log + `who accessed chunk N`
- [x] Defense-in-depth ACL guard
- [x] Automated red-team suite (exact / paraphrase / injection probes)
- [ ] Run red-team at `--n 500` on the real corpus; record "0 leaks across N attacks"
- [ ] Add group-based ACLs (e.g. `group:legal` from a mapping file) and tests
- [ ] Write-up: *"The semantic cache bug that leaks your data"*

### Next up (found on real data)
- [ ] **Raise safe cache sharing above ~5%.** Try provenance tightening: `sources = cited ∪ {uncited
      chunks with 5-gram overlap with the answer}`. Re-run `eval redteam` with the real LLM
      (paraphrase leaks can't show up with the extractive FakeLLM) and plot leaks vs. authorized hit rate.

## Week 3: GraphRAG
- [x] Header + heuristic entity graph, ACL-filtered traversal
- [ ] Generate 40 multi-hop questions (`gen-questions --method llm`), review them by hand
- [ ] Ablation: hybrid vs hybrid+graph on multi-hop recall
- [ ] Optional: LLM entity extraction on the top 5k chunks; compare against the heuristic graph

## Week 4: Cost and latency
- [x] Semantic cache (4 modes) + router + cost accounting
- [ ] `eval cache` on real Claude API calls; record cost saved and p95
- [ ] `eval answers --compare-routing`: quality vs cost for small / large / routed
- [ ] Fit router weights on judge labels (logistic regression on the 5 features)
- [ ] Prompt caching for the static system prompt once it passes the minimum cacheable length

## Week 5: Ship it
- [ ] Deploy (Fly.io / Render / a small VM) with the Docker image
- [ ] Record a 60-second demo GIF of the two-user view
- [ ] Final README numbers, blog post, resume bullets

## Stretch
- Swap `VectorIndex` for Qdrant with payload filtering; benchmark at full 500k emails
- GitHub connector (public/private repos map naturally onto ACLs)
- Streaming answers in the UI
- Time-aware retrieval ("what did we know as of Aug 2001?")
