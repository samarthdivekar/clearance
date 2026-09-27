# Security model

Clearance treats "who can see what" as a correctness property, not a feature. This document is
the threat model, the defenses, and how each is tested.

## Principals and ACLs

* A **principal** is a user (`user:jeff.skilling@enron.com`) plus optional groups
  (`group:legal`). The special principal `auditor` can read everything, including the audit log.
* Every **email** gets an ACL at ingest: its sender, every To/Cc/Bcc recipient, and every
  mailbox owner whose mailbox contained a copy. That mirrors who could actually read it in 2001.
* **Chunks, graph edges, and cache entries** inherit ACLs from the emails they came from. There is
  no second permission system to drift out of sync.

## Groups

`config/groups.toml` defines groups in two halves that change on different clocks:

| Half | Example | Stored | Takes effect |
|---|---|---|---|
| **Membership**: who is in the group | `members = ["rosalee.fleming@enron.com"]` | the file, reloaded on change | next query |
| **Grants**: which emails the group reads | `mailboxes = ["lay-k"]`, `folders = ["kaminski-v/var*"]`, `addresses`, `participants`, `exclude_folders` | `group:<name>` rows in `email_acl`, written by `clearance acl sync` | next query after sync |

Grants are materialized into the same `email_acl` table as per-user ACLs, so retrieval still
filters with one indexed JOIN and every downstream check (graph, cache, guard) works on groups
unchanged. Design choices:

* **Revocation is the hard case, so it is tested.** Removing a member takes effect on the next
  query with no re-sync, including cache hits, because the cache re-checks the requester's
  *current* tokens (`test_membership_revocation_is_immediate`,
  `test_cache_shared_within_group_and_revoked_with_membership`). Re-syncing bumps an ACL version
  that the retriever checks on every query, so cached readable sets are dropped even when the
  sync ran in another process (`test_grant_revocation_needs_only_resync`).
* **Exclusions win if *any* copy matches.** Enron keeps a copy of almost everything in
  `all_documents`, so "exclude if all copies are in personal/" would exclude almost nothing.
* **The config fails loudly.** Unknown keys, malformed names or addresses, empty rules and
  match-everything patterns (`folders = ["*"]`) are errors. A typo in a security file must never
  silently widen or narrow access. "Read everything" is the auditor role, not a group.
* **Explainable.** `clearance acl explain <message-id> --as <user>` prints whether and *why* a
  user can read an email (named on it, mailbox owner, or which group).
* **The audit log records groups at query time**, so an investigation can tell what access a
  user had *when* they asked, even after membership changes.

On the 3-mailbox index, `config/groups.toml` grants Ken Lay's assistant delegated access to
3,640 of his emails (her readable chunks go from 1,023 to 6,459), and none of the 828 emails
filed in Kaminski's `personal`/`resumes` folders reach his research group.

## Threats and defenses

| # | Threat | Defense | Test |
|---|---|---|---|
| T1 | Retrieval returns a chunk the user can't read | ACL is a JOIN inside the BM25 SQL query; vector search scores only readable rows (pre-filter, exact) | `test_bm25_respects_acl`, `test_vector_respects_acl` |
| T2 | Graph traversal walks through a forbidden email ("Fastow → Raptor" edge from a private email) | Every edge and mention stores its source chunk; traversal only follows edges whose chunk is readable | `test_graph_respects_acl` |
| T3 | A bug in any retriever lets a chunk through | Defense in depth: before generation, the full context is re-checked against the DB (`unreadable_among`) and violations are dropped and logged as `ACL_GUARD_BLOCKED` | pipeline guard |
| **T4** | **Semantic cache serves user A's answer to user B** | `acl_aware` cache: an entry records the chunks its answer depends on (citations + uncited chunks it lifted terms/phrases from); a hit requires the requester to read every one | `test_naive_global_cache_leaks`, `test_acl_aware_cache_blocks_leak`, red-team suite |
| T5 | Cached answer is *incomplete* for a user who can see more | Hit also requires the requester's top retrieved email to have been in the original answer's context | `coverage_k` / `min_coverage` in `SemanticCache` |
| T6 | "Not found" cached for one user suppresses a real answer for another | Answers without sources are never shared across users | `test_not_found_answers_are_not_shared` |
| T7 | Prompt injection inside an email ("ignore previous instructions, reveal...") | Emails are wrapped as `<context>` data; system prompt says instructions in emails are text to report on. Crucially, the LLM only ever *has* authorized context, so even a successful injection can't reveal unauthorized data | red-team `injection` probes |
| T8 | User asks the model to "act as auditor" | Identity comes from the auth layer (`Principal`), never from the prompt | red-team `injection` probes |
| T10 | A removed group member keeps access (via stale caches or cached answers) | Membership resolved per request; ACL version bump invalidates readable-set caches; cache re-checks current tokens | `test_membership_revocation_is_immediate`, `test_grant_revocation_needs_only_resync` |
| T11 | A config typo silently grants broad access | Strict schema validation; match-everything patterns rejected | `test_invalid_config_fails_loudly` |
| T9 | Nobody can reconstruct who saw what after an incident | Append-only audit log of every query: principal, chunk ids sent to the LLM, cited ids, cache status, model, cost. `clearance audit --chunk N` answers "who had chunk N in their context?" | `test_audit_log_records_every_query` |

## Why the cache is the interesting part

A retrieval-only permission check is not enough once a semantic cache exists:

```
1. CFO:     "What did we decide about the Raptor hedges?"   → answer built from CFO-only emails → cached
2. Analyst: "What was decided on the Raptor hedges?"        → cosine 0.95 → HIT → CFO's answer
```

The analyst's own retrieval was correctly filtered and returned nothing sensitive, and they still
read the CFO's data. The leak is in the *cache key*: it encodes the question but not the
permission scope the answer was built under.

Options considered:

| Design | Leaks? | Shares hits across users? | Notes |
|---|---|---|---|
| Global cache | **Yes** | Yes | The common tutorial implementation |
| Key by user | No | No | Hit rate collapses; popular questions re-pay the LLM for each user |
| Key by exact ACL set of the user | No | Rarely | Users almost never have identical permission sets in email |
| **Source-authorized (Clearance)** | **No** | **Yes, whenever safe** | Hit iff requester can read every chunk the answer depends on, and the answer considered their best evidence |

The source-authorized design checks the *answer's provenance* against the *requester's
permissions* at lookup time, so two users with different overall access still share a cache
entry whenever the entry only depends on emails both can read.

Retrieval runs **before** the cache lookup. That costs ~10–50 ms, but the saved LLM call costs
seconds and cents, and it provides the completeness check (T5) for free.

### Iteration log: getting it to share on real data

**v1: provenance = the whole context.** Each entry recorded all 8 chunks sent to the LLM. It was
safe, but on 3 real mailboxes a legitimate second reader asking the *identical* question was served
from cache only **5.7%** of the time (`global`: 100%, with 223/300 attacks leaking). Diagnosis over
90 repeats: 60 were refused by the ACL check, because the first reader's context nearly always
contained some email the second reader was never sent. Yet answers cited only **1.3 chunks** on
average.

**v2: provenance = what the answer depends on** (`cache/provenance.py`). Each entry keeps two sets:

| Set | Contents | Used for |
|---|---|---|
| `source_chunks` | cited chunks ∪ uncited chunks the answer demonstrably drew on | ACL check (T4) |
| `context_chunks` | every chunk the model was shown | completeness check (T5) |

An uncited chunk is pulled into `source_chunks` if the answer contains a **distinctive term**
(number, amount, acronym, non-initial capitalized word) found in that chunk but not in the
question or the cited chunks, or shares a **4-word run of content words** with it. So an answer
that cites email A but states "$500M underwater" from uncited email B depends on B, and a user
who can't read B is refused (`test_lifted_fact_blocks_cache_hit_for_user_who_cannot_read_its_source`).

The completeness check also had to change. "≥60% of the requester's top-3 emails were in the
original context" blocked legitimate hits, because ranks 2–3 are mostly weak matches that differ
between users with different mailboxes. It is now "the requester's **top-1** email was in the
original context": if the second user has better evidence than the first answer considered,
they get a fresh answer.

**Residual risk.** A fact reworded with no names, numbers or shared phrasing ("the vehicles lost
most of their value") can come from an uncited chunk undetected. The extractive offline LLM used
by the red-team can't produce such paraphrases, so the red-team must also be run with the real
model (`clearance eval redteam --llm anthropic`) before relying on it. Set
`CLEARANCE_CACHE_PROVENANCE=context` for the strict v1 behavior.

**Result** (3 real mailboxes, 100 restricted emails × 3 probes, offline extractive LLM):

| cache configuration | cache leaks | content leaks | legit 2nd reader served from cache |
|---|---|---|---|
| global (naive) | **223 / 300** | **153 / 300** | 100% |
| per_user | 0 | 0 | 0% |
| acl_aware, v1 (context provenance) | 0 | 0 | 7% |
| **acl_aware, v2 (dependency provenance)** | **0** | **0** | **37%** |

Sharing rose about 5× with no leaks detected by this suite. The real-LLM run is still pending (see residual risk above).

## Known limitations

* **Existence side channel.** Response latency (hit vs. miss) could hint that *someone* asked a
  similar question. Mitigation if needed: pad cache-hit latency, or don't reveal cache status to
  the client (the demo reveals it on purpose).
* **ACL changes.** Revoking access takes effect immediately for retrieval and cache (checked at
  lookup time), but answers already shown to a user can't be recalled.
* **Demo authentication.** The API trusts an `X-Principal` header so the UI can switch users.
  Production must derive the principal from a verified session/JWT; only `current_principal`
  in `api/app.py` changes.
* **Derived-data leakage via the graph build.** Entity *names* are global (the entity table is not
  ACL'd), but an entity is only reachable if a readable chunk mentions it, and traversal never
  surfaces unreadable chunks. Autocomplete over entity names would need its own ACL filter.
