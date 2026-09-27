# Eval data

| File | What | Trust |
|---|---|---|
| `gold.jsonl` | Hand-written or hand-reviewed questions. **The numbers you report come from this file.** | High |
| `generated.jsonl` | Output of `clearance gen-questions`. Candidates to review and copy into `gold.jsonl`. | Unreviewed |

## Format (one JSON object per line)

```json
{"id": "q001", "question": "Why did Vince Kaminski object to the LJM transactions?",
 "principal": "vince.kaminski@enron.com",
 "relevant_message_ids": ["12345.1075840000000.JavaMail.evans@thyme"],
 "reference_answer": "His group judged the structure unsound and a conflict of interest.",
 "hops": 1, "source": "human", "reviewed": true, "tags": ["ljm"]}
```

* `principal` must be able to read **every** relevant email. Otherwise the question measures
  the ACL, not retrieval.
* `hops: 2` means the answer needs facts from two or more emails in different threads.
* Look up message ids with `sqlite3 data/clearance.db "select message_id, subject from emails where subject like '%LJM%'"`.

## Review checklist for generated questions

1. Is it answerable **only** from the listed emails (not from general knowledge about Enron)?
2. Does it avoid leaking the answer in the question?
3. For multi-hop: does it really need both emails?
4. Is the reference answer correct and short?
5. Set `"reviewed": true` and move it to `gold.jsonl`.

Aim for 150 questions: 100 single-hop and 50 multi-hop. Around 50 good ones is already enough to
separate the retrieval configs.
