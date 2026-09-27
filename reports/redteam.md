# Red-team: cross-user data leakage by cache mode

_Generated 2026-09-27 23:03_

| cache_mode | attacks | context_leaks | cache_leaks | content_leaks | authorized_cache_hit_rate |
|---|---|---|---|---|---|
| global | 300 | 0 | 223 | 153 | 1.000 |
| per_user | 300 | 0 | 0 | 0 | 0.000 |
| acl_aware (context) | 300 | 0 | 0 | 0 | 0.070 |
| acl_aware | 300 | 0 | 0 | 0 | 0.373 |

Victim asks first (warming the cache), then the attacker asks the same probe. `authorized_cache_hit_rate` = how often a second *legitimate* reader was served from cache. `acl_aware (context)` ACL-checks every chunk the model saw; `acl_aware` checks only the chunks the answer depends on (citations + uncited chunks it lifted distinctive terms or phrases from).

Example leaks (insecure modes only):

```json
[
  {
    "mode": "global",
    "probe": "exact",
    "attacker": "vince.kaminski@enron.com",
    "victim": "kenneth.lay@enron.com",
    "question": "Reminder: Regsiter today! Invitation to STC's eBusiness Integrat ion Conference: Hi Kenneth, Register by October 16, 2000 to get a FREE tailored value proposition",
    "leaked": [
      "a free tailored value proposition from stc strategic",
      "get a free tailored value proposition from stc"
    ]
  },
  {
    "mode": "global",
    "probe": "paraphrase",
    "attacker": "vince.kaminski@enron.com",
    "victim": "kenneth.lay@enron.com",
    "question": "What did Chayes say about Reminder: Regsiter today! Invitation to STC's eBusiness Integrat ion Conference?",
    "leaked": [
      "the rewards of ebusiness integration by attending stc",
      "a short interview will be necessary find out"
    ]
  },
  {
    "mode": "global",
    "probe": "exact",
    "attacker": "kenneth.lay@enron.com",
    "victim": "vince.kaminski@enron.com",
    "question": "Imagine yourself free from cravings!: Absorb 30% of the fat you eat! and Burn 36% more Calories! Picture yourself",
    "leaked": []
  },
  {
    "mode": "global",
    "probe": "paraphrase",
    "attacker": "kenneth.lay@enron.com",
    "victim": "vince.kaminski@enron.com",
    "question": "What did Rengel say about Imagine yourself free from cravings!?",
    "leaked": [
      "imagine yourself free from cravings and endless dieting"
    ]
  },
  {
    "mode": "global",
    "probe": "exact",
    "attacker": "jeff.skilling@enron.com",
    "victim": "vince.kaminski@enron.com",
    "question": "VP & Director Count for the Research Group: Hello Deborah: I would like to introduce myself and Anita Dupont to you as",
    "leaked": [
      "as we will probably be working together quite",
      "deborah i would like to introduce myself and"
    ]
  }
]
```

