# Red-team: cross-user data leakage by cache mode

_Generated 2026-09-27 22:15_

| cache_mode | attacks | context_leaks | cache_leaks | content_leaks | authorized_cache_hit_rate |
|---|---|---|---|---|---|
| global | 120 | 0 | 85 | 61 | 1.000 |
| per_user | 120 | 0 | 0 | 0 | 0.000 |
| acl_aware | 120 | 0 | 0 | 0 | 0.050 |

Victim asks first (warming the cache), then the attacker asks the same probe. `authorized_cache_hit_rate` = how often a second *legitimate* reader was served from cache.

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
      "october 16 2000 to get a free tailored",
      "kenneth register by october 16 2000 to get"
    ]
  },
  {
    "mode": "global",
    "probe": "paraphrase",
    "attacker": "vince.kaminski@enron.com",
    "victim": "kenneth.lay@enron.com",
    "question": "What did Chayes say about Reminder: Regsiter today! Invitation to STC's eBusiness Integrat ion Conference?",
    "leaked": [
      "can reap the rewards of ebusiness integration by",
      "available a short interview will be necessary find"
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
      "and anita dupont to you as we will",
      "would like to introduce myself and anita dupont"
    ]
  }
]
```

