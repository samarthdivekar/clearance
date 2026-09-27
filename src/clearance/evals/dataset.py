"""Gold question sets (JSONL).

Each line:
    {"id": "q001", "question": "...", "principal": "vince.kaminski@enron.com",
     "relevant_message_ids": ["<...>"], "reference_answer": "...", "hops": 1,
     "source": "human" | "generated", "reviewed": true}

Relevance is judged at the *email* level (message ids), which is stable across re-chunking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GoldQuestion:
    id: str
    question: str
    principal: str
    relevant_message_ids: list[str]
    reference_answer: str = ""
    hops: int = 1
    source: str = "human"
    reviewed: bool = False
    tags: list[str] = field(default_factory=list)


def load(path: Path | str) -> list[GoldQuestion]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            out.append(GoldQuestion(**json.loads(line)))
    return out


def save(questions: list[GoldQuestion], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")
