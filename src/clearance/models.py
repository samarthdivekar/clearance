"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Email:
    message_id: str
    date: str  # ISO-8601
    sender: str
    to: list[str]
    cc: list[str]
    bcc: list[str]
    subject: str
    body: str
    owners: set[str] = field(default_factory=set)  # mailbox owners (email addresses)
    folders: set[str] = field(default_factory=set)

    @property
    def recipients(self) -> list[str]:
        return [*self.to, *self.cc, *self.bcc]

    @property
    def thread_key(self) -> str:
        subject = self.subject.lower().strip()
        for prefix in ("re:", "fw:", "fwd:", "re :", "fw :"):
            while subject.startswith(prefix):
                subject = subject[len(prefix):].strip()
        return subject

    def acl(self) -> set[str]:
        """Principals allowed to read this email: the sender, every recipient, and mailbox owners."""
        people = {self.sender, *self.recipients, *self.owners}
        return {f"user:{p}" for p in people if p}


@dataclass
class Chunk:
    id: int
    email_id: int
    ord: int
    text: str
    # Denormalized for prompting / display
    message_id: str = ""
    subject: str = ""
    sender: str = ""
    date: str = ""


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    sources: dict[str, float] = field(default_factory=dict)  # retriever name -> rank/score


@dataclass
class Citation:
    n: int
    chunk_id: int
    message_id: str
    subject: str
    sender: str
    date: str
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    model: str
    route_reason: str
    cache_status: str  # "miss" | "hit" | "bypass"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    retrieved_chunk_ids: list[int] = field(default_factory=list)
    security_events: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": [c.__dict__ for c in self.citations],
            "model": self.model,
            "route_reason": self.route_reason,
            "cache_status": self.cache_status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 1),
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "security_events": self.security_events,
        }
