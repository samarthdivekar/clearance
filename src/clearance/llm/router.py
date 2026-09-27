"""Cost-aware model routing.

A transparent, feature-based difficulty score decides whether a question goes to the small
(cheap, fast) model or the large one. Features are computed from the question *and* the
retrieved context, so "What did Vince say about Raptor?" (one relevant email) routes small,
while "Compare what Vince and Jeff said about Raptor over 2001" (many emails, comparison,
timeline) routes large. Every decision is logged with its reason, and `clearance eval router`
measures quality/cost against always-small and always-large baselines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clearance.models import ScoredChunk

_MULTI_HOP_CUES = re.compile(
    r"\b(why|compare|comparison|difference|relationship|timeline|over time|evolve|changed|"
    r"both|between|before|after|who .* that|which .* that|summari[sz]e|overall|explain|"
    r"implications?|risk|strategy)\b",
    re.I,
)
_THREAD_PREFIX = re.compile(r"^((re|fw|fwd)\s*:\s*)+")


@dataclass
class RouteDecision:
    model: str
    score: float
    reason: str
    tier: str  # "small" | "large"


class Router:
    def __init__(self, small_model: str, large_model: str, threshold: float = 0.5):
        self.small_model = small_model
        self.large_model = large_model
        self.threshold = threshold

    @staticmethod
    def features(question: str, context: list[ScoredChunk]) -> dict[str, float]:
        words = len(question.split())
        distinct_emails = len({c.chunk.email_id for c in context})
        distinct_threads = len({_THREAD_PREFIX.sub("", c.chunk.subject.lower()) for c in context})
        graph_hits = sum(1 for c in context if "graph" in c.sources)
        return {
            "length": min(words / 30, 1.0),
            "reasoning_cues": min(len(_MULTI_HOP_CUES.findall(question)) / 2, 1.0),
            "spread": min(max(distinct_threads - 1, 0) / 5, 1.0),
            "multi_email": min(max(distinct_emails - 2, 0) / 6, 1.0),
            "graph": min(graph_hits / 4, 1.0),
        }

    WEIGHTS = {"length": 0.15, "reasoning_cues": 0.40, "spread": 0.20, "multi_email": 0.10, "graph": 0.15}

    def route(self, question: str, context: list[ScoredChunk]) -> RouteDecision:
        f = self.features(question, context)
        score = sum(self.WEIGHTS[k] * v for k, v in f.items())
        top = sorted(f.items(), key=lambda kv: -self.WEIGHTS[kv[0]] * kv[1])[:2]
        why = ", ".join(f"{k}={v:.2f}" for k, v in top)
        if score >= self.threshold:
            return RouteDecision(self.large_model, score, f"difficulty {score:.2f} ≥ {self.threshold} ({why})", "large")
        return RouteDecision(self.small_model, score, f"difficulty {score:.2f} < {self.threshold} ({why})", "small")
