"""End-to-end answer quality with an LLM judge, and router vs. single-model baselines.

For each gold question the pipeline answers as the question's principal; a judge model scores
faithfulness (claims supported by the retrieved context), relevance, and correctness against
the reference answer. Running with --compare-routing replays the set with always-small,
always-large and routed generation to show the quality/cost trade-off.

Requires CLEARANCE_LLM_PROVIDER=anthropic (the judge must be a real model).
"""

from __future__ import annotations

from clearance.evals import report
from clearance.evals.dataset import GoldQuestion
from clearance.evals.metrics import mean
from clearance.llm.prompts import JUDGE_PROMPT, JUDGE_SYSTEM, format_context
from clearance.pipeline import RAGService
from clearance.security.acl import Principal

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "relevance": {"type": "number"},
        "correct": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["faithfulness", "relevance", "correct", "rationale"],
    "additionalProperties": False,
}


def judge(service: RAGService, q: GoldQuestion, answer_text: str, chunk_ids: list[int]) -> dict:
    chunks = list(service.db.get_chunks(chunk_ids).values())
    ref = f"<reference_answer>{q.reference_answer}</reference_answer>\n" if q.reference_answer else ""
    prompt = JUDGE_PROMPT.format(
        context=format_context(chunks), question=q.question, answer=answer_text, reference_block=ref
    )
    data, _ = service.llm.complete_json(JUDGE_SYSTEM, prompt, service.settings.judge_model, JUDGE_SCHEMA)
    return data


def run(service: RAGService, questions: list[GoldQuestion], compare_routing: bool = False) -> list[dict]:
    s = service.settings
    variants = [("routed", None)]
    if compare_routing:
        variants = [("always small", s.small_model), ("always large", s.large_model), ("routed", None)]
    original_mode = service.cache.mode
    service.cache.mode = "off"  # measure generation, not cache
    rows = []
    try:
        for label, force in variants:
            faith, rel, correct, cost, large = [], [], [], 0.0, 0
            for q in questions:
                ans = service.ask(q.question, Principal.from_email(q.principal), force_model=force)
                cost += ans.cost_usd
                large += ans.model == s.large_model
                grade = judge(service, q, ans.text, ans.retrieved_chunk_ids)
                faith.append(float(grade["faithfulness"]))
                rel.append(float(grade["relevance"]))
                correct.append(1.0 if grade["correct"] else 0.0)
            n = max(len(questions), 1)
            rows.append({
                "generation": label,
                "faithfulness": mean(faith),
                "relevance": mean(rel),
                "correct": mean(correct),
                "large_share": large / n,
                "cost_per_query_usd": cost / n,
            })
            print(f"  {label:<14} {rows[-1]}")
    finally:
        service.cache.mode = original_mode
    report.write(
        "answer_quality",
        "Answer quality (LLM judge) and routing trade-off",
        rows,
        notes=f"Judge: {s.judge_model}. {len(questions)} questions. Faithfulness = share of claims supported by retrieved context.",
    )
    return rows
