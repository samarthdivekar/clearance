"""Generate candidate gold questions from the corpus.

* ``--method llm``: Claude writes a single-hop question per sampled email, and multi-hop
  questions for pairs of emails that share a (non-hub) graph entity. Ground truth (which emails
  answer it) is known by construction. Output is marked ``reviewed: false`` - spot-check and
  edit before trusting the numbers; `eval_data/README.md` describes the review checklist.
* ``--method offline``: known-item questions from headers + a distinctive body phrase. Weaker,
  but free and useful for smoke-testing retrieval.

The principal for each question is a reader of *all* its relevant emails, so the question is
answerable under ACLs.
"""

from __future__ import annotations

import random
import re

from clearance.evals.dataset import GoldQuestion
from clearance.graph.extract import person_label
from clearance.llm.prompts import MULTI_HOP_PROMPT, QUESTION_GEN_SYSTEM, SINGLE_HOP_PROMPT
from clearance.pipeline import RAGService

QA_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
    "additionalProperties": False,
}


def _email_text(row) -> str:
    return f"From: {row['sender']}\nDate: {row['date'][:10]}\nSubject: {row['subject']}\n\n{row['body'][:3000]}"


def _readers(service: RAGService, email_id: int) -> set[str]:
    return {p[5:] for p in service.db.email_acl(email_id) if p.startswith("user:")}


def _pick_principal(readers: set[str], owners: set[str], rng: random.Random) -> str | None:
    preferred = sorted(readers & owners) or sorted(readers)
    return rng.choice(preferred) if preferred else None


def _candidate_emails(service: RAGService, rng: random.Random, n: int):
    rows = service.db.conn.execute(
        "SELECT id, message_id, sender, date, subject, body FROM emails WHERE length(body) BETWEEN 300 AND 6000"
    ).fetchall()
    rng.shuffle(rows)
    return rows[: n * 3]


def generate_offline(service: RAGService, n: int = 100, seed: int = 3) -> list[GoldQuestion]:
    rng = random.Random(seed)
    owners = set(service.db.mailbox_owners())
    out: list[GoldQuestion] = []
    for row in _candidate_emails(service, rng, n):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", row["body"]) if 8 <= len(s.split()) <= 30]
        principal = _pick_principal(_readers(service, row["id"]), owners, rng)
        if not sentences or not principal or not row["subject"]:
            continue
        s = rng.choice(sentences)
        words = s.split()
        phrase = " ".join(words[: min(8, len(words))])
        out.append(GoldQuestion(
            id=f"off{len(out):04d}",
            question=f"What did {person_label(row['sender'])} write about \"{row['subject']}\" mentioning {phrase.lower()}?",
            principal=principal,
            relevant_message_ids=[row["message_id"]],
            reference_answer=s,
            source="generated-offline",
        ))
        if len(out) >= n:
            break
    return out


def generate_llm(service: RAGService, n_single: int = 80, n_multi: int = 40, seed: int = 3) -> list[GoldQuestion]:
    rng = random.Random(seed)
    owners = set(service.db.mailbox_owners())
    model = service.settings.small_model
    out: list[GoldQuestion] = []

    for row in _candidate_emails(service, rng, n_single):
        principal = _pick_principal(_readers(service, row["id"]), owners, rng)
        if not principal:
            continue
        data, _ = service.llm.complete_json(QUESTION_GEN_SYSTEM, SINGLE_HOP_PROMPT.format(email=_email_text(row)), model, QA_SCHEMA)
        if data.get("question"):
            out.append(GoldQuestion(f"s{len(out):04d}", data["question"], principal, [row["message_id"]],
                                    data.get("answer", ""), hops=1, source="generated-llm"))
        if sum(q.hops == 1 for q in out) >= n_single:
            break

    # Multi-hop: two emails from different threads sharing a mid-frequency entity.
    ents = service.db.conn.execute(
        """SELECT m.entity_id, e.label, COUNT(DISTINCT c.email_id) AS n_emails
           FROM mentions m JOIN entities e ON e.id = m.entity_id JOIN chunks c ON c.id = m.chunk_id
           WHERE e.type != 'person' GROUP BY m.entity_id HAVING n_emails BETWEEN 2 AND 30"""
    ).fetchall()
    rng.shuffle(ents)
    for ent in ents:
        emails = service.db.conn.execute(
            """SELECT DISTINCT em.id, em.message_id, em.sender, em.date, em.subject, em.body, em.thread_key
               FROM mentions m JOIN chunks c ON c.id = m.chunk_id JOIN emails em ON em.id = c.email_id
               WHERE m.entity_id = ?""",
            (ent["entity_id"],),
        ).fetchall()
        pairs = [(a, b) for a in emails for b in emails if a["id"] < b["id"] and a["thread_key"] != b["thread_key"]]
        rng.shuffle(pairs)
        for a, b in pairs[:3]:
            common = _readers(service, a["id"]) & _readers(service, b["id"])
            principal = _pick_principal(common, owners, rng)
            if not principal:
                continue
            prompt = MULTI_HOP_PROMPT.format(entity=ent["label"], email_a=_email_text(a), email_b=_email_text(b))
            data, _ = service.llm.complete_json(QUESTION_GEN_SYSTEM, prompt, model, QA_SCHEMA)
            if data.get("question"):
                out.append(GoldQuestion(f"m{len(out):04d}", data["question"], principal,
                                        [a["message_id"], b["message_id"]], data.get("answer", ""),
                                        hops=2, source="generated-llm", tags=[ent["label"]]))
            break
        if sum(q.hops == 2 for q in out) >= n_multi:
            break
    return out
