"""Tightened cache provenance: shares more, but must still catch facts lifted from uncited chunks."""

import re

from clearance.cache.provenance import answer_dependencies, distinctive_terms
from clearance.cache.semantic import SemanticCache
from clearance.llm.client import LLMResult
from clearance.models import Chunk

Q = "What is wrong with the Raptor structure?"


class ScriptedLLM:
    """Cites the context email containing `cite_word`, and says `answer_template` with that number."""

    def __init__(self, cite_word: str, answer_template: str):
        self.cite_word = cite_word
        self.answer_template = answer_template

    def complete(self, system, user, model, max_tokens=16000, effort=None):
        n = next(n for n, body in re.findall(r'<email n="(\d+)">\n(.*?)\n</email>', user, re.S) if self.cite_word in body)
        return LLMResult(self.answer_template.format(n=n), model, 100, 20)


def _use(service, llm, provenance):
    service.llm = llm
    service.settings = service.settings.with_overrides(cache_provenance=provenance)
    service.cache = SemanticCache(service.db, mode="acl_aware", threshold=0.8)


def test_distinctive_terms():
    terms = distinctive_terms("The deal lost $500 million in 2001 via LJM, per Andrew Fastow [2].")
    assert {"$500", "2001", "ljm", "andrew", "fastow"} <= terms
    assert "deal" not in terms and "the" not in terms


def test_dependencies_include_uncited_chunk_the_answer_lifted_a_fact_from():
    a = Chunk(1, 1, 0, "From: x | Subject: LJM\nThe structure is unsound.")
    b = Chunk(2, 2, 0, "From: y | Subject: Raptor\nThe vehicles are underwater by 500 million dollars.")
    c = Chunk(3, 3, 0, "From: z | Subject: Lunch\nSandwiches at noon.")
    deps = answer_dependencies("It is unsound [1] and 500 million dollars underwater.", Q, [a, b, c], {1})
    assert deps == [1, 2]
    assert answer_dependencies("It is unsound [1].", Q, [a, b, c], {1}) == [1]


def test_lifted_fact_blocks_cache_hit_for_user_who_cannot_read_its_source(service, people):
    # Cites Kaminski's LJM email, but states the $500M figure that is only in Fastow's private email.
    _use(service, ScriptedLLM("unsound", "The structure is unsound [{n}] and underwater by 500 million dollars."), "dependencies")
    service.ask(Q, people["fastow"])
    attempt = service.ask(Q, people["kaminski"])  # Kaminski can read the LJM email, not the Raptor one
    assert attempt.cache_status != "hit"
    assert any(e.startswith("CACHE_ACL_BLOCKED") for e in attempt.security_events)


def test_dependency_provenance_shares_where_context_provenance_refuses(service, people):
    llm = ScriptedLLM("unsound", "Kaminski's group judged the structure unsound [{n}].")

    _use(service, llm, "context")
    service.ask(Q, people["fastow"])
    assert service.ask(Q, people["kaminski"]).cache_status != "hit"  # context held Fastow-only email

    _use(service, llm, "dependencies")
    service.cache.clear()
    service.ask(Q, people["fastow"])
    shared = service.ask(Q, people["kaminski"])
    assert shared.cache_status == "hit"
    assert shared.cost_usd == 0
    assert not service.db.unreadable_among(people["kaminski"], shared.retrieved_chunk_ids)

