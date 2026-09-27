from clearance.embeddings import HashEmbedder
from clearance.graph.extract import extract_text_entities, person_label
from clearance.llm.router import Router
from clearance.retrieval.retriever import RetrievalConfig, rrf


def test_rrf_rewards_agreement():
    fused = rrf({"a": [(1, 9.0), (2, 8.0)], "b": [(2, 0.9), (3, 0.8)]})
    assert fused[0][0] == 2
    assert fused[0][2] == {"a": 2, "b": 1}


def test_hash_embedder_similarity():
    e = HashEmbedder()
    v = e.embed(["raptor hedges underwater", "raptor hedges are underwater", "gas curve model"])
    assert v[0] @ v[1] > v[0] @ v[2]


def test_entity_extraction():
    ents = {label for label, _ in extract_text_entities("header\nThe LJM partnership and Chewco Investments LP met with Arthur Andersen.")}
    assert "LJM" in ents
    assert any("Chewco" in e for e in ents)


def test_person_label():
    assert person_label("jeff.skilling@enron.com") == "Jeff Skilling"


def test_graph_seeds_match_people_by_surname(service):
    labels = service.retriever.graph.describe("What did Kaminski say?")
    assert "Vince Kaminski" in labels


def test_hybrid_finds_relevant_email(service, people):
    cfg = RetrievalConfig(mode="hybrid", use_graph=True, top_k=3)
    results = service.retriever.retrieve("gas forward curve recalibration", people["kaminski"], cfg)
    assert "recalibration" in results[0].chunk.text


def test_router_sends_simple_questions_small_and_hard_ones_large(service, people):
    router = Router("small", "large", threshold=0.35)
    cfg = RetrievalConfig(mode="hybrid", use_graph=True, top_k=6)
    simple = "gas curve model"
    hard = "Compare why Kaminski and Fastow disagreed about Raptor and LJM, and explain the timeline and the risk"
    assert router.route(simple, service.retriever.retrieve(simple, people["kaminski"], cfg)).tier == "small"
    assert router.route(hard, service.retriever.retrieve(hard, people["lay"], cfg)).tier == "large"


def test_answer_has_citations(service, people):
    ans = service.ask("Why is the Raptor structure unsound?", people["kaminski"])
    assert ans.citations
    assert all(c.message_id for c in ans.citations)
    assert ans.cost_usd > 0 and ans.model
