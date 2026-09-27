"""CLI smoke tests: ingest the synthetic maildir and run commands end to end, offline."""

import pytest

from clearance import cli
from clearance.evals import report


@pytest.fixture
def env(tmp_path, maildir, monkeypatch):
    monkeypatch.setenv("CLEARANCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLEARANCE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CLEARANCE_EMBEDDER", "hash")
    monkeypatch.setattr(report, "REPORTS_DIR", tmp_path / "reports")
    cli.main(["ingest", "--maildir", str(maildir), "--users", "all", "--embedder", "hash"])
    return tmp_path


def test_ask(env, capsys):
    cli.main(["ask", "Why is the Raptor structure unsound?", "--as", "vince.kaminski@enron.com"])
    out = capsys.readouterr().out
    assert "unsound" in out and "cache=miss" in out


def test_eval_redteam_and_cache(env):
    cli.main(["eval", "redteam", "--n", "5"])
    cli.main(["eval", "cache", "--n", "3"])
    assert (env / "reports" / "redteam.md").exists()
    assert (env / "reports" / "cache_bench.md").exists()


def test_gen_questions_offline_then_retrieval_eval(env):
    out = env / "gen.jsonl"
    cli.main(["gen-questions", "--method", "offline", "--n-single", "3", "--out", str(out)])
    if out.read_text().strip():
        cli.main(["eval", "retrieval", "--gold", str(out)])
