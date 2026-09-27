"""Group-based permissions: grants, exclusions, revocation, cache and audit behavior."""

import os
import time

import pytest

from clearance import cli
from clearance.cache.semantic import SemanticCache
from clearance.retrieval.retriever import RetrievalConfig
from clearance.security import audit
from clearance.security.groups import (
    Directory,
    GrantRule,
    GroupConfigError,
    explain,
    sync_group_acls,
)

from .conftest import FASTOW

ASSISTANT = "rosalee.fleming@enron.com"
ASSISTANT_2 = "second.assistant@enron.com"
OUTSIDER = "someone.else@enron.com"
RAPTOR_Q = "Are the Raptor vehicles underwater before the earnings call?"
BM25 = RetrievalConfig(mode="bm25", use_graph=False, top_k=10)

EXEC_OFFICE = f"""
[groups.exec-office]
description = "Lay's office"
members = ["{ASSISTANT}", "{ASSISTANT_2}"]

[[groups.exec-office.grants]]
mailboxes = ["lay-k"]
"""


def _write(service, text: str) -> None:
    path = service.settings.groups_path
    path.write_text(text, encoding="utf-8")
    stamp = time.time() + (getattr(_write, "n", 0) + 1)  # force a distinct mtime on every write
    _write.n = getattr(_write, "n", 0) + 1
    os.utime(path, (stamp, stamp))


def _configure(service, text: str) -> dict[str, int]:
    _write(service, text)
    return sync_group_acls(service.db, service.directory)


def _sees_raptor(service, email: str) -> bool:
    results = service.retriever.retrieve(RAPTOR_Q, service.principal(email), BM25)
    return any("underwater" in r.chunk.text for r in results)


# ------------------------------------------------------------------ config validation

@pytest.mark.parametrize(
    "bad, message",
    [
        ('[groups.x]\nmembrs = ["a@enron.com"]', "unknown keys"),
        ('[groups."Bad Name"]\nmembers = []', "group names"),
        ('[groups.x]\nmembers = ["not-an-email"]', "not an email"),
        ("[groups.x]\n[[groups.x.grants]]\nexclude_folders = ['*/personal']", "at least one"),
        ("[groups.x]\n[[groups.x.grants]]\nfolders = ['*']", "every email"),
        ("[groups.x]\n[[groups.x.grants]]\nmailbox = ['lay-k']", "unknown keys"),
        ('[group.x]\nmembers = []', "unknown top-level"),
    ],
)
def test_invalid_config_fails_loudly(bad, message):
    with pytest.raises(GroupConfigError, match=message):
        Directory.from_toml(bad)


def test_grant_rule_fields():
    folders = ["lay-k/inbox", "lay-k/all_documents"]
    people = {"a@enron.com", "legal.dept@enron.com"}
    assert GrantRule(mailboxes=("lay-k",)).matches(folders, people)
    assert GrantRule(folders=("lay-k/in*",)).matches(folders, people)
    assert GrantRule(addresses=("legal.dept@enron.com",)).matches(folders, people)
    assert GrantRule(participants=("a@enron.com",)).matches(folders, people)
    assert not GrantRule(mailboxes=("skilling-j",)).matches(folders, people)
    # Exclusion wins if ANY copy is filed in an excluded folder.
    assert not GrantRule(mailboxes=("lay-k",), exclude_folders=("*/inbox",)).matches(folders, people)


# ------------------------------------------------------------------ grants

def test_group_grant_gives_access_to_non_recipient(service, people):
    assert not _sees_raptor(service, ASSISTANT)
    counts = _configure(service, EXEC_OFFICE)
    assert counts["exec-office"] > 0
    assert service.principal(ASSISTANT).groups == {"exec-office"}
    assert _sees_raptor(service, ASSISTANT)
    assert not _sees_raptor(service, OUTSIDER)


def test_exclusion_blocks_grant(service):
    _configure(service, EXEC_OFFICE + 'exclude_folders = ["lay-k/inbox*"]\n')
    assert not _sees_raptor(service, ASSISTANT)  # the Raptor email is filed in lay-k/inbox


def test_membership_revocation_is_immediate(service):
    _configure(service, EXEC_OFFICE)
    assert _sees_raptor(service, ASSISTANT)
    _write(service, EXEC_OFFICE.replace(f'"{ASSISTANT}", ', ""))  # remove from group, no re-sync
    assert service.principal(ASSISTANT).groups == frozenset()
    assert not _sees_raptor(service, ASSISTANT)


def test_grant_revocation_needs_only_resync(service):
    _configure(service, EXEC_OFFICE)
    assert _sees_raptor(service, ASSISTANT)  # readable set is now cached in the retriever
    before = service.db.acl_version()
    _configure(service, EXEC_OFFICE.replace('mailboxes = ["lay-k"]', 'mailboxes = ["analyst-j"]'))
    assert service.db.acl_version() > before
    assert not _sees_raptor(service, ASSISTANT)  # cache invalidated by the version bump


# ------------------------------------------------------------------ cache

def test_cache_shared_within_group_and_revoked_with_membership(service, people):
    _configure(service, EXEC_OFFICE)
    service.cache = SemanticCache(service.db, mode="acl_aware", threshold=0.8)
    assert service.ask(RAPTOR_Q, service.principal(ASSISTANT)).cache_status == "miss"
    assert service.ask(RAPTOR_Q, service.principal(ASSISTANT_2)).cache_status == "hit"

    outsider = service.ask(RAPTOR_Q, service.principal(OUTSIDER))
    assert outsider.cache_status != "hit" and "underwater" not in outsider.text

    _write(service, EXEC_OFFICE.replace(f', "{ASSISTANT_2}"', ""))
    revoked = service.ask(RAPTOR_Q, service.principal(ASSISTANT_2))
    assert revoked.cache_status != "hit" and "underwater" not in revoked.text


# ------------------------------------------------------------------ explain, audit, CLI

def test_explain(service, people):
    _configure(service, EXEC_OFFICE)
    msg_id = service.db.conn.execute("SELECT message_id FROM emails WHERE subject = 'Raptor hedges'").fetchone()[0]
    via_group = explain(service.db, service.directory, service.principal(ASSISTANT), msg_id)
    assert via_group.readable and via_group.matched == ["group:exec-office"]
    assert "exec-office" in via_group.reasons[0]
    direct = explain(service.db, service.directory, service.principal(FASTOW), msg_id)
    assert direct.readable and direct.matched == [f"user:{FASTOW}"]
    assert not explain(service.db, service.directory, service.principal(OUTSIDER), msg_id).readable


def test_audit_records_groups_at_query_time(service):
    _configure(service, EXEC_OFFICE)
    service.ask(RAPTOR_Q, service.principal(ASSISTANT))
    assert audit.recent(service.db, 1)[0]["principal_groups"] == ["exec-office"]


def test_cli_acl_commands(tmp_path, maildir, monkeypatch, capsys):
    groups = tmp_path / "groups.toml"
    groups.write_text(EXEC_OFFICE, encoding="utf-8")
    monkeypatch.setenv("CLEARANCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLEARANCE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("CLEARANCE_EMBEDDER", "hash")
    monkeypatch.setenv("CLEARANCE_GROUPS_FILE", str(groups))
    cli.main(["ingest", "--maildir", str(maildir), "--users", "all", "--embedder", "hash"])  # syncs groups
    assert "group grants" in capsys.readouterr().out
    cli.main(["acl", "groups"])
    assert ASSISTANT in capsys.readouterr().out
    cli.main(["acl", "sync"])
    assert "group:exec-office" in capsys.readouterr().out
    cli.main(["acl", "explain", "1.1075840000000.JavaMail.evans@thyme", "--as", ASSISTANT])
    out = capsys.readouterr().out
    assert out.startswith("CAN read") and "exec-office" in out


def test_repo_groups_file_is_valid():
    d = Directory(__import__("pathlib").Path(__file__).parents[1] / "config" / "groups.toml")
    assert {"exec-office-lay", "exec-office-skilling", "research", "legal"} <= set(d.groups)
