"""Test fixtures: a tiny synthetic Enron-style maildir with deliberately restricted emails."""

from __future__ import annotations

from pathlib import Path

import pytest

from clearance.config import Settings
from clearance.embeddings import HashEmbedder
from clearance.ingest.pipeline import run_ingest
from clearance.llm.client import FakeLLM
from clearance.pipeline import RAGService
from clearance.security.acl import Principal

LAY = "kenneth.lay@enron.com"
FASTOW = "andrew.fastow@enron.com"
KAMINSKI = "vince.kaminski@enron.com"
ANALYST = "jane.analyst@enron.com"

RAPTOR_SECRET = "The Raptor vehicles are underwater by 500 million dollars and must be restructured before the third quarter earnings call."


def _msg(n: int, sender: str, to: list[str], subject: str, body: str, date: str, cc: list[str] | None = None) -> str:
    headers = [
        f"Message-ID: <{n}.1075840000000.JavaMail.evans@thyme>",
        f"Date: {date}",
        f"From: {sender}",
        f"To: {', '.join(to)}",
        f"Subject: {subject}",
    ]
    if cc:
        headers.append(f"Cc: {', '.join(cc)}")
    headers += ["Mime-Version: 1.0", "Content-Type: text/plain; charset=us-ascii", "X-From: x", "X-To: y"]
    return "\n".join(headers) + "\n\n" + body + "\n"


# (mailbox, folder, filename, message)
CORPUS = [
    ("fastow-a", "sent", "1.", _msg(1, FASTOW, [LAY], "Raptor hedges",
        f"Ken,\n\n{RAPTOR_SECRET}\n\nPlease keep this between us.\n\nAndy", "Mon, 13 Aug 2001 09:15:00 -0700 (PDT)")),
    # Same email in the CEO's inbox (different Message-ID, same content) -> must dedup and merge owners.
    ("lay-k", "inbox", "1.", _msg(101, FASTOW, [LAY], "Raptor hedges",
        f"Ken,\n\n{RAPTOR_SECRET}\n\nPlease keep this between us.\n\nAndy", "Mon, 13 Aug 2001 09:15:00 -0700 (PDT)")),
    ("kaminski-v", "sent", "1.", _msg(2, KAMINSKI, [FASTOW, LAY], "LJM concerns",
        "My research group believes the LJM partnership creates a conflict of interest. "
        "The Raptor structure is unsound because it is hedged with Enron's own stock.",
        "Tue, 14 Aug 2001 10:00:00 -0700 (PDT)")),
    ("lay-k", "sent", "2.", _msg(3, LAY, [ANALYST], "Company update",
        "Enron stock is an incredible bargain at current prices. Our third quarter is looking great.",
        "Wed, 26 Sep 2001 11:00:00 -0700 (PDT)")),
    ("analyst-j", "sent", "1.", _msg(4, ANALYST, [KAMINSKI], "Model review",
        "The natural gas forward curve model needs recalibration before Friday. Volatility inputs look stale.",
        "Thu, 06 Sep 2001 14:30:00 -0700 (PDT)")),
    # Reply with quoted history: the quoted Raptor text must NOT be indexed under this email's ACL.
    ("lay-k", "sent", "3.", _msg(5, LAY, [FASTOW], "Re: Raptor hedges",
        "Agreed. Schedule a meeting with Jeff Skilling this week.\n\n"
        f"-----Original Message-----\nFrom: Fastow, Andrew\nSent: Monday\n\n{RAPTOR_SECRET}",
        "Mon, 13 Aug 2001 12:00:00 -0700 (PDT)")),
    # Forward: forwarded content is kept, and the forward's recipient may read it.
    ("kaminski-v", "sent", "2.", _msg(6, KAMINSKI, [ANALYST], "Fw: Weather desk",
        "Jane, see below.\n\n---------------------- Forwarded by Vince J Kaminski/HOU/ECT on 09/10/2001 ---------------------------\n\n"
        "The weather derivatives desk launches in Houston next month with a 20 million dollar trading limit.",
        "Mon, 10 Sep 2001 08:00:00 -0700 (PDT)")),
    ("analyst-j", "inbox", "1.", _msg(7, LAY, [ANALYST], "Company update",
        "Enron stock is an incredible bargain at current prices. Our third quarter is looking great.",
        "Wed, 26 Sep 2001 11:00:00 -0700 (PDT)")),
]


def write_maildir(root: Path) -> Path:
    maildir = root / "maildir"
    for mailbox, folder, name, msg in CORPUS:
        path = maildir / mailbox / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(msg, encoding="latin-1")
    return maildir


@pytest.fixture
def maildir(tmp_path: Path) -> Path:
    return write_maildir(tmp_path)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        llm_provider="fake",
        embedder="hash",
        retrieval_mode="hybrid",
        use_graph=True,
        use_reranker=False,
        cache_mode="acl_aware",
        cache_threshold=0.9,
        top_k=4,
    )


@pytest.fixture
def service(settings: Settings, maildir: Path) -> RAGService:
    run_ingest(settings, maildir, embedder=HashEmbedder())
    svc = RAGService.from_settings(settings, llm=FakeLLM(), embedder=HashEmbedder())
    yield svc
    svc.db.close()


@pytest.fixture
def people() -> dict[str, Principal]:
    return {
        "lay": Principal.from_email(LAY),
        "fastow": Principal.from_email(FASTOW),
        "kaminski": Principal.from_email(KAMINSKI),
        "analyst": Principal.from_email(ANALYST),
        "auditor": Principal.from_email("auditor"),
    }
