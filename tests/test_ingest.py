from clearance.ingest.chunker import chunk_email
from clearance.ingest.parser import clean_body, parse_maildir
from clearance.models import Email

from .conftest import ANALYST, FASTOW, KAMINSKI, LAY, RAPTOR_SECRET


def test_dedup_merges_mailbox_owners(maildir):
    emails, owners = parse_maildir(maildir, progress=False)
    raptor = [e for e in emails if e.subject == "Raptor hedges"]
    assert len(raptor) == 1, "same email in two mailboxes must be stored once"
    assert raptor[0].owners == {FASTOW, LAY}


def test_owner_address_resolved_from_sent_folder(maildir):
    _, owners = parse_maildir(maildir, progress=False)
    assert owners["fastow-a"] == FASTOW
    assert owners["kaminski-v"] == KAMINSKI
    assert owners["analyst-j"] == ANALYST


def test_aliases_are_canonicalized(tmp_path):
    from .conftest import _msg

    box = tmp_path / "maildir" / "kaminski-v"
    (box / "sent").mkdir(parents=True)
    (box / "sent" / "1.").write_text(_msg(1, KAMINSKI, [LAY], "a", "Body text one here.", "Mon, 13 Aug 2001 09:15:00 -0700"))
    (box / "inbox").mkdir()
    (box / "inbox" / "1.").write_text(_msg(2, LAY, ["vkaminski@aol.com"], "b", "Body text two here.", "Tue, 14 Aug 2001 09:15:00 -0700"))
    emails, _ = parse_maildir(tmp_path / "maildir", progress=False)
    to_aol = next(e for e in emails if e.subject == "b")
    assert to_aol.to == [KAMINSKI]


def test_boilerplate_keeps_first_occurrence():
    from clearance.ingest.parser import strip_boilerplate

    footer = "This message is confidential and may be privileged. If you are not the intended recipient please delete it now."
    emails = [Email(str(i), f"2001-01-{i + 1:02d}", "a@x.com", [], [], [], "s", f"Unique body {i}.\n\n{footer}") for i in range(25)]
    assert strip_boilerplate(emails, min_count=20) == 24
    assert footer in emails[0].body and footer not in emails[1].body


def test_eight_bit_headers_parse():
    from clearance.ingest.parser import parse_message

    raw = b"Message-ID: <x>\nDate: Mon, 13 Aug 2001 09:15:00 -0700\nFrom: a@enron.com\nTo: b@enron.com\nSubject: Caf\xe9 meeting\n\nSee you there."
    em = parse_message(raw)
    assert em is not None and em.to == ["b@enron.com"] and "meeting" in em.subject


def test_reply_quotes_are_stripped(maildir):
    emails, _ = parse_maildir(maildir, progress=False)
    reply = next(e for e in emails if e.subject == "Re: Raptor hedges")
    assert "Skilling" in reply.body
    assert "underwater" not in reply.body


def test_forwarded_content_is_kept():
    body = "FYI\n\n----- Forwarded by X on 1/1/2001 -----\n\nDesk launches in Houston."
    assert "Desk launches in Houston" in clean_body(body)


def test_acl_includes_sender_recipients_and_owners():
    em = Email("id", "2001-01-01", "a@enron.com", ["b@enron.com"], ["c@enron.com"], [], "s", "body", owners={"d@enron.com"})
    assert em.acl() == {"user:a@enron.com", "user:b@enron.com", "user:c@enron.com", "user:d@enron.com"}


def test_chunks_carry_header_context():
    em = Email("id", "2001-08-13T09:15:00", FASTOW, [LAY], [], [], "Raptor hedges", RAPTOR_SECRET)
    chunks = chunk_email(em)
    assert len(chunks) == 1
    assert chunks[0].startswith(f"From: {FASTOW}")
    assert "Subject: Raptor hedges" in chunks[0]


def test_long_bodies_split_with_overlap():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    em = Email("id", "", FASTOW, [LAY], [], [], "long", body)
    chunks = chunk_email(em, max_words=150, overlap_words=20)
    assert len(chunks) > 3
    assert all(len(c.split("\n", 1)[1].split()) <= 150 + 20 for c in chunks)
