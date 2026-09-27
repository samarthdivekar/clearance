"""Parse the CMU Enron maildir (`maildir/<user>/<folder>/<n>.`) into deduplicated `Email`s.

Things that make this corpus messy, and how they are handled:

* The same email is stored many times (sender's `sent`, `all_documents`, each recipient's
  inbox...), each copy with a *different* Message-ID. We dedup on a content key
  (sender, date, subject, body hash) and merge the mailbox owners of every copy into the ACL.
* Replies quote the whole thread. Quoted text (``-----Original Message-----``, ``>`` lines,
  "On ... wrote:") is stripped so each fact is indexed once, under the ACL of the email that
  originally carried it. Forwarded content is *kept*: forwarding is how information legitimately
  reached new people, so the forward's recipients should be able to retrieve it.
* Mailbox directory names (``skilling-j``) are not email addresses, executives' sent folders are
  full of mail their assistants wrote, and people use several addresses. `resolve_owner` maps
  each mailbox to a canonical address plus aliases, and every email is canonicalized.
* Legal disclaimers and signatures repeat across thousands of emails and swamp retrieval.
  `strip_boilerplate` keeps only their first occurrence.
"""

from __future__ import annotations

import email
import email.policy
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from clearance.models import Email

SENT_FOLDERS = {"sent", "sent_items", "_sent_mail", "sent_mail"}

_REPLY_MARKERS = [
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}", re.I | re.M),
    re.compile(r"^\s*On .{5,120} wrote:\s*$", re.I | re.M),
    re.compile(r"^\s*_{10,}\s*$", re.M),  # Outlook separator line
    re.compile(r"^\s*From:\s.+\n\s*Sent:\s", re.I | re.M),
]
_FORWARD_HEADER = re.compile(r"^\s*-{5,}\s*Forwarded by .*?-{5,}\s*$", re.I | re.M)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_ADDRESS_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)


def _header(msg, name: str) -> str:
    """Header value as str. compat32 returns `Header` objects for undecodable (8-bit) values."""
    value = msg.get(name)
    return "" if value is None else str(value)


def normalize_address(addr: str) -> str:
    addr = addr.strip().strip("<>\"'").lower()
    return addr if _ADDRESS_RE.match(addr) else ""


def parse_addresses(value: str | None) -> list[str]:
    if not value:
        return []
    value = re.sub(r"\s+", " ", value)
    out = []
    for _, addr in getaddresses([value]):
        norm = normalize_address(addr)
        if norm and norm not in out:
            out.append(norm)
    return out


def clean_body(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    # Cut at the first reply marker (quoted history of earlier emails).
    cut = len(body)
    for pattern in _REPLY_MARKERS:
        m = pattern.search(body)
        if m and m.start() < cut:
            cut = m.start()
    body = body[:cut]
    lines = [ln for ln in body.split("\n") if not ln.lstrip().startswith(">")]
    body = "\n".join(lines)
    body = _FORWARD_HEADER.sub("[Forwarded message]", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def parse_message(raw: bytes, owner_dir: str = "", folder: str = "") -> Email | None:
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    sender_list = parse_addresses(_header(msg, "From"))
    if not sender_list:
        return None
    try:
        date = parsedate_to_datetime(_header(msg, "Date")).isoformat()
    except (TypeError, ValueError, IndexError):
        date = ""
    payload = msg.get_payload(decode=True) if not msg.is_multipart() else None
    if payload is None:
        parts = [p.get_payload(decode=True) or b"" for p in msg.walk() if p.get_content_type() == "text/plain"]
        payload = b"\n".join(parts)
    body = clean_body(payload.decode("latin-1", errors="replace"))
    subject = re.sub(r"\s+", " ", _header(msg, "Subject") or "").strip()
    message_id = (_header(msg, "Message-ID") or "").strip().strip("<>")
    em = Email(
        message_id=message_id,
        date=date,
        sender=sender_list[0],
        to=parse_addresses(_header(msg, "To")),
        cc=parse_addresses(_header(msg, "Cc")),
        bcc=parse_addresses(_header(msg, "Bcc")),
        subject=subject,
        body=body,
    )
    if folder:
        em.folders.add(f"{owner_dir}/{folder}")
    return em


def content_key(em: Email) -> str:
    h = hashlib.sha1()
    for part in (em.sender, em.date, em.subject, em.body):
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def iter_mailbox_files(user_dir: Path) -> Iterator[tuple[str, Path]]:
    for path in sorted(user_dir.rglob("*")):
        if path.is_file():
            folder = path.parent.relative_to(user_dir).as_posix()
            yield folder, path


def resolve_owner(user_dir: Path, sample: int = 300) -> tuple[str, set[str]]:
    """Resolve a mailbox directory (``lay-k``) to (canonical address, aliases).

    The naive rule "most frequent From in sent items" picks assistants: Ken Lay's and Jeff
    Skilling's sent folders are dominated by the assistants who wrote on their behalf. Instead
    we collect addresses from sent-From and inbox-To, keep those whose local part contains the
    surname from the directory name, and prefer the one that also starts with the first
    initial on an enron.com domain. Every matching address becomes an alias of the owner
    (e.g. vince.kaminski@enron.com, j.kaminski@enron.com, vkaminski@aol.com).
    """
    surname, _, initial = user_dir.name.partition("-")
    counts: Counter[str] = Counter()
    for sub in user_dir.iterdir():
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if name not in SENT_FOLDERS and name != "inbox":
            continue
        for i, path in enumerate(p for p in sub.iterdir() if p.is_file()):
            if i >= sample:
                break
            msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.compat32)
            header = "From" if name in SENT_FOLDERS else "To"
            counts.update(parse_addresses(_header(msg, header))[:1 if header == "From" else None])
    matching = [a for a, _ in counts.most_common() if surname and surname in a.split("@")[0]]

    def rank(addr: str) -> tuple:
        local, domain = addr.split("@")
        return (domain != "enron.com", not (initial and local.startswith(initial)), "." not in local, -counts[addr])

    if matching:
        canonical = sorted(matching, key=rank)[0]
        return canonical, set(matching)
    if counts:
        top = counts.most_common(1)[0][0]
        return top, {top}
    fallback = f"{user_dir.name}@enron.com"
    return fallback, {fallback}


def strip_boilerplate(emails: list[Email], min_count: int = 20, min_words: int = 12) -> int:
    """Remove paragraphs repeated across many emails (legal disclaimers, signatures, footers).

    The first occurrence (by date) is kept, so a widely forwarded announcement stays
    retrievable once instead of flooding every result list.
    """
    def norm(p: str) -> str:
        return re.sub(r"\s+", " ", p).strip().lower()

    counts: Counter[str] = Counter()
    for em in emails:
        counts.update({norm(p) for p in _PARAGRAPH_SPLIT.split(em.body) if len(p.split()) >= min_words})
    frequent = {p for p, n in counts.items() if n >= min_count}
    if not frequent:
        return 0
    seen: set[str] = set()
    removed = 0
    for em in sorted(emails, key=lambda e: e.date):
        kept = []
        for p in _PARAGRAPH_SPLIT.split(em.body):
            key = norm(p)
            if key in frequent:
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
            kept.append(p)
        em.body = "\n\n".join(kept).strip()
    return removed


def _canonicalize(em: Email, alias_map: dict[str, str]) -> None:
    def c(addr: str) -> str:
        return alias_map.get(addr, addr)

    em.sender = c(em.sender)
    em.to = list(dict.fromkeys(c(a) for a in em.to))
    em.cc = list(dict.fromkeys(c(a) for a in em.cc))
    em.bcc = list(dict.fromkeys(c(a) for a in em.bcc))


def parse_maildir(
    maildir: Path,
    users: Iterable[str] | None = None,
    max_emails: int | None = None,
    progress: bool = True,
) -> tuple[list[Email], dict[str, str]]:
    """Parse and dedup mailboxes. Returns (emails, {mailbox_dir: owner_address})."""
    maildir = Path(maildir)
    user_dirs = sorted(p for p in maildir.iterdir() if p.is_dir())
    if users:
        wanted = set(users)
        missing = wanted - {p.name for p in user_dirs}
        if missing and progress:
            print(f"  skipping mailboxes not found in {maildir}: {', '.join(sorted(missing))}")
        user_dirs = [p for p in user_dirs if p.name in wanted]

    by_key: dict[str, Email] = {}
    owners: dict[str, str] = {}
    alias_map: dict[str, str] = {}
    for user_dir in user_dirs:
        owner, aliases = resolve_owner(user_dir)
        owners[user_dir.name] = owner
        alias_map.update({a: owner for a in aliases})
        n_files = 0
        for folder, path in iter_mailbox_files(user_dir):
            try:
                em = parse_message(path.read_bytes(), user_dir.name, folder)
            except Exception as e:  # one malformed file must not abort a 500k-file ingest
                if progress:
                    print(f"  skipped unparseable {path}: {type(e).__name__}: {e}")
                continue
            if em is None or (not em.body and not em.subject):
                continue
            n_files += 1
            key = content_key(em)
            existing = by_key.get(key)
            if existing is None:
                em.owners.add(owner)
                by_key[key] = em
            else:
                existing.owners.add(owner)
                existing.folders |= em.folders
            if max_emails and len(by_key) >= max_emails:
                break
        if progress:
            print(f"  {user_dir.name:<16} owner={owner:<36} files={n_files:>6}  unique so far={len(by_key)}")
        if max_emails and len(by_key) >= max_emails:
            break

    # Sampling can miss rarely used aliases (vkaminski@aol.com): also match every address in the
    # corpus against "<first initial>[given name][._]<surname>@" for each mailbox owner.
    patterns = {
        owner: re.compile(rf"^{re.escape(d.partition('-')[2][:1])}[a-z]*[._]?{re.escape(d.partition('-')[0])}@")
        for d, owner in owners.items() if d.partition("-")[2]
    }
    for em in by_key.values():
        for addr in (em.sender, *em.recipients):
            if addr not in alias_map:
                for owner, pat in patterns.items():
                    if pat.match(addr):
                        alias_map[addr] = owner
                        break
    for em in by_key.values():
        _canonicalize(em, alias_map)
    removed = strip_boilerplate(list(by_key.values()))
    if progress:
        print(f"  identity: {len(alias_map)} addresses -> {len(owners)} owners; boilerplate paragraphs removed: {removed}")

    # Some copies lack a Message-ID; synthesize a stable one from the content key.
    emails = []
    seen_ids: set[str] = set()
    for key, em in by_key.items():
        if not em.message_id or em.message_id in seen_ids:
            em.message_id = f"{key}@clearance.local"
        seen_ids.add(em.message_id)
        emails.append(em)
    emails.sort(key=lambda e: e.date)
    return emails, owners
