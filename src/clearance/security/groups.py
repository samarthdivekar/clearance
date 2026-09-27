"""Group-based access: who is in which group, and which emails each group may read.

Two independent halves, defined in a TOML file (default ``config/groups.toml``):

* **Membership** (``members``): resolved on every request from the file, reloaded when it
  changes. Removing someone from a group takes effect on their next query, including for cache
  hits, because every check compares against the requester's *current* tokens.
* **Grants** (``[[groups.<name>.grants]]``): which emails the group may read. Materialized as
  ``group:<name>`` rows in ``email_acl`` by `sync_group_acls` (``clearance acl sync``), so
  retrieval keeps filtering with one indexed JOIN. Re-syncing replaces all group rows, so
  removed grants are revoked too.

A grant rule matches an email if ANY of its fields match:

    mailboxes    = ["lay-k"]                 # every email found in these mailboxes (delegate access)
    folders      = ["kaminski-v/risk*"]      # fnmatch on "<mailbox>/<folder>"
    addresses    = ["legal.dept@enron.com"]  # sent to/from a distribution-list address
    participants = ["mark.haedicke@enron.com"]  # sent or received by these people
    exclude_folders = ["*/personal*"]        # ...unless ANY copy is filed in these folders

The file is security configuration: unknown keys, malformed names and empty rules are errors,
not warnings. A typo must never silently widen (or narrow) access.
"""

from __future__ import annotations

import fnmatch
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from clearance.security.acl import Principal
from clearance.store.db import Database

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_EMAIL_RE = re.compile(r"^[^@\s*?]+@[^@\s*?]+\.[a-z]{2,}$")
_RULE_KEYS = {"mailboxes", "folders", "addresses", "participants", "exclude_folders"}
_GROUP_KEYS = {"description", "members", "grants"}


class GroupConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GrantRule:
    mailboxes: tuple[str, ...] = ()
    folders: tuple[str, ...] = ()
    addresses: tuple[str, ...] = ()
    participants: tuple[str, ...] = ()
    exclude_folders: tuple[str, ...] = ()

    def matches(self, folders: list[str], people: set[str]) -> bool:
        # Exclusion wins if ANY copy is filed there: Enron keeps a copy of almost everything in
        # all_documents, so "all copies excluded" would almost never exclude anything.
        if any(fnmatch.fnmatch(f, pat) for f in folders for pat in self.exclude_folders):
            return False
        mailboxes = {f.split("/", 1)[0] for f in folders}
        return bool(
            (self.mailboxes and mailboxes & set(self.mailboxes))
            or (self.folders and any(fnmatch.fnmatch(f, pat) for f in folders for pat in self.folders))
            or (self.addresses and people & set(self.addresses))
            or (self.participants and people & set(self.participants))
        )


@dataclass(frozen=True)
class Group:
    name: str
    description: str = ""
    members: frozenset[str] = frozenset()
    grants: tuple[GrantRule, ...] = ()


def _str_list(value, where: str, email: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise GroupConfigError(f"{where}: expected a list of strings")
    out = tuple(v.strip().lower() for v in value)
    if email:
        bad = [v for v in out if not _EMAIL_RE.match(v)]
        if bad:
            raise GroupConfigError(f"{where}: not an email address: {bad}")
    return out


def parse_groups(data: dict) -> dict[str, Group]:
    unknown_top = set(data) - {"groups"}
    if unknown_top:
        raise GroupConfigError(f"unknown top-level keys: {sorted(unknown_top)}")
    groups: dict[str, Group] = {}
    for name, spec in (data.get("groups") or {}).items():
        where = f"groups.{name}"
        if not _NAME_RE.match(name):
            raise GroupConfigError(f"{where}: group names must match {_NAME_RE.pattern}")
        if not isinstance(spec, dict):
            raise GroupConfigError(f"{where}: expected a table")
        unknown = set(spec) - _GROUP_KEYS
        if unknown:
            raise GroupConfigError(f"{where}: unknown keys {sorted(unknown)}")
        rules = []
        for i, rule in enumerate(spec.get("grants", [])):
            rwhere = f"{where}.grants[{i}]"
            if not isinstance(rule, dict):
                raise GroupConfigError(f"{rwhere}: expected a table")
            unknown = set(rule) - _RULE_KEYS
            if unknown:
                raise GroupConfigError(f"{rwhere}: unknown keys {sorted(unknown)}")
            parsed = GrantRule(
                mailboxes=_str_list(rule.get("mailboxes", []), f"{rwhere}.mailboxes"),
                folders=_str_list(rule.get("folders", []), f"{rwhere}.folders"),
                addresses=_str_list(rule.get("addresses", []), f"{rwhere}.addresses", email=True),
                participants=_str_list(rule.get("participants", []), f"{rwhere}.participants", email=True),
                exclude_folders=_str_list(rule.get("exclude_folders", []), f"{rwhere}.exclude_folders"),
            )
            if not (parsed.mailboxes or parsed.folders or parsed.addresses or parsed.participants):
                raise GroupConfigError(f"{rwhere}: a grant needs at least one of mailboxes/folders/addresses/participants")
            if any(p in {"*", "*/*"} for p in parsed.folders):
                raise GroupConfigError(f"{rwhere}: a folder pattern matching every email is not allowed; use the auditor role")
            rules.append(parsed)
        groups[name] = Group(
            name=name,
            description=str(spec.get("description", "")),
            members=frozenset(_str_list(spec.get("members", []), f"{where}.members", email=True)),
            grants=tuple(rules),
        )
    return groups


class Directory:
    """Resolves user identities to `Principal`s with their current group memberships."""

    def __init__(self, path: Path | None = None, groups: dict[str, Group] | None = None):
        self.path = Path(path) if path else None
        self._mtime: float | None = None
        self.groups: dict[str, Group] = groups or {}
        if groups is None:
            self._maybe_reload()

    @classmethod
    def from_toml(cls, text: str) -> Directory:
        return cls(groups=parse_groups(tomllib.loads(text)))

    def _maybe_reload(self) -> None:
        if self.path is None:
            return
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime == self._mtime:
            return
        self.groups = parse_groups(tomllib.loads(self.path.read_text(encoding="utf-8"))) if mtime else {}
        self._mtime = mtime

    def groups_for(self, user: str) -> frozenset[str]:
        self._maybe_reload()
        user = user.strip().lower()
        return frozenset(g.name for g in self.groups.values() if user in g.members)

    def principal(self, email: str) -> Principal:
        base = Principal.from_email(email)
        if base.is_auditor:
            return base
        return Principal(user=base.user, groups=self.groups_for(base.user))


def sync_group_acls(db: Database, directory: Directory) -> dict[str, int]:
    """Replace every ``group:*`` row in email_acl with what the current grants imply."""
    directory._maybe_reload()
    conn = db.conn
    rows = conn.execute("SELECT id, sender, recipients, folders FROM emails").fetchall()
    grants: list[tuple[int, str]] = []
    counts: dict[str, int] = {name: 0 for name in directory.groups}
    for r in rows:
        rec = json.loads(r["recipients"] or "{}")
        people = {r["sender"], *rec.get("to", []), *rec.get("cc", []), *rec.get("bcc", [])}
        folders = json.loads(r["folders"] or "[]")
        for g in directory.groups.values():
            if any(rule.matches(folders, people) for rule in g.grants):
                grants.append((r["id"], f"group:{g.name}"))
                counts[g.name] += 1
    with conn:
        conn.execute("DELETE FROM email_acl WHERE principal LIKE 'group:%'")
        conn.executemany("INSERT OR IGNORE INTO email_acl (email_id, principal) VALUES (?, ?)", grants)
        db.bump_acl_version()
    return counts


@dataclass
class Explanation:
    readable: bool
    principal_tokens: list[str]
    acl: list[str]
    matched: list[str]
    reasons: list[str] = field(default_factory=list)


def explain(db: Database, directory: Directory, principal: Principal, message_id: str) -> Explanation:
    """Why can (or can't) this principal read this email?"""
    row = db.conn.execute("SELECT id FROM emails WHERE message_id = ?", (message_id,)).fetchone()
    if row is None:
        raise KeyError(f"no email with message id {message_id!r}")
    acl = db.email_acl(row["id"])
    tokens = principal.tokens()
    if principal.is_auditor:
        return Explanation(True, sorted(tokens), sorted(acl), ["*"], ["auditor role reads everything"])
    matched = sorted(tokens & acl)
    reasons = []
    for tok in matched:
        if tok.startswith("user:"):
            reasons.append(f"{tok[5:]} is the sender, a recipient, or a mailbox owner of this email")
        else:
            g = directory.groups.get(tok[6:])
            reasons.append(f"member of group '{tok[6:]}'" + (f" ({g.description})" if g and g.description else ""))
    return Explanation(bool(matched), sorted(tokens), sorted(acl), matched, reasons)
