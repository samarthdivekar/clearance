"""Access control.

Every email carries an ACL: a set of principal tokens such as ``user:jeff.skilling@enron.com``
or ``group:legal``. A principal can read an email when its own tokens intersect the ACL.
Chunks, graph edges and cache entries all inherit ACLs from the emails they came from, so a
single check (`can_read_emails`) guards every path to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WILDCARD = "*"


@dataclass(frozen=True)
class Principal:
    user: str
    groups: frozenset[str] = field(default_factory=frozenset)
    is_auditor: bool = False

    @classmethod
    def from_email(cls, email: str, groups: list[str] | None = None) -> Principal:
        email = email.strip().lower()
        if email in {"auditor", "auditor@enron.com"}:
            return cls(user="auditor@enron.com", is_auditor=True)
        return cls(user=email, groups=frozenset(groups or []))

    def tokens(self) -> set[str]:
        if self.is_auditor:
            return {WILDCARD}
        return {f"user:{self.user}", *(f"group:{g}" for g in self.groups)}

    @property
    def id(self) -> str:
        return self.user


def can_read(principal: Principal, acl: set[str]) -> bool:
    if principal.is_auditor:
        return True
    return bool(principal.tokens() & acl)


class AccessDenied(Exception):
    """Raised when a chunk that the principal cannot read reaches the generation step."""
