"""SQLite storage: emails, chunks, ACLs, full-text index, knowledge graph, audit log and cache.

SQLite keeps the project zero-infrastructure. FTS5 gives BM25 for free, and ACL filtering is a
plain JOIN, so the permission check happens *inside* the retrieval query rather than after it.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from clearance.models import Chunk, Email
from clearance.security.acl import Principal

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS emails (
    id          INTEGER PRIMARY KEY,
    message_id  TEXT UNIQUE NOT NULL,
    date        TEXT,
    sender      TEXT,
    recipients  TEXT,          -- JSON {"to": [...], "cc": [...], "bcc": [...]}
    subject     TEXT,
    body        TEXT,
    thread_key  TEXT,
    owners      TEXT,          -- JSON list of mailbox owners
    folders     TEXT           -- JSON list of folders
);
CREATE INDEX IF NOT EXISTS idx_emails_thread ON emails(thread_key);

CREATE TABLE IF NOT EXISTS email_acl (
    email_id   INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    principal  TEXT NOT NULL,
    PRIMARY KEY (email_id, principal)
);
CREATE INDEX IF NOT EXISTS idx_acl_principal ON email_acl(principal);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    email_id  INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    ord       INTEGER NOT NULL,
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_email ON chunks(email_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);

-- Knowledge graph. Every edge and mention points at the chunk it was extracted from, so the
-- graph inherits that chunk's ACL: a principal can only traverse edges they could read.
CREATE TABLE IF NOT EXISTS entities (
    id    INTEGER PRIMARY KEY,
    name  TEXT UNIQUE NOT NULL,   -- normalized (lowercase)
    label TEXT NOT NULL,          -- display form
    type  TEXT NOT NULL           -- person | org | project | other
);
CREATE TABLE IF NOT EXISTS mentions (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id  INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(chunk_id);
CREATE TABLE IF NOT EXISTS edges (
    src      INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst      INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    rel      TEXT NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

-- Bumped whenever ACL rows change (group sync); caches of readable sets compare against it.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id               INTEGER PRIMARY KEY,
    ts               TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    principal        TEXT NOT NULL,
    principal_groups TEXT,     -- JSON list of the principal's groups at query time
    question         TEXT NOT NULL,
    retrieved        TEXT,     -- JSON list of chunk ids sent to the LLM
    cited            TEXT,     -- JSON list of chunk ids cited in the answer
    cache_status     TEXT,
    model            TEXT,
    route_reason     TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    cost_usd         REAL,
    latency_ms       REAL,
    security_events  TEXT      -- JSON list
);

CREATE TABLE IF NOT EXISTS cache_entries (
    id             INTEGER PRIMARY KEY,
    created_at     REAL NOT NULL,
    owner          TEXT NOT NULL,   -- principal who produced the entry
    question       TEXT NOT NULL,
    embedding      BLOB NOT NULL,
    answer         TEXT NOT NULL,   -- JSON serialized Answer payload
    source_chunks  TEXT NOT NULL,   -- JSON list of chunk ids the answer depends on (ACL check)
    context_chunks TEXT,            -- JSON list of every chunk id the model was shown (coverage check)
    hits           INTEGER DEFAULT 0
);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(cache_entries)")}
        if "context_chunks" not in cols:
            self.conn.execute("ALTER TABLE cache_entries ADD COLUMN context_chunks TEXT")
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(audit_log)")}
        if "principal_groups" not in cols:
            self.conn.execute("ALTER TABLE audit_log ADD COLUMN principal_groups TEXT")
        self.conn.commit()

    def acl_version(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'acl_version'").fetchone()
        return int(row[0]) if row else 0

    def bump_acl_version(self) -> None:
        self.conn.execute(
            """INSERT INTO meta (key, value) VALUES ('acl_version', '1')
               ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"""
        )

    # One connection per thread: FastAPI serves requests from a thread pool.
    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------ ingest
    def reset_corpus(self) -> None:
        c = self.conn
        for table in ("edges", "mentions", "entities", "cache_entries", "chunks", "email_acl", "emails"):
            c.execute(f"DELETE FROM {table}")
        c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self.bump_acl_version()
        c.commit()

    def insert_email(self, email: Email, acl: Iterable[str]) -> int:
        cur = self.conn.execute(
            """INSERT INTO emails (message_id, date, sender, recipients, subject, body, thread_key, owners, folders)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                email.message_id,
                email.date,
                email.sender,
                json.dumps({"to": email.to, "cc": email.cc, "bcc": email.bcc}),
                email.subject,
                email.body,
                email.thread_key,
                json.dumps(sorted(email.owners)),
                json.dumps(sorted(email.folders)),
            ),
        )
        email_id = cur.lastrowid
        self.conn.executemany(
            "INSERT OR IGNORE INTO email_acl (email_id, principal) VALUES (?, ?)",
            [(email_id, p) for p in acl],
        )
        return email_id

    def insert_chunk(self, email_id: int, ord_: int, text: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO chunks (email_id, ord, text) VALUES (?, ?, ?)", (email_id, ord_, text)
        )
        self.conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, text))
        return cur.lastrowid

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ reads
    def chunk_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def email_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]

    def all_chunks(self) -> list[tuple[int, str]]:
        return [(r["id"], r["text"]) for r in self.conn.execute("SELECT id, text FROM chunks ORDER BY id")]

    def get_chunks(self, ids: Iterable[int]) -> dict[int, Chunk]:
        ids = list(ids)
        if not ids:
            return {}
        out: dict[int, Chunk] = {}
        for batch_start in range(0, len(ids), 500):
            batch = ids[batch_start : batch_start + 500]
            q = f"""SELECT c.id, c.email_id, c.ord, c.text, e.message_id, e.subject, e.sender, e.date
                    FROM chunks c JOIN emails e ON e.id = c.email_id
                    WHERE c.id IN ({",".join("?" * len(batch))})"""
            for r in self.conn.execute(q, batch):
                out[r["id"]] = Chunk(
                    id=r["id"], email_id=r["email_id"], ord=r["ord"], text=r["text"],
                    message_id=r["message_id"], subject=r["subject"] or "", sender=r["sender"] or "",
                    date=r["date"] or "",
                )
        return out

    def chunk_email_ids(self, chunk_ids: Iterable[int]) -> dict[int, int]:
        ids = sorted(set(chunk_ids))
        if not ids:
            return {}
        q = f"SELECT id, email_id FROM chunks WHERE id IN ({','.join('?' * len(ids))})"
        return {r["id"]: r["email_id"] for r in self.conn.execute(q, ids)}

    def email_ids_for_message_ids(self, message_ids: Iterable[str]) -> dict[str, int]:
        ids = list(message_ids)
        if not ids:
            return {}
        q = f"SELECT id, message_id FROM emails WHERE message_id IN ({','.join('?' * len(ids))})"
        return {r["message_id"]: r["id"] for r in self.conn.execute(q, ids)}

    def get_email(self, email_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()

    def email_acl(self, email_id: int) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT principal FROM email_acl WHERE email_id = ?", (email_id,))}

    # ------------------------------------------------------------------ access control
    def readable_chunk_ids(self, principal: Principal) -> set[int]:
        if principal.is_auditor:
            return {r[0] for r in self.conn.execute("SELECT id FROM chunks")}
        tokens = sorted(principal.tokens())
        q = f"""SELECT c.id FROM chunks c
                WHERE c.email_id IN (SELECT email_id FROM email_acl WHERE principal IN ({",".join("?" * len(tokens))}))"""
        return {r[0] for r in self.conn.execute(q, tokens)}

    def unreadable_among(self, principal: Principal, chunk_ids: Iterable[int]) -> set[int]:
        """Return the subset of `chunk_ids` the principal may NOT read (used as a final guard)."""
        chunk_ids = set(chunk_ids)
        if principal.is_auditor or not chunk_ids:
            return set()
        tokens = sorted(principal.tokens())
        ids = sorted(chunk_ids)
        q = f"""SELECT DISTINCT c.id FROM chunks c JOIN email_acl a ON a.email_id = c.email_id
                WHERE c.id IN ({",".join("?" * len(ids))}) AND a.principal IN ({",".join("?" * len(tokens))})"""
        readable = {r[0] for r in self.conn.execute(q, [*ids, *tokens])}
        return chunk_ids - readable

    # ------------------------------------------------------------------ BM25
    @staticmethod
    def fts_query(text: str) -> str:
        words = [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOPWORDS]
        return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))

    def bm25_search(self, query: str, principal: Principal, limit: int) -> list[tuple[int, float]]:
        match = self.fts_query(query)
        if not match:
            return []
        if principal.is_auditor:
            q = """SELECT rowid AS id, bm25(chunks_fts) AS s FROM chunks_fts
                   WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?"""
            rows = self.conn.execute(q, (match, limit))
        else:
            tokens = sorted(principal.tokens())
            # ACL filter is part of the query: unauthorized rows never leave SQLite.
            q = f"""SELECT f.rowid AS id, bm25(chunks_fts) AS s
                    FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                    WHERE chunks_fts MATCH ?
                      AND c.email_id IN (SELECT email_id FROM email_acl
                                         WHERE principal IN ({",".join("?" * len(tokens))}))
                    ORDER BY s LIMIT ?"""
            rows = self.conn.execute(q, (match, *tokens, limit))
        # FTS5 bm25() is "lower is better"; flip sign so higher is better everywhere.
        return [(r["id"], -r["s"]) for r in rows]

    # ------------------------------------------------------------------ principals
    def top_principals(self, limit: int = 25) -> list[tuple[str, int]]:
        q = """SELECT substr(principal, 6) AS user, COUNT(*) AS n FROM email_acl
               WHERE principal LIKE 'user:%' AND principal LIKE '%@enron.com'
               GROUP BY principal ORDER BY n DESC LIMIT ?"""
        return [(r["user"], r["n"]) for r in self.conn.execute(q, (limit,))]

    def mailbox_owners(self) -> list[str]:
        owners: set[str] = set()
        for (raw,) in self.conn.execute("SELECT DISTINCT owners FROM emails"):
            owners.update(json.loads(raw or "[]"))
        return sorted(owners)


_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in is it its me my of on or our
    she so that the their them they this to was we were what when where which who whom why will with
    you your did do does about into than then there these those how can could would should
    """.split()
)
