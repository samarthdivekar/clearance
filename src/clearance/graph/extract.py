"""Build the knowledge graph from emails.

Two sources of structure:

1. **Headers (free, exact):** people are email addresses; ``sender -[emailed]-> recipient``.
2. **Text (heuristic, cheap):** organizations, acronyms/code names (LJM, JEDI, Raptor) and
   people's names are pulled out of chunk text with patterns, then kept only if they recur
   across the corpus (a document-frequency floor removes most noise). Entities co-mentioned in
   a chunk are linked. `graph/llm_extract.py` offers an optional LLM-based extractor for higher
   precision on a subset.

Every mention and edge stores the chunk it came from. At query time the traversal only follows
edges whose chunk the principal can read, so the graph cannot become a side channel.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter, defaultdict

from clearance.store.db import Database

ORG_SUFFIX = r"(?:Inc|Corp|Corporation|LLC|L\.?P\.?|Ltd|Company|Co\.|Energy|Bank|Partners|Capital|Power|Gas|Group|Holdings|Commission)"
_ORG_RE = re.compile(rf"\b((?:[A-Z][A-Za-z&\-]+\s){{1,4}}{ORG_SUFFIX})\b")
_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z&]{1,6}[A-Z0-9])\b")
_CAPS_PHRASE_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b")
_LOWER_WORD_RE = re.compile(r"\b[a-z]{2,}\b")
_PROJECT_RE = re.compile(r"\b(?:Project|project|Deal|deal)\s+([A-Z][a-zA-Z]+)\b")

STOP_ACRONYMS = frozenset(
    """FYI ASAP PM AM EST CST PST CDT EDT PDT THE AND FOR RE FW FWD NOTE URL HTML HTTP WWW USA US
    OK TBD CEO CFO COO VP SVP EVP MBA II III IV PS NY TX MW MWH ETC NA ID NO YES LLC INC CORP""".split()
)
COMMON_WORDS = frozenset(
    """the a an this that these those it its is was are were be been will would can could should
    what when where which who why how and or but if for of to in on at by with from about as all
    any each our your their we you they he she i me my us them his her not no yes so do does did
    have has had there here then than also just only very more most some such new""".split()
)
STOP_PHRASES = frozenset(
    """thank you best regards kind regards please let let me thanks again good morning happy new
    new year dear all see attached original message""".split()
)


def person_label(address: str) -> str:
    local = address.split("@")[0]
    parts = [p for p in re.split(r"[._\-]+", local) if p and not p.isdigit()]
    return " ".join(p.capitalize() for p in parts) or address


def extract_text_entities(text: str) -> set[tuple[str, str]]:
    """Return {(label, type)} found in a chunk body (header line excluded)."""
    body = text.split("\n", 1)[1] if "\n" in text else ""
    found: set[tuple[str, str]] = set()
    for m in _ORG_RE.finditer(body):
        found.add((m.group(1).strip(), "org"))
    for m in _ACRONYM_RE.finditer(body):
        tok = m.group(1)
        if tok not in STOP_ACRONYMS:
            found.add((tok, "project" if len(tok) >= 3 else "other"))
    for m in _PROJECT_RE.finditer(body):
        if m.group(1).lower() not in COMMON_WORDS:
            found.add((m.group(1), "project"))
    for m in _CAPS_PHRASE_RE.finditer(body):
        phrase = m.group(1)
        if phrase.lower() not in STOP_PHRASES and not any(w.lower() in STOP_PHRASES for w in phrase.split()[:1]):
            found.add((phrase, "person_or_org"))
    return found


def build_graph(db: Database, min_df: int = 2, max_df_ratio: float = 0.05, progress: bool = True) -> dict:
    conn = db.conn
    for table in ("edges", "mentions", "entities"):
        conn.execute(f"DELETE FROM {table}")

    rows = conn.execute(
        """SELECT c.id AS chunk_id, c.ord, c.text, e.sender, e.recipients
           FROM chunks c JOIN emails e ON e.id = c.email_id ORDER BY c.id"""
    ).fetchall()
    n_chunks = max(len(rows), 1)

    # Pass 1: candidate text entities + document frequency.
    chunk_entities: dict[int, set[tuple[str, str]]] = {}
    df: Counter[str] = Counter()
    lower_df: Counter[str] = Counter()
    for r in rows:
        ents = extract_text_entities(r["text"])
        chunk_entities[r["chunk_id"]] = ents
        df.update({label.lower() for label, _ in ents})
        lower_df.update(set(_LOWER_WORD_RE.findall(r["text"])))
    max_df = max(min_df + 1, int(n_chunks * max_df_ratio))
    keep = {
        name for name, n in df.items()
        if min_df <= n <= max_df and name not in COMMON_WORDS
        # An all-caps token that is also a common lowercase word is shouting, not an acronym.
        and not (" " not in name and lower_df.get(name, 0) >= max(3, n // 2))
    }

    entity_ids: dict[str, int] = {}

    def entity_id(label: str, type_: str) -> int:
        name = label.lower()
        if name not in entity_ids:
            cur = conn.execute(
                "INSERT INTO entities (name, label, type) VALUES (?, ?, ?)", (name, label, type_)
            )
            entity_ids[name] = cur.lastrowid
        return entity_ids[name]

    mentions: set[tuple[int, int]] = set()
    edges: list[tuple[int, int, str, int]] = []
    for r in rows:
        cid = r["chunk_id"]
        sender_id = entity_id(r["sender"], "person")
        mentions.add((sender_id, cid))
        chunk_ents = [sender_id]
        if r["ord"] == 0:  # header edges once per email, anchored on its first chunk
            rec = json.loads(r["recipients"] or "{}")
            for addr in itertools.chain(rec.get("to", []), rec.get("cc", [])):
                rid = entity_id(addr, "person")
                mentions.add((rid, cid))
                edges.append((sender_id, rid, "emailed", cid))
        for label, type_ in chunk_entities[cid]:
            if label.lower() in keep:
                eid = entity_id(label, type_)
                mentions.add((eid, cid))
                chunk_ents.append(eid)
        # Co-mention edges; cap the fan-out so long chunks don't create O(n^2) edges.
        uniq = list(dict.fromkeys(chunk_ents))[:10]
        for a, b in itertools.combinations(uniq, 2):
            edges.append((a, b, "co_mentioned", cid))

    conn.executemany("INSERT OR IGNORE INTO mentions (entity_id, chunk_id) VALUES (?, ?)", sorted(mentions))
    conn.executemany("INSERT INTO edges (src, dst, rel, chunk_id) VALUES (?, ?, ?, ?)", edges)

    # Person entities get a readable label ("jeff.skilling@enron.com" -> "Jeff Skilling").
    for name, eid in entity_ids.items():
        if "@" in name:
            conn.execute("UPDATE entities SET label = ? WHERE id = ?", (person_label(name), eid))
    conn.commit()

    stats = {"entities": len(entity_ids), "mentions": len(mentions), "edges": len(edges)}
    if progress:
        print(f"  graph: {stats['entities']} entities, {stats['mentions']} mentions, {stats['edges']} edges")
    return stats


def entity_type_counts(db: Database) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in db.conn.execute("SELECT type, COUNT(*) AS n FROM entities GROUP BY type"):
        out[r["type"]] = r["n"]
    return dict(out)
