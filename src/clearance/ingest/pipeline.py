"""Ingest: maildir -> dedup + ACL -> chunks + FTS -> embeddings -> knowledge graph."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from clearance.config import Settings
from clearance.embeddings import Embedder, load_embedder
from clearance.graph.extract import build_graph
from clearance.ingest.chunker import chunk_email
from clearance.ingest.parser import parse_maildir
from clearance.store.db import Database
from clearance.store.vectors import VectorIndex

# Mailboxes of people central to the Enron story; a good default subset (~40-60k unique emails).
DEFAULT_USERS = [
    "lay-k", "skilling-j", "kaminski-v", "delainey-d", "kitchen-l", "lavorato-j", "haedicke-m",
    "whalley-g", "buy-r", "kean-s", "dasovich-j", "shackleton-s", "taylor-m", "derrick-j",
    "beck-s", "mcconnell-m", "shankman-j", "presto-k", "sager-e", "steffes-j",
]


def run_ingest(
    settings: Settings,
    maildir: Path,
    users: list[str] | None = None,
    max_emails: int | None = None,
    embedder: Embedder | None = None,
    build_kg: bool = True,
) -> dict:
    t0 = time.time()
    print(f"[1/4] Parsing {maildir} ...")
    emails, owners = parse_maildir(maildir, users=users, max_emails=max_emails)
    print(f"      {len(emails)} unique emails from {len(owners)} mailboxes")

    print("[2/4] Writing emails, ACLs and chunks ...")
    db = Database(settings.db_path)
    db.reset_corpus()
    n_chunks = 0
    for em in emails:
        email_id = db.insert_email(em, em.acl())
        for i, text in enumerate(chunk_email(em)):
            db.insert_chunk(email_id, i, text)
            n_chunks += 1
    db.commit()
    print(f"      {n_chunks} chunks")

    print("[3/4] Embedding chunks ...")
    embedder = embedder or load_embedder(settings.embedder, settings.embed_model)
    rows = db.all_chunks()
    ids = np.array([cid for cid, _ in rows], dtype=np.int64)
    texts = [t for _, t in rows]
    vecs = []
    for start in range(0, len(texts), 2048):
        vecs.append(embedder.embed(texts[start : start + 2048]))
        print(f"      {min(start + 2048, len(texts))}/{len(texts)}")
    matrix = np.vstack(vecs) if vecs else np.zeros((0, embedder.dim), dtype=np.float32)
    VectorIndex(ids, matrix, embedder.name).save(settings.vectors_path)

    graph_stats = {}
    if build_kg:
        print("[4/4] Building knowledge graph ...")
        graph_stats = build_graph(db)

    stats = {
        "emails": len(emails),
        "mailboxes": len(owners),
        "chunks": n_chunks,
        "embedder": embedder.name,
        "seconds": round(time.time() - t0, 1),
        **graph_stats,
    }
    print(f"Done in {stats['seconds']}s: {stats}")
    return stats
