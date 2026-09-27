"""Command-line interface: `clearance <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from clearance.config import get_settings
from clearance.ingest.pipeline import DEFAULT_USERS

DATASET_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"


def _users_arg(value: str | None) -> list[str] | None:
    if value in (None, "default"):
        return DEFAULT_USERS
    if value == "all":
        return None
    return [u.strip() for u in value.split(",") if u.strip()]


def cmd_download(args) -> None:
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "enron_mail_20150507.tar.gz"
    if not archive.exists():
        print(f"Downloading {DATASET_URL} (~423 MB) ...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    users = _users_arg(args.users)
    prefixes = tuple(f"maildir/{u}/" for u in users) if users else ("maildir/",)
    print(f"Extracting {'all mailboxes' if users is None else f'{len(users)} mailboxes'} to {dest / 'maildir'} ...")
    n = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.startswith(prefixes):
                tar.extract(member, dest, filter="data")
                n += 1
                if n % 10000 == 0:
                    print(f"  {n} files")
    print(f"Extracted {n} files.")


def cmd_ingest(args) -> None:
    from clearance.ingest.pipeline import run_ingest

    settings = get_settings().with_overrides(embedder=args.embedder)
    run_ingest(settings, Path(args.maildir), users=_users_arg(args.users), max_emails=args.max_emails,
               build_kg=not args.no_graph)


def _service(args, **overrides):
    from clearance.pipeline import RAGService

    values = {
        "llm_provider": getattr(args, "llm", None),
        "cache_mode": getattr(args, "cache", None),
        "retrieval_mode": getattr(args, "mode", None),
    }
    values.update(overrides)
    settings = get_settings().with_overrides(**values)
    return RAGService.from_settings(settings)


def cmd_ask(args) -> None:
    from clearance.security.acl import Principal

    svc = _service(args)
    ans = svc.ask(args.question, Principal.from_email(args.as_user))
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2))
        return
    print(f"\n{ans.text}\n")
    for c in ans.citations:
        print(f"  [{c.n}] {c.date}  {c.sender}  \"{c.subject}\"")
    print(f"\n  model={ans.model}  cache={ans.cache_status}  cost=${ans.cost_usd:.5f}  latency={ans.latency_ms:.0f}ms")
    print(f"  route: {ans.route_reason}")
    for e in ans.security_events:
        print(f"  SECURITY: {e}")


def cmd_users(args) -> None:
    from clearance.store.db import Database

    db = Database(get_settings().db_path)
    print("Mailbox owners (good demo principals):")
    for o in db.mailbox_owners():
        print(f"  {o}")
    print("\nMost-connected addresses:")
    for user, n in db.top_principals(args.limit):
        print(f"  {n:>6}  {user}")
    print("\nSpecial: 'auditor' (reads everything, can read the audit log)")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("clearance.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_audit(args) -> None:
    from clearance.security import audit
    from clearance.store.db import Database

    db = Database(get_settings().db_path)
    if args.chunk is not None:
        rows = audit.who_accessed(db, args.chunk)
    else:
        rows = audit.recent(db, args.limit)
    print(json.dumps(rows, indent=2, default=str))
    print(json.dumps(audit.metrics(db), indent=2))


def cmd_gen(args) -> None:
    from clearance.evals import dataset, generate

    svc = _service(args)
    if args.method == "llm":
        qs = generate.generate_llm(svc, args.n_single, args.n_multi)
    else:
        qs = generate.generate_offline(svc, args.n_single)
    dataset.save(qs, args.out)
    print(f"Wrote {len(qs)} questions to {args.out} (review them before trusting eval numbers)")


def cmd_eval(args) -> None:
    from clearance.evals import dataset

    if args.suite == "retrieval":
        from clearance.evals import retrieval_eval

        svc = _service(args)
        retrieval_eval.run(svc, dataset.load(args.gold), k=args.k)
    elif args.suite == "redteam":
        from clearance.evals import redteam

        svc = _service(args, llm_provider=args.llm or "fake")
        rows = redteam.run(svc, n_targets=args.n)
        secure = next(r for r in rows if r["cache_mode"] == "acl_aware")
        if secure["context_leaks"] or secure["cache_leaks"] or secure["content_leaks"]:
            sys.exit("FAIL: leaks detected in acl_aware mode")
    elif args.suite == "cache":
        from clearance.evals import cache_bench

        svc = _service(args, llm_provider=args.llm or "fake")
        cache_bench.run(svc, n_emails=args.n)
    elif args.suite == "answers":
        from clearance.evals import answer_eval

        svc = _service(args)
        answer_eval.run(svc, dataset.load(args.gold), compare_routing=args.compare_routing)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="clearance", description="Permission-aware GraphRAG over Enron email.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download the CMU Enron dataset and extract mailboxes")
    d.add_argument("--dest", default="data/raw")
    d.add_argument("--users", default="default", help="'default', 'all', or comma-separated mailbox dirs")
    d.set_defaults(func=cmd_download)

    i = sub.add_parser("ingest", help="parse, chunk, embed and graph the corpus")
    i.add_argument("--maildir", default="data/raw/maildir")
    i.add_argument("--users", default="default")
    i.add_argument("--max-emails", type=int)
    i.add_argument("--embedder", choices=["local", "hash"])
    i.add_argument("--no-graph", action="store_true")
    i.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask", help="ask a question as a given user")
    a.add_argument("question")
    a.add_argument("--as", dest="as_user", required=True, help="email address, or 'auditor'")
    a.add_argument("--llm", choices=["anthropic", "fake"])
    a.add_argument("--cache", choices=["off", "global", "per_user", "acl_aware"])
    a.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_ask)

    u = sub.add_parser("users", help="list demo principals")
    u.add_argument("--limit", type=int, default=20)
    u.set_defaults(func=cmd_users)

    s = sub.add_parser("serve", help="run the API + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    au = sub.add_parser("audit", help="print the audit log and metrics")
    au.add_argument("--limit", type=int, default=20)
    au.add_argument("--chunk", type=int, help="who had this chunk sent to the LLM?")
    au.set_defaults(func=cmd_audit)

    g = sub.add_parser("gen-questions", help="generate candidate gold questions")
    g.add_argument("--method", choices=["llm", "offline"], default="offline")
    g.add_argument("--n-single", type=int, default=80)
    g.add_argument("--n-multi", type=int, default=40)
    g.add_argument("--out", default="eval_data/generated.jsonl")
    g.add_argument("--llm", choices=["anthropic", "fake"])
    g.set_defaults(func=cmd_gen)

    e = sub.add_parser("eval", help="run an eval suite")
    e.add_argument("suite", choices=["retrieval", "redteam", "cache", "answers"])
    e.add_argument("--gold", default="eval_data/gold.jsonl")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--n", type=int, default=100, help="targets (redteam) or shared emails (cache)")
    e.add_argument("--llm", choices=["anthropic", "fake"])
    e.add_argument("--compare-routing", action="store_true")
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
