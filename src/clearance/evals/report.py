"""Write eval results as Markdown tables + JSON under reports/."""

from __future__ import annotations

import json
import time
from pathlib import Path

REPORTS_DIR = Path("reports")


def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "_no rows_\n"
    columns = columns or list(rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write(name: str, title: str, rows: list[dict], notes: str = "", extra: dict | None = None) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    md = f"# {title}\n\n_Generated {stamp}_\n\n{md_table(rows)}\n{notes}\n"
    md_path = REPORTS_DIR / f"{name}.md"
    md_path.write_text(md, encoding="utf-8")
    (REPORTS_DIR / f"{name}.json").write_text(
        json.dumps({"title": title, "generated": stamp, "rows": rows, **(extra or {})}, indent=2), encoding="utf-8"
    )
    print(md)
    print(f"-> {md_path}")
    return md_path
