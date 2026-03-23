#!/usr/bin/env python3
"""
Build a JSON corpus and a date-sorted CSV from articles/*.txt.

Each record:
  { "id": int, "title": str, "author": str, "date": "YYYY-MM-DD", "text": str }

IDs are auto-increment starting at 1 in **chronological order** (earliest date first),
so the first row is the earliest article (from 2008 onward in your corpus). Articles
with no parseable date are placed last, sorted by source path.

Outputs (defaults):
  articles_corpus.json  — UTF-8 JSON array
  articles_by_date.csv  — same fields + id, CSV quoted for Excel/pandas

Usage:
  python3 articles_json_and_csv.py
  python3 articles_json_and_csv.py --json out.json --csv out.csv --compact-json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parent
ARTICLES = THESIS_ROOT / "articles"


def parse_article_file(text: str) -> dict[str, str]:
    """Same layout as articles_to_csv.py; returns title, author, date, article_text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    author_idx = None
    for i, line in enumerate(lines):
        if line.startswith("作者：") or line.startswith("作者:"):
            author_idx = i
            break

    if author_idx is not None:
        title_block = "\n".join(lines[:author_idx]).strip()
        author = re.sub(r"^作者[：:]\s*", "", lines[author_idx]).strip()
        next_i = author_idx + 1
        date = ""
        if next_i < len(lines) and (
            lines[next_i].startswith("时间：") or lines[next_i].startswith("时间:")
        ):
            dm = re.search(r"(\d{4}-\d{2}-\d{2})", lines[next_i])
            if dm:
                date = dm.group(1)
            next_i += 1
        while next_i < len(lines) and lines[next_i].strip() == "":
            next_i += 1
        body = "\n".join(lines[next_i:]).strip()
        return {
            "title": title_block,
            "author": author,
            "date": date,
            "article_text": body,
        }

    time_idx = None
    for i, line in enumerate(lines):
        if line.startswith("时间：") or line.startswith("时间:"):
            time_idx = i
            break
    if time_idx is None:
        return {
            "title": "",
            "author": "",
            "date": "",
            "article_text": text.strip(),
        }

    title_block = "\n".join(lines[:time_idx]).strip()
    dm = re.search(r"(\d{4}-\d{2}-\d{2})", lines[time_idx])
    date = dm.group(1) if dm else ""
    next_i = time_idx + 1
    while next_i < len(lines) and lines[next_i].strip() == "":
        next_i += 1
    body = "\n".join(lines[next_i:]).strip()
    return {
        "title": title_block,
        "author": "",
        "date": date,
        "article_text": body,
    }


def sort_key(row: dict) -> tuple:
    d = row["date"]
    if not d:
        return ("9999-99-99", row["_source"])
    return (d, row["_source"])


def main():
    p = argparse.ArgumentParser(description="Articles → JSON + date-sorted CSV")
    p.add_argument("--json", type=Path, default=THESIS_ROOT / "articles_corpus.json")
    p.add_argument("--csv", type=Path, default=THESIS_ROOT / "articles_by_date.csv")
    p.add_argument(
        "--compact-json",
        action="store_true",
        help="Single-line JSON (smaller file)",
    )
    p.add_argument(
        "--excel-bom",
        action="store_true",
        help="UTF-8 BOM on CSV for Excel (Windows)",
    )
    args = p.parse_args()

    if not ARTICLES.is_dir():
        raise SystemExit(f"Not found: {ARTICLES}")

    rows: list[dict] = []
    for path in sorted(ARTICLES.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_article_file(raw)
        rows.append(
            {
                "title": parsed["title"],
                "author": parsed["author"],
                "date": parsed["date"],
                "article_text": parsed["article_text"],
                "_source": str(path.relative_to(THESIS_ROOT)),
            }
        )

    rows.sort(key=sort_key)

    out_json = []
    for i, r in enumerate(rows, start=1):
        rec = {
            "id": i,
            "title": r["title"],
            "author": r["author"],
            "date": r["date"],
            "text": r["article_text"],
        }
        out_json.append(rec)

    indent = None if args.compact_json else 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=indent)
        if indent is not None:
            f.write("\n")

    enc = "utf-8-sig" if args.excel_bom else "utf-8"
    fieldnames = ["id", "title", "author", "date", "text"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="", encoding=enc) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for rec in out_json:
            w.writerow(rec)

    earliest = next((r["date"] for r in out_json if r["date"]), "")
    no_date = sum(1 for r in out_json if not r["date"])
    print(f"Wrote {len(out_json)} records")
    print(f"  JSON: {args.json}")
    print(f"  CSV (date ascending): {args.csv}")
    print(f"  Earliest dated article: {earliest or '(none)'}")
    if no_date:
        print(f"  Articles with no date (listed last): {no_date}")


if __name__ == "__main__":
    main()
