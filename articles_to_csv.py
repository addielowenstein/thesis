#!/usr/bin/env python3
"""
Export all article .txt files to a single CSV.

Columns: author, date, title, article_text, source_file (optional path for traceability)

Why this approach:
  - csv.writer with quoting=csv.QUOTE_MINIMAL safely escapes commas, quotes, and newlines
    inside fields (standard for R, pandas, Excel with UTF-8 import).
  - UTF-8 with utf-8-sig BOM optional for Excel on Windows (--excel-bom).

Usage:
  python3 articles_to_csv.py -o articles_corpus.csv
  python3 articles_to_csv.py -o articles_corpus.csv --excel-bom
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parent
ARTICLES = THESIS_ROOT / "articles"


def parse_article_file(text: str) -> dict[str, str]:
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


def main():
    p = argparse.ArgumentParser(description="Export articles/*.txt to CSV")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=THESIS_ROOT / "articles_corpus.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--excel-bom",
        action="store_true",
        help="Write UTF-8 BOM first byte for Excel (Windows) to recognize encoding",
    )
    p.add_argument(
        "--no-source",
        action="store_true",
        help="Omit source_file column",
    )
    args = p.parse_args()

    if not ARTICLES.is_dir():
        raise SystemExit(f"Not found: {ARTICLES}")

    paths = sorted(ARTICLES.rglob("*.txt"))
    encoding = "utf-8-sig" if args.excel_bom else "utf-8"

    fieldnames = ["author", "date", "title", "article_text"]
    if not args.no_source:
        fieldnames.append("source_file")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", newline="", encoding=encoding) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="ignore",
        )
        w.writeheader()
        for path in paths:
            raw = path.read_text(encoding="utf-8", errors="replace")
            row = parse_article_file(raw)
            if not args.no_source:
                row["source_file"] = str(path.relative_to(THESIS_ROOT))
            w.writerow(row)

    print(f"Wrote {len(paths)} rows to {args.output}")


if __name__ == "__main__":
    main()
