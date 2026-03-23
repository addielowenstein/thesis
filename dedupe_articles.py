#!/usr/bin/env python3
"""
Remove duplicate article .txt files: same title + author + body keeps one copy only.

Keeps the lexicographically first path in each duplicate group; deletes the rest.
Writes a CSV log of deleted files (default: dedupe_deleted_log.csv).

Usage:
  python3 dedupe_articles.py           # dry-run: print counts only
  python3 dedupe_articles.py --apply # actually delete files
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

THESIS_ROOT = Path(__file__).resolve().parent
ARTICLES = THESIS_ROOT / "articles"
LOG_DEFAULT = THESIS_ROOT / "dedupe_deleted_log.csv"


def parse_article(text: str) -> tuple[str, str, str] | None:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    author_idx = None
    for i, line in enumerate(lines):
        if line.startswith("作者：") or line.startswith("作者:"):
            author_idx = i
            break
    if author_idx is not None:
        title = "\n".join(lines[:author_idx]).strip()
        author = re.sub(r"^作者[：:]\s*", "", lines[author_idx]).strip()
        next_i = author_idx + 1
        if next_i < len(lines) and (
            lines[next_i].startswith("时间：") or lines[next_i].startswith("时间:")
        ):
            next_i += 1
        while next_i < len(lines) and lines[next_i].strip() == "":
            next_i += 1
        body = "\n".join(lines[next_i:]).strip()
        return (title, author, body)
    time_idx = None
    for i, line in enumerate(lines):
        if line.startswith("时间：") or line.startswith("时间:"):
            time_idx = i
            break
    if time_idx is None:
        return None
    title = "\n".join(lines[:time_idx]).strip()
    next_i = time_idx + 1
    while next_i < len(lines) and lines[next_i].strip() == "":
        next_i += 1
    body = "\n".join(lines[next_i:]).strip()
    return (title, "", body)


def main():
    parser = argparse.ArgumentParser(description="Deduplicate identical articles under articles/")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (default: dry-run only)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_DEFAULT,
        help=f"CSV log of deleted paths (default: {LOG_DEFAULT})",
    )
    args = parser.parse_args()

    if not ARTICLES.is_dir():
        raise SystemExit(f"Not found: {ARTICLES}")

    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    parse_fail: list[tuple[str, str]] = []

    for p in sorted(ARTICLES.rglob("*.txt")):
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            parse_fail.append((str(p), str(e)))
            continue
        parsed = parse_article(raw)
        if parsed is None:
            parse_fail.append((str(p), "no 作者/时间 metadata"))
            continue
        groups[parsed].append(p)

    to_delete: list[tuple[Path, Path, int]] = []  # (path, kept_path, group_size)
    for key, paths in groups.items():
        if len(paths) <= 1:
            continue
        paths_sorted = sorted(paths)
        keeper = paths_sorted[0]
        for dup in paths_sorted[1:]:
            to_delete.append((dup, keeper, len(paths)))

    print(f"Articles root: {ARTICLES}")
    print(f"Unique content keys: {len(groups)}")
    print(f"Total .txt files: {sum(len(v) for v in groups.values())}")
    print(f"Duplicate groups: {sum(1 for v in groups.values() if len(v) > 1)}")
    print(f"Files to delete (extra copies): {len(to_delete)}")
    if parse_fail:
        print(f"Parse failures (skipped, not deleted): {len(parse_fail)}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to delete duplicates.")
        return

    deleted = 0
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["deleted_path", "kept_path", "group_size"])
        for dup, keeper, gsize in to_delete:
            try:
                dup.unlink()
                w.writerow([str(dup.relative_to(THESIS_ROOT)), str(keeper.relative_to(THESIS_ROOT)), gsize])
                deleted += 1
            except OSError as e:
                w.writerow([str(dup), f"ERROR: {e}", gsize])

    print(f"\nDeleted {deleted} files. Log: {args.log}")


if __name__ == "__main__":
    main()
