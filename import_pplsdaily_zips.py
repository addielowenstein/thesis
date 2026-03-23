#!/usr/bin/env python3
"""
Unzip People's Daily page archives from PPLSDAILY and append .txt articles under thesis/articles/.

Skips the range already imported earlier: 05-pg2.3.zip through 64-pg20.3.zip (inclusive),
when zips are sorted naturally by leading number (00, 01, … 65-pg21.1, …).

Each zip contains index.html + pic/; same parsing as extract_articles_to_txt.py.

Usage:
  python import_pplsdaily_zips.py
  python import_pplsdaily_zips.py --dry-run
  python import_pplsdaily_zips.py --pplsdaily /path/to/PPLSDAILY
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from extract_articles_to_txt import (
    ARTICLES_ROOT,
    detect_year_from_html,
    process_html_file,
)

DEFAULT_PPLSDAILY = Path.home() / "Downloads" / "PPLSDAILY"
SKIP_FIRST = "05-pg2.3.zip"
SKIP_LAST = "64-pg20.3.zip"


def natural_zip_key(p: Path) -> tuple:
    """Sort 00-pg1.1, … 99-pg…, 100-pg33.2, …"""
    stem = p.stem
    m = re.match(r"^(\d+)-(.*)$", stem)
    if m:
        return (int(m.group(1)), m.group(2))
    return (10**9, stem)


def main():
    parser = argparse.ArgumentParser(description="Import new PPLSDAILY zips into thesis/articles.")
    parser.add_argument(
        "--pplsdaily",
        type=Path,
        default=DEFAULT_PPLSDAILY,
        help=f"Folder containing *.zip (default: {DEFAULT_PPLSDAILY})",
    )
    parser.add_argument("--dry-run", action="store_true", help="List zips only, do not extract.")
    args = parser.parse_args()

    ppl = args.pplsdaily
    if not ppl.is_dir():
        raise SystemExit(f"Not a directory: {ppl}")

    zips = sorted(ppl.glob("*.zip"), key=natural_zip_key)
    if not zips:
        raise SystemExit(f"No .zip files in {ppl}")

    names = [z.name for z in zips]
    try:
        i_first = names.index(SKIP_FIRST)
        i_last = names.index(SKIP_LAST)
    except ValueError as e:
        raise SystemExit(
            f"Could not find {SKIP_FIRST} or {SKIP_LAST} in {ppl}. "
            f"First zip: {names[0]}, last: {names[-1]}"
        ) from e

    to_import = [z for i, z in enumerate(zips) if i < i_first or i > i_last]
    print(f"PPLSDAILY: {len(zips)} zips total; skipping already-imported [{SKIP_FIRST} … {SKIP_LAST}] "
          f"({i_last - i_first + 1} zips); will import {len(to_import)} zips.\n")

    if args.dry_run:
        for z in to_import[:20]:
            print(f"  would import: {z.name}")
        if len(to_import) > 20:
            print(f"  … and {len(to_import) - 20} more")
        return

    articles_root = Path(ARTICLES_ROOT)
    articles_root.mkdir(parents=True, exist_ok=True)

    total_articles = 0
    errors: list[str] = []

    for zi, zp in enumerate(to_import):
        stem = zp.stem
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                names_in = zf.namelist()
                if "index.html" not in names_in:
                    errors.append(f"{zp.name}: no index.html")
                    continue
                raw = zf.read("index.html").decode("utf-8", errors="replace")
                year = detect_year_from_html(raw)
                out_dir = articles_root / year
                out_dir.mkdir(parents=True, exist_ok=True)

                # Skip if this page was already extracted (idempotent re-run)
                probe = out_dir / f"{stem}_001.txt"
                if probe.is_file():
                    print(f"[{zi+1}/{len(to_import)}] skip (exists): {zp.name} -> {year}/")
                    continue

                with tempfile.TemporaryDirectory(prefix="ppls_") as td:
                    td_path = Path(td)
                    zf.extractall(td_path)
                    html_path = td_path / "index.html"
                    n = process_html_file(html_path, out_dir, stem)
                    total_articles += n
                    print(f"[{zi+1}/{len(to_import)}] {zp.name} -> {year}/ ({n} articles)")
        except Exception as e:
            errors.append(f"{zp.name}: {e}")
            print(f"[{zi+1}/{len(to_import)}] ERROR {zp.name}: {e}")

    print(f"\nDone. Wrote {total_articles} new article .txt files under {articles_root}.")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:30]:
            print(f"  {e}")
        if len(errors) > 30:
            print(f"  … and {len(errors) - 30} more")


if __name__ == "__main__":
    main()
