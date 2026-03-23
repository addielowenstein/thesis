#!/usr/bin/env python3
"""
Extract each article from index.html files in thesis/articles into separate .txt files.
Each <div class="detail"> becomes one .txt with title, author, date, and plain-text body.
Replaces the articles folder so it only contains .txt files organized by date.
"""
import re
import shutil
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
from typing import Optional
from collections import Counter

ARTICLES_ROOT = Path(__file__).resolve().parent / "articles"
ENCODING = "utf-8"


class TextExtractor(HTMLParser):
    """Strip HTML and collect text, preserving paragraph breaks."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True
        elif tag in ("p", "br", "div") and not self.in_script:
            if self.parts and not self.parts[-1].endswith("\n\n"):
                self.parts.append("\n\n" if tag == "p" else "\n")

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if not self.in_script and data.strip():
            self.parts.append(data.strip())

    def get_text(self):
        text = " ".join(self.parts)
        text = re.sub(r"\n\n+", "\n\n", text)
        text = re.sub(r" +", " ", text)
        return text.strip()


def extract_text(html_fragment: str) -> str:
    p = TextExtractor()
    p.feed(html_fragment)
    return unescape(p.get_text())


def parse_detail_block(block: str) -> Optional[dict]:
    """Parse one <div class="detail"> block. Returns dict with title, subtitle, author, date, body or None."""
    # Title: first h2, then optional h3
    h2_m = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.DOTALL)
    h3_m = re.search(r"<h3[^>]*>(.*?)</h3>", block, re.DOTALL)
    title = extract_text(h2_m.group(1)) if h2_m else ""
    subtitle = extract_text(h3_m.group(1)) if h3_m else ""

    # Author and date from <p>作者：... 时间：YYYY-MM-DD</p>
    author = ""
    date = ""
    author_m = re.search(r"作者[：:]\s*([^&]+?)(?:\s*&nbsp\s*)?\s*时间", block)
    if author_m:
        author = author_m.group(1).strip()
    date_m = re.search(r"时间[：:]\s*(\d{4}-\d{2}-\d{2})", block)
    if date_m:
        date = date_m.group(1)

    # Body: content of <div class="detail-p" id="detail-p">...</div>
    body_m = re.search(r'<div class="detail-p"[^>]*>(.*?)</div>\s*</div>\s*<!--', block, re.DOTALL)
    if not body_m:
        body_m = re.search(r'<div class="detail-p"[^>]*>(.*?)</div>\s*</div>', block, re.DOTALL)
    body = extract_text(body_m.group(1)) if body_m else ""

    if not title and not body:
        return None
    return {
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "date": date,
        "body": body,
    }


def split_detail_blocks(html: str) -> list[str]:
    """Split HTML into list of <div class="detail">...</div> blocks."""
    parts = re.split(r"<!--文章正文-->", html)
    blocks = []
    for part in parts[1:]:  # skip everything before first comment
        part = part.strip()
        if '<div class="detail"' not in part:
            continue
        start = part.find('<div class="detail"')
        if start == -1:
            start = part.find("<div")
        if start == -1:
            continue
        depth = 0
        pos = start
        while pos < len(part):
            if part[pos : pos + 4] == "<div":
                depth += 1
                pos += 4
            elif part[pos : pos + 6] == "</div>":
                depth -= 1
                pos += 6
                if depth == 0:
                    blocks.append(part[start:pos])
                    break
            else:
                pos += 1
        else:
            blocks.append(part[start:])
    return blocks


def safe_filename(s: str, max_len: int = 80) -> str:
    """Make a string safe for use as filename (strip path chars, limit length)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", s)
    s = s.strip() or "untitled"
    return s[:max_len]


def detect_year_from_html(html: str) -> str:
    """Infer publication year from 时间：YYYY-MM-DD lines (mode of all dates found)."""
    years = re.findall(r"时间[：:]\s*(\d{4})-\d{2}-\d{2}", html)
    if not years:
        years = re.findall(r"(\d{4})-\d{2}-\d{2}", html)
    if not years:
        # e.g. pic paths: pic/101/2015/12/...
        years = re.findall(r"pic/101/(\d{4})/", html)
    if not years:
        return "unknown"
    return Counter(years).most_common(1)[0][0]


def process_html_file(html_path: Path, out_dir: Path, source_id: str) -> int:
    """Process one index.html; write one .txt per article. Returns count of articles written."""
    raw = html_path.read_text(encoding=ENCODING)
    blocks = split_detail_blocks(raw)
    count = 0
    for i, block in enumerate(blocks):
        article = parse_detail_block(block)
        if not article or (not article["title"] and not article["body"]):
            continue
        lines = []
        if article["title"]:
            lines.append(article["title"])
        if article["subtitle"]:
            lines.append(article["subtitle"])
        if article["author"]:
            lines.append(f"作者：{article['author']}")
        if article["date"]:
            lines.append(f"时间：{article['date']}")
        if lines:
            lines.append("")
        lines.append(article["body"] or "")
        text = "\n".join(lines)
        out_name = f"{source_id}_{i + 1:03d}.txt"
        out_path = out_dir / out_name
        out_path.write_text(text, encoding=ENCODING)
        count += 1
    return count


def main():
    articles_root = Path(ARTICLES_ROOT)
    if not articles_root.is_dir():
        print(f"Articles root not found: {articles_root}")
        return

    # Collect (date_dir, source_folder) for every index.html
    to_process = []
    for date_dir in articles_root.rglob("*"):
        if not date_dir.is_dir():
            continue
        # date_dir is like articles/2013/11/06
        try:
            rel = date_dir.relative_to(articles_root)
            parts = rel.parts
            if len(parts) != 3 or not all(p.isdigit() for p in parts):
                continue
        except ValueError:
            continue
        for item in date_dir.iterdir():
            if item.is_dir():
                html_file = item / "index.html"
                if html_file.is_file():
                    to_process.append((date_dir, item, item.name))

    # Build new structure: write .txt into a temp sibling, then replace
    total = 0
    for date_dir, source_folder, source_id in to_process:
        n = process_html_file(source_folder / "index.html", date_dir, source_id)
        total += n
        print(f"  {date_dir.relative_to(articles_root)}/{source_id}: {n} articles")

    # Remove original subfolders (the zip-name folders containing index.html and pic/)
    for date_dir, source_folder, _ in to_process:
        shutil.rmtree(source_folder, ignore_errors=True)

    print(f"\nDone. Extracted {total} articles to .txt files. Removed original HTML/pic folders.")


if __name__ == "__main__":
    main()
