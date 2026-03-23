"""
Stance sentiment by year: (1) China's response/role in global financial crisis,
(2) Other countries' response/role. Samples 100 articles from 2013, 100 from 2014,
100 from 2015; runs Chinese BERT sentiment on extracted excerpts; logs results.

Set SAMPLE_SIZE = 10 for a quick run (≈3–4 per year); set to None for full 300.
"""
import os
import re
import csv
import random
import time
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

THESIS_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = THESIS_ROOT / "articles"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SENTIMENT_MODEL = "jackietung/bert-base-chinese-finetuned-sentiment"

# Quick run: 10 articles total (≈3–4 per year). None = use *all* articles.
SAMPLE_SIZE = None
PER_YEAR_CAP = None

CHINA_KEYWORDS = ("中国", "我国", "中国政府", "中方", "中国央行", "中国经济", "国内")
CRISIS_KEYWORDS = ("金融危机", "经济危机", "危机", "金融", "经济", "复苏", "应对", "刺激", "衰退", "全球")
OTHER_COUNTRY_KEYWORDS = ("美国", "欧洲", "欧盟", "西方", "日本", "英国", "各国", "发达国家", "美欧", "华尔街", "国际")


def get_article_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8").strip()


def split_sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"[。；\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def has_any(s: str, keywords: tuple) -> bool:
    return any(k in s for k in keywords)


def extract_china_crisis_text(text: str) -> str:
    sents = split_sentences(text)
    chosen = [s for s in sents if has_any(s, CHINA_KEYWORDS) and has_any(s, CRISIS_KEYWORDS)]
    return " ".join(chosen) if chosen else ""


def extract_other_countries_crisis_text(text: str) -> str:
    sents = split_sentences(text)
    chosen = [s for s in sents if has_any(s, OTHER_COUNTRY_KEYWORDS) and has_any(s, CRISIS_KEYWORDS)]
    return " ".join(chosen) if chosen else ""


def load_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir=str(_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, cache_dir=str(_CACHE_DIR))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def run_sentiment_on_text(tokenizer, model, text: str, max_len: int = 512):
    if not text or not text.strip():
        return None, None
    inp = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True,
    )
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    probs = torch.softmax(out.logits, dim=-1)
    pred_id = probs.argmax(dim=1).item()
    conf = probs[0, pred_id].item()
    label = model.config.id2label.get(pred_id, str(pred_id))
    return label, conf


def sample_articles_by_year():
    """Sample articles from each year (2013, 2014, 2015).

    - If PER_YEAR_CAP is not None: cap at that many per year (with random sample if needed).
    - If SAMPLE_SIZE is not None: down-sample to that many total (stratified across years).
    - If both are None: use *all* available .txt articles under each year.
    """
    rng = random.Random(42)
    by_year = {}
    for year in [2013, 2014, 2015]:
        year_dir = ARTICLES_DIR / str(year)
        if not year_dir.is_dir():
            continue
        files = sorted(year_dir.glob("*.txt"))
        if PER_YEAR_CAP is not None and len(files) > PER_YEAR_CAP:
            files = rng.sample(files, PER_YEAR_CAP)
        by_year[year] = files

    if SAMPLE_SIZE is not None:
        # Take SAMPLE_SIZE total, stratified across years
        pool = []
        for year in [2013, 2014, 2015]:
            for p in by_year.get(year, []):
                pool.append((year, p))
        if len(pool) > SAMPLE_SIZE:
            pool = rng.sample(pool, SAMPLE_SIZE)
        return pool

    # Full run: 100+100+100 (or all available per year)
    pool = []
    for year in [2013, 2014, 2015]:
        for p in by_year.get(year, []):
            pool.append((year, p))
    return sorted(pool, key=lambda x: (x[0], str(x[1])))


def main():
    print(f"Device: {DEVICE}")
    print("Loading tokenizer and sentiment model ...")
    tokenizer, model = load_models()

    articles = sample_articles_by_year()
    print(f"Analyzing stance sentiment for {len(articles)} articles (by year) ...")
    if SAMPLE_SIZE:
        print(f"  (Quick run: SAMPLE_SIZE={SAMPLE_SIZE}. Set to None for 100+100+100.)")
    print()

    log_path = Path(__file__).resolve().parent / "stance_sentiment_by_year.csv"
    start = time.perf_counter()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "year",
            "article",
            "china_stance_sentiment",
            "china_confidence",
            "china_excerpt_len",
            "other_countries_stance_sentiment",
            "other_countries_confidence",
            "other_excerpt_len",
        ])

        for i, (year, path) in enumerate(articles):
            text = get_article_text(path)
            try:
                rel = path.relative_to(ARTICLES_DIR)
            except ValueError:
                rel = path.name

            china_text = extract_china_crisis_text(text)
            other_text = extract_other_countries_crisis_text(text)

            china_label, china_conf = run_sentiment_on_text(tokenizer, model, china_text) if china_text else (None, None)
            other_label, other_conf = run_sentiment_on_text(tokenizer, model, other_text) if other_text else (None, None)

            w.writerow([
                year,
                str(rel),
                china_label or "N/A",
                f"{china_conf:.4f}" if china_conf is not None else "",
                len(china_text),
                other_label or "N/A",
                f"{other_conf:.4f}" if other_conf is not None else "",
                len(other_text),
            ])

            print(f"[{i+1}/{len(articles)}] {year} {rel}")
            print(f"  1) China's response/role:     {china_label or 'N/A'}" + (f" (conf: {china_conf:.3f})" if china_conf is not None else ""))
            print(f"  2) Other countries' role:   {other_label or 'N/A'}" + (f" (conf: {other_conf:.3f})" if other_conf is not None else ""))
            print()

    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f} s. Log: {log_path}")


if __name__ == "__main__":
    main()
