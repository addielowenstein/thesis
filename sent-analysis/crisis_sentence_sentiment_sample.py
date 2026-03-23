"""
Extract sentences about the global financial crisis (金融危机 / 国际金融危机 / 全球金融危机)
and run Chinese BERT sentiment on each. By default samples 100 articles (stratified by year);
use --all to process all 931 articles. Output: CSV with article, year, sentence, sentiment, confidence.
"""
import os
import re
import csv
import time
import random
import argparse
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

THESIS_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = THESIS_ROOT / "articles"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SENTIMENT_MODEL = "jackietung/bert-base-chinese-finetuned-sentiment"

# Sentences must contain one of these to count as "about the global financial crisis"
CRISIS_TERMS = ("金融危机", "国际金融危机", "全球金融危机")


def get_article_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore").strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences (by 。 ； and newline)."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"[。；\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def sentences_about_crisis(text: str) -> list[str]:
    """Return sentences that mention the global financial crisis specifically."""
    sents = split_sentences(text)
    return [s for s in sents if any(term in s for term in CRISIS_TERMS)]


def load_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir=str(_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, cache_dir=str(_CACHE_DIR))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def run_sentiment_on_text(tokenizer, model, text: str, max_len: int = 512):
    """Run sentiment on one text. Returns (label, confidence) or (None, None) if empty."""
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


def main():
    parser = argparse.ArgumentParser(description="Crisis sentence sentiment: sample (default) or all articles.")
    parser.add_argument("--all", action="store_true", help="Process all articles (default: sample 100)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    # Group articles by year
    by_year = {}
    for path in sorted(ARTICLES_DIR.rglob("*.txt")):
        try:
            rel = path.relative_to(ARTICLES_DIR)
        except ValueError:
            rel = Path(path.name)
        year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        if year not in by_year:
            by_year[year] = []
        by_year[year].append(path)

    total = sum(len(v) for v in by_year.values())
    if args.all:
        articles_to_process = [p for paths in by_year.values() for p in paths]
        out_path = Path(__file__).resolve().parent / "crisis_sentence_sentiment_full.csv"
        print(f"Processing all {len(articles_to_process)} articles from years: {sorted(by_year.keys())}")
    else:
        SAMPLE_SIZE = 100
        sampled = []
        for year in sorted(by_year.keys()):
            paths = by_year[year]
            n = max(1, round(SAMPLE_SIZE * len(paths) / total))
            n = min(n, len(paths))
            rng = random.Random(42)
            sampled.extend(rng.sample(paths, n))
        if len(sampled) > SAMPLE_SIZE:
            sampled = sampled[:SAMPLE_SIZE]
        elif len(sampled) < SAMPLE_SIZE and total >= SAMPLE_SIZE:
            need = SAMPLE_SIZE - len(sampled)
            pool = [p for p in ARTICLES_DIR.rglob("*.txt") if p not in sampled]
            rng = random.Random(43)
            sampled.extend(rng.sample(pool, min(need, len(pool))))
        articles_to_process = list(dict.fromkeys(sampled))[:SAMPLE_SIZE]
        out_path = Path(__file__).resolve().parent / "crisis_sentence_sentiment_sample.csv"
        print(f"Sampled {len(articles_to_process)} articles from years: {sorted(by_year.keys())}")

    print("Loading tokenizer and sentiment model ...")
    tokenizer, model = load_models()
    start = time.perf_counter()
    n_sentences = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article", "year", "sentence", "sentiment", "confidence"])

        for i, path in enumerate(articles_to_process):
            try:
                rel = path.relative_to(ARTICLES_DIR)
            except ValueError:
                rel = Path(path.name)
            year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            article_id = str(rel)

            text = get_article_text(path)
            crisis_sents = sentences_about_crisis(text)

            for sent in crisis_sents:
                label, conf = run_sentiment_on_text(tokenizer, model, sent)
                w.writerow([
                    article_id,
                    year,
                    sent,
                    label if label else "",
                    f"{conf:.4f}" if conf is not None else "",
                ])
                n_sentences += 1

            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(articles_to_process)}] articles processed, {n_sentences} sentences so far ...")

    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f} s. {n_sentences} sentences written to {out_path}")


if __name__ == "__main__":
    main()
