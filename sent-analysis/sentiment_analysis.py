"""
Sentiment analysis on all articles using a Chinese BERT sentiment model.

Next steps (what this script does):
  1. Load the same bert-base-chinese tokenizer + a pretrained sentiment classifier.
  2. For each article: tokenize into 512-token chunks (reuse your chunking).
  3. Run the model on each chunk (in batches for speed), then aggregate to one
     sentiment per article (mean probability over chunks → label).
  4. Save results to CSV: article path, sentiment label, confidence.

Run (with tokenizer/model cache):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python sentiment_analysis.py

First run (downloads sentiment model): ensure network and run without OFFLINE.
"""
import os
import csv
import time
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Shared with bert-analysis.py (same tokenizer + chunking)
THESIS_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = THESIS_ROOT / "articles"
CHUNK_LEN = 512


def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir=str(_CACHE_DIR))


def get_article_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8").strip()


def tokenize_article_chunked(tokenizer, text: str, max_length: int = CHUNK_LEN, return_tensors=None):
    enc = tokenizer.encode(text, add_special_tokens=False)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0
    body_len = max_length - 2
    chunks = []
    for i in range(0, len(enc), body_len):
        piece = enc[i : i + body_len]
        input_ids = [cls_id] + piece + [sep_id] + [pad_id] * (max_length - len(piece) - 2)
        attention_mask = [1] * (len(piece) + 2) + [0] * (max_length - len(piece) - 2)
        chunks.append({"input_ids": input_ids, "attention_mask": attention_mask})
    if return_tensors == "pt":
        for c in chunks:
            c["input_ids"] = torch.tensor([c["input_ids"]], dtype=torch.long)
            c["attention_mask"] = torch.tensor([c["attention_mask"]], dtype=torch.long)
    return chunks

# General Chinese sentiment (bert-base-chinese fine-tuned; 2 classes typical)
SENTIMENT_MODEL = "jackietung/bert-base-chinese-finetuned-sentiment"
BATCH_SIZE = 16  # chunks per forward pass
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_sentiment_model():
    tokenizer = load_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL,
        cache_dir=str(_CACHE_DIR),
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def run_sentiment_on_chunks(model, chunks_pt, batch_size=BATCH_SIZE):
    """Run model on list of chunk dicts (input_ids, attention_mask tensors). Returns (probs, labels)."""
    all_probs = []
    for i in range(0, len(chunks_pt), batch_size):
        batch = chunks_pt[i : i + batch_size]
        input_ids = torch.cat([c["input_ids"] for c in batch], dim=0).to(DEVICE)
        attention_mask = torch.cat([c["attention_mask"] for c in batch], dim=0).to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu())
    probs = torch.cat(all_probs, dim=0)
    return probs


def sentiment_for_article(tokenizer, model, text):
    """One sentiment per article: chunk text, run model on chunks, aggregate by mean probability."""
    chunks = tokenize_article_chunked(tokenizer, text, return_tensors="pt")
    if not chunks:
        return None, None
    # Stack to batch-friendly: each chunk has shape (1, 512)
    probs = run_sentiment_on_chunks(model, chunks)
    mean_probs = probs.mean(dim=0)
    pred_id = mean_probs.argmax().item()
    confidence = mean_probs[pred_id].item()
    id2label = model.config.id2label
    label = id2label.get(pred_id, str(pred_id))
    return label, confidence


def main():
    print(f"Device: {DEVICE}")
    print("Loading tokenizer and sentiment model ...")
    tokenizer, model = load_sentiment_model()

    article_files = sorted(ARTICLES_DIR.rglob("*.txt"))
    max_articles = None  # all
    if max_articles is not None and len(article_files) > max_articles:
        article_files = article_files[:max_articles]
    print(f"Running sentiment on {len(article_files)} articles ...")

    out_path = Path(__file__).resolve().parent / "sentiment_results.csv"
    start = time.perf_counter()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article", "sentiment", "confidence", "n_chunks"])

        for i, path in enumerate(article_files):
            text = get_article_text(path)
            chunks = tokenize_article_chunked(tokenizer, text)
            n_chunks = len(chunks)
            if not chunks:
                w.writerow([str(path), "", 0.0, 0])
                continue
            label, conf = sentiment_for_article(tokenizer, model, text)
            rel = path.relative_to(ARTICLES_DIR) if path.is_relative_to(ARTICLES_DIR) else path
            w.writerow([str(rel), label or "", conf or 0.0, n_chunks])
            if (i + 1) % 200 == 0 or i == 0 or i == len(article_files) - 1:
                print(f"  {i + 1}/{len(article_files)} ...")

    elapsed = time.perf_counter() - start
    print(f"\nDone in {elapsed:.2f} s. Results: {out_path}")
    print(f"  ({elapsed / len(article_files):.3f} s per article)")


if __name__ == "__main__":
    main()
