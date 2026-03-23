"""
Stance sentiment analysis: (1) China's response/role in global financial crisis,
(2) Other countries' response/role. Uses Chinese BERT sentiment on extracted
relevant sentences; logs results per article. Starts with 10 articles.
"""
import os
import re
import csv
import time
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

THESIS_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = THESIS_ROOT / "articles"
CHUNK_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SENTIMENT_MODEL = "jackietung/bert-base-chinese-finetuned-sentiment"

# Keywords to find sentences about China vs other countries in crisis context
CHINA_KEYWORDS = ("中国", "我国", "中国政府", "中方", "中国央行", "中国经济", "国内")
CRISIS_KEYWORDS = ("金融危机", "经济危机", "危机", "金融", "经济", "复苏", "应对", "刺激", "衰退", "全球")
OTHER_COUNTRY_KEYWORDS = ("美国", "欧洲", "欧盟", "西方", "日本", "英国", "各国", "发达国家", "美欧", "华尔街", "国际")


def get_article_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8").strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences (by 。 ； and double newline)."""
    text = re.sub(r"\s+", " ", text.strip())
    # Split on sentence-ending punctuation
    parts = re.split(r"[。；\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def has_any(s: str, keywords: tuple) -> bool:
    return any(k in s for k in keywords)


def extract_china_crisis_text(text: str) -> str:
    """Concatenate sentences that mention China + crisis/finance."""
    sents = split_sentences(text)
    chosen = [s for s in sents if has_any(s, CHINA_KEYWORDS) and has_any(s, CRISIS_KEYWORDS)]
    return " ".join(chosen) if chosen else ""


def extract_other_countries_crisis_text(text: str) -> str:
    """Concatenate sentences that mention other countries + crisis/finance."""
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
    """Run sentiment on one text (truncate to max_len). Returns (label, confidence) or (None, None) if empty."""
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
    print(f"Device: {DEVICE}")
    print("Loading tokenizer and sentiment model ...")
    tokenizer, model = load_models()

    # Use all article .txt files under articles/ (currently ~931)
    article_files = sorted(ARTICLES_DIR.rglob("*.txt"))
    assert article_files, "No articles found."
    print(f"Analyzing stance sentiment for {len(article_files)} articles ...\n")

    log_path = Path(__file__).resolve().parent / "stance_sentiment_log.csv"
    start = time.perf_counter()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "article",
            "china_stance_sentiment",
            "china_confidence",
            "china_excerpt_len",
            "other_countries_stance_sentiment",
            "other_countries_confidence",
            "other_excerpt_len",
        ])

        for i, path in enumerate(article_files):
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
                str(rel),
                china_label or "N/A",
                f"{china_conf:.4f}" if china_conf is not None else "",
                len(china_text),
                other_label or "N/A",
                f"{other_conf:.4f}" if other_conf is not None else "",
                len(other_text),
            ])

            print(f"[{i+1}/{len(article_files)}] {rel}")
            print(f"  1) China's response/role:     {china_label or 'N/A'} (conf: {china_conf:.3f})" if china_conf is not None else "  1) China's response/role:     N/A (no matching sentences)")
            print(f"  2) Other countries' role:   {other_label or 'N/A'} (conf: {other_conf:.3f})" if other_conf is not None else "  2) Other countries' role:   N/A (no matching sentences)")
            print()

    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f} s. Log: {log_path}")


if __name__ == "__main__":
    main()
