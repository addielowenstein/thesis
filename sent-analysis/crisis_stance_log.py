"""
Crisis stance: pro-China vs anti-China re global financial crisis.

Mapping (Chinese BERT sentiment → stance):
- Pro-China: crisis as opportunity / China's response effective (China text → 正面);
             other countries suffering / not responding effectively (other text → 負面).
- Anti-China: crisis negative for China / problems with China's handling (China text → 負面);
              other countries doing well (other text → 正面).
- Neutral: otherwise.

Same 100 articles (or set N_ARTICLES); start with 10 for a quick run.
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
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SENTIMENT_MODEL = "jackietung/bert-base-chinese-finetuned-sentiment"

# Labels from the sentiment model (typical: 正面 positive, 負面 negative, 中性 neutral)
POSITIVE_LABEL = "正面"
NEGATIVE_LABEL = "負面"
NEUTRAL_LABEL = "中性"

CHINA_KEYWORDS = ("中国", "我国", "中国政府", "中方", "中国央行", "中国经济", "国内")
CRISIS_KEYWORDS = ("金融危机", "经济危机", "危机", "金融", "经济", "复苏", "应对", "刺激", "衰退", "全球")
OTHER_COUNTRY_KEYWORDS = ("美国", "欧洲", "欧盟", "西方", "日本", "英国", "各国", "发达国家", "美欧", "华尔街", "国际")

N_ARTICLES = 100  # set to None for all articles


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


def stance_from_sentiments(china_label, china_conf, other_label, other_conf):
    """
    Pro-China: China 正面 (opportunity/effective) or other 負面 (others suffering/ineffective).
    Anti-China: China 負面 (negative for China/poor handling) or other 正面 (others doing well).
    """
    pro_score = 0.0
    anti_score = 0.0
    if china_label == POSITIVE_LABEL and china_conf is not None:
        pro_score += china_conf
    elif china_label == NEGATIVE_LABEL and china_conf is not None:
        anti_score += china_conf
    if other_label == NEGATIVE_LABEL and other_conf is not None:
        pro_score += other_conf  # others suffering/ineffective → pro-China
    elif other_label == POSITIVE_LABEL and other_conf is not None:
        anti_score += other_conf  # others doing well → anti-China

    if pro_score > anti_score:
        return "pro-China", pro_score, anti_score
    if anti_score > pro_score:
        return "anti-China", pro_score, anti_score
    return "neutral", pro_score, anti_score


def main():
    print(f"Device: {DEVICE}")
    print("Loading tokenizer and sentiment model ...")
    tokenizer, model = load_models()

    article_files = sorted(ARTICLES_DIR.rglob("*.txt"))
    if N_ARTICLES is not None:
        article_files = article_files[:N_ARTICLES]
    assert article_files, "No articles found."
    print(f"Crisis stance (pro-/anti-China) for {len(article_files)} articles ...\n")

    log_path = Path(__file__).resolve().parent / "crisis_stance_log.csv"
    start = time.perf_counter()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "article",
            "stance",
            "pro_score",
            "anti_score",
            "china_sentiment",
            "china_confidence",
            "other_countries_sentiment",
            "other_confidence",
            "china_excerpt_len",
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

            stance, pro_score, anti_score = stance_from_sentiments(china_label, china_conf, other_label, other_conf)
            if not china_text and not other_text:
                stance = "N/A"

            w.writerow([
                str(rel),
                stance,
                f"{pro_score:.4f}",
                f"{anti_score:.4f}",
                china_label or "",
                f"{china_conf:.4f}" if china_conf is not None else "",
                other_label or "",
                f"{other_conf:.4f}" if other_conf is not None else "",
                len(china_text),
                len(other_text),
            ])

            print(f"[{i+1}/{len(article_files)}] {rel}")
            print(f"  Stance: {stance}  (pro_score={pro_score:.3f}, anti_score={anti_score:.3f})")
            print(f"  China excerpt: {china_label or 'N/A'} (conf: {china_conf:.3f})" if china_conf is not None else "  China excerpt: N/A")
            print(f"  Other countries: {other_label or 'N/A'} (conf: {other_conf:.3f})" if other_conf is not None else "  Other countries: N/A")
            print()

    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f} s. Log: {log_path}")


if __name__ == "__main__":
    main()
