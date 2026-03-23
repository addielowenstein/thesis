"""
Classify sentiment of crisis coverage in People's Daily using Qwen with custom rules.

(1) China impact: negative = only damages; positive = China effectively handling (with or without damages).
(2) Other countries: negative = only suffering; positive = effectively handling (even with damages).

Samples 100 articles (stratified by year). Two modes:

  --local   Use open-source Qwen2.5 via Hugging Face (no API key). Slower on CPU.
  (default) Use Qwen API: set DASHSCOPE_API_KEY or OPENAI_API_KEY + QWEN_BASE_URL.
"""
import os
import re
import csv
import time
import random
import json
import argparse
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
THESIS_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = THESIS_ROOT / "articles"
SAMPLE_SIZE = 100
MAX_EXCERPT_LEN = 3500  # chars per excerpt to stay within context
LOCAL_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # small enough for CPU; use 3B/7B if you have GPU

# Same keywords as stance_sentiment_log for consistency
CHINA_KEYWORDS = ("中国", "我国", "中国政府", "中方", "中国央行", "中国经济", "国内")
CRISIS_KEYWORDS = ("金融危机", "经济危机", "危机", "金融", "经济", "复苏", "应对", "刺激", "衰退", "全球")
OTHER_COUNTRY_KEYWORDS = ("美国", "欧洲", "欧盟", "西方", "日本", "英国", "各国", "发达国家", "美欧", "华尔街", "国际")


def get_article_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="ignore").strip()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"[。；\n]+", text)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def has_any(s: str, keywords: tuple) -> bool:
    return any(k in s for k in keywords)


def extract_china_crisis_text(text: str, max_len: int = MAX_EXCERPT_LEN) -> str:
    sents = split_sentences(text)
    chosen = [s for s in sents if has_any(s, CHINA_KEYWORDS) and has_any(s, CRISIS_KEYWORDS)]
    out = " ".join(chosen) if chosen else ""
    return out[:max_len] if len(out) > max_len else out


def extract_other_countries_crisis_text(text: str, max_len: int = MAX_EXCERPT_LEN) -> str:
    sents = split_sentences(text)
    chosen = [s for s in sents if has_any(s, OTHER_COUNTRY_KEYWORDS) and has_any(s, CRISIS_KEYWORDS)]
    out = " ".join(chosen) if chosen else ""
    return out[:max_len] if len(out) > max_len else out


SYSTEM_PROMPT = """You are a sentiment classifier for Chinese news articles about the global financial crisis (国际金融危机/全球金融危机). You will receive excerpts from one article: (A) sentences about China and the crisis, (B) sentences about other countries (e.g. US, Europe) and the crisis. Classify according to these rules only.

(1) China impact — choose exactly one:
  - positive: The article mentions China successfully or effectively handling/responding to the crisis (e.g. stimulus, reform, stabilizing economy, helping others), either alone or in addition to any description of damages.
  - negative: The article only describes damages or negative effects of the crisis on China, with no mention of China handling the crisis effectively.
  - neutral: No relevant content about China and the crisis in the excerpts.

(2) Other countries impact — choose exactly one:
  - positive: The article describes other countries (or regions like US, Europe, EU) effectively handling or responding to the crisis, even if there are also sentences about damages.
  - negative: The article describes other countries suffering from the crisis (damages, recession, unemployment, etc.) without describing them handling it effectively.
  - neutral: No relevant content about other countries and the crisis in the excerpts.

Respond with a JSON object only, no other text:
{"china":"positive|negative|neutral","other_countries":"positive|negative|neutral"}"""


def build_user_prompt(china_excerpt: str, other_excerpt: str) -> str:
    parts = []
    if china_excerpt:
        parts.append("【与中国及危机相关的句子】\n" + china_excerpt)
    else:
        parts.append("【与中国及危机相关的句子】\n（无）")
    if other_excerpt:
        parts.append("\n【与其他国家及危机相关的句子】\n" + other_excerpt)
    else:
        parts.append("\n【与其他国家及危机相关的句子】\n（无）")
    parts.append("\n请按规则对上述内容分类，只输出JSON。")
    return "\n".join(parts)


def _parse_json_response(content: str) -> tuple[str, str]:
    """Parse model output to (china_sentiment, other_countries_sentiment)."""
    content = (content or "").strip()
    if "```" in content:
        content = re.sub(r"^.*?```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```.*$", "", content)
    content = content.strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to find a JSON object in the text
        match = re.search(r"\{[^{}]*\"china\"[^{}]*\"other_countries\"[^{}]*\}", content)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return ("error", "error")
        else:
            return ("error", "error")
    china = data.get("china", "neutral").lower()
    other = data.get("other_countries", "neutral").lower()
    if china not in ("positive", "negative", "neutral"):
        china = "neutral"
    if other not in ("positive", "negative", "neutral"):
        other = "neutral"
    return (china, other)


def call_qwen_api(china_excerpt: str, other_excerpt: str, client, model: str) -> tuple[str, str]:
    """Call Qwen via OpenAI-compatible API. Returns (china_sentiment, other_countries_sentiment)."""
    user = build_user_prompt(china_excerpt, other_excerpt)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _parse_json_response(content)
    except Exception:
        return ("error", "error")


def call_qwen_local(china_excerpt: str, other_excerpt: str, tokenizer, model, device: str) -> tuple[str, str]:
    """Run open-source Qwen2.5 locally. No API key needed."""
    user = build_user_prompt(china_excerpt, other_excerpt)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with __import__("torch").no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse_json_response(response)
    except Exception:
        return ("error", "error")


def get_sampled_articles():
    """Return list of 100 article paths stratified by year."""
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
    return list(dict.fromkeys(sampled))[:SAMPLE_SIZE]


def main():
    parser = argparse.ArgumentParser(description="Qwen crisis sentiment: API (default) or --local (no API key).")
    parser.add_argument("--local", action="store_true", help="Use open-source Qwen2.5 locally (Hugging Face); no API key.")
    args = parser.parse_args()

    articles_to_process = get_sampled_articles()
    out_path = Path(__file__).resolve().parent / "qwen_crisis_sentiment.csv"

    if args.local:
        os.environ.setdefault("HF_HOME", str(_CACHE_DIR))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        print(f"Loading local model {LOCAL_MODEL_ID} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, cache_dir=str(_CACHE_DIR), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, cache_dir=str(_CACHE_DIR), trust_remote_code=True).to(device)
        model.eval()
        print(f"Processing {len(articles_to_process)} articles (stratified by year) ...")
        start = time.perf_counter()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["article", "year", "china_sentiment", "other_countries_sentiment"])
            for i, path in enumerate(articles_to_process):
                try:
                    rel = path.relative_to(ARTICLES_DIR)
                except ValueError:
                    rel = Path(path.name)
                year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
                article_id = str(rel)
                text = get_article_text(path)
                china_text = extract_china_crisis_text(text)
                other_text = extract_other_countries_crisis_text(text)
                china_sent, other_sent = call_qwen_local(china_text, other_text, tokenizer, model, device)
                w.writerow([article_id, year, china_sent, other_sent])
                if (i + 1) % 10 == 0:
                    print(f"[{i+1}/{len(articles_to_process)}] {article_id} -> China: {china_sent}, Others: {other_sent}")
        elapsed = time.perf_counter() - start
        print(f"Done in {elapsed:.2f} s. Output: {out_path}")
        return

    # API mode
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Install openai: pip install openai")
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY or OPENAI_API_KEY. Or use --local to run open-source Qwen with no API key.")
    base_url = os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("QWEN_MODEL", "qwen3.5-plus")
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    print(f"Using API: {model}, base_url: {base_url}")
    print(f"Processing {len(articles_to_process)} articles (stratified by year) ...")
    start = time.perf_counter()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article", "year", "china_sentiment", "other_countries_sentiment"])
        for i, path in enumerate(articles_to_process):
            try:
                rel = path.relative_to(ARTICLES_DIR)
            except ValueError:
                rel = Path(path.name)
            year = rel.parts[0] if len(rel.parts) > 1 else "unknown"
            article_id = str(rel)
            text = get_article_text(path)
            china_text = extract_china_crisis_text(text)
            other_text = extract_other_countries_crisis_text(text)
            china_sent, other_sent = call_qwen_api(china_text, other_text, client, model)
            w.writerow([article_id, year, china_sent, other_sent])
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(articles_to_process)}] {article_id} -> China: {china_sent}, Others: {other_sent}")
            time.sleep(0.3)
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f} s. Output: {out_path}")


if __name__ == "__main__":
    main()
