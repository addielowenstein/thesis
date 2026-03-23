"""
BERT-based tokenization for Mandarin articles.
Uses bert-base-chinese (Google), the standard BERT for Chinese text.
"""
import os
import time
from pathlib import Path

# Cache models inside project so we don't need write access to ~/.cache
_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE_DIR))

from transformers import AutoTokenizer

# Standard BERT for Mandarin; proven for Chinese NLP (character/subword tokenization)
MODEL_NAME = "bert-base-chinese"

# Path to thesis root (parent of sent-analysis)
THESIS_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = THESIS_ROOT / "articles"


def load_tokenizer():
    """Load the Chinese BERT tokenizer (downloads on first run)."""
    return AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(_CACHE_DIR))


def get_article_text(filepath: Path) -> str:
    """Read article file and return its full text (for tokenization)."""
    return filepath.read_text(encoding="utf-8").strip()


CHUNK_LEN = 512


def tokenize_article(tokenizer, text: str, max_length: int = 512, return_tensors=None):
    """
    Tokenize a single article's text with the BERT tokenizer (single segment; truncates).
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors=return_tensors,
        return_attention_mask=True,
    )


def tokenize_article_chunked(
    tokenizer,
    text: str,
    max_length: int = CHUNK_LEN,
    return_tensors=None,
):
    """
    Tokenize an article into 512-token chunks (no truncation). Long articles become
    multiple chunks, each [CLS] + tokens + [SEP] + padding to max_length.

    Returns:
        List of dicts, each with "input_ids" and "attention_mask" of length max_length.
    """
    enc = tokenizer.encode(text, add_special_tokens=False)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0
    body_len = max_length - 2  # room for [CLS] and [SEP]

    chunks = []
    for i in range(0, len(enc), body_len):
        piece = enc[i : i + body_len]
        input_ids = [cls_id] + piece + [sep_id]
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        attention_mask = [1] * (len(piece) + 2) + [0] * pad_len
        chunks.append({"input_ids": input_ids, "attention_mask": attention_mask})

    if return_tensors == "pt":
        import torch
        for c in chunks:
            c["input_ids"] = torch.tensor([c["input_ids"]], dtype=torch.long)
            c["attention_mask"] = torch.tensor([c["attention_mask"]], dtype=torch.long)

    return chunks


def main():
    print(f"Loading tokenizer: {MODEL_NAME} ...")
    tokenizer = load_tokenizer()

    # All articles (optionally cap for a quick run)
    article_files = sorted(ARTICLES_DIR.rglob("*.txt"))
    max_articles = 1000  # set to None to process all
    if max_articles is not None and len(article_files) > max_articles:
        article_files = article_files[:max_articles]
    assert article_files, "No .txt articles found under articles/"

    print(f"Tokenizing {len(article_files)} articles in 512-token chunks ...")
    start = time.perf_counter()
    total_chunks = 0
    for i, path in enumerate(article_files):
        text = get_article_text(path)
        chunks = tokenize_article_chunked(tokenizer, text)
        total_chunks += len(chunks)
        if (i + 1) % 200 == 0 or i == 0 or i == len(article_files) - 1:
            print(f"  {i + 1}/{len(article_files)} articles, {total_chunks} chunks so far ...")
    elapsed = time.perf_counter() - start

    print(f"\nTotal: {total_chunks} chunks from {len(article_files)} articles")
    print(f"Total time: {elapsed:.2f} s  ({elapsed / len(article_files):.3f} s per article)")


if __name__ == "__main__":
    main()
