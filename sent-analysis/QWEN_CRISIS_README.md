# Qwen crisis sentiment

Script: **`qwen_crisis_sentiment.py`**

Uses **Qwen** to classify how the global financial crisis is framed in a **sample of 100 articles** (stratified by year) from People’s Daily. Two modes:

- **`--local`** — Run the **open-source** Qwen2.5 model locally (Hugging Face). **No API key.** Uses `Qwen/Qwen2.5-1.5B-Instruct` by default (reasonable on CPU).
- **Default** — Call Qwen via an **API** (e.g. Alibaba DashScope). Requires an API key.

## Classification rules

- **China impact**
  - **Negative:** The article only describes damages of the crisis on China.
  - **Positive:** The article mentions China successfully or effectively handling the crisis (alone or in addition to damages).

- **Other countries impact**
  - **Negative:** Describes other countries suffering from the crisis without describing them handling it effectively.
  - **Positive:** Describes other countries effectively handling the crisis, even if damages are also mentioned.

- **Neutral:** No relevant excerpt for that dimension (no China+crisis or no other-countries+crisis sentences).

## Run locally (no API key)

Uses the open-source Qwen2.5 model from Hugging Face. Slower on CPU; fine on GPU.

```bash
cd sent-analysis
.venv/bin/python qwen_crisis_sentiment.py --local
```

Requires `transformers` and `torch` (already in `requirements.txt`). Model is cached under `sent-analysis/.cache/huggingface`.

## Run via API (Qwen3.5 or other hosted Qwen)

1. Set an API key (e.g. Alibaba DashScope):
   ```bash
   export DASHSCOPE_API_KEY=sk-...
   ```
   Or for a custom endpoint:
   ```bash
   export OPENAI_API_KEY=your-key
   export QWEN_BASE_URL=https://your-endpoint/v1
   export QWEN_MODEL=your-model-name
   ```
2. Install the OpenAI client: `pip install openai`
3. Run without `--local`:
   ```bash
   .venv/bin/python qwen_crisis_sentiment.py
   ```

Output: **`qwen_crisis_sentiment.csv`** with columns:
- `article` — path (e.g. `2014/39-pg12.2_005.txt`)
- `year` — 2013 / 2014 / 2015
- `china_sentiment` — positive | negative | neutral
- `other_countries_sentiment` — positive | negative | neutral

Excerpts are the same sentence sets used in `stance_sentiment_log.py` (China+crisis keywords and other-countries+crisis keywords), truncated to 3500 characters per dimension to fit context limits.
