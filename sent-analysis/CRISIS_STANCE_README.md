# How to Interpret the Crisis Stance Ratings (English)

This guide explains how to read the **crisis_stance_log.csv** results for the 10 articles (and for any larger run).

---

## What We Measured

For each article we:

1. **Pulled out two types of sentences** (only those that mention the global financial crisis / economy):
   - **China + crisis:** sentences that talk about China (中国, 我国, etc.) and the crisis, recovery, or economic policy.
   - **Other countries + crisis:** sentences that talk about other countries (美国, 欧洲, etc.) and the crisis, recovery, or economic policy.

2. **Ran a Chinese BERT sentiment model** on each of those two texts. The model gives one of three labels:
   - **正面** = *positive* (favorable, good, effective, opportunity)
   - **負面** = *negative* (unfavorable, bad, ineffective, problem)
   - **中性** = *neutral* (neither clearly positive nor negative)

3. **Turned those labels into a single “stance”** for the article, using your rules:
   - **Pro-China:** The article frames the crisis as an *opportunity* or China’s response as *effective*, **or** it frames *other countries as suffering or not responding effectively*.
   - **Anti-China:** The article frames the crisis as *bad for China* or *China’s handling as problematic*, **or** it frames *other countries as doing well*.
   - **Neutral:** The relevant sentences don’t clearly lean pro or anti (mostly neutral sentiment, or no relevant text).

---

## How to Read the CSV Columns

| Column | Meaning in English |
|--------|--------------------|
| **article** | Which article (file path under `articles/`). |
| **stance** | Overall rating: **pro-China**, **anti-China**, **neutral**, or **N/A** (no crisis-related sentences found). |
| **pro_score** | Strength of “pro-China” signal (0–1). Comes from: positive sentiment on China text, or negative sentiment on other-countries text. |
| **anti_score** | Strength of “anti-China” signal (0–1). Comes from: negative sentiment on China text, or positive sentiment on other-countries text. |
| **china_sentiment** | Sentiment of the *China + crisis* excerpt: 正面 / 負面 / 中性 (or blank if no such text). |
| **china_confidence** | How confident the model is in that China sentiment (0–1). |
| **other_countries_sentiment** | Sentiment of the *other countries + crisis* excerpt: 正面 / 負面 / 中性 (or blank if none). |
| **other_confidence** | Model confidence for that other-countries sentiment (0–1). |
| **china_excerpt_len** | Length (in characters) of the China+crisis text we analyzed (0 if none). |
| **other_excerpt_len** | Length of the other-countries+crisis text (0 if none). |

**Stance is decided by comparing pro_score and anti_score:**  
- If **pro_score > anti_score** → **pro-China**  
- If **anti_score > pro_score** → **anti-China**  
- If they’re equal (usually both 0) → **neutral**

---

## Interpreting the 10 Articles (Your Current Run)

- **Articles 1, 5, 7, 9, 10**  
  - **Stance: neutral** (pro_score and anti_score both 0).  
  - There was **no** China+crisis text (china_excerpt_len = 0). The other-countries text was all **中性** (neutral), so it didn’t add any pro- or anti-China weight.  
  - **In plain English:** The article didn’t give us clear pro- or anti-China framing in the parts we analyzed.

- **Articles 2, 3, 4, 8**  
  - **Stance: neutral.**  
  - Both China+crisis and other-countries+crisis excerpts were **中性**. So again no net pro- or anti-China signal.  
  - **In plain English:** The tone in both the China-related and other-countries-related crisis sentences was neutral.

- **Article 6**  
  - **Stance: pro-China** (pro_score ≈ 0.45, anti_score = 0).  
  - China excerpt: **中性**. Other-countries excerpt: **負面** (negative) with confidence ~0.45.  
  - **In plain English:** The article’s discussion of *other countries* in the crisis was read as negative (e.g. suffering, ineffective). By your rules, that counts as **pro-China**, so the overall stance is pro-China.

---

## Short Glossary

- **正面** = positive (favorable / effective / opportunity).  
- **負面** = negative (unfavorable / ineffective / problem).  
- **中性** = neutral.  
- **pro_score** = evidence for “pro-China” framing (crisis as opportunity / China effective, or others suffering/ineffective).  
- **anti_score** = evidence for “anti-China” framing (crisis bad for China / China’s handling poor, or others doing well).

If you want the same logic applied to more articles (e.g. 100), change **N_ARTICLES** in `crisis_stance_log.py` and re-run the script; the way you interpret the ratings stays the same.
