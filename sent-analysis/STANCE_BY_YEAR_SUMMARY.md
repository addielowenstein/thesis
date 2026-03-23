# Stance Sentiment by Year: Summary and Trends

**Data:** 300 articles (100 from 2013, 100 from 2014, 100 from 2015), People’s Daily.  
**Measures:** (1) Sentiment toward **China’s response/role** in the global financial crisis; (2) Sentiment toward **other countries’ response/role**. Labels: 正面 (positive), 中性 (neutral), 負面 (negative), or N/A (no matching sentences).

---

## Overall (300 articles)

| Stance | China's response/role | Other countries' response/role |
|--------|------------------------|---------------------------------|
| **正面** (positive) | 1 (0.3%) | 1 (0.3%) |
| **中性** (neutral)  | 211 (70.3%) | 258 (86.0%) |
| **負面** (negative) | 22 (7.3%) | 41 (13.7%) |
| **N/A** (no excerpt) | 66 (22.0%) | 0 (0%) |

- **China:** Most articles with a China+crisis excerpt are **neutral**; a small share are **negative**. Only one article is **positive**. About **22%** of articles have no China+crisis sentences (N/A).
- **Other countries:** Almost all articles have an other-countries+crisis excerpt. Sentiment is mostly **neutral**; **negative** is more common than for China (14% vs 7% among non-N/A).

---

## By year

### China's response/role

| Year | N/A | 中性 | 負面 | 正面 |
|------|-----|-----|------|------|
| 2013 | 23% | 67% | **10%** | 0% |
| 2014 | 26% | 68% | 6% | 0% |
| 2015 | 17% | 76% | 6% | 1% |

- **Negative (負面) on China** is **highest in 2013 (10%)**, then **6% in 2014 and 2015**. So the share of articles framing China’s role/handling negatively is a bit lower in the later years.
- **N/A** is **lowest in 2015 (17%)**, i.e. more articles in 2015 contain China+crisis sentences.
- **Positive (正面)** appears only once, in 2015.

### Other countries' response/role

| Year | 中性 | 負面 | 正面 |
|------|-----|------|------|
| 2013 | 90% | **10%** | 0% |
| 2014 | 81% | **19%** | 0% |
| 2015 | 87% | 12% | 1% |

- **Negative (負面) on other countries** is **highest in 2014 (19%)**, then 12% in 2015 and 10% in 2013. So **2014** stands out as the year with the most negative framing of others’ response/role.
- **Positive (正面)** on others appears only once, in 2015.

---

## Trends and interpretation

1. **China: slight softening of negative framing over time**  
   The share of articles with **negative** sentiment toward China’s response/role falls from **10% (2013)** to **6% (2014 and 2015)**. Neutral remains dominant; positive is almost absent.

2. **Other countries: 2014 as peak of negative framing**  
   The share **negative** on other countries rises from **10% (2013)** to **19% (2014)**, then drops to **12% (2015)**. So coverage is most critical of other countries’ crisis response in **2014**, and a bit less so again in 2015.

3. **Contrast China vs others**  
   In **2014**, negative sentiment is **much more often** directed at other countries (19%) than at China (6%). That fits a narrative where others are portrayed as suffering or ineffective relative to China.

4. **Coverage of China in crisis discourse**  
   The **drop in N/A for China** from 2013–2014 (23–26%) to **17% in 2015** suggests that by 2015, more articles explicitly discuss China in a crisis/finance context, though still mostly in a neutral tone.

5. **Caveats**  
   Labels come from a general Chinese BERT sentiment model on keyword-extracted excerpts, not from a custom “pro-/anti-China” or “effectiveness” classifier. Sample is 100 articles per year; percentages can shift with a different sample or full corpus.

---

## Bottom line

- **China:** Mostly neutral; **negative share falls from 2013 (10%) to 2014–2015 (6%)**; positive is negligible.
- **Other countries:** Mostly neutral; **negative share peaks in 2014 (19%)**, then is 12% in 2015 and 10% in 2013.
- **2014** is the year where other countries are framed most negatively relative to both China and to the other two years, consistent with a “China faring better / others faring worse” narrative in that year.
