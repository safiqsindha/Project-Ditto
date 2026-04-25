# Session Log

## Session 10 — 2026-04-25: Public-facing methodology correction disclosure

**Action:** Added methodology correction disclosure to public-facing documents.

**Files modified:**
- `README.md` — added "## Methodology Correction (April 2026)" section after the Hypothesis section; includes corrected numbers, links to `CORRECTED_SCORING.md` and `src/scorer_corrected.py`
- `RESULTS.md` — added correction notice in the Status block at the top; split the Headline Numbers section into "Original numbers" and "Corrected numbers" subsections with a side-by-side table; added explanatory paragraph
- `CHANGELOG.md` — added `[0.1.1] — 2026-04-25` entry documenting both the corrected scorer (Session 9) and this disclosure (Session 10)

**Files NOT modified (per constraints):**
- `src/scorer.py` — original scorer retained unmodified for reproducibility
- `src/scorer_corrected.py` — corrected scorer unchanged
- `results/scored_corrected.json` — corrected output unchanged
- `CORRECTED_SCORING.md` — correction document unchanged

**Findings preserved:**

| Model | Published (original) | Corrected | Tier change? |
|---|---|---|---|
| Haiku 4.5 | gap 0.059, moderate_positive | gap 0.066, moderate_positive | None |
| Sonnet 4.6 | gap 0.201, strong_positive | gap 0.206, strong_positive | None |

Both Haiku (moderate_positive) and Sonnet (strong_positive) findings are publicly visible with both original and corrected numbers. The methodology correction is disclosed as additive: original numbers remain in the published record.

---

## Session 9 — 2026-04-25: v2 Methodology Correction Applied to v1

**Reason:** The v2 methodology review identified three statistical issues in `src/scorer.py`. Because v1 used the same scorer, the correction was applied here to verify that v1's published results are robust to the methodology fix.

**Issues addressed:**
1. Layer 1: two-sample proportion z-test (unpaired) → McNemar's test (paired)
2. Layer 2: Welch's t-test (unpaired) → paired t-test
3. No multiple-comparisons correction → Bonferroni across 2 primary cells (haiku, sonnet)
4. No explicit (chain_id, eval_seed) pair alignment → enforced

**Files added (additive — no primary artifacts modified):**
- `src/scorer_corrected.py` — corrected scorer
- `results/scored_corrected.json` — output of corrected scorer
- `CORRECTED_SCORING.md` — side-by-side comparison document
- `SESSION_LOG.md` — this file

**Do v1 headline results hold under corrected scoring?**

**YES.** Both outcome tiers are unchanged:

| Model | Published (original) | Corrected | Tier change? |
|---|---|---|---|
| Haiku 4.5 | gap 0.059, moderate_positive | gap 0.066, moderate_positive | None |
| Sonnet 4.6 | gap 0.201, strong_positive | gap 0.206, strong_positive | None |

Pre-registered minimum criterion (gap ≥ 0.05, p < 0.05 on at least one model): met under both uncorrected and Bonferroni-corrected (n=2) tests. Corrected p-values are p ≪ 10⁻²⁵ for haiku and p ≪ 10⁻²¹² for sonnet — far beyond any significance threshold.

**Do any public-facing v1 documents need updating?**

No headline change is required. A methodology note in the README (or a new section in RESULTS.md) pointing to `CORRECTED_SCORING.md` is recommended but requires co-author approval before being added to the public repo.

---

## Sessions 1–8 — 2026-04-21: Full pipeline + evaluation

Sessions 1–8 are documented in `CHANGELOG.md` and commit history. Brief summary:

- Sessions 1–4: Environment, data acquisition, translation (frozen at `T-v1.1-frozen`), chain construction
- Session 5: Evaluation runner; primary T=0 s42 run (haiku + sonnet, 19,224 chains, 38,448 requests)
- Session 6: Batch runner and multi-seed variance study (T=0.5, seeds 1337 & 7919)
- Session 7: Scorer; Layer 1/2/3 metrics; primary results confirmed strong_positive (sonnet), moderate_positive (haiku)
- Session 8: Full evaluation pipeline confirmed; multi-seed variance study results identical outcome tiers across all three conditions
