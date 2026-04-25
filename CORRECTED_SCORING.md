# Corrected Scoring — v1 Methodological Review

**Date:** 2026-04-25  
**Reviewer:** Applied v2 methodology correction to v1 data  
**Status:** v1 headline results **confirmed** under corrected scoring

---

## Why this correction was applied

The v2 methodology review identified three statistical issues in the original `src/scorer.py`. Because v1 used the same scorer, the correction is applied here to verify that v1's published results are robust.

The three issues:

| # | Issue | Original (v1) | Corrected |
|---|---|---|---|
| 1 | Layer 1 test | Two-sample proportion z-test (unpaired) | McNemar's test (paired, continuity-corrected) |
| 2 | Layer 2 test | Welch's t-test (unpaired) | Paired t-test |
| 3 | Multiple comparisons | None | Bonferroni across 2 primary cells |

A fourth structural fix was also applied: explicit `(chain_id, eval_seed)` pair alignment is now enforced. The original scorer pooled real and shuffled results independently without verifying alignment.

**What was NOT changed:** v1's filter rule (reference-miss exclusion) is preserved and applied symmetrically — a pair enters Layer 1 analysis only if both the real and shuffled evaluations obtained a non-None reference lookup. V2's actionable-constraint-type filter is **not** imported; v1 had no such filter.

---

## Data structure and pair alignment

| Item | Value |
|---|---|
| Real chains | 4,806 (p1 + p2 perspectives across 2,500 matches) |
| Shuffled chains | 14,418 (3 shuffle seeds × 4,806) |
| Eval configs | T=0 seed 42 (primary); T=0.5 seed 1337; T=0.5 seed 7919 |
| Total pairs built | 86,508 (4,806 × 3 shuffle seeds × 6 eval configs) |
| Pairs excluded (no real eval) | 0 |
| Pairs excluded (no shuffled eval) | 0 |

### Reference-miss breakdown (all pairs, all configs combined)

| Category | Count | % of total |
|---|---:|---:|
| Both conditions: ref hit ✓ | 64,116 | 74.1% |
| Real miss only | 5,196 | 6.0% |
| Shuffled miss only | 12,564 | 14.5% |
| Both miss | 4,632 | 5.4% |
| **Total pairs** | **86,508** | 100% |

Shuffled-only misses (14.5%) exceed real-only misses (6.0%) by ~2.4×, consistent with the asymmetric reference-miss finding reported in RESULTS.md §"Robustness check": shuffling destroys information-reveal sequence, causing ~2× higher reference-miss rates on shuffled chains.

**Primary T=0 s42 Layer 1 pairs per model:** 10,686 (out of 14,418 potential; 74.1% valid).  
Original scorer used: 4,260 real + 11,552 shuffled independently (not paired).

---

## Side-by-side comparison: Layer 1 (primary T=0, seed 42)

### Haiku 4.5

| Metric | Original (z-test, unpaired) | Corrected (McNemar, paired) | Change |
|---|---|---|---|
| Test | Two-sample proportion z-test | McNemar continuity-corrected | — |
| Real rate | 0.3023 | 0.3088 | +0.0065 |
| Shuffled rate | 0.2432 | 0.2428 | −0.0004 |
| **Gap** | **0.0591** | **0.0660** | **+0.007** |
| Test statistic | z = 7.52 | χ² = 118.31, df=1 | — |
| p-value (raw) | ≈ 0 (p ≪ 10⁻¹⁰) | ≈ 0 (p ≪ 10⁻²⁶) | — |
| Bonferroni-corrected p (n=2) | N/A | ≈ 0 (p ≪ 10⁻²⁵) | — |
| Significant at α=0.05? | ✅ | ✅ | unchanged |
| Significant at Bonferroni α=0.025? | N/A | ✅ | — |
| n_real / n_shuffled (original) | 4,260 / 11,552 | — | — |
| n_pairs (corrected) | — | 10,686 | — |
| **Outcome tier** | **moderate_positive** | **moderate_positive** | **unchanged** |
| Layer 1 threshold met (gap ≥ 0.05, p < 0.05)? | ✅ | ✅ | unchanged |

### Sonnet 4.6

| Metric | Original (z-test, unpaired) | Corrected (McNemar, paired) | Change |
|---|---|---|---|
| Test | Two-sample proportion z-test | McNemar continuity-corrected | — |
| Real rate | 0.4453 | 0.4496 | +0.0043 |
| Shuffled rate | 0.2447 | 0.2432 | −0.0015 |
| **Gap** | **0.2006** | **0.2063** | **+0.006** |
| Test statistic | z = 24.45 | χ² = 981.93, df=1 | — |
| p-value (raw) | ≈ 0 (p ≪ 10⁻¹⁰) | ≈ 0 (p ≪ 10⁻²¹³) | — |
| Bonferroni-corrected p (n=2) | N/A | ≈ 0 (p ≪ 10⁻²¹²) | — |
| Significant at α=0.05? | ✅ | ✅ | unchanged |
| Significant at Bonferroni α=0.025? | N/A | ✅ | — |
| n_real / n_shuffled (original) | 4,260 / 11,552 | — | — |
| n_pairs (corrected) | — | 10,686 | — |
| **Outcome tier** | **strong_positive** | **strong_positive** | **unchanged** |
| Strong-positive threshold met (gap ≥ 0.08, p < 0.01)? | ✅ | ✅ | unchanged |

---

## Side-by-side comparison: Layer 2 (primary T=0, seed 42)

### Haiku 4.5

| Metric | Original (Welch t, unpaired) | Corrected (paired t) | Change |
|---|---|---|---|
| Test | Welch's t-test | Paired t-test | — |
| Real mean | 0.1643 | 0.1643 | 0.0000 |
| Shuffled mean | 0.1345 | 0.1345 | 0.0000 |
| **Gap** | **0.0298** | **0.0298** | **0.0000** |
| t-statistic | 6.04 | 8.81 | larger (paired) |
| p-value | ≈ 0 | ≈ 0 | unchanged |
| n (original) | 4,260 real / 11,552 shuffled | — | — |
| n_pairs (corrected) | — | 14,418 | — |
| Layer 2 threshold met (gap ≥ 0.04)? | ❌ 0.0298 | ❌ 0.0298 | unchanged |

### Sonnet 4.6

| Metric | Original (Welch t, unpaired) | Corrected (paired t) | Change |
|---|---|---|---|
| Test | Welch's t-test | Paired t-test | — |
| Real mean | 0.1683 | 0.1683 | 0.0000 |
| Shuffled mean | 0.1062 | 0.1062 | 0.0000 |
| **Gap** | **0.0621** | **0.0621** | **0.0000** |
| t-statistic | 14.09 | 20.32 | larger (paired) |
| p-value | ≈ 0 | ≈ 0 | unchanged |
| n (original) | 4,260 real / 11,552 shuffled | — | — |
| n_pairs (corrected) | — | 14,418 | — |
| Layer 2 threshold met (gap ≥ 0.04)? | ✅ 0.0621 | ✅ 0.0621 | unchanged |

---

## Multi-seed variance study (T=0.5, corrected)

Corrected Layer 1 gaps per config (McNemar, symmetric filter):

| Model | T=0 s42 (corrected) | T=0.5 s1337 (corrected) | T=0.5 s7919 (corrected) | Spread |
|---|---:|---:|---:|---:|
| Haiku 4.5 | +0.0660 | +0.0647 | +0.0675 | 0.0028 |
| **Sonnet 4.6** | **+0.2063** | **+0.2068** | **+0.2083** | **0.0020** |

All outcome tiers are identical to the original multi-seed results. Corrected gaps are consistently ~0.006–0.007 higher than original across all conditions for haiku and ~0.005–0.006 higher for sonnet — a uniform upward shift driven by the symmetric filter excluding more shuffled-miss pairs.

---

## Assessment: do v1's published headline results hold?

### Sonnet — strong_positive (published: gap 0.201, p ≪ 10⁻¹⁰)

**YES. Confirmed and strengthened.**

- Corrected gap: 0.2063 (within +0.006 of published 0.2006)
- McNemar χ² = 981.93 → p ≪ 10⁻²¹³
- Bonferroni-corrected (n=2) p ≪ 10⁻²¹² — clears corrected α=0.025 with extreme margin
- Outcome tier: **strong_positive** (unchanged)
- The corrected gap (0.206) is squarely within the "similar to original" range; no headline change warranted

### Haiku — moderate_positive (published: gap 0.059, p ≪ 10⁻¹⁰)

**YES. Confirmed and slightly strengthened.**

- Corrected gap: 0.0660 (within +0.007 of published 0.0591)
- McNemar χ² = 118.31 → p ≪ 10⁻²⁶
- Bonferroni-corrected (n=2) p ≪ 10⁻²⁵ — clears corrected α=0.025 with extreme margin
- Outcome tier: **moderate_positive** (unchanged)
- Gap increased from 0.059 to 0.066, closer to the strong_positive threshold (0.08)
- The pre-registered Layer 1 threshold (gap ≥ 0.05, p < 0.05) is met under both uncorrected and Bonferroni-corrected tests

### Pre-registration minimum criterion

> "gap ≥ 0.05 on top-3 match rate, p < 0.05, on at least one model"

Under corrected scoring with Bonferroni correction (n=2): **MET on both models.**

---

## Cells that changed status

**None.** No cell moved from significant to non-significant, changed outcome tier, or changed direction.

The corrected gaps are ~0.006–0.007 larger than published (not smaller). The corrected p-values are orders of magnitude more extreme. This is the expected direction: the original unpaired test was conservative relative to the paired test for these data, and the symmetric filter removes the noisiest pairs from both conditions equally.

---

## Unexpected findings

**Corrected gap is slightly larger than published** for haiku: 0.066 vs 0.059 (+0.7 pp). This deserves a brief note.

The original scorer computed rates independently: real rate = (real hits) / (real submissions with ref hit). The corrected scorer computes rates over aligned pairs where both conditions hit the reference. Since shuffled chains miss the reference ~2.4× more often than real chains, the symmetric filter excludes a disproportionately large number of shuffled-only-miss pairs. These are plausibly the shuffled chains in the most unusual states — states the model is least likely to score correctly in either condition. Dropping them produces a slightly higher real rate (0.3088 vs 0.3023) and nearly unchanged shuffled rate (0.2428 vs 0.2432), widening the gap.

This is consistent with the interpretation in RESULTS.md §"Robustness check": the asymmetric reference-miss rate is itself evidence of the structural difference between real and shuffled chains.

---

## Recommendation for v1 public repo

The corrected gaps fall within the range described as "similar to original" (sonnet gap 0.206 vs published 0.201; haiku gap 0.066 vs published 0.059). No headline finding changes.

**Recommended action:** Add a brief methodology note to v1's README (or a new section in RESULTS.md) pointing to `CORRECTED_SCORING.md` and `results/scored_corrected.json`. The note should state:

> A post-publication methodological review (applying v2's paired-test corrections to v1's data) confirmed that v1's headline results hold under McNemar's test, paired t-test, and Bonferroni correction across 2 primary cells. Corrected gaps: haiku 0.066 (published 0.059), sonnet 0.206 (published 0.201). Outcome tiers unchanged. See `CORRECTED_SCORING.md`.

**Co-author decision required** before modifying the public README, per the integrity constraints on this review. No other primary artifacts (RESULTS.md, original scorer, reference distributions) should be modified.

---

## Files added by this review

| File | Description |
|---|---|
| `src/scorer_corrected.py` | Corrected scorer: McNemar, paired t-test, Bonferroni |
| `results/scored_corrected.json` | Output of corrected scorer |
| `CORRECTED_SCORING.md` | This document |
| `SESSION_LOG.md` | Session record |
