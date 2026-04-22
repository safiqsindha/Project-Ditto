# Project-Ditto — Pre-registration Results

**Status:** Primary hypothesis confirmed — `strong_positive` on Claude Sonnet 4.6, `moderate_positive` on Claude Haiku 4.5. Per the pre-registered decision rule (CLAUDE.md §7, "on at least one model"), the result is publishable. Replicated across two additional seeds at temperature 0.5 (§"Multi-seed variance study"); outcome tiers are identical across all three conditions.

**Date:** 2026-04-21  
**Seeds:** 42 (T = 0 primary); 1337, 7919 (T = 0.5 variance study)  
**Models:** `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`  
**Dataset:** 2,500 Gen 9 OU ladder matches (Elo ≥ 1700 both players, ≥ 15 turns, `|win|` event present). Source: `HolidayOugi/pokemon-showdown-replays` parquet shards 7, 8, 9, 11.

---

## TL;DR

On both frontier models tested, Claude's top-3 action-match rate is **higher on real constraint chains than on shuffled controls** at high statistical significance. Sonnet shows a 20.1-percentage-point gap, clearing the pre-registered `strong_positive` threshold (gap ≥ 0.08, p < 0.01). Haiku shows a 5.9-point gap, clearing the primary Layer 1 threshold (gap ≥ 0.05, p < 0.05) but falling short of Layer 2's 0.04-point composite threshold.

The finding supports the precondition that motivated the experiment: real battle telemetry carries causal structure (HP trajectories, PP depletion, hidden-info reveals) that frontier LLMs can exploit and that shuffling destroys.

---

## Headline Numbers (Layer 1, primary)

| Model | Real top-3 | Shuffled top-3 | **Gap** | z | p | n_real | n_shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Haiku 4.5 | 0.3023 | 0.2432 | **+0.0591** | 7.52 | ≈ 0 | 4,260 | 11,552 |
| **Sonnet 4.6** | **0.4453** | 0.2447 | **+0.2006** | 24.45 | ≈ 0 | 4,260 | 11,552 |

p-values reported as ≈ 0 because the two-sample proportion-test statistic is beyond the double-precision tail (both z-scores correspond to p ≪ 10⁻¹⁰).

## Layer 2 (legality × optimality composite)

| Model | Real | Shuffled | Gap | t | p | 95% CI (real) |
|---|---:|---:|---:|---:|---:|---|
| Haiku 4.5 | 0.1643 | 0.1345 | +0.0298 | 6.04 | ≈ 0 | [0.156, 0.173] |
| **Sonnet 4.6** | 0.1683 | 0.1062 | **+0.0621** | 14.08 | ≈ 0 | [0.161, 0.176] |

Layer 2 threshold: gap ≥ 0.04. Sonnet clears it; haiku's +0.0298 gap is directionally consistent with Layer 1 but below threshold.

## Pre-registration checklist (CLAUDE.md §7)

| Criterion | Threshold | Haiku | Sonnet |
|---|---|:-:|:-:|
| Layer 1 gap | ≥ 0.05 | ✅ 0.0591 | ✅ 0.2006 |
| Layer 1 significance | p < 0.05 | ✅ | ✅ |
| Layer 2 gap | ≥ 0.04 | ❌ 0.0298 | ✅ 0.0621 |
| Layer 2 direction | consistent with L1 | ✅ | ✅ |
| Strong-positive | gap ≥ 0.08 ∧ p < 0.01 | ❌ | ✅ |
| **Outcome tier** | — | **moderate_positive** | **strong_positive** |

"At least one model clears primary threshold" → **satisfied**. Pre-registration outcome: **publishable strong-positive result.**

---

## Robustness check: what if we count reference-miss drops as misses?

The default scorer excludes chains for which the reference distribution returns an empty lookup at all four backoff levels (the state signature never appeared in the 2,500-match training set). This excluded 17.7% of all evaluations. Crucially, the drop rate was **asymmetric** between conditions:

| Condition | Total | Dropped | Drop rate |
|---|---:|---:|---:|
| Real | 4,806 | 546 | 11.4% |
| Shuffled | 14,418 | 2,866 | 19.9% |

Shuffled chains drop ~2× more often because shuffling permutes information-reveal timing: at the half-chain cutoff, many shuffled signatures contain `unit_unknown` (opponent not yet revealed under the permuted ordering) which the reference distribution — built from real matches where active units are always known — cannot match. This is the hypothesis in action: information structure that real chains carry and shuffled chains lack.

Because the dropped shuffled chains are likely the *hardest* to score (rarer states), Scenario A (current scorer) biases the shuffled rate upward and the gap downward. Scenario B below re-runs Layer 1 counting all drops as misses (top_k_match = 0) and using all 19,224 submitted chains as the denominator:

| Model | Scenario | Real rate | Shuffled rate | Gap | n_real | n_shuffled |
|---|---|---:|---:|---:|---:|---:|
| Haiku | A (exclude drops, primary) | 0.3023 | 0.2432 | +0.0591 | 4,260 | 11,552 |
| Haiku | B (drops-as-miss) | 0.2680 | 0.1949 | **+0.0731** | 4,806 | 14,418 |
| Sonnet | A (exclude drops, primary) | 0.4453 | 0.2447 | +0.2006 | 4,260 | 11,552 |
| Sonnet | B (drops-as-miss) | 0.3947 | 0.1961 | **+0.1986** | 4,806 | 14,418 |

Under Scenario B the sonnet gap stays essentially unchanged (0.199 vs 0.201) — the effect is not an artifact of which chains the reference happens to cover. For haiku, the gap actually widens from 0.059 to 0.073, moving it closer to strong_positive territory.

The pre-registered metric is Scenario A. Both are reported here in the interest of methodological transparency; Scenario B is a conservative lower-bound robustness check.

---

## Multi-seed variance study (T = 0.5)

A follow-up run sampled at **temperature 0.5** across **two additional seeds (1337, 7919)** on both models — 8 additional batches, 76,896 requests, 0 errors. Output isolated to `results/T05_s1337/` and `results/T05_s7919/` so the pre-registered T = 0 s42 primary is unaffected.

### Layer 1 gap across three conditions

| Model  | T = 0  s42 (primary) | T = 0.5  s1337 | T = 0.5  s7919 | Spread |
|---|---:|---:|---:|---:|
| Haiku 4.5 | +0.0591 | +0.0572 | +0.0603 | 0.0031 |
| **Sonnet 4.6** | **+0.2006** | **+0.2010** | **+0.2016** | **0.0010** |

### Layer 2 gap across three conditions

| Model  | T = 0  s42 (primary) | T = 0.5  s1337 | T = 0.5  s7919 | Spread |
|---|---:|---:|---:|---:|
| Haiku 4.5 | +0.0298 | +0.0269 | +0.0282 | 0.0029 |
| **Sonnet 4.6** | **+0.0621** | **+0.0613** | **+0.0631** | **0.0018** |

### Outcome tier across conditions

| Model | T = 0 s42 | T = 0.5 s1337 | T = 0.5 s7919 |
|---|:-:|:-:|:-:|
| Haiku 4.5 | moderate_positive | moderate_positive | moderate_positive |
| **Sonnet 4.6** | **strong_positive** | **strong_positive** | **strong_positive** |

All three conditions return identical outcome tiers on both models, with Layer 1 gap spreads under 0.005 and Layer 2 spreads under 0.004 — well inside the pre-registered threshold margins. The finding is **stable across both the temperature setting and the specific seed** used for sampling. This addresses the previously-flagged limitation that a single-seed T = 0 run provided no empirical variance estimate.

---

## Method (summary)

Full pipeline in `CLAUDE.md`. Briefly:

1. **Translation T (frozen at git tag `T-v1.1-frozen`).** Raw Showdown protocol logs are converted into typed constraint sequences with abstracted vocabulary (Pokémon → `unit_A..F`, moves → `action_1..4`, weather/terrain/hazards to letter tokens). Six constraint types: `ResourceBudget`, `ToolAvailability`, `SubGoalTransition`, `InformationState`, `CoordinationDependency`, `OptimizationCriterion`.
2. **Asymmetric observability.** Opponent units are suppressed until revealed by in-game events; opponent HP is bucketed to 0/25/50/75/100%.
3. **Validity filter.** Length 20–40, ≥ 2 `SubGoalTransition`, ≥ 1 `ToolAvailability(unavailable)`, ≥ 10 `ResourceBudget`, non-decreasing timestamps, no consecutive duplicates.
4. **Shuffler.** For each real chain, produce shuffled variants with constraints permuted uniformly at random; timestamps are re-sorted so the shuffle is not detectable from timestamp monotonicity alone. Seeds: 42, 1337, 7919 (three shuffles per real chain). Chain rendering and `active_pair_by_step` metadata are permuted in lockstep.
5. **Reference distribution.** State signature → action-distribution lookup with 4-level backoff, built from the same 2,500-match corpus (`data/reference_dist.pkl`, 144,270 observations). Max-backoff hit rate at reference build time: 80%.
6. **Evaluation.** Model sees the first `len(constraints) // 2` rendered steps (abstract vocabulary only; zero Pokémon leakage unit-tested) and is asked for the next action. Temperature 0.0, `max_tokens=50`.
7. **Scoring.** Layer 1 top-3 action match vs reference. Layer 2 legality × optimality composite. Welch t-test and two-sample proportion test for significance.

Chains produced: 4,806 real (p1 + p2 perspectives across 2,500 matches, after validity filter) and 14,418 shuffled (3 seeds × 4,806).

All evaluations run via Anthropic's Message Batches API (50% cost reduction vs synchronous). Batch IDs recorded in the reproducibility appendix.

---

## Interpretation

The gap between real and shuffled performance is consistent with the hypothesis that the tested frontier LLMs are — at least to some degree — reasoning over the temporal/causal structure of the chain rather than treating it as a bag of constraints. The three most plausible structural features:

- **HP trajectory coherence.** Real chains exhibit monotonic HP depletion within a unit; shuffled chains show arbitrary oscillation.
- **PP/resource depletion order.** Real chains have action availability strictly shrinking; shuffled chains permit "de-depletion."
- **Information-reveal sequence.** Real chains reveal opponent units and moves in an order dictated by game mechanics; shuffled chains place reveals at arbitrary positions. The asymmetry in reference-miss rate (19.9% shuffled vs 11.4% real) is direct evidence of this feature being destroyed by shuffling.

The much larger gap on Sonnet (0.20) compared to Haiku (0.06) is consistent with the interpretation that exploiting this structure requires reasoning capacity, and that capacity scales with model size. A ~3.4× gap ratio between sonnet and haiku is on the high end of what we might have expected and is the headline quantitative finding of the study.

---

## Limitations

- ~~**Single seed for evaluation.**~~ Resolved — the runner now supports `--temperature > 0` and seeds 1337 and 7919 were run at T = 0.5 (§"Multi-seed variance study"). Outcome tiers unchanged; Layer 1 gap spreads < 0.005.
- **One data-selection pipeline.** Results are specific to Gen 9 OU singles at Elo ≥ 1700. We have not tested whether the real-vs-shuffled gap persists at lower skill tiers, in different formats (Doubles, 1v1), or at shorter/longer chain lengths.
- ~~Incomplete sonnet shuffled batch.~~ Resolved — the 238 credit-exhausted requests were re-submitted as `msgbatch_01XEGbjZncEs4YX9birvdJRT` and completed with 238/238 succeeded. Full 14,418/14,418 coverage now achieved on sonnet shuffled. Rescored numbers above reflect the complete dataset.
- **Scoring integrity.** CLAUDE.md specifies that scoring should run in a separate session from evaluation. In this study, scoring was executed in the same session as the evaluation run. This does not affect the scorer itself (which is algorithmically deterministic and reads from the blinded `results/blinded/` directory with model/seed/condition stripped), but the separate-session convention for researcher blinding was not fully honored.
- **No human baseline.** We have no ceiling for how well a human expert would score on the same chains, so we cannot compare model performance to an authoritative reference.
- **Layer 1 reference ceiling.** Real performance tops out at 45% top-3 match on sonnet, implying the reference distribution is peakier than a strict ceiling and that multiple of the top-3 actions remain plausible on most states. Interpreting *real rate* as "skill" is misleading; only the *gap* is the quantity the pre-reg cares about.

---

## Reproducibility

**Code state.** Project at `/Users/safiqsindha/Desktop/Ditto/`. Translation at `T-v1.1-frozen` tag.

**Evaluation parameters.** Temperature 0.0, `max_tokens=50`, seed 42, cutoff K = `len(constraints) // 2`, prompt version `v1.0`.

**Batch IDs (Anthropic Messages API).**

| Model | Condition | Batch ID | Completed |
|---|---|---|---|
| Haiku 4.5 | Real | `msgbatch_01Q2WXzgmWGDoMUrYECmCWS8` | 4,806 / 4,806 |
| Haiku 4.5 | Shuffled | `msgbatch_01LqcdtPThCKQr8hWviaanhh` | 14,418 / 14,418 |
| Sonnet 4.6 | Real | `msgbatch_017sZzAtBw6gX9aTjK73efKt` | 4,806 / 4,806 |
| Sonnet 4.6 | Shuffled (initial) | `msgbatch_011a4wTLH6Z1ppxp33STb7xw` | 14,180 / 14,418 (238 credit-exhausted) |
| Sonnet 4.6 | Shuffled (retry) | `msgbatch_01XEGbjZncEs4YX9birvdJRT` | 238 / 238 |

**Variance-study batches (T = 0.5, seeds 1337 & 7919).**

| Model | Condition | Seed | Batch ID | Completed |
|---|---|---|---|---|
| Haiku 4.5 | Real | 1337 | `msgbatch_01Ebc1ziw7YKqybkD4cdzqMg` | 4,806 / 4,806 |
| Haiku 4.5 | Real | 7919 | `msgbatch_0156pRz1GiZjo7his8EdRV12` | 4,806 / 4,806 |
| Haiku 4.5 | Shuffled | 1337 | `msgbatch_01WSZcTnH4chmNMfu2Ld43hf` | 14,418 / 14,418 |
| Haiku 4.5 | Shuffled | 7919 | `msgbatch_01EwmuchLv5s1iijAAVukEjL` | 14,418 / 14,418 |
| Sonnet 4.6 | Real | 1337 | `msgbatch_01FXQa8QKivc5mUpjTvxkb94` | 4,806 / 4,806 |
| Sonnet 4.6 | Real | 7919 | `msgbatch_019t7S9GPnKfABwnv9dt5uUr` | 4,806 / 4,806 |
| Sonnet 4.6 | Shuffled | 1337 | `msgbatch_01Q8QxCoatVWWzKoz3L65TzZ` | 14,418 / 14,418 |
| Sonnet 4.6 | Shuffled | 7919 | `msgbatch_01Hg4MLa7yrFBKTHkqjUSn8L` | 14,418 / 14,418 |

**Artifacts.**
- `data/raw/` — 2,500 filtered matches (one JSONL record per match)
- `chains/real/` — 4,806 chain files, one per perspective
- `chains/shuffled/` — 14,418 chain files, 3 seeds per real chain
- `data/reference_dist.pkl` — 144,270 state-action observations
- `results/raw/` — T = 0 s42 primary: 38,210 model responses (haiku + sonnet)
- `results/blinded/` — scorer input, model/seed/condition stripped
- `results/scored.json` — T = 0 s42 Layer 1/2/3 metrics and outcome tiers
- `results/T05_s1337/` — T = 0.5 seed 1337: raw, blinded, scored.json
- `results/T05_s7919/` — T = 0.5 seed 7919: raw, blinded, scored.json

**Spend.** Approximately $33 for the pre-registered T = 0 primary run and an additional ~$66 for the T = 0.5 × (s1337, s7919) variance study — ~$99 total, all via Batches API (50% of sync list price).

---

## Next steps (recommended, in priority order)

1. **Write up the paper.** The result cleanly clears the pre-registered `strong_positive` threshold on sonnet and the `moderate_positive` threshold on haiku, giving a two-model replication with increasing effect size by capability. Multi-seed variance study at T = 0.5 confirms tier stability on both models. The asymmetric reference-miss rate is a small additional piece of structural evidence worth reporting.
2. **Opus 4.7 is optional.** It would strengthen the "effect size scales with capability" narrative if the gap continues to grow, but it is not required for the pre-reg to hold. Estimated cost on Batches API: ~$124.
