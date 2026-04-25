# Project-Ditto

An evaluation-only research prototype testing whether frontier LLMs reason better on real Pokémon Showdown battle telemetry than on shuffled versions of the same data.

Follow-up work: Project Ditto v2 — applying this methodology to programming-task trajectories.

## Hypothesis

Real battle chains have causal consistency (HP trajectories, PP depletion, hidden-information reveals) that models can exploit. Shuffled chains violate this consistency. A real-vs-shuffled gap ≥ 0.05 on top-3 action-match rate (p < 0.05) is the pre-registered success threshold.

## Methodology Correction (April 2026)

Post-hoc methodological review identified that the original scoring used unpaired statistical tests (two-sample proportion z-test for Layer 1, Welch's t-test for Layer 2) on data that is inherently paired — each real chain has corresponding shuffled variants produced from the same source match. The review also identified the absence of multiple-comparisons correction across the two primary model cells.

Corrected analysis applies McNemar's test (Layer 1), paired t-test (Layer 2), and Bonferroni correction across the two primary cells (Haiku, Sonnet). The corrected analysis preserves both published findings:

- **Sonnet 4.6:** gap 0.206, Bonferroni-corrected p ≪ 10⁻²¹² (strong_positive, unchanged)
- **Haiku 4.5:** gap 0.066, Bonferroni-corrected p ≪ 10⁻²⁵ (moderate_positive, unchanged)

Both findings clear the pre-registered minimum publishable threshold by substantial margins. The methodology correction does not change the qualitative or quantitative headline of this study; it provides statistically appropriate inference for paired data.

Implementation: [`src/scorer_corrected.py`](src/scorer_corrected.py)  
Full comparison: [`CORRECTED_SCORING.md`](CORRECTED_SCORING.md)

---

## Quick start

```bash
pip install -r requirements.txt

# Generate synthetic data (no internet needed)
python scripts/generate_synthetic_data.py --n 2500 --out data/raw/

# Build chains + reference distribution (from raw logs — works without chains)
python scripts/build_chains.py --data data/raw/ --out-real chains/real/ --out-shuffled chains/shuffled/
python scripts/build_reference.py build-raw --raw data/raw/ --out data/reference_dist.pkl

# Dry-run evaluation (no API key needed)
python -m src.runner --model haiku --chains chains/real/ --seed 42 --dry-run --n 5

# Full evaluation (requires ANTHROPIC_API_KEY in .env)
python -m src.runner --model haiku --chains chains/real/ --seed 42
python -m src.runner --model opus  --chains chains/real/ --seed 42
```

## Pipeline overview

```
data/raw/*.jsonl  →  scripts/build_chains.py  →  chains/real/ + chains/shuffled/
                                                       ↓
                  scripts/build_reference.py  →  data/reference_dist.pkl
                                                       ↓
                          src/runner.py        →  results/raw/
                                                       ↓
                          src/scorer.py        →  results/scored.json
```

See [`CLAUDE.md`](CLAUDE.md) for full architecture documentation and [`showdown prototype spec v1.pdf`](showdown%20prototype%20spec%20v1.pdf) for the complete research specification.

## Tests

```bash
pytest tests/
```
