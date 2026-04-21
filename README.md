# Project-Ditto

An evaluation-only research prototype testing whether frontier LLMs reason better on real Pokémon Showdown battle telemetry than on shuffled versions of the same data.

## Hypothesis

Real battle chains have causal consistency (HP trajectories, PP depletion, hidden-information reveals) that models can exploit. Shuffled chains violate this consistency. A real-vs-shuffled gap ≥ 0.05 on top-3 action-match rate (p < 0.05) is the pre-registered success threshold.

## Quick start

```bash
pip install -r requirements.txt

# Generate synthetic data (no internet needed)
python scripts/generate_synthetic_data.py --n 2500 --out data/raw/

# Build chains + reference distribution
python scripts/build_chains.py --data data/raw/ --out-real chains/real/ --out-shuffled chains/shuffled/
python scripts/build_reference.py build --chains chains/real/ --out data/reference_dist.pkl

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
