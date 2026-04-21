# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project-Ditto** is a research prototype testing whether frontier LLMs reason better on real Pokémon Showdown battle telemetry than on shuffled versions of the same telemetry. The experiment is evaluation-only (~$0 cost on Claude Max) and is designed to falsify a precondition for a larger training experiment.

**Hypothesis:** Real battle chains have causal consistency (HP trajectories, PP depletion, hidden-info reveals) that models exploit. Shuffled chains violate this consistency. A real-vs-shuffled gap ≥ 0.05 on top-3 action-match rate (p < 0.05) is the pre-registered success threshold.

The spec is in `showdown prototype spec v1.pdf` on the main branch.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic battle data (no internet needed)
python scripts/generate_synthetic_data.py --n 2500 --out data/raw/

# Acquire real data (internet required — run once)
python scripts/acquire_data.py --source huggingface --out data/raw/

# Build constraint chains from raw data
python scripts/build_chains.py --data data/raw/ --out-real chains/real/ --out-shuffled chains/shuffled/

# Build reference action distribution (primary: from raw logs)
python scripts/build_reference.py build-raw --raw data/raw/ --out data/reference_dist.pkl
# Alt: build from chain files after build_chains.py has run
# python scripts/build_reference.py build --chains chains/real/ --out data/reference_dist.pkl
python scripts/build_reference.py check --dist data/reference_dist.pkl --chains chains/real/

# Dry-run evaluation (no API key needed)
python -m src.runner --model haiku --chains chains/real/ --seed 42 --dry-run --n 5

# Full evaluation run (requires ANTHROPIC_API_KEY in .env)
python -m src.runner --model haiku --chains chains/real/ --seed 42
python -m src.runner --model opus --chains chains/real/ --seed 42

# Score results
python -m src.scorer --results results/raw/ --dist data/reference_dist.pkl \
  --chains-real chains/real/ --chains-shuffled chains/shuffled/ --out results/scored.json

# Run tests
pytest tests/
pytest tests/test_translation.py::TestSwitchPlayerSide  # single test class
```

## Architecture

### Pipeline (in order)

1. **`src/parser.py`** — parses raw Showdown protocol text into `BattleLog` / `BattleEvent`. `constraint_events()` filters to the 12 events that emit constraints. `filter_match()` applies Elo ≥ 1700, ≥ 15 turns, Gen 9 OU, `|win|` event present.

2. **`src/translation.py`** — Translation function T (frozen at git tag `T-v1.1-frozen`). Converts `BattleEvent` objects into typed constraint objects via `translate_match(events, perspective)`. Six constraint types: `ResourceBudget`, `ToolAvailability`, `SubGoalTransition`, `InformationState`, `CoordinationDependency`, `OptimizationCriterion`. Pokémon names → `unit_A..unit_F`, moves → `action_1..action_4`; weather → `weather_A..E`, terrain → `terrain_A..E`, hazards → `hazard_type_A..D`. **Do not modify after the T-v1.1-frozen tag.**

3. **`src/observability.py`** — Asymmetric rendering: opponent units hidden until revealed, opponent HP bucketed to 0/25/50/75/100%.

4. **`src/filter.py`** — `is_valid_chain()`: length 20–40, ≥ 2 `SubGoalTransition`, ≥ 1 `ToolAvailability(unavailable)`, ≥ 10 `ResourceBudget`, timestamps non-decreasing, no consecutive duplicates.

5. **`src/shuffler.py`** — `shuffle_chain(chain, seed)`: permutes constraints, preserves monotonic timestamps, suffixes chain ID with `_shuffled_{seed}`.

6. **`src/renderer.py`** — `render_chain(constraints, perspective) → str`: abstract English rendering. `check_pokemon_leakage()` validates zero Pokémon vocabulary in output.

7. **`src/reference.py`** — `ReferenceDistribution`: state-signature → action distribution lookup with 4-level backoff. State signature = (active_pair, hp_brackets, status_effects, field_conditions, turn_bucket).

8. **`src/prompt_builder.py`** — Fixed versioned prompt template (`PROMPT_VERSION = "v1.0"`). `build_prompt(rendered_steps, cutoff_k)`.

9. **`src/runner.py`** — Evaluation runner: loads chains, calls Anthropic API (Haiku 4.5 + Opus 4.7), saves raw + blinded results, handles rate-limit backoff.

10. **`src/scorer.py`** — Layer 1 (top-3 match rate), Layer 2 (legality × optimality), Layer 3 (subset breakdowns). `classify_outcome_tier()` maps gap + p-value to {strong_positive, moderate_positive, weak_mixed, null, reversed}.

### Key invariants

- **T is frozen.** `src/translation.py` and `src/renderer.py` must not change after the `T-v1.1-frozen` git tag. Any divergence must be recorded and reported.
- **Scorer is blinded.** Run scoring in a separate session; the scorer never sees model identity, real/shuffled label, or seed.
- **No Pokémon vocabulary in rendered chains.** Unit-tested via `check_pokemon_leakage()`.
- **Cutoff K = len(constraints) // 2** (half-chain), unless overridden by `--cutoff-k`.
- **Evaluation models:** `claude-haiku-4-5-20251001` and `claude-opus-4-7`. Seeds: 42, 1337, 7919. Temperature: 0.0, max_tokens: 50.

### Data flow

```
data/raw/*.jsonl          (raw match records with Showdown log text)
  → scripts/build_chains.py
chains/real/*.jsonl       (one file per chain, one JSON record per file)
chains/shuffled/*.jsonl   (one shuffled variant per real chain per seed)
  → scripts/build_reference.py
data/reference_dist.pkl   (state→action lookup table)
  → src/runner.py
results/raw/              (model responses with full metadata)
results/blinded/          (stripped for scorer)
  → src/scorer.py
results/scored.json       (Layer 1/2/3 metrics + outcome tier)
```

### Synthetic data

Because the real PokéChamp / HuggingFace datasets require internet access, `scripts/generate_synthetic_data.py` generates structurally valid (but not semantically meaningful) Showdown logs for pipeline development. Replace with real data before drawing conclusions.

### Pre-registered success thresholds (§7)

- **Layer 1 (primary):** real-vs-shuffled gap ≥ 0.05 on top-3 match rate, p < 0.05 (two-sample proportion test), on at least one model.
- **Layer 2:** gap ≥ 0.04 on legality × optimality composite, consistent direction with Layer 1.
- Outcome tiers: strong_positive (gap ≥ 0.08, p < 0.01) → publishable; moderate_positive → replicate at 500 chains; null → publishable negative; reversed → different research direction.
