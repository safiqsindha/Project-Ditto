# Changelog

## [Unreleased]

## [0.1.0] — 2026-04-21

### Added

**Session 1 — Environment + data acquisition**
- `pyproject.toml` and `requirements.txt` (Python ≥ 3.11)
- `src/parser.py` — Showdown protocol parser; `BattleLog`, `BattleEvent`, `filter_match()`, `constraint_events()`
- `scripts/generate_synthetic_data.py` — generates structurally valid Showdown logs offline (no network required)
- `scripts/acquire_data.py` — downloads real data from PokéChamp or HuggingFace `pokemon-showdown-replays`

**Session 2 — Reference action distribution**
- `src/reference.py` — `StateSignature` (5-component game state with 4-level backoff), `ReferenceDistribution` (state → action distribution lookup), `extract_state_signature()`
- `scripts/build_reference.py` — builds and checks the distribution from chain files or raw logs

**Session 3 — Translation function T** *(frozen at tag `T-v1.0-frozen`)*
- `src/translation.py` — all 12 Showdown event→constraint mappings; six constraint types: `ResourceBudget`, `ToolAvailability`, `SubGoalTransition`, `InformationState`, `CoordinationDependency`, `OptimizationCriterion`; Pokémon names → `unit_A..unit_F`, moves → `action_1..action_4`
- `src/renderer.py` — `render_chain()` abstract English rendering; `check_pokemon_leakage()` validation
- `tests/test_translation.py` — 64 unit tests covering all 12 event mappings

**Session 4 — Chain construction + validity filter**
- `src/observability.py` — `apply_asymmetric_observability()`: opponent units hidden until revealed, opponent HP bucketed to 0/25/50/75/100%
- `src/filter.py` — `is_valid_chain()`: length 20–40, ≥2 `SubGoalTransition`, ≥1 unavailable `ToolAvailability`, ≥10 `ResourceBudget`, non-decreasing timestamps, no consecutive duplicates
- `src/shuffler.py` — `shuffle_chain(chain, seed)`: seeded permutation with monotonic timestamp reassignment
- `scripts/build_chains.py` — end-to-end pipeline: parse → translate → observability → window → filter → shuffle → JSONL

**Session 5 — Evaluation runner**
- `src/prompt_builder.py` — versioned prompt template `v1.0`; `build_prompt()`, `cutoff_rendered()`
- `src/runner.py` — evaluation runner for `claude-haiku-4-5-20251001` and `claude-opus-4-7`; seeds 42/1337/7919; temperature 0.0; rate-limit backoff; raw + blinded result saving

**Scorer (Session 7 prep)**
- `src/scorer.py` — Layer 1 (top-3 match rate, two-sample proportion test), Layer 2 (legality × optimality, Welch's t-test), Layer 3 (subset breakdowns by chain length / archetype / composition / cutoff), `classify_outcome_tier()`
