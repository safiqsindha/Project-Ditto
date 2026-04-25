"""
Corrected scorer v1 — paired tests with symmetric reference-miss filter.

Addresses the same methodological issues identified in the v2 methodology
review (see CORRECTED_SCORING.md), applied here to v1's data:

  Issue 1: Layer 1 used a two-sample proportion z-test (unpaired)
           → corrected to McNemar's test (paired)
  Issue 2: Layer 2 used Welch's t-test (unpaired)
           → corrected to paired t-test
  Issue 3: No multiple-comparisons correction across primary cells
           → Bonferroni correction (2 primary cells: haiku, sonnet)
  Issue 4: No enforcement of (chain_id, eval_seed) pair alignment
           → pairs are now explicitly constructed and aligned

Filter rule (SYMMETRIC APPLICATION of v1's original filter):
    A pair enters the Layer 1 analysis IFF:
      - real chain's reference lookup returned a non-None top_k_match, AND
      - shuffled chain's reference lookup returned a non-None top_k_match.
    Both conditions must hold symmetrically. This mirrors the original
    scorer's per-condition reference-miss exclusion but applied to the pair
    rather than each condition independently.

    V1 has no actionable-subset filter (unlike v2); the only filter is
    reference-miss exclusion. V2's constraint-type filter is NOT introduced
    here — only v2's statistical methodology is adopted.

Primary cells: haiku, sonnet (2 cells; corrected α = 0.05 / 2 = 0.025)

Original src/scorer.py is NOT modified. This file is additive.

Usage:
    python -m src.scorer_corrected \\
        --results results/raw/ \\
        --dist data/reference_dist.pkl \\
        --chains-real chains/real/ \\
        --chains-shuffled chains/shuffled/ \\
        --out results/scored_corrected.json

    # To include variance-study runs alongside the primary:
    python -m src.scorer_corrected \\
        --results results/ \\
        --dist data/reference_dist.pkl \\
        --chains-real chains/real/ \\
        --chains-shuffled chains/shuffled/ \\
        --out results/scored_corrected.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from src.normalize import normalize_action
from src.reference import ReferenceDistribution, extract_state_signature
from src.scorer import (
    TOP_K,
    classify_outcome_tier,
    score_layer1,
    score_layer2,
)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemar_test(
    real_matches: list[int],
    shuffled_matches: list[int],
) -> dict[str, Any]:
    """
    McNemar's test (continuity-corrected) for paired binary Layer 1 outcomes.

    Contingency cells:
      b (n_10): real matched, shuffled did not — "discordant real-match"
      c (n_01): shuffled matched, real did not — "discordant shuffled-match"

    Gap is computed over all pairs (matching the original scorer's estimand).
    """
    n = len(real_matches)
    if n == 0:
        return {"error": "empty sample"}
    if n != len(shuffled_matches):
        return {"error": "list length mismatch"}

    n11 = sum(r == 1 and s == 1 for r, s in zip(real_matches, shuffled_matches))
    b   = sum(r == 1 and s == 0 for r, s in zip(real_matches, shuffled_matches))
    c   = sum(r == 0 and s == 1 for r, s in zip(real_matches, shuffled_matches))
    n00 = sum(r == 0 and s == 0 for r, s in zip(real_matches, shuffled_matches))

    if b + c == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = float((abs(b - c) - 1) ** 2) / float(b + c)
        p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=1))

    real_rate = sum(real_matches) / n
    shuf_rate = sum(shuffled_matches) / n

    return {
        "n_pairs": n,
        "n11_concordant_both_match": n11,
        "b_discordant_real_match": b,
        "c_discordant_shuffled_match": c,
        "n00_concordant_neither": n00,
        "real_rate": round(real_rate, 4),
        "shuffled_rate": round(shuf_rate, 4),
        "gap": round(real_rate - shuf_rate, 4),
        "chi2_stat": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "significant_05": bool(p_value < 0.05),
        "test": "mcnemar_continuity_corrected",
    }


def paired_ttest(
    real_scores: list[float],
    shuffled_scores: list[float],
) -> dict[str, Any]:
    """Paired t-test (scipy.stats.ttest_rel) for Layer 2 coupled scores."""
    n = len(real_scores)
    if n < 2:
        return {"error": "insufficient data (need >=2 pairs)"}
    if n != len(shuffled_scores):
        return {"error": "list length mismatch"}

    t, p = stats.ttest_rel(real_scores, shuffled_scores)
    diffs = [r - s for r, s in zip(real_scores, shuffled_scores)]

    return {
        "n_pairs": n,
        "real_mean": round(float(np.mean(real_scores)), 4),
        "shuffled_mean": round(float(np.mean(shuffled_scores)), 4),
        "gap": round(float(np.mean(real_scores)) - float(np.mean(shuffled_scores)), 4),
        "mean_diff": round(float(np.mean(diffs)), 4),
        "t_stat": round(float(t), 4),
        "p_value": round(float(p), 6),
        "significant_05": bool(p < 0.05),
        "test": "paired_ttest",
    }


def apply_bonferroni(p_value: float, n_tests: int) -> float:
    """Bonferroni-corrected p-value (clamped to [0, 1])."""
    return min(1.0, p_value * n_tests)


# ---------------------------------------------------------------------------
# Pair-alignment helpers
# ---------------------------------------------------------------------------

def base_chain_id(chain_id: str) -> str:
    """Strip the _shuffled_{seed} suffix to get the base (real) chain_id."""
    if "_shuffled_" in chain_id:
        return chain_id.split("_shuffled_")[0]
    return chain_id


def shuffle_seed_from_chain_id(chain_id: str) -> int | None:
    """Return the shuffle seed embedded in a shuffled chain_id, or None."""
    if "_shuffled_" not in chain_id:
        return None
    try:
        return int(chain_id.split("_shuffled_")[1])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main corrected scoring pipeline
# ---------------------------------------------------------------------------

def score_all_corrected(
    results_dirs: list[Path],
    dist_path: Path,
    chains_real_dir: Path,
    chains_shuffled_dir: Path,
) -> dict[str, Any]:
    """
    Load all raw results, build aligned pairs, and compute corrected statistics.

    Alignment key: (base_chain_id, model, eval_seed, shuffle_seed)
    Symmetric filter: both real and shuffled must have non-None top_k_match.
    """
    # --- Reference distribution ---
    ref_dist = ReferenceDistribution.load(dist_path)

    # --- Load all raw results from every results directory ---
    all_results: list[dict] = []
    for results_dir in results_dirs:
        for rfile in sorted(results_dir.glob("**/*.json")):
            with open(rfile) as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            if not isinstance(data, dict) or "chain_id" not in data:
                continue
            all_results.append(data)

    print(f"Loaded {len(all_results)} raw results")

    # --- Index chains ---
    chain_index: dict[str, dict] = {}
    for cfile in chains_real_dir.glob("**/*.jsonl"):
        with open(cfile) as f:
            chain = json.loads(f.readline())
            chain_index[chain["chain_id"]] = chain
    for cfile in chains_shuffled_dir.glob("**/*.jsonl"):
        with open(cfile) as f:
            chain = json.loads(f.readline())
            chain_index[chain["chain_id"]] = chain

    print(f"Indexed {len(chain_index)} chains")

    # --- Per-evaluation scoring ---
    scored_by_chain: dict[str, dict] = defaultdict(dict)
    skipped_no_chain = 0

    for r in all_results:
        cid = r.get("chain_id", "")
        chain = chain_index.get(cid, {})
        if not chain:
            skipped_no_chain += 1
            continue

        model = r.get("model", "unknown")
        seed = r.get("seed", 0)
        temperature = r.get("temperature", 0.0)
        model_action = r.get("response", "").strip()
        cutoff_k = r.get("cutoff_k", 15)

        # Reference lookup
        dist: dict[str, float] = {}
        try:
            sig = extract_state_signature(chain, cutoff_k)
            if sig is not None:
                _top_k, dist, _backoff = ref_dist.lookup(sig)
        except (KeyError, AttributeError, TypeError):
            pass

        l1 = score_layer1(model_action, dist)
        l2 = score_layer2(model_action, chain, cutoff_k, dist)

        key = (model, seed, temperature)
        scored_by_chain[cid][key] = {
            "l1": l1.get("top_k_match"),          # None if reference miss
            "l2": l2.get("coupled", 0.0),
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }

    if skipped_no_chain:
        print(f"[scorer_corrected] WARN: skipped {skipped_no_chain} results with no chain")

    # --- Build aligned pairs ---
    all_chain_ids = set(scored_by_chain.keys())
    real_chain_ids = {cid for cid in all_chain_ids if "_shuffled_" not in cid}
    shuffled_chain_ids = {cid for cid in all_chain_ids if "_shuffled_" in cid}

    shuffled_by_base: dict[str, list[str]] = defaultdict(list)
    for cid in shuffled_chain_ids:
        shuffled_by_base[base_chain_id(cid)].append(cid)

    @dataclass
    class Pair:
        base_cid: str
        model: str
        eval_seed: int
        temperature: float
        shuffle_seed: int | None
        real_l1: int | None
        real_l2: float
        shuffled_l1: int | None
        shuffled_l2: float

    pairs: list[Pair] = []
    n_excl_no_real = 0
    n_excl_no_shuffled = 0

    for base_cid in sorted(real_chain_ids):
        real_scored = scored_by_chain.get(base_cid, {})
        shuf_cids = shuffled_by_base.get(base_cid, [])

        for shuf_cid in shuf_cids:
            shuf_scored = scored_by_chain.get(shuf_cid, {})
            sh_seed = shuffle_seed_from_chain_id(shuf_cid)

            all_eval_keys = set(real_scored.keys()) | set(shuf_scored.keys())
            for ek in sorted(all_eval_keys):
                real_entry = real_scored.get(ek)
                shuf_entry = shuf_scored.get(ek)

                if real_entry is None:
                    n_excl_no_real += 1
                    continue
                if shuf_entry is None:
                    n_excl_no_shuffled += 1
                    continue

                pairs.append(Pair(
                    base_cid=base_cid,
                    model=ek[0],
                    eval_seed=ek[1],
                    temperature=ek[2],
                    shuffle_seed=sh_seed,
                    real_l1=real_entry["l1"],
                    real_l2=real_entry["l2"],
                    shuffled_l1=shuf_entry["l1"],
                    shuffled_l2=shuf_entry["l2"],
                ))

    no_shuffled_bases = len(real_chain_ids - set(shuffled_by_base.keys()))

    print(f"Built {len(pairs)} pairs from {len(real_chain_ids)} real chains")
    print(f"Excluded: {n_excl_no_real} (missing real eval), "
          f"{n_excl_no_shuffled} (missing shuffled eval), "
          f"{no_shuffled_bases} base chains had no shuffled variants")

    # --- Bucket structure ---
    def _new_bucket() -> dict[str, list]:
        return {"real_l1": [], "shuf_l1": [], "real_l2": [], "shuf_l2": []}

    # Separate primary (T=0) from variance-study runs
    per_model_primary: dict[str, dict[str, list]] = defaultdict(_new_bucket)
    per_model_all: dict[str, dict[str, list]] = defaultdict(_new_bucket)
    per_model_per_config: dict[tuple, dict[str, list]] = defaultdict(_new_bucket)

    n_pairs_ref_miss_both = 0
    n_pairs_ref_miss_real_only = 0
    n_pairs_ref_miss_shuffled_only = 0
    n_pairs_both_valid = 0

    for p in pairs:
        model = p.model
        config_key = f"T{p.temperature}_seed{p.eval_seed}"

        real_valid = p.real_l1 is not None
        shuf_valid = p.shuffled_l1 is not None

        if not real_valid and not shuf_valid:
            n_pairs_ref_miss_both += 1
        elif not real_valid:
            n_pairs_ref_miss_real_only += 1
        elif not shuf_valid:
            n_pairs_ref_miss_shuffled_only += 1
        else:
            n_pairs_both_valid += 1

        # Layer 1: only include if BOTH have non-None top_k_match (symmetric filter)
        if real_valid and shuf_valid:
            per_model_all[model]["real_l1"].append(p.real_l1)
            per_model_all[model]["shuf_l1"].append(p.shuffled_l1)
            per_model_per_config[(model, config_key)]["real_l1"].append(p.real_l1)
            per_model_per_config[(model, config_key)]["shuf_l1"].append(p.shuffled_l1)
            if p.temperature == 0.0:
                per_model_primary[model]["real_l1"].append(p.real_l1)
                per_model_primary[model]["shuf_l1"].append(p.shuffled_l1)

        # Layer 2: always include (score_layer2 always returns a numeric value)
        per_model_all[model]["real_l2"].append(p.real_l2)
        per_model_all[model]["shuf_l2"].append(p.shuffled_l2)
        per_model_per_config[(model, config_key)]["real_l2"].append(p.real_l2)
        per_model_per_config[(model, config_key)]["shuf_l2"].append(p.shuffled_l2)
        if p.temperature == 0.0:
            per_model_primary[model]["real_l2"].append(p.real_l2)
            per_model_primary[model]["shuf_l2"].append(p.shuffled_l2)

    # --- Summarise each bucket ---
    def _summarise(bucket: dict[str, list]) -> dict[str, Any]:
        l1 = mcnemar_test(bucket["real_l1"], bucket["shuf_l1"])
        l2 = paired_ttest(bucket["real_l2"], bucket["shuf_l2"])
        gap  = l1.get("gap", 0.0) if "error" not in l1 else 0.0
        pval = l1.get("p_value", 1.0) if "error" not in l1 else 1.0
        return {
            "layer1": l1,
            "layer2": l2,
            "outcome_tier_uncorrected": classify_outcome_tier(gap, pval),
        }

    primary_stats = {m: _summarise(b) for m, b in per_model_primary.items()}
    all_stats     = {m: _summarise(b) for m, b in per_model_all.items()}
    config_stats  = {f"{m}::{c}": _summarise(b) for (m, c), b in per_model_per_config.items()}

    # --- Bonferroni correction: 2 primary cells (haiku, sonnet) ---
    # Primary metric is Layer 1 on T=0 s42 results
    PRIMARY_CELLS = ["haiku", "sonnet"]
    N_PRIMARY = len(PRIMARY_CELLS)

    bonferroni_primary: dict[str, Any] = {}
    for model_name in PRIMARY_CELLS:
        cell_stats = primary_stats.get(model_name, {})
        l1 = cell_stats.get("layer1", {})
        raw_p  = l1.get("p_value", 1.0) if "error" not in l1 else 1.0
        corr_p = apply_bonferroni(raw_p, N_PRIMARY)
        gap = l1.get("gap", float("nan")) if "error" not in l1 else float("nan")
        bonferroni_primary[model_name] = {
            "gap": gap,
            "uncorrected_p": round(raw_p, 6),
            "bonferroni_corrected_p": round(corr_p, 6),
            "n_primary_tests": N_PRIMARY,
            "corrected_alpha": round(0.05 / N_PRIMARY, 4),
            "meets_preregistered_threshold_bonferroni": bool(
                not math.isnan(gap) and gap >= 0.05 and corr_p < 0.05
            ),
            "meets_preregistered_threshold_uncorrected": bool(
                not math.isnan(gap) and gap >= 0.05 and raw_p < 0.05
            ),
        }

    # Add Bonferroni-adjusted outcome tier to primary stats
    for model_name in PRIMARY_CELLS:
        if model_name in primary_stats:
            l1 = primary_stats[model_name].get("layer1", {})
            gap    = l1.get("gap", 0.0) if "error" not in l1 else 0.0
            corr_p = bonferroni_primary.get(model_name, {}).get("bonferroni_corrected_p", 1.0)
            primary_stats[model_name]["outcome_tier_bonferroni"] = classify_outcome_tier(gap, corr_p)

    preregistered_min_met_bonf = any(
        v["meets_preregistered_threshold_bonferroni"] for v in bonferroni_primary.values()
    )
    preregistered_min_met_uncorr = any(
        v["meets_preregistered_threshold_uncorrected"] for v in bonferroni_primary.values()
    )

    return {
        "scorer_version": "corrected_v1",
        "filter_rule": "symmetric_reference_miss_exclusion",
        "description": (
            "Both real and shuffled must have non-None top_k_match (reference hit). "
            "No actionable-type filter (v1 had none). "
            "Layer 1: McNemar continuity-corrected. "
            "Layer 2: paired t-test. "
            "Bonferroni: 2 primary cells (haiku, sonnet), corrected alpha=0.025."
        ),
        "test_layer1": "mcnemar_continuity_corrected",
        "test_layer2": "paired_ttest",
        "n_results_loaded": len(all_results),
        "n_real_chains": len(real_chain_ids),
        "n_pairs_total": len(pairs),
        "n_pairs_both_valid_l1": n_pairs_both_valid,
        "n_pairs_ref_miss_both": n_pairs_ref_miss_both,
        "n_pairs_ref_miss_real_only": n_pairs_ref_miss_real_only,
        "n_pairs_ref_miss_shuffled_only": n_pairs_ref_miss_shuffled_only,
        "n_excl_no_real_eval": n_excl_no_real,
        "n_excl_no_shuffled_eval": n_excl_no_shuffled,
        "n_base_chains_no_shuffled_variant": no_shuffled_bases,
        "correction_method": "bonferroni",
        "n_primary_cells": N_PRIMARY,
        "per_model_primary_T0": primary_stats,
        "per_model_all_conditions": all_stats,
        "per_model_per_config": config_stats,
        "bonferroni_primary_2_cells": bonferroni_primary,
        "preregistered_min_criterion_met_bonferroni": preregistered_min_met_bonf,
        "preregistered_min_criterion_met_uncorrected": preregistered_min_met_uncorr,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Corrected scorer v1 (paired tests + Bonferroni)"
    )
    parser.add_argument(
        "--results", type=Path, nargs="+",
        default=[Path("results/raw")],
        help="One or more results directories (searched recursively for *.json)",
    )
    parser.add_argument("--dist",           type=Path, default=Path("data/reference_dist.pkl"))
    parser.add_argument("--chains-real",    type=Path, default=Path("chains/real"))
    parser.add_argument("--chains-shuffled",type=Path, default=Path("chains/shuffled"))
    parser.add_argument("--out",            type=Path, default=Path("results/scored_corrected.json"))
    args = parser.parse_args()

    scored = score_all_corrected(
        results_dirs=args.results,
        dist_path=args.dist,
        chains_real_dir=args.chains_real,
        chains_shuffled_dir=args.chains_shuffled,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"\nResults written to {args.out}")

    # Summary table: primary cells (T=0)
    print("\n=== Corrected v1 scoring — primary 2 cells (Layer 1, McNemar) ===")
    print(f"  Filter: symmetric reference-miss exclusion (both conditions must have ref hit)")
    print(f"  Bonferroni correction: n_cells={len(scored['bonferroni_primary_2_cells'])}, corrected alpha=0.025")
    print()
    hdr = f"  {'Cell':10s}  {'n_pairs':>8}  {'gap':>7}  {'uncorr_p':>10}  {'bonf_p':>10}  {'tier_bonf':>18}  {'meets_bonf':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for model_name in ["haiku", "sonnet"]:
        bv = scored["bonferroni_primary_2_cells"].get(model_name, {})
        ps = scored["per_model_primary_T0"].get(model_name, {})
        l1 = ps.get("layer1", {})
        n  = l1.get("n_pairs", "?") if "error" not in l1 else "?"
        tier = ps.get("outcome_tier_bonferroni", "?")
        print(
            f"  {model_name:10s}  {str(n):>8}  {bv.get('gap', float('nan')):>7.4f}"
            f"  {bv.get('uncorrected_p', 1.0):>10.2e}"
            f"  {bv.get('bonferroni_corrected_p', 1.0):>10.2e}"
            f"  {tier:>18}"
            f"  {str(bv.get('meets_preregistered_threshold_bonferroni', False)):>10}"
        )

    print(f"\nPre-registered min (gap>=0.05, Bonferroni p<0.05): "
          f"{'MET' if scored['preregistered_min_criterion_met_bonferroni'] else 'NOT MET'}")
    print(f"Pre-registered min (gap>=0.05, uncorrected p<0.05): "
          f"{'MET' if scored['preregistered_min_criterion_met_uncorrected'] else 'NOT MET'}")

    print("\n=== Layer 2 (paired t-test) — primary T=0 results ===")
    for model_name in ["haiku", "sonnet"]:
        ps = scored["per_model_primary_T0"].get(model_name, {})
        l2 = ps.get("layer2", {})
        print(
            f"  {model_name:10s}  gap={l2.get('gap', 'N/A')}  "
            f"t={l2.get('t_stat', 'N/A')}  p={l2.get('p_value', 'N/A')}  "
            f"n_pairs={l2.get('n_pairs', 'N/A')}"
        )

    print("\n=== Per-config breakdown ===")
    for cfg_key in sorted(scored["per_model_per_config"].keys()):
        cs = scored["per_model_per_config"][cfg_key]
        l1 = cs.get("layer1", {})
        n  = l1.get("n_pairs", "?") if "error" not in l1 else "?"
        gap = l1.get("gap", float("nan")) if "error" not in l1 else float("nan")
        pv  = l1.get("p_value", 1.0) if "error" not in l1 else 1.0
        tier = cs.get("outcome_tier_uncorrected", "?")
        print(f"  {cfg_key:40s}  n={str(n):>7}  gap={gap:>7.4f}  p={pv:>8.2e}  tier={tier}")
