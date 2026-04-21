"""
Session 7: Scorer — blinded evaluation of model responses.

Runs in a separate context with no knowledge of model identity, real/shuffled
label, or seed. Computes Layer 1, Layer 2, and Layer 3 metrics.

Usage:
    python -m src.scorer \
        --results results/raw/ \
        --dist data/reference_dist.pkl \
        --chains-real chains/real/ \
        --chains-shuffled chains/shuffled/ \
        --out results/scored.json
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


# ---------------------------------------------------------------------------
# Layer 1: Objective action-match rate
# ---------------------------------------------------------------------------

TOP_K = 3  # top-3 match rate (spec §7.1)


def score_layer1(
    model_action: str,
    reference_dist: dict[str, float],
    k: int = TOP_K,
) -> dict[str, float]:
    """
    Compute top-k match and probability mass for one evaluation.

    Parameters
    ----------
    model_action    : the action string the model proposed
    reference_dist  : {action: probability} from ReferenceDistribution.lookup()
    k               : top-k threshold (default 3)

    Returns
    -------
    {"top_k_match": 0|1, "probability_mass": float, "reference_entropy": float}
    """
    if not reference_dist:
        return {"top_k_match": None, "probability_mass": None, "reference_entropy": None}

    norm_action = normalize_action(model_action)
    norm_dist = {normalize_action(a): p for a, p in reference_dist.items()}

    sorted_actions = sorted(norm_dist, key=norm_dist.get, reverse=True)
    top_k_actions = sorted_actions[:k]

    top_k_match = 1 if norm_action in top_k_actions else 0
    prob_mass = norm_dist.get(norm_action, 0.0)

    # Entropy of reference distribution (diagnostic)
    probs = list(reference_dist.values())
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    return {
        "top_k_match": top_k_match,
        "probability_mass": prob_mass,
        "reference_entropy": round(entropy, 4),
    }


# ---------------------------------------------------------------------------
# Layer 2: Legality × optimality
# ---------------------------------------------------------------------------

def score_layer2(
    model_action: str,
    chain: dict,
    cutoff_k: int,
    reference_dist: dict[str, float],
) -> dict[str, float]:
    """
    Compute constraint satisfaction rate and coupled composite metric.

    Legality: did the model propose an action that is permitted by active
    constraints at step K? A fainted/unavailable unit cannot act.

    Optimality proxy: among legal actions in the reference distribution,
    what fraction of mass does the model's action capture?

    Returns {"legality": float, "optimality_proxy": float, "coupled": float}
    """
    constraints = chain.get("constraints", [])[:cutoff_k]

    # Determine which units are currently unavailable (fainted / benched as unavailable)
    unavailable_units: set[str] = set()
    available_units: set[str] = set()
    for c in constraints:
        if c.get("type") == "ToolAvailability":
            tool = c.get("tool", "")
            state = c.get("state", "")
            if state == "unavailable":
                unavailable_units.add(tool)
                available_units.discard(tool)
            elif state == "available":
                available_units.add(tool)
                unavailable_units.discard(tool)

    # Check legality: if model_action mentions an unavailable unit, it's illegal
    legality = 1.0
    norm_action = normalize_action(model_action)
    unavailable_norm = {u.lower() for u in unavailable_units}
    for unit in unavailable_norm:
        if unit in norm_action:
            legality = 0.0
            break

    # Optimality proxy among legal actions
    norm_ref = {normalize_action(a): p for a, p in reference_dist.items()}
    legal_dist = {
        a: p for a, p in norm_ref.items()
        if not any(u in a for u in unavailable_norm)
    }
    total_legal_mass = sum(legal_dist.values())
    if total_legal_mass > 0 and legal_dist:
        legal_dist_normalised = {a: p / total_legal_mass for a, p in legal_dist.items()}
        optimality_proxy = legal_dist_normalised.get(norm_action, 0.0)
    else:
        optimality_proxy = 0.0

    coupled = legality * optimality_proxy

    return {
        "legality": legality,
        "optimality_proxy": round(optimality_proxy, 4),
        "coupled": round(coupled, 4),
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def two_sample_proportion_test(
    real_matches: list[int],
    shuffled_matches: list[int],
) -> dict[str, float]:
    """
    Two-sample proportion test (z-test) for Layer 1 top-k match rates.

    Parameters
    ----------
    real_matches     : list of 0/1 top-k match outcomes for real chains
    shuffled_matches : list of 0/1 top-k match outcomes for shuffled chains

    Returns {"real_rate", "shuffled_rate", "gap", "z_stat", "p_value", "significant_05"}
    """
    n1, n2 = len(real_matches), len(shuffled_matches)
    if n1 == 0 or n2 == 0:
        return {"error": "empty sample"}

    p1 = sum(real_matches) / n1
    p2 = sum(shuffled_matches) / n2
    gap = p1 - p2

    # Pooled proportion
    p_pool = (sum(real_matches) + sum(shuffled_matches)) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = gap / se if se > 0 else 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "real_rate": round(p1, 4),
        "shuffled_rate": round(p2, 4),
        "gap": round(gap, 4),
        "z_stat": round(z, 4),
        "p_value": round(p_value, 4),
        "significant_05": bool(p_value < 0.05),
        "n_real": n1,
        "n_shuffled": n2,
    }


def welch_ttest(real_scores: list[float], shuffled_scores: list[float]) -> dict[str, float]:
    """Welch's t-test for Layer 2 coupled metric."""
    if len(real_scores) < 2 or len(shuffled_scores) < 2:
        return {"error": "insufficient data"}
    t, p = stats.ttest_ind(real_scores, shuffled_scores, equal_var=False)
    ci_real = stats.t.interval(0.95, len(real_scores) - 1,
                               loc=np.mean(real_scores), scale=stats.sem(real_scores))
    return {
        "real_mean": round(float(np.mean(real_scores)), 4),
        "shuffled_mean": round(float(np.mean(shuffled_scores)), 4),
        "gap": round(float(np.mean(real_scores)) - float(np.mean(shuffled_scores)), 4),
        "t_stat": round(float(t), 4),
        "p_value": round(float(p), 4),
        "significant_05": bool(p < 0.05),
        "ci_real_95": (round(float(ci_real[0]), 4), round(float(ci_real[1]), 4)),
    }


# ---------------------------------------------------------------------------
# Layer 3: Subset breakdown
# ---------------------------------------------------------------------------

def classify_chain(chain: dict) -> dict[str, str]:
    """Classify a chain into Layer 3 subsets."""
    constraints = chain.get("constraints", [])
    n = len(constraints)

    # Length bucket
    if n <= 25:
        length_bucket = "short_20_25"
    elif n <= 32:
        length_bucket = "medium_26_32"
    else:
        length_bucket = "long_33_40"

    # Constraint-type composition
    n_resource = sum(1 for c in constraints if c.get("type") == "ResourceBudget")
    n_subgoal = sum(1 for c in constraints if c.get("type") == "SubGoalTransition")
    if n_resource > n_subgoal:
        composition = "resource_dominated"
    elif n_subgoal > n_resource:
        composition = "subgoal_dominated"
    else:
        composition = "balanced"

    # Archetype heuristic
    n_faint_tool = sum(
        1 for c in constraints
        if c.get("type") == "ToolAvailability" and c.get("state") == "unavailable" and c.get("recover_in") is None
    )
    n_boost = sum(
        1 for c in constraints
        if c.get("type") == "ResourceBudget" and "boost" in c.get("resource", "")
    )
    if n_faint_tool >= 4:
        archetype = "aggressive"
    elif n_boost >= 3:
        archetype = "setup"
    else:
        archetype = "stall"

    return {
        "length_bucket": length_bucket,
        "composition": composition,
        "archetype": archetype,
    }


# ---------------------------------------------------------------------------
# Outcome tier classification
# ---------------------------------------------------------------------------

def classify_outcome_tier(gap: float, p_value: float) -> str:
    """Classify the Layer 1 result into an outcome tier (§7.4)."""
    if gap < -0.01:
        return "reversed"
    if abs(gap) < 0.02 and p_value > 0.1:
        return "null"
    if gap >= 0.08 and p_value < 0.01:
        return "strong_positive"
    if gap >= 0.05 and p_value < 0.05:
        return "moderate_positive"
    return "weak_mixed"


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------

def score_all(
    results_dir: Path,
    dist_path: Path,
    chains_real_dir: Path,
    chains_shuffled_dir: Path,
) -> dict[str, Any]:
    """
    Load all raw results and compute Layer 1, 2, 3 metrics.

    Returns a comprehensive results dict suitable for JSON serialisation.
    """
    # Load reference distribution (must use ReferenceDistribution.load, NOT pickle.load
    # directly — the pickle file stores a payload dict, not the object itself)
    ref_dist = ReferenceDistribution.load(dist_path)

    # Load all raw results
    results: list[dict] = []
    for rfile in sorted(results_dir.glob("*.json")):
        with open(rfile) as f:
            results.append(json.load(f))

    print(f"Loaded {len(results)} raw results")

    # Index chains for fast lookup
    chain_index: dict[str, dict] = {}
    for cfile in chains_real_dir.glob("*.jsonl"):
        with open(cfile) as f:
            chain = json.loads(f.readline())
            chain_index[chain["chain_id"]] = chain
    for cfile in chains_shuffled_dir.glob("*.jsonl"):
        with open(cfile) as f:
            chain = json.loads(f.readline())
            chain_index[chain["chain_id"]] = chain

    # Score each result
    per_model: dict[str, dict[str, list]] = defaultdict(lambda: {
        "real_l1": [], "shuffled_l1": [],
        "real_l2": [], "shuffled_l2": [],
    })

    per_subset: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        chain_id = r.get("chain_id", "")
        model = r.get("model", "unknown")
        is_real = "_shuffled_" not in chain_id
        condition = "real" if is_real else "shuffled"
        model_action = r.get("response", "").strip()
        cutoff_k = r.get("cutoff_k", 15)

        chain = chain_index.get(chain_id, {})
        if not chain:
            continue

        # Get reference distribution for this step
        dist: dict[str, float] = {}
        try:
            sig = extract_state_signature(chain, cutoff_k)
            if sig is not None:
                _top_k, dist, _backoff = ref_dist.lookup(sig)
        except (KeyError, AttributeError, TypeError) as exc:
            print(f"[scorer] lookup failed for chain_id={chain_id}: {exc}")

        # Layer 1
        l1 = score_layer1(model_action, dist)
        if l1["top_k_match"] is not None:
            per_model[model][f"{condition}_l1"].append(l1["top_k_match"])

        # Layer 2
        l2 = score_layer2(model_action, chain, cutoff_k, dist)
        per_model[model][f"{condition}_l2"].append(l2["coupled"])

        # Layer 3 subsets
        subsets = classify_chain(chain)
        for subset_name, subset_val in subsets.items():
            key = f"{subset_name}:{subset_val}"
            per_subset[model][f"{key}_{condition}_l1"].append(l1.get("top_k_match", 0) or 0)

    # Aggregate Layer 1 + 2 statistics per model
    model_stats: dict[str, Any] = {}
    for model_name, data in per_model.items():
        l1_test = two_sample_proportion_test(data["real_l1"], data["shuffled_l1"])
        l2_test = welch_ttest(data["real_l2"], data["shuffled_l2"])
        gap = l1_test.get("gap", 0.0)
        pval = l1_test.get("p_value", 1.0)
        tier = classify_outcome_tier(gap, pval)
        model_stats[model_name] = {
            "layer1": l1_test,
            "layer2": l2_test,
            "outcome_tier": tier,
        }

    return {
        "per_model": model_stats,
        "n_results": len(results),
        "per_subset": {m: dict(d) for m, d in per_subset.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/raw"))
    parser.add_argument("--dist", type=Path, default=Path("data/reference_dist.pkl"))
    parser.add_argument("--chains-real", type=Path, default=Path("chains/real"))
    parser.add_argument("--chains-shuffled", type=Path, default=Path("chains/shuffled"))
    parser.add_argument("--out", type=Path, default=Path("results/scored.json"))
    args = parser.parse_args()

    scored = score_all(args.results, args.dist, args.chains_real, args.chains_shuffled)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"\nResults written to {args.out}")
    print("\n=== Outcome summary ===")
    for model, stats_data in scored["per_model"].items():
        tier = stats_data.get("outcome_tier", "unknown")
        gap = stats_data["layer1"].get("gap", "N/A")
        pval = stats_data["layer1"].get("p_value", "N/A")
        print(f"  {model}: tier={tier}, gap={gap}, p={pval}")
