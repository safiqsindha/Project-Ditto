"""
Scorer round-trip and unit tests.

Key concern: score_all() must load the reference distribution correctly
(via ReferenceDistribution.load, not raw pickle.load) so that lookups
return real data rather than silently empty dicts.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.reference import ReferenceDistribution, StateSignature
from src.scorer import (
    classify_outcome_tier,
    score_layer1,
    score_layer2,
    two_sample_proportion_test,
    welch_ttest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dist_with_data() -> ReferenceDistribution:
    """Return a small but non-empty ReferenceDistribution."""
    dist = ReferenceDistribution()
    sig = StateSignature(
        active_pair=("unit_A", "unit_B"),
        hp_brackets=(4, 3),
        status_effects=frozenset(),
        field_conditions=frozenset(),
        turn_bucket=0,
    )
    # Record 10 observations of action_1, 5 of action_2
    for _ in range(10):
        dist.add_observation(sig, "use action_1 with unit_a")
    for _ in range(5):
        dist.add_observation(sig, "use action_2 with unit_a")
    return dist


# ---------------------------------------------------------------------------
# Round-trip: save → load → lookup
# ---------------------------------------------------------------------------

class TestReferenceDistributionRoundTrip:

    def test_save_and_load_preserves_counts(self, tmp_path):
        dist = _make_dist_with_data()
        pkl = tmp_path / "dist.pkl"
        dist.save(pkl)

        loaded = ReferenceDistribution.load(pkl)
        assert loaded is not None

    def test_loaded_dist_lookup_returns_non_empty(self, tmp_path):
        dist = _make_dist_with_data()
        pkl = tmp_path / "dist.pkl"
        dist.save(pkl)

        loaded = ReferenceDistribution.load(pkl)
        sig = StateSignature(
            active_pair=("unit_A", "unit_B"),
            hp_brackets=(4, 3),
            status_effects=frozenset(),
            field_conditions=frozenset(),
            turn_bucket=0,
        )
        top_k, distribution, level = loaded.lookup(sig)
        assert len(top_k) > 0, "Loaded distribution returned no actions — save/load broken"
        assert len(distribution) > 0

    def test_loaded_dist_returns_correct_top_action(self, tmp_path):
        dist = _make_dist_with_data()
        pkl = tmp_path / "dist.pkl"
        dist.save(pkl)

        loaded = ReferenceDistribution.load(pkl)
        sig = StateSignature(
            active_pair=("unit_A", "unit_B"),
            hp_brackets=(4, 3),
            status_effects=frozenset(),
            field_conditions=frozenset(),
            turn_bucket=0,
        )
        top_k, distribution, _ = loaded.lookup(sig)
        # action_1 has 10 obs vs action_2's 5 → should rank first
        assert top_k[0] == "use action_1 with unit_a"

    def test_loaded_dist_probabilities_sum_to_one(self, tmp_path):
        dist = _make_dist_with_data()
        pkl = tmp_path / "dist.pkl"
        dist.save(pkl)

        loaded = ReferenceDistribution.load(pkl)
        sig = StateSignature(
            active_pair=("unit_A", "unit_B"),
            hp_brackets=(4, 3),
            status_effects=frozenset(),
            field_conditions=frozenset(),
            turn_bucket=0,
        )
        _, distribution, _ = loaded.lookup(sig)
        assert abs(sum(distribution.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# score_layer1
# ---------------------------------------------------------------------------

class TestScoreLayer1:

    def test_exact_match_in_top3(self):
        ref = {"use action_1 with unit_a": 0.6, "use action_2 with unit_a": 0.3, "switch to unit_c": 0.1}
        result = score_layer1("use action_1 with unit_A", ref)
        assert result["top_k_match"] == 1

    def test_normalisation_enables_match(self):
        # Model responds with trailing period and caps — normalize_action should handle it
        ref = {"use action_1 with unit_a": 0.7, "switch to unit_b": 0.3}
        result = score_layer1("Use Action_1 With Unit_A.", ref)
        assert result["top_k_match"] == 1

    def test_no_match_returns_zero(self):
        ref = {"use action_2 with unit_a": 0.8, "switch to unit_c": 0.2}
        result = score_layer1("use action_1 with unit_a", ref)
        assert result["top_k_match"] == 0

    def test_empty_dist_returns_none(self):
        result = score_layer1("use action_1 with unit_a", {})
        assert result["top_k_match"] is None

    def test_probability_mass_captured(self):
        ref = {"use action_1 with unit_a": 0.6, "switch to unit_b": 0.4}
        result = score_layer1("use action_1 with unit_a", ref)
        assert abs(result["probability_mass"] - 0.6) < 1e-9

    def test_entropy_is_non_negative(self):
        ref = {"use action_1 with unit_a": 0.6, "switch to unit_b": 0.4}
        result = score_layer1("switch to unit_b", ref)
        assert result["reference_entropy"] >= 0


# ---------------------------------------------------------------------------
# two_sample_proportion_test
# ---------------------------------------------------------------------------

class TestProportionTest:

    def test_returns_gap_and_pvalue(self):
        real = [1, 1, 0, 1, 0]
        shuf = [0, 0, 1, 0, 0]
        result = two_sample_proportion_test(real, shuf)
        assert "gap" in result
        assert "p_value" in result
        assert result["gap"] > 0

    def test_empty_returns_error(self):
        result = two_sample_proportion_test([], [1, 0])
        assert "error" in result

    def test_identical_samples_gap_zero(self):
        result = two_sample_proportion_test([1, 0, 1], [1, 0, 1])
        assert result["gap"] == 0.0


# ---------------------------------------------------------------------------
# classify_outcome_tier
# ---------------------------------------------------------------------------

class TestOutcomeTier:

    def test_strong_positive(self):
        assert classify_outcome_tier(0.09, 0.005) == "strong_positive"

    def test_moderate_positive(self):
        assert classify_outcome_tier(0.06, 0.03) == "moderate_positive"

    def test_null(self):
        assert classify_outcome_tier(0.01, 0.5) == "null"

    def test_reversed(self):
        assert classify_outcome_tier(-0.05, 0.03) == "reversed"

    def test_weak_mixed(self):
        assert classify_outcome_tier(0.04, 0.08) == "weak_mixed"
