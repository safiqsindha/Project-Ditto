"""
Fix 11: Verify that extract_state_signature handles causally-inconsistent
(shuffled) chains correctly.

Shuffling permutes constraints so causal ordering is broken, but
extract_state_signature must still return a non-None StateSignature
(it just extracts whatever state information is visible in the prefix).
"""

from __future__ import annotations

import random

import pytest

from src.reference import StateSignature, extract_state_signature
from src.shuffler import shuffle_chain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chain(chain_id: str = "test_p1") -> dict:
    """Construct a minimal chain dict that passes the signature extractor."""
    constraints = [
        # Turn 1: switch in unit_A for focal player
        {"type": "ToolAvailability", "timestamp": 1, "tool": "unit_A", "state": "available", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 1, "resource": "hp_unit_A", "amount": 1.0,  "decay": "none", "recover_in": None},
        # Turn 1: opponent brings unit_B
        {"type": "ToolAvailability", "timestamp": 1, "tool": "unit_B", "state": "available", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 1, "resource": "hp_unit_B", "amount": 0.9,  "decay": "none", "recover_in": None},
        # Turn 2: damage events
        {"type": "ResourceBudget",   "timestamp": 2, "resource": "hp_unit_A", "amount": 0.7,  "decay": "none", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 2, "resource": "hp_unit_B", "amount": 0.6,  "decay": "none", "recover_in": None},
        # Turn 3: PP event
        {"type": "ResourceBudget",   "timestamp": 3, "resource": "pp_action_1", "amount": 0.875, "decay": "monotone_decrease", "recover_in": None},
        # Turn 5: time milestone
        {"type": "ResourceBudget",   "timestamp": 5, "resource": "match_time_remaining", "amount": 0.917, "decay": "monotone_decrease", "recover_in": None},
        # Turn 6: status
        {"type": "ResourceBudget",   "timestamp": 6, "resource": "status_unit_A", "amount": 0.5, "decay": "none", "recover_in": None},
        # Turn 7: weather
        {"type": "OptimizationCriterion", "timestamp": 7, "objective": "weather_raindance", "weight_shift": "weather_A"},
        # Turn 8: faint + subgoal
        {"type": "ToolAvailability",  "timestamp": 8, "tool": "unit_B", "state": "unavailable", "recover_in": None},
        {"type": "SubGoalTransition", "timestamp": 8, "from_phase": "initial", "to_phase": "forced_switch_required", "trigger": "own_faint"},
        # Turn 9: switch in
        {"type": "ToolAvailability",  "timestamp": 9, "tool": "unit_C", "state": "available", "recover_in": None},
        {"type": "ResourceBudget",    "timestamp": 9, "resource": "hp_unit_C", "amount": 1.0,  "decay": "none", "recover_in": None},
        # Turn 10: time milestone
        {"type": "ResourceBudget",    "timestamp": 10, "resource": "match_time_remaining", "amount": 0.833, "decay": "monotone_decrease", "recover_in": None},
        # Turn 11: hazard
        {"type": "CoordinationDependency", "timestamp": 11, "role": "field_side_p2", "dependency": "hazard_type_A", "expected_action": "hazard_response"},
        # Turns 12-20: padding to reach length >= 20
        {"type": "ResourceBudget",   "timestamp": 12, "resource": "hp_unit_A", "amount": 0.5, "decay": "none", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 13, "resource": "hp_unit_A", "amount": 0.4, "decay": "none", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 14, "resource": "pp_action_2", "amount": 0.75, "decay": "monotone_decrease", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 15, "resource": "hp_unit_C", "amount": 0.8, "decay": "none", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 16, "resource": "hp_unit_A", "amount": 0.3, "decay": "none", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 17, "resource": "match_time_remaining", "amount": 0.72, "decay": "monotone_decrease", "recover_in": None},
        {"type": "ResourceBudget",   "timestamp": 18, "resource": "hp_unit_C", "amount": 0.7, "decay": "none", "recover_in": None},
        {"type": "SubGoalTransition","timestamp": 19, "from_phase": "forced_switch_required", "to_phase": "recovery", "trigger": "opponent_switch"},
        {"type": "ResourceBudget",   "timestamp": 20, "resource": "hp_unit_A", "amount": 0.2, "decay": "none", "recover_in": None},
    ]
    return {
        "chain_id": chain_id,
        "match_id": "test_match",
        "perspective": "p1",
        "constraints": constraints,
        "rendered": "",
        "cutoff_k": len(constraints) // 2,
    }


class TestExtractStateSignatureShuffled:

    def test_real_chain_returns_non_none(self):
        chain = _make_chain()
        sig = extract_state_signature(chain, 10)
        assert sig is not None

    def test_real_chain_signature_is_state_signature(self):
        chain = _make_chain()
        sig = extract_state_signature(chain, 10)
        assert isinstance(sig, StateSignature)

    def test_shuffled_chain_returns_non_none(self):
        """Shuffled chains must still produce a valid (non-None) signature."""
        chain = _make_chain()
        shuffled = shuffle_chain(chain, seed=42)
        sig = extract_state_signature(shuffled, 10)
        assert sig is not None

    def test_shuffled_chain_signature_is_state_signature(self):
        chain = _make_chain()
        shuffled = shuffle_chain(chain, seed=1337)
        sig = extract_state_signature(shuffled, 10)
        assert isinstance(sig, StateSignature)

    def test_multiple_seeds_all_return_non_none(self):
        chain = _make_chain()
        for seed in [42, 1337, 7919, 0, 99]:
            shuffled = shuffle_chain(chain, seed=seed)
            sig = extract_state_signature(shuffled, len(shuffled["constraints"]) // 2)
            assert sig is not None, f"Got None signature for seed={seed}"

    def test_shuffled_signature_has_valid_hp_brackets(self):
        chain = _make_chain()
        shuffled = shuffle_chain(chain, seed=42)
        sig = extract_state_signature(shuffled, 12)
        assert sig is not None
        for bracket in sig.hp_brackets:
            assert 0 <= bracket <= 4

    def test_shuffled_signature_has_frozenset_fields(self):
        chain = _make_chain()
        shuffled = shuffle_chain(chain, seed=42)
        sig = extract_state_signature(shuffled, 12)
        assert sig is not None
        assert isinstance(sig.status_effects, frozenset)
        assert isinstance(sig.field_conditions, frozenset)

    def test_shuffled_signature_is_hashable(self):
        """StateSignature must be usable as a dict key (required for lookup)."""
        chain = _make_chain()
        shuffled = shuffle_chain(chain, seed=42)
        sig = extract_state_signature(shuffled, 12)
        assert sig is not None
        d: dict = {}
        d[sig.to_key(level=0)] = "test"
        assert len(d) == 1

    def test_out_of_bounds_step_returns_none(self):
        chain = _make_chain()
        sig = extract_state_signature(chain, 9999)
        assert sig is None

    def test_empty_chain_step_zero_returns_none(self):
        chain = {"chain_id": "empty", "perspective": "p1", "constraints": []}
        sig = extract_state_signature(chain, 0)
        assert sig is None
