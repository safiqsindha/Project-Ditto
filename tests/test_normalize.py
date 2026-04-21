"""Tests for src/normalize.py — action string normalization."""

import pytest
from src.normalize import normalize_action


class TestNormalizeAction:
    def test_lowercases(self):
        assert normalize_action("USE action_1 WITH unit_A") == "use action_1 with unit_a"

    def test_strips_whitespace(self):
        assert normalize_action("  use action_2 with unit_b  ") == "use action_2 with unit_b"

    def test_collapses_spaces(self):
        assert normalize_action("use  action_3  with  unit_c") == "use action_3 with unit_c"

    def test_removes_trailing_period(self):
        assert normalize_action("switch to unit_c.") == "switch to unit_c"

    def test_removes_comma(self):
        assert normalize_action("use action_1, with unit_a") == "use action_1 with unit_a"

    def test_removes_exclamation(self):
        assert normalize_action("switch to unit_b!") == "switch to unit_b"

    def test_empty_string(self):
        assert normalize_action("") == ""

    def test_already_normalized(self):
        assert normalize_action("use action_2 with unit_b") == "use action_2 with unit_b"

    def test_switch_canonical(self):
        assert normalize_action("Switch to unit_C") == "switch to unit_c"

    def test_uppercase_action(self):
        assert normalize_action("USE ACTION_4 WITH UNIT_F") == "use action_4 with unit_f"

    def test_preserves_underscores(self):
        result = normalize_action("use action_1 with unit_a")
        assert "_" in result

    def test_removes_parentheses(self):
        assert normalize_action("use action_1 (move)") == "use action_1 move"

    def test_tabs_collapsed(self):
        # tabs become spaces via re.sub on non-alphanumeric
        assert normalize_action("use\taction_1") == "use action_1"

    def test_mixed_case_unit(self):
        assert normalize_action("Switch To Unit_D") == "switch to unit_d"

    def test_newlines_stripped(self):
        result = normalize_action("use action_2\nwith unit_b")
        assert "\n" not in result

    def test_numeric_digits_preserved(self):
        result = normalize_action("use action_3 with unit_e")
        assert "3" in result
        assert "e" in result

    def test_double_space_after_strip(self):
        assert normalize_action("switch  to   unit_a") == "switch to unit_a"

    def test_only_spaces(self):
        assert normalize_action("   ") == ""

    def test_brackets_removed(self):
        assert normalize_action("[use action_1 with unit_a]") == "use action_1 with unit_a"

    def test_consistent_with_reference_lookup(self):
        # Both sides must normalise identically for a lookup to succeed
        model_response = "  Use Action_2 With Unit_B. "
        ref_key = "use action_2 with unit_b"
        assert normalize_action(model_response) == normalize_action(ref_key)
