"""
Unit tests for src/translation.py — covers all 12 event-to-constraint mappings.

Each test:
  - Constructs a minimal fixture BattleEvent
  - Calls translate_event() with a fresh TranslationContext
  - Asserts correct constraint type, field names, and numeric values
"""

from __future__ import annotations

import pytest

from src.parser import BattleEvent
from src.translation import (
    CoordinationDependency,
    InformationState,
    OptimizationCriterion,
    ResourceBudget,
    SubGoalTransition,
    ToolAvailability,
    TranslationContext,
    translate_event,
    translate_match,
)
from src.renderer import render_chain, check_pokemon_leakage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(perspective: str = "p1") -> TranslationContext:
    """Return a fresh translation context."""
    return TranslationContext(perspective=perspective)


def _event(etype: str, args: list[str], turn: int = 1, player: str | None = None) -> BattleEvent:
    return BattleEvent(
        type=etype,
        player=player,
        args=args,
        raw_line=f"|{etype}|{'|'.join(args)}",
        turn=turn,
    )


# ---------------------------------------------------------------------------
# Mapping 1 & 2: |switch| — player's side
# ---------------------------------------------------------------------------

class TestSwitchPlayerSide:
    """|switch| on perspective's side emits ToolAvailability constraints."""

    def test_first_switch_emits_available(self):
        ctx = _ctx("p1")
        evt = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        results = translate_event(evt, ctx)

        avail = [c for c in results if isinstance(c, ToolAvailability) and c.state == "available"]
        assert len(avail) == 1
        assert avail[0].tool == "unit_A"
        assert avail[0].timestamp == 1

    def test_second_switch_marks_previous_unavailable(self):
        ctx = _ctx("p1")
        # First switch in unit_A
        evt1 = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(evt1, ctx)

        # Second switch in unit_B
        evt2 = _event("switch", ["p1a: Rotom-Wash", "Rotom-Wash, L80", "260/260"], turn=3)
        results = translate_event(evt2, ctx)

        unavail = [c for c in results if isinstance(c, ToolAvailability) and c.state == "unavailable"]
        avail = [c for c in results if isinstance(c, ToolAvailability) and c.state == "available"]

        assert len(unavail) == 1
        assert unavail[0].tool == "unit_A"    # previous unit benched
        assert unavail[0].recover_in == 0     # benched = recoverable

        assert len(avail) == 1
        assert avail[0].tool == "unit_B"      # new unit in

    def test_hp_recorded_on_switch(self):
        ctx = _ctx("p1")
        evt = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "245/350"], turn=2)
        translate_event(evt, ctx)
        assert ctx.get_hp_max("p1", "unit_A") == 350


# ---------------------------------------------------------------------------
# Mapping 2: |switch| — opponent's side
# ---------------------------------------------------------------------------

class TestSwitchOpponentSide:
    """Opponent switch emits SubGoalTransition + InformationState."""

    def test_subgoal_and_info_emitted(self):
        ctx = _ctx("p1")
        evt = _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=2)
        results = translate_event(evt, ctx)

        sgt = [c for c in results if isinstance(c, SubGoalTransition)]
        info = [c for c in results if isinstance(c, InformationState)]

        assert len(sgt) == 1
        assert sgt[0].trigger == "opponent_switch"
        assert sgt[0].from_phase == "initial"
        assert "unit_A" in sgt[0].to_phase

        assert len(info) == 1
        assert "unit_A" in info[0].observable_added

    def test_tool_availability_also_emitted(self):
        ctx = _ctx("p1")
        evt = _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=2)
        results = translate_event(evt, ctx)
        avail = [c for c in results if isinstance(c, ToolAvailability) and c.state == "available"]
        assert len(avail) == 1

    def test_phase_updates_on_consecutive_opponent_switches(self):
        ctx = _ctx("p1")
        evt1 = _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=2)
        translate_event(evt1, ctx)

        evt2 = _event("switch", ["p2a: Corviknight", "Corviknight, L80", "370/370"], turn=4)
        results2 = translate_event(evt2, ctx)
        sgt = [c for c in results2 if isinstance(c, SubGoalTransition)]
        assert len(sgt) == 1
        assert "unit_A" in sgt[0].from_phase   # prior phase references old unit
        assert "unit_B" in sgt[0].to_phase


# ---------------------------------------------------------------------------
# Mapping 3: |faint| — player's side
# ---------------------------------------------------------------------------

class TestFaint:
    def test_faint_permanent_unavailability(self):
        ctx = _ctx("p1")
        # Establish unit first
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("faint", ["p1a: Garchomp"], turn=3)
        results = translate_event(evt, ctx)

        unavail = [c for c in results if isinstance(c, ToolAvailability)]
        assert len(unavail) >= 1
        perm = [c for c in unavail if c.recover_in is None and c.state == "unavailable"]
        assert len(perm) == 1
        assert perm[0].tool == "unit_A"

    def test_faint_own_unit_triggers_forced_switch_phase(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("faint", ["p1a: Garchomp"], turn=3)
        results = translate_event(evt, ctx)

        sgt = [c for c in results if isinstance(c, SubGoalTransition)]
        assert len(sgt) == 1
        assert sgt[0].to_phase == "forced_switch_required"
        assert sgt[0].trigger == "own_faint"

    def test_faint_opponent_unit_no_forced_switch(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=1)
        translate_event(sw, ctx)

        evt = _event("faint", ["p2a: Dragapult"], turn=3)
        results = translate_event(evt, ctx)

        sgt = [c for c in results if isinstance(c, SubGoalTransition)]
        assert len(sgt) == 0   # opponent faint doesn't trigger own forced_switch


# ---------------------------------------------------------------------------
# Mapping 4 & 5: |-damage| and |-heal|
# ---------------------------------------------------------------------------

class TestDamageHeal:
    def test_damage_hp_fraction(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-damage", ["p1a: Garchomp", "p1a: Garchomp|245/350"], turn=2)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert len(rb) == 1
        assert rb[0].resource == "hp_unit_A"
        assert abs(rb[0].amount - (245 / 350)) < 0.001

    def test_damage_to_zero(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-damage", ["p1a: Garchomp", "p1a: Garchomp|0 fnt"], turn=2)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert rb[0].amount == 0.0

    def test_heal_hp_fraction(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Blissey", "Blissey, L80", "620/620"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-heal", ["p1a: Blissey", "p1a: Blissey|465/620"], turn=3)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert rb[0].resource == "hp_unit_A"
        assert abs(rb[0].amount - (465 / 620)) < 0.001

    def test_damage_opponent_unit(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p2a: Kingambit", "Kingambit, L80", "380/380"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-damage", ["p2a: Kingambit", "p2a: Kingambit|190/380"], turn=2)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert rb[0].resource == "hp_unit_A"
        assert abs(rb[0].amount - 0.5) < 0.001


# ---------------------------------------------------------------------------
# Mapping 6: |-status|
# ---------------------------------------------------------------------------

class TestStatus:
    @pytest.mark.parametrize("status_token,expected_severity", [
        ("par", 0.5),
        ("brn", 0.5),
        ("slp", 1.0),
        ("frz", 1.0),
        ("psn", 0.3),
        ("tox", 0.3),
    ])
    def test_status_severity(self, status_token, expected_severity):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-status", ["p1a: Garchomp", status_token], turn=2)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert len(rb) == 1
        assert rb[0].resource == "status_unit_A"
        assert abs(rb[0].amount - expected_severity) < 0.001

    def test_sleep_decay_label(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-status", ["p1a: Garchomp", "slp"], turn=2)
        results = translate_event(evt, ctx)
        rb = results[0]
        assert rb.decay == "monotone_decrease_if_turn_based"

    def test_burn_decay_label_is_none(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-status", ["p1a: Garchomp", "brn"], turn=2)
        results = translate_event(evt, ctx)
        rb = results[0]
        assert rb.decay == "none"


# ---------------------------------------------------------------------------
# Mapping 7: |-boost| / |-unboost|
# ---------------------------------------------------------------------------

class TestBoostUnboost:
    def test_boost_stage_normalised(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-boost", ["p1a: Garchomp", "atk", "2"], turn=2)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert len(rb) == 1
        assert "boost_atk_unit_A" == rb[0].resource
        assert abs(rb[0].amount - (2 / 6.0)) < 0.001

    def test_unboost_same_magnitude(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-unboost", ["p1a: Garchomp", "spe", "1"], turn=3)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert abs(rb[0].amount - (1 / 6.0)) < 0.001
        assert "spe" in rb[0].resource

    def test_boost_stage_6_clamps_to_1(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-boost", ["p1a: Garchomp", "spa", "6"], turn=4)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert rb[0].amount == 1.0

    def test_boost_unit_label_in_resource(self):
        ctx = _ctx("p2")
        sw = _event("switch", ["p1a: Amoonguss", "Amoonguss, L80", "280/280"], turn=1)
        translate_event(sw, ctx)

        evt = _event("-boost", ["p1a: Amoonguss", "def", "1"], turn=2)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert "unit_A" in rb[0].resource


# ---------------------------------------------------------------------------
# Mapping 8: |move| — PP depletion
# ---------------------------------------------------------------------------

class TestMove:
    def test_pp_decrements_each_use(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        move_evt = _event("move", ["p1a: Garchomp", "Earthquake", "p2a: Target"], turn=2)
        results1 = translate_event(move_evt, ctx)
        rb1 = [c for c in results1 if isinstance(c, ResourceBudget)][0]
        assert rb1.resource == "pp_action_1"

        results2 = translate_event(move_evt, ctx)
        rb2 = [c for c in results2 if isinstance(c, ResourceBudget)][0]

        # Second use should show less remaining
        assert rb2.amount < rb1.amount

    def test_pp_resource_uses_action_label(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        move_evt = _event("move", ["p1a: Garchomp", "Earthquake", "p2a: Target"], turn=2)
        results = translate_event(move_evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)][0]
        assert rb.resource.startswith("pp_action_")

    def test_different_moves_get_different_action_labels(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        move1 = _event("move", ["p1a: Garchomp", "Earthquake", "p2a: Target"], turn=2)
        move2 = _event("move", ["p1a: Garchomp", "Dragon Claw", "p2a: Target"], turn=3)

        r1 = translate_event(move1, ctx)
        r2 = translate_event(move2, ctx)

        resource1 = r1[0].resource
        resource2 = r2[0].resource
        assert resource1 != resource2

    def test_pp_decay_label(self):
        ctx = _ctx("p1")
        sw = _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1)
        translate_event(sw, ctx)

        move_evt = _event("move", ["p1a: Garchomp", "Earthquake", "p2a: Target"], turn=2)
        results = translate_event(move_evt, ctx)
        rb = [c for c in results if isinstance(c, ResourceBudget)][0]
        assert rb.decay == "monotone_decrease"


# ---------------------------------------------------------------------------
# Mapping 9: |-sidestart| — hazards
# ---------------------------------------------------------------------------

class TestSidestart:
    @pytest.mark.parametrize("condition_raw,expected_dep", [
        ("Stealth Rock", "hazard_entry_rock"),
        ("Spikes",       "hazard_entry_spike"),
        ("Sticky Web",   "hazard_entry_web"),
        ("Toxic Spikes", "hazard_entry_toxic_spike"),
    ])
    def test_hazard_emits_coordination_dependency(self, condition_raw, expected_dep):
        ctx = _ctx("p1")
        evt = _event("-sidestart", ["p1: Player", condition_raw], turn=3)
        results = translate_event(evt, ctx)

        cd = [c for c in results if isinstance(c, CoordinationDependency)]
        assert len(cd) == 1
        assert cd[0].dependency == expected_dep

    def test_hazard_role_reflects_side(self):
        ctx = _ctx("p1")
        evt = _event("-sidestart", ["p2: Opponent", "Stealth Rock"], turn=3)
        results = translate_event(evt, ctx)

        cd = [c for c in results if isinstance(c, CoordinationDependency)][0]
        assert cd.role == "field_side_p2"

    def test_unknown_hazard_still_emits(self):
        ctx = _ctx("p1")
        evt = _event("-sidestart", ["p1: Player", "Future Sight Screen"], turn=5)
        results = translate_event(evt, ctx)

        cd = [c for c in results if isinstance(c, CoordinationDependency)]
        assert len(cd) == 1
        assert cd[0].dependency.startswith("hazard_")


# ---------------------------------------------------------------------------
# Mapping 10: |-weather|
# ---------------------------------------------------------------------------

class TestWeather:
    @pytest.mark.parametrize("weather_token,expected_obj_part,expected_shift_part", [
        ("RainDance", "rain",  "water_amplify"),
        ("SunnyDay",  "sun",   "fire_amplify"),
        ("Sandstorm", "sand",  "rock_steel"),
        ("Snow",      "snow",  "ice_buffer"),
    ])
    def test_weather_emits_optimization_criterion(self, weather_token, expected_obj_part, expected_shift_part):
        ctx = _ctx("p1")
        evt = _event("-weather", [weather_token], turn=4)
        results = translate_event(evt, ctx)

        oc = [c for c in results if isinstance(c, OptimizationCriterion)]
        assert len(oc) == 1
        assert expected_obj_part in oc[0].objective.lower()
        assert expected_shift_part in oc[0].weight_shift.lower()

    def test_weather_none_emits_neutral(self):
        ctx = _ctx("p1")
        evt = _event("-weather", ["none"], turn=8)
        results = translate_event(evt, ctx)

        oc = [c for c in results if isinstance(c, OptimizationCriterion)]
        assert len(oc) == 1
        assert oc[0].weight_shift == "neutral"


# ---------------------------------------------------------------------------
# Mapping 11: |-fieldstart| — terrain / trick room
# ---------------------------------------------------------------------------

class TestFieldstart:
    @pytest.mark.parametrize("field_token,expected_obj_part,expected_shift_part", [
        ("Electric Terrain", "electric",  "electric_amplify"),
        ("Grassy Terrain",   "grassy",    "grass_amplify"),
        ("Misty Terrain",    "misty",     "dragon_reduce"),
        ("Psychic Terrain",  "psychic",   "psychic_amplify"),
        ("Trick Room",       "trickroom", "speed_inversion"),
    ])
    def test_field_emits_optimization_criterion(self, field_token, expected_obj_part, expected_shift_part):
        ctx = _ctx("p1")
        evt = _event("-fieldstart", [field_token], turn=5)
        results = translate_event(evt, ctx)

        oc = [c for c in results if isinstance(c, OptimizationCriterion)]
        assert len(oc) == 1
        assert expected_obj_part.lower() in oc[0].objective.lower()
        assert expected_shift_part.lower() in oc[0].weight_shift.lower()


# ---------------------------------------------------------------------------
# Mapping 12: |turn| — every 5 turns
# ---------------------------------------------------------------------------

class TestTurnBudget:
    def test_turn_5_emits_budget(self):
        ctx = _ctx("p1")
        evt = _event("turn", [], turn=5)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert len(rb) == 1
        assert rb[0].resource == "match_time_remaining"
        assert abs(rb[0].amount - (1.0 - 5 / 60.0)) < 0.001

    def test_turn_10_emits_budget(self):
        ctx = _ctx("p1")
        evt = _event("turn", [], turn=10)
        results = translate_event(evt, ctx)

        rb = [c for c in results if isinstance(c, ResourceBudget)]
        assert len(rb) == 1
        assert abs(rb[0].amount - (1.0 - 10 / 60.0)) < 0.001

    def test_non_multiple_of_5_no_budget(self):
        ctx = _ctx("p1")
        for t in [1, 2, 3, 4, 6, 7, 8, 9, 11]:
            evt = _event("turn", [], turn=t)
            results = translate_event(evt, ctx)
            rb = [c for c in results if isinstance(c, ResourceBudget)]
            assert len(rb) == 0, f"Unexpected budget at turn {t}"

    def test_turn_budget_decay_label(self):
        ctx = _ctx("p1")
        evt = _event("turn", [], turn=5)
        results = translate_event(evt, ctx)
        rb = results[0]
        assert rb.decay == "monotone_decrease"

    def test_turn_60_amount_zero(self):
        ctx = _ctx("p1")
        evt = _event("turn", [], turn=60)
        results = translate_event(evt, ctx)
        rb = results[0]
        assert rb.amount == 0.0

    def test_turn_beyond_60_clamps_to_zero(self):
        ctx = _ctx("p1")
        evt = _event("turn", [], turn=65)
        results = translate_event(evt, ctx)
        rb = results[0]
        assert rb.amount == 0.0


# ---------------------------------------------------------------------------
# translate_match integration
# ---------------------------------------------------------------------------

class TestTranslateMatch:
    def test_translate_match_returns_ordered_constraints(self):
        events = [
            _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1),
            _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=1),
            _event("-damage", ["p1a: Garchomp", "p1a: Garchomp|245/350"], turn=2),
            _event("move",    ["p1a: Garchomp", "Earthquake", "p2a: Dragapult"], turn=2),
            _event("turn",    [], turn=5),
        ]
        constraints = translate_match(events, perspective="p1")
        # Should have at least one of each relevant type
        types = {type(c) for c in constraints}
        assert ToolAvailability in types
        assert ResourceBudget in types
        assert SubGoalTransition in types

    def test_translate_match_causal_order(self):
        """Constraints should be non-decreasing in timestamp."""
        events = [
            _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1),
            _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=1),
            _event("-damage", ["p1a: Garchomp", "p1a: Garchomp|200/350"], turn=2),
            _event("turn",    [], turn=5),
            _event("turn",    [], turn=10),
        ]
        constraints = translate_match(events, perspective="p1")
        timestamps = [c.timestamp for c in constraints]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------

class TestRenderer:
    def test_render_chain_no_pokemon_vocab(self):
        """Rendered output must not contain any raw Pokémon names."""
        events = [
            _event("switch", ["p1a: Garchomp",  "Garchomp, L80",  "350/350"], turn=1),
            _event("switch", ["p2a: Dragapult", "Dragapult, L80", "290/290"], turn=1),
            _event("-damage", ["p1a: Garchomp", "p1a: Garchomp|245/350"], turn=2),
            _event("-status", ["p2a: Dragapult", "par"], turn=3),
            _event("-weather", ["RainDance"], turn=4),
        ]
        constraints = translate_match(events, perspective="p1")
        rendered = render_chain(constraints, perspective="p1")
        leaked = check_pokemon_leakage(rendered, {"Garchomp", "Dragapult", "RainDance"})
        assert leaked == [], f"Pokémon vocabulary leaked: {leaked}"

    def test_render_chain_contains_step_numbers(self):
        events = [
            _event("switch", ["p1a: Garchomp", "Garchomp, L80", "350/350"], turn=1),
            _event("-damage", ["p1a: Garchomp", "p1a: Garchomp|245/350"], turn=2),
        ]
        constraints = translate_match(events, perspective="p1")
        rendered = render_chain(constraints, perspective="p1")
        assert "Step 1" in rendered
        assert "Step 2" in rendered

    def test_render_chain_empty(self):
        rendered = render_chain([], perspective="p1")
        assert "no constraints" in rendered

    def test_check_pokemon_leakage_catches_name(self):
        leaked = check_pokemon_leakage("unit_A fought Garchomp", {"Garchomp"})
        assert "Garchomp" in leaked

    def test_check_pokemon_leakage_case_insensitive(self):
        leaked = check_pokemon_leakage("GARCHOMP was damaged", {"Garchomp"})
        assert "Garchomp" in leaked

    def test_check_pokemon_leakage_no_false_positive(self):
        # "sand" is in Sandstorm's weight_shift descriptor but is not a Pokémon name
        leaked = check_pokemon_leakage("weight_shift=rock_steel_ground_buffer", {"Sandslash"})
        assert leaked == []

    def test_render_tool_availability_permanent(self):
        from src.renderer import render_constraint
        c = ToolAvailability(timestamp=3, tool="unit_B", state="unavailable", recover_in=None)
        rendered = render_constraint(c)
        assert "permanent" in rendered
        assert "unit_B" in rendered

    def test_render_resource_budget_shows_percentage(self):
        from src.renderer import render_constraint
        c = ResourceBudget(timestamp=2, resource="hp_unit_A", amount=0.7, decay="none", recover_in=None)
        rendered = render_constraint(c)
        assert "70.0%" in rendered

    def test_render_subgoal_transition(self):
        from src.renderer import render_constraint
        c = SubGoalTransition(timestamp=1, from_phase="initial", to_phase="vs_unit_A", trigger="opponent_switch")
        rendered = render_constraint(c)
        assert "initial" in rendered
        assert "vs_unit_A" in rendered
        assert "opponent_switch" in rendered

    def test_render_coordination_dependency(self):
        from src.renderer import render_constraint
        c = CoordinationDependency(timestamp=4, role="field_side_p2", dependency="hazard_entry_rock", expected_action="hazard_response")
        rendered = render_constraint(c)
        assert "field_side_p2" in rendered
        assert "hazard_entry_rock" in rendered

    def test_render_optimization_criterion(self):
        from src.renderer import render_constraint
        c = OptimizationCriterion(timestamp=5, objective="weather_rain", weight_shift="water_amplify_fire_reduce")
        rendered = render_constraint(c)
        assert "weather_rain" in rendered
        assert "water_amplify" in rendered
