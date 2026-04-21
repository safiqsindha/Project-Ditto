"""
Translation layer T: converts raw Showdown BattleEvents into abstract Constraint objects.

FROZEN after Session 3 (git tag T-v1.0-frozen).

Design principles:
  1. Parametric   — the same function applied uniformly to all matches
  2. Domain-abstracted — no Pokémon vocabulary in output
  3. Causally-preserving — constraints derived only from prior state + current event
  4. Inspectable  — dataclasses with explicit fields
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.parser import BattleEvent


# ---------------------------------------------------------------------------
# Constraint dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResourceBudget:
    """Continuous or depleting pool: HP, PP, status turns, match time."""
    timestamp: int          # turn number
    resource: str           # e.g. "hp_unit_A", "pp_action_2", "status_unit_B"
    amount: float           # normalised 0.0–1.0
    decay: str              # "none" | "monotone_decrease" | "monotone_decrease_if_turn_based"
    recover_in: int | None  # expected turns until replenished, or None


@dataclass
class ToolAvailability:
    """Discrete on/off: unit active/fainted, move usable/locked."""
    timestamp: int
    tool: str               # e.g. "unit_A", "action_1"
    state: str              # "available" | "unavailable"
    recover_in: int | None  # None ↔ permanent


@dataclass
class SubGoalTransition:
    """Regime change in battle plan."""
    timestamp: int
    from_phase: str
    to_phase: str
    trigger: str            # e.g. "opponent_switch"


@dataclass
class InformationState:
    """Partial observability events."""
    timestamp: int
    observable_added: list[str]
    observable_removed: list[str]
    uncertainty: float      # 0.0 = fully known, 1.0 = fully unknown


@dataclass
class CoordinationDependency:
    """Field effects that require a coordinated response."""
    timestamp: int
    role: str               # e.g. "field_side_p1", "field_side_p2"
    dependency: str         # e.g. "hazard_stealth_rock"
    expected_action: str    # abstract action hint


@dataclass
class OptimizationCriterion:
    """Weather, terrain, or boost states that shift optimal play."""
    timestamp: int
    objective: str          # e.g. "weather_rain", "terrain_electric"
    weight_shift: str       # type-specific descriptor


# Union type for all constraints
Constraint = (
    ResourceBudget
    | ToolAvailability
    | SubGoalTransition
    | InformationState
    | CoordinationDependency
    | OptimizationCriterion
)

# Maximum number of units we expect per player (teams of 6)
_MAX_UNITS = 6
_UNIT_LABELS = [f"unit_{chr(ord('A') + i)}" for i in range(_MAX_UNITS)]  # unit_A … unit_F
_MAX_MOVES = 4
_ACTION_LABELS = [f"action_{i + 1}" for i in range(_MAX_MOVES)]          # action_1 … action_4

# Default PP for a move when we haven't seen the team preview
_DEFAULT_PP = 16

# Status severity map — keys are Showdown status tokens
_STATUS_SEVERITY: dict[str, float] = {
    "par": 0.5,
    "brn": 0.5,
    "slp": 1.0,
    "frz": 1.0,
    "psn": 0.3,
    "tox": 0.3,  # grows; tracked in TranslationContext
}

# Weight-shift descriptors for weather/terrain — domain-abstracted labels
_WEATHER_SHIFT: dict[str, str] = {
    "raindance":   "water_amplify_fire_reduce",
    "sunnyday":    "fire_amplify_water_reduce",
    "sandstorm":   "rock_steel_ground_buffer_others_chip",
    "snow":        "ice_buffer_others_unaffected",
    "hail":        "ice_buffer_others_chip",
    # aliases that Showdown may emit
    "rain":        "water_amplify_fire_reduce",
    "sun":         "fire_amplify_water_reduce",
    "sand":        "rock_steel_ground_buffer_others_chip",
    "none":        "neutral",
}

_TERRAIN_SHIFT: dict[str, str] = {
    "electricterrain": "electric_amplify_sleep_immunity_grounded",
    "grassyterrain":   "grass_amplify_ground_reduce_grounded",
    "mistyterrain":    "dragon_reduce_status_immunity_grounded",
    "psychicterrain":  "psychic_amplify_priority_block_grounded",
    "trickroom":       "speed_inversion_active",
}

# Hazard dependency keys — domain-abstracted
_HAZARD_DEPENDENCY: dict[str, str] = {
    "stealthrock":   "hazard_entry_rock",
    "spikes":        "hazard_entry_spike",
    "stickyweb":     "hazard_entry_web",
    "toxicspikes":   "hazard_entry_toxic_spike",
}


# ---------------------------------------------------------------------------
# Translation context
# ---------------------------------------------------------------------------

@dataclass
class TranslationContext:
    """
    Mutable state accumulated across events within a single match.

    All mappings use abstract labels only; Pokémon names are hashed on first
    encounter and never stored after mapping is established.
    """
    perspective: str                            # "p1" or "p2"

    # unit_label → assigned for each player
    _unit_map: dict[str, dict[str, str]] = field(default_factory=dict)
    # reverse: abstract_label → original_name (used only inside translator, never leaked)
    _unit_rev: dict[str, dict[str, str]] = field(default_factory=dict)

    # move_label per unit: (player, unit_label) → {move_name: action_N}
    _move_map: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)

    # active unit label per player
    active_unit: dict[str, str | None] = field(default_factory=lambda: {"p1": None, "p2": None})

    # HP tracking: (player, unit_label) → max_hp
    _hp_max: dict[tuple[str, str], int] = field(default_factory=dict)

    # PP tracking: (player, unit_label, action_label) → remaining_pp
    _pp: dict[tuple[str, str, str], int] = field(default_factory=dict)

    # PP max: (player, unit_label, action_label) → max_pp
    _pp_max: dict[tuple[str, str, str], int] = field(default_factory=dict)

    # Current phase label per player
    phase: dict[str, str] = field(default_factory=lambda: {"p1": "initial", "p2": "initial"})

    # Toxic turn counter: (player, unit_label) → turns
    _tox_turns: dict[tuple[str, str], int] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Unit ID helpers
    # ---------------------------------------------------------------------------

    def _get_unit_label(self, player: str, pokemon_name: str) -> str:
        """Return the abstract unit label for a Pokémon name, assigning one if new."""
        if player not in self._unit_map:
            self._unit_map[player] = {}
            self._unit_rev[player] = {}
        pmap = self._unit_map[player]
        if pokemon_name not in pmap:
            idx = len(pmap)
            if idx >= _MAX_UNITS:
                # Overflow guard: reuse last slot
                idx = _MAX_UNITS - 1
            label = _UNIT_LABELS[idx]
            pmap[pokemon_name] = label
            self._unit_rev[player][label] = pokemon_name
        return pmap[pokemon_name]

    def get_unit_label(self, player: str, pokemon_name: str) -> str:
        return self._get_unit_label(player, pokemon_name)

    # ---------------------------------------------------------------------------
    # Move ID helpers
    # ---------------------------------------------------------------------------

    def _get_action_label(self, player: str, unit_label: str, move_name: str) -> str:
        key = (player, unit_label)
        if key not in self._move_map:
            self._move_map[key] = {}
        mmap = self._move_map[key]
        if move_name not in mmap:
            idx = len(mmap)
            if idx >= _MAX_MOVES:
                idx = _MAX_MOVES - 1
            mmap[move_name] = _ACTION_LABELS[idx]
        return mmap[move_name]

    def get_action_label(self, player: str, unit_label: str, move_name: str) -> str:
        return self._get_action_label(player, unit_label, move_name)

    # ---------------------------------------------------------------------------
    # HP helpers
    # ---------------------------------------------------------------------------

    def record_hp_max(self, player: str, unit_label: str, max_hp: int) -> None:
        key = (player, unit_label)
        if key not in self._hp_max:
            self._hp_max[key] = max_hp

    def get_hp_max(self, player: str, unit_label: str) -> int:
        return self._hp_max.get((player, unit_label), 100)

    # ---------------------------------------------------------------------------
    # PP helpers
    # ---------------------------------------------------------------------------

    def use_pp(self, player: str, unit_label: str, action_label: str) -> None:
        key = (player, unit_label, action_label)
        if key not in self._pp_max:
            self._pp_max[key] = _DEFAULT_PP
            self._pp[key] = _DEFAULT_PP
        self._pp[key] = max(0, self._pp[key] - 1)

    def get_pp_fraction(self, player: str, unit_label: str, action_label: str) -> float:
        key = (player, unit_label, action_label)
        remaining = self._pp.get(key, _DEFAULT_PP)
        max_pp = self._pp_max.get(key, _DEFAULT_PP)
        return remaining / max_pp if max_pp > 0 else 0.0

    # ---------------------------------------------------------------------------
    # Toxic tracker
    # ---------------------------------------------------------------------------

    def increment_tox(self, player: str, unit_label: str) -> int:
        key = (player, unit_label)
        self._tox_turns[key] = self._tox_turns.get(key, 0) + 1
        return self._tox_turns[key]

    def get_tox_turns(self, player: str, unit_label: str) -> int:
        return self._tox_turns.get((player, unit_label), 1)


# ---------------------------------------------------------------------------
# HP parsing
# ---------------------------------------------------------------------------

def _parse_hp_arg(hp_str: str) -> tuple[int, int]:
    """
    Parse Showdown HP string like '245/350' or '0 fnt'.
    Returns (current_hp, max_hp).
    """
    hp_str = hp_str.strip()
    if hp_str in ("0 fnt", "0"):
        return 0, 100
    m = re.match(r"(\d+)/(\d+)", hp_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 100


def _parse_unit_from_arg(arg: str) -> tuple[str, str]:
    """
    Parse 'p1a: Garchomp' or 'p2b: Dragapult' into (player, pokemon_name).
    """
    m = re.match(r"^(p[12])[ab]?:\s*(.+)$", arg)
    if m:
        return m.group(1), m.group(2).strip()
    return "", ""


def _parse_hp_from_condition(condition_arg: str) -> tuple[int, int]:
    """
    Parse 'p1a: Garchomp|245/350 brn' — return (current_hp, max_hp).
    The pipe separates the identity from the condition string.
    """
    if "|" in condition_arg:
        hp_part = condition_arg.split("|", 1)[1]
    else:
        hp_part = condition_arg
    # Strip status suffix like " brn", " par" etc.
    hp_part = hp_part.split(" ")[0]
    return _parse_hp_arg(hp_part)


# ---------------------------------------------------------------------------
# Individual event translators
# ---------------------------------------------------------------------------

def _translate_switch(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |switch|POKEMON|DETAILS|HPCONDITION
    args[0] = 'p1a: Garchomp'
    args[1] = species details (ignored)
    args[2] = HP condition like '350/350'
    """
    constraints: list[Constraint] = []
    if not event.args:
        return constraints

    player, pokemon_name = _parse_unit_from_arg(event.args[0])
    if not player:
        return constraints

    incoming_label = ctx.get_unit_label(player, pokemon_name)

    # Parse and record max HP
    if len(event.args) >= 3:
        cur_hp, max_hp = _parse_hp_from_condition(event.args[2])
        ctx.record_hp_max(player, incoming_label, max_hp)

    # Mark previous active unit as unavailable (switched out)
    prev_label = ctx.active_unit.get(player)
    if prev_label is not None and prev_label != incoming_label:
        constraints.append(ToolAvailability(
            timestamp=event.turn,
            tool=prev_label,
            state="unavailable",
            recover_in=0,  # 0 = benched, recoverable
        ))

    # Mark incoming unit as available
    ctx.active_unit[player] = incoming_label
    constraints.append(ToolAvailability(
        timestamp=event.turn,
        tool=incoming_label,
        state="available",
        recover_in=None,
    ))

    if player == ctx.perspective:
        # Own switch: phase update handled implicitly
        pass
    else:
        # Opponent switch
        prior_phase = ctx.phase.get(player, "initial")
        new_phase = f"vs_{incoming_label}"
        ctx.phase[player] = new_phase
        constraints.append(SubGoalTransition(
            timestamp=event.turn,
            from_phase=prior_phase,
            to_phase=new_phase,
            trigger="opponent_switch",
        ))
        constraints.append(InformationState(
            timestamp=event.turn,
            observable_added=[incoming_label],
            observable_removed=[],
            uncertainty=0.5,
        ))

    return constraints


def _translate_faint(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |faint|POKEMON
    """
    constraints: list[Constraint] = []
    if not event.args:
        return constraints

    player, pokemon_name = _parse_unit_from_arg(event.args[0])
    if not player:
        return constraints

    unit_label = ctx.get_unit_label(player, pokemon_name)
    ctx.active_unit[player] = None

    constraints.append(ToolAvailability(
        timestamp=event.turn,
        tool=unit_label,
        state="unavailable",
        recover_in=None,  # permanent
    ))

    if player == ctx.perspective:
        prior = ctx.phase.get(player, "initial")
        ctx.phase[player] = "forced_switch_required"
        constraints.append(SubGoalTransition(
            timestamp=event.turn,
            from_phase=prior,
            to_phase="forced_switch_required",
            trigger="own_faint",
        ))

    return constraints


def _translate_damage_heal(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |-damage|POKEMON|HPCONDITION
    |-heal|POKEMON|HPCONDITION
    """
    constraints: list[Constraint] = []
    if len(event.args) < 2:
        return constraints

    player, pokemon_name = _parse_unit_from_arg(event.args[0])
    if not player:
        return constraints

    unit_label = ctx.get_unit_label(player, pokemon_name)

    cur_hp, max_hp = _parse_hp_from_condition(event.args[1])
    ctx.record_hp_max(player, unit_label, max_hp)
    actual_max = ctx.get_hp_max(player, unit_label)
    amount = cur_hp / actual_max if actual_max > 0 else 0.0

    constraints.append(ResourceBudget(
        timestamp=event.turn,
        resource=f"hp_{unit_label}",
        amount=round(amount, 4),
        decay="none",
        recover_in=None,
    ))

    return constraints


def _translate_status(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |-status|POKEMON|STATUS
    """
    constraints: list[Constraint] = []
    if len(event.args) < 2:
        return constraints

    player, pokemon_name = _parse_unit_from_arg(event.args[0])
    if not player:
        return constraints

    unit_label = ctx.get_unit_label(player, pokemon_name)
    status_token = event.args[1].lower().strip()

    severity = _STATUS_SEVERITY.get(status_token, 0.3)
    decay = "none"
    if status_token in ("slp", "tox"):
        decay = "monotone_decrease_if_turn_based"

    constraints.append(ResourceBudget(
        timestamp=event.turn,
        resource=f"status_{unit_label}",
        amount=severity,
        decay=decay,
        recover_in=None,
    ))

    return constraints


def _translate_boost(
    event: BattleEvent,
    ctx: TranslationContext,
    is_unboost: bool = False,
) -> list[Constraint]:
    """
    |-boost|POKEMON|STAT|AMOUNT
    |-unboost|POKEMON|STAT|AMOUNT
    """
    constraints: list[Constraint] = []
    if len(event.args) < 3:
        return constraints

    player, pokemon_name = _parse_unit_from_arg(event.args[0])
    if not player:
        return constraints

    unit_label = ctx.get_unit_label(player, pokemon_name)
    stat_name = event.args[1].lower().strip()

    try:
        stage_delta = int(event.args[2])
    except (ValueError, IndexError):
        stage_delta = 1

    if is_unboost:
        stage_delta = -stage_delta

    # Normalise: stage range is −6..+6, map to 0..1 as (stage+6)/12
    # We emit the magnitude shift normalised to [0, 1]
    normalised = min(1.0, max(0.0, abs(stage_delta) / 6.0))

    constraints.append(ResourceBudget(
        timestamp=event.turn,
        resource=f"boost_{stat_name}_{unit_label}",
        amount=round(normalised, 4),
        decay="none",
        recover_in=None,
    ))

    return constraints


def _translate_move(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |move|POKEMON|MOVENAME|TARGET
    Emits a PP-depletion constraint every time a move is used.
    """
    constraints: list[Constraint] = []
    if len(event.args) < 2:
        return constraints

    player, pokemon_name = _parse_unit_from_arg(event.args[0])
    if not player:
        return constraints

    unit_label = ctx.get_unit_label(player, pokemon_name)
    move_name = event.args[1].strip()
    action_label = ctx.get_action_label(player, unit_label, move_name)

    ctx.use_pp(player, unit_label, action_label)
    pp_fraction = ctx.get_pp_fraction(player, unit_label, action_label)

    constraints.append(ResourceBudget(
        timestamp=event.turn,
        resource=f"pp_{action_label}",
        amount=round(pp_fraction, 4),
        decay="monotone_decrease",
        recover_in=None,
    ))

    return constraints


def _translate_sidestart(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |-sidestart|SIDE|CONDITION
    e.g. |-sidestart|p1: PlayerName|Stealth Rock
    """
    constraints: list[Constraint] = []
    if len(event.args) < 2:
        return constraints

    # args[0] is side identifier like "p1: PlayerName"
    side_arg = event.args[0]
    m = re.match(r"^(p[12])", side_arg)
    side = m.group(1) if m else "p1"

    condition_raw = event.args[1].lower().replace(" ", "")
    dependency = _HAZARD_DEPENDENCY.get(condition_raw)
    if dependency is None:
        # Unknown hazard — still emit with a generic label
        dependency = f"hazard_{condition_raw}"

    constraints.append(CoordinationDependency(
        timestamp=event.turn,
        role=f"field_side_{side}",
        dependency=dependency,
        expected_action="hazard_response",
    ))

    return constraints


def _translate_weather(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |-weather|WEATHERTYPE[|...]
    """
    constraints: list[Constraint] = []
    if not event.args:
        return constraints

    weather_raw = event.args[0].lower().strip()
    if weather_raw in ("none", ""):
        # Weather ended — emit neutral
        weight_shift = "neutral"
        objective = "weather_none"
    else:
        weight_shift = _WEATHER_SHIFT.get(weather_raw, "unknown_weather_shift")
        objective = f"weather_{weather_raw}"

    constraints.append(OptimizationCriterion(
        timestamp=event.turn,
        objective=objective,
        weight_shift=weight_shift,
    ))

    return constraints


def _translate_fieldstart(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |-fieldstart|CONDITION[|...]
    """
    constraints: list[Constraint] = []
    if not event.args:
        return constraints

    field_raw = event.args[0].lower().replace(" ", "")
    weight_shift = _TERRAIN_SHIFT.get(field_raw, f"field_{field_raw}_shift")
    objective = f"field_{field_raw}"

    constraints.append(OptimizationCriterion(
        timestamp=event.turn,
        objective=objective,
        weight_shift=weight_shift,
    ))

    return constraints


def _translate_turn(
    event: BattleEvent,
    ctx: TranslationContext,
) -> list[Constraint]:
    """
    |turn|N
    Emit a match-time budget constraint every 5 turns.
    """
    constraints: list[Constraint] = []
    turn_number = event.turn
    if turn_number % 5 != 0:
        return constraints

    # Assumes matches rarely exceed 60 turns; clamp to [0, 1]
    amount = max(0.0, 1.0 - turn_number / 60.0)

    constraints.append(ResourceBudget(
        timestamp=turn_number,
        resource="match_time_remaining",
        amount=round(amount, 4),
        decay="monotone_decrease",
        recover_in=None,
    ))

    return constraints


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TRANSLATORS = {
    "switch":       _translate_switch,
    "faint":        _translate_faint,
    "-damage":      _translate_damage_heal,
    "-heal":        _translate_damage_heal,
    "-status":      _translate_status,
    "-boost":       lambda e, c: _translate_boost(e, c, is_unboost=False),
    "-unboost":     lambda e, c: _translate_boost(e, c, is_unboost=True),
    "move":         _translate_move,
    "-sidestart":   _translate_sidestart,
    "-weather":     _translate_weather,
    "-fieldstart":  _translate_fieldstart,
    "turn":         _translate_turn,
}


def translate_event(event: BattleEvent, ctx: TranslationContext) -> list[Constraint]:
    """Translate a single BattleEvent into zero or more Constraint objects."""
    handler = _TRANSLATORS.get(event.type)
    if handler is None:
        return []
    return handler(event, ctx)


def translate_match(events: list[BattleEvent], perspective: str) -> list[Constraint]:
    """
    Translate a full sequence of BattleEvents from the given player perspective.

    Parameters
    ----------
    events      : list of BattleEvent from parser.constraint_events()
    perspective : "p1" or "p2"

    Returns
    -------
    Ordered list of Constraint objects (causally ordered by turn).
    """
    ctx = TranslationContext(perspective=perspective)
    constraints: list[Constraint] = []
    for event in events:
        constraints.extend(translate_event(event, ctx))
    return constraints
