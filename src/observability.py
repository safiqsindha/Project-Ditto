"""
Asymmetric observability filter (§4.5).

Renders constraint chains from one player's perspective, applying the
partial-information rules that govern what is and is not visible to that player.
"""

from __future__ import annotations

from src.translation import (
    Constraint,
    InformationState,
    ResourceBudget,
    ToolAvailability,
    SubGoalTransition,
    CoordinationDependency,
    OptimizationCriterion,
)


def bucket_hp(amount: float) -> float:
    """Bucket 0.0-1.0 HP to {0.0, 0.25, 0.5, 0.75, 1.0}."""
    if amount <= 0:
        return 0.0
    if amount <= 0.25:
        return 0.25
    if amount <= 0.5:
        return 0.5
    if amount <= 0.75:
        return 0.75
    return 1.0


def _opponent_player(perspective: str) -> str:
    """Return the other player label."""
    return "p2" if perspective == "p1" else "p1"


def _unit_belongs_to_opponent(tool_or_resource: str, revealed: set[str], perspective: str) -> bool:
    """
    Determine if a tool/resource string references a unit that belongs to the opponent.

    Unit labels look like unit_A, unit_B, ... — the translation layer assigns them
    independently per player, so we cannot infer ownership from the label alone.
    We use the revealed set as the source of truth: any unit that has appeared in
    an InformationState.observable_added is an opponent unit (the translator only
    emits InformationState for opponent switches).

    Additionally, we assume any unit referenced in ToolAvailability or ResourceBudget
    that has ever appeared in observable_added is an opponent unit.
    """
    # Extract unit label from resource strings like "hp_unit_A", "status_unit_B",
    # "boost_atk_unit_C", "pp_action_1" (pp is own), or bare "unit_A"
    import re
    m = re.search(r"(unit_[A-F])", tool_or_resource)
    if m:
        return m.group(1) in revealed
    return False


def apply_asymmetric_observability(
    constraints: list[Constraint],
    perspective: str,
) -> list[Constraint]:
    """
    Filter/modify constraints to reflect partial information from perspective player's POV.

    Rules:
    - player's own units: full visibility (keep all constraints)
    - opponent units:
      - ToolAvailability: only emit AFTER the unit has been revealed (first switch-in)
      - ResourceBudget(hp_*): bucket opponent HP to 0/25/50/75/100% brackets
      - ResourceBudget(status_*): keep as-is (visible on battle screen)
      - InformationState: keep (it's what makes opponent units get revealed)
    """
    # Track which opponent unit labels have been revealed via InformationState
    # or ToolAvailability events (the translator emits InformationState on opponent switch)
    revealed_opponent_units: set[str] = set()

    result: list[Constraint] = []

    for c in constraints:
        if isinstance(c, InformationState):
            # Always keep InformationState; also record revealed units
            for unit in c.observable_added:
                revealed_opponent_units.add(unit)
            result.append(c)

        elif isinstance(c, ToolAvailability):
            # Check if this tool is an opponent unit
            if _unit_belongs_to_opponent(c.tool, revealed_opponent_units, perspective):
                # Only emit if revealed
                if c.tool in revealed_opponent_units:
                    result.append(c)
                # else suppress
            else:
                # Could be own unit or a non-unit tool — keep it
                # Also: if this tool is a new unit we haven't seen, check if it
                # appears in InformationState (opponent) or not (own)
                # At this point any unit NOT in revealed_opponent_units is treated as own
                result.append(c)

        elif isinstance(c, ResourceBudget):
            resource = c.resource
            # Determine if this is an opponent unit's resource
            if _unit_belongs_to_opponent(resource, revealed_opponent_units, perspective):
                # Opponent unit: apply rules
                if resource.startswith("hp_"):
                    # Bucket HP
                    bucketed = bucket_hp(c.amount)
                    result.append(ResourceBudget(
                        timestamp=c.timestamp,
                        resource=c.resource,
                        amount=bucketed,
                        decay=c.decay,
                        recover_in=c.recover_in,
                    ))
                elif resource.startswith("status_"):
                    # Keep as-is (visible on battle screen)
                    result.append(c)
                else:
                    # Other opponent resources (boost_*, pp_*): suppress
                    # These are not directly observable before the unit acts
                    pass
            else:
                # Own unit or non-unit resource: full visibility
                result.append(c)

        else:
            # SubGoalTransition, CoordinationDependency, OptimizationCriterion:
            # these are field-level or own-player events — always visible
            result.append(c)

    return result
