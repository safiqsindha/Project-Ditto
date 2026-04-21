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


def _find_opponent_reveal_indices(constraints: list[Constraint]) -> dict[str, int]:
    """
    Return a mapping from opponent unit label to the index of its first InformationState
    revelation in the constraint list.

    The translator emits InformationState only for opponent switch-ins, so every unit
    that ever appears in InformationState.observable_added is an opponent unit.
    """
    first_reveal: dict[str, int] = {}
    for i, c in enumerate(constraints):
        if isinstance(c, InformationState):
            for unit in c.observable_added:
                if unit not in first_reveal:
                    first_reveal[unit] = i
    return first_reveal


def _find_opponent_pre_reveal_ta_indices(
    constraints: list[Constraint],
    first_reveal: dict[str, int],
) -> set[int]:
    """
    Identify the indices of ToolAvailability constraints that correspond to an
    opponent unit's switch-in BEFORE that unit has been revealed.

    The translator emits for an opponent switch:
        [ToolAvailability(incoming, available), SubGoalTransition, InformationState(incoming)]
    in that order (with an optional prior ToolAvailability(prev, unavailable) before).

    We detect these 'pre-reveal TAs' by looking backwards from each InformationState:
    the TA immediately preceding SubGoalTransition at the InformationState is the one
    to suppress.
    """
    suppress_indices: set[int] = set()

    for i, c in enumerate(constraints):
        if not isinstance(c, InformationState):
            continue
        revealed_units = c.observable_added
        if not revealed_units:
            continue

        # Walk backwards from i to find the ToolAvailability that triggered this reveal.
        # Pattern: ..., TA(incoming avail), [TA(prev unavail),] SubGoalTransition, InformationState
        # We look at i-1 and i-2 for SubGoalTransition, then the TA before that.
        # In practice the translator emits: TA_avail, SubGoalTransition, InformationState
        # (prev TA_unavail is emitted before TA_avail)
        j = i - 1
        while j >= 0 and isinstance(constraints[j], SubGoalTransition):
            j -= 1
        # j now points to the ToolAvailability for the incoming unit (available)
        if j >= 0 and isinstance(constraints[j], ToolAvailability):
            ta = constraints[j]
            if ta.tool in revealed_units and ta.state == "available":
                # This TA is the opponent's pre-reveal switch-in; suppress if before reveal
                if j < first_reveal.get(ta.tool, i):
                    suppress_indices.add(j)

    return suppress_indices


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
    # Two-pass approach to handle the label collision between own and opponent units.
    # Both players use labels unit_A..unit_F, so we cannot distinguish own from
    # opponent purely by label. Instead:
    #   1. Pre-scan to find which indices hold pre-reveal opponent switch-in TAs.
    #   2. Collect all units that appear in InformationState (= all opponent units).
    #   3. Apply suppression and bucketing rules.

    first_reveal = _find_opponent_reveal_indices(constraints)
    opponent_units: set[str] = set(first_reveal.keys())
    suppress_ta_indices = _find_opponent_pre_reveal_ta_indices(constraints, first_reveal)

    result: list[Constraint] = []

    for i, c in enumerate(constraints):
        if isinstance(c, InformationState):
            # Always keep; also no modification needed
            result.append(c)

        elif isinstance(c, ToolAvailability):
            if i in suppress_ta_indices:
                # Pre-reveal opponent switch-in TA — suppress
                continue
            # Post-reveal or own-unit TA: keep as-is
            result.append(c)

        elif isinstance(c, ResourceBudget):
            resource = c.resource

            # Extract unit label from resource string (e.g. "hp_unit_A" -> "unit_A")
            import re
            m = re.search(r"(unit_[A-F])", resource)
            unit_label = m.group(1) if m else None

            if unit_label and unit_label in opponent_units:
                if resource.startswith("hp_"):
                    # Bucket opponent HP
                    bucketed = bucket_hp(c.amount)
                    result.append(ResourceBudget(
                        timestamp=c.timestamp,
                        resource=c.resource,
                        amount=bucketed,
                        decay=c.decay,
                        recover_in=c.recover_in,
                    ))
                elif resource.startswith("status_"):
                    # Status is visible on screen — keep as-is
                    result.append(c)
                else:
                    # Other opponent resources (boost_*, etc.) — suppress
                    pass
            else:
                # Own unit resource or non-unit resource — full visibility
                result.append(c)

        else:
            # SubGoalTransition, CoordinationDependency, OptimizationCriterion:
            # field-level or own-player events — always visible
            result.append(c)

    return result
