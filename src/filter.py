"""
Chain validity filter (§5.2).

Checks whether a constraint chain meets the quality criteria required
for inclusion in the evaluation dataset.
"""

from __future__ import annotations

import dataclasses

from src.translation import (
    Constraint,
    ResourceBudget,
    SubGoalTransition,
    ToolAvailability,
)


def is_valid_chain(constraints: list[Constraint]) -> bool:
    """
    A chain is valid if ALL of:
    - length between 20 and 40 (inclusive)
    - contains >= 2 SubGoalTransition events
    - contains >= 1 ToolAvailability with state='unavailable'
    - contains >= 10 ResourceBudget events
    - timestamps are strictly non-decreasing
    - no two consecutive constraints are identical (no stuttering)
    """
    n = len(constraints)

    # Length check
    if n < 20 or n > 40:
        return False

    # Count constraint types
    subgoal_count = 0
    unavailable_count = 0
    resource_count = 0

    for c in constraints:
        if isinstance(c, SubGoalTransition):
            subgoal_count += 1
        elif isinstance(c, ToolAvailability) and c.state == "unavailable":
            unavailable_count += 1
        elif isinstance(c, ResourceBudget):
            resource_count += 1

    if subgoal_count < 2:
        return False
    if unavailable_count < 1:
        return False
    if resource_count < 10:
        return False

    # Timestamps strictly non-decreasing
    for i in range(1, n):
        t_prev = getattr(constraints[i - 1], "timestamp", 0)
        t_curr = getattr(constraints[i], "timestamp", 0)
        if t_curr < t_prev:
            return False

    # No two consecutive identical constraints (no stuttering)
    for i in range(1, n):
        if type(constraints[i]) == type(constraints[i - 1]):
            if dataclasses.asdict(constraints[i]) == dataclasses.asdict(constraints[i - 1]):
                return False

    return True


def filter_chains(chains: list[dict]) -> list[dict]:
    """Apply is_valid_chain to a list of chain dicts.

    Each chain dict must have a 'constraints' key containing a list of
    Constraint dataclass instances (not serialized dicts).
    """
    return [chain for chain in chains if is_valid_chain(chain["constraints"])]
