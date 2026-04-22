"""
Shuffler: produces a control-condition shuffled variant for each real chain (§5.3).

The shuffled chain has its constraints permuted uniformly at random, but
timestamps are re-assigned to preserve monotonic ordering so the shuffle
cannot be detected from timestamp order alone.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

from src.translation import Constraint


def _get_timestamps(constraints: list) -> list[int]:
    """Extract ordered timestamps from the original chain (handles both dicts and dataclasses)."""
    result = []
    for c in constraints:
        if isinstance(c, dict):
            result.append(c.get("timestamp", 0))
        else:
            result.append(getattr(c, "timestamp", 0))
    return result


def _set_timestamp(constraint, ts: int):
    """Return a copy of the constraint with the timestamp replaced (handles both formats)."""
    if isinstance(constraint, dict):
        new = dict(constraint)
        new["timestamp"] = ts
        return new
    d = dataclasses.asdict(constraint)
    d["timestamp"] = ts
    return type(constraint)(**d)


def shuffle_chain(chain: dict, seed: int) -> dict:
    """
    Produce one shuffled variant of a real chain.

    Rules:
    - Permute constraints uniformly at random (seeded)
    - Reassign timestamps to preserve monotonic ordering (so shuffled chains
      don't reveal themselves via out-of-order timestamps)
    - Preserve InformationState semantics — the event moves but keeps its content
    - Chain ID: {original_id}_shuffled_{seed}
    - Preserve match_id field
    """
    original_constraints: list[Constraint] = chain["constraints"]

    # Collect the sorted (non-decreasing) sequence of timestamps from the original
    original_timestamps = _get_timestamps(original_constraints)
    sorted_timestamps = sorted(original_timestamps)

    # Permute the constraints via an explicit permutation so we can apply the
    # same permutation to any parallel per-step metadata (e.g. active_pair_by_step).
    rng = random.Random(seed)
    perm = list(range(len(original_constraints)))
    rng.shuffle(perm)
    shuffled_constraints = [original_constraints[i] for i in perm]

    # Re-assign timestamps in sorted order so the sequence is non-decreasing
    reassigned: list[Constraint] = []
    for i, c in enumerate(shuffled_constraints):
        new_ts = sorted_timestamps[i]
        reassigned.append(_set_timestamp(c, new_ts))

    # Build shuffled chain dict
    original_id = chain["chain_id"]
    shuffled_chain = dict(chain)  # shallow copy of top-level fields
    shuffled_chain["chain_id"] = f"{original_id}_shuffled_{seed}"
    shuffled_chain["match_id"] = chain["match_id"]
    shuffled_chain["constraints"] = reassigned
    if "active_pair_by_step" in chain:
        orig_pairs = chain["active_pair_by_step"]
        shuffled_chain["active_pair_by_step"] = [orig_pairs[i] for i in perm]

    # The rendered field and action_at_step will be (re-)computed by the pipeline
    # based on the shuffled constraints, so we don't copy them blindly from the original.
    # (The pipeline sets these fields before saving.)

    return shuffled_chain
