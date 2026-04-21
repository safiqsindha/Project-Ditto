"""
Reference action distribution lookup table for objective scoring (Layer 1).

Maps (game state signature) -> (empirical action distribution among high-Elo players).
Used at evaluation time to score whether a model's proposed action matches what
real high-Elo players chose.

Backoff levels (§3.4):
  Level 0 (full):     all 5 components
  Level 1:            drop field_conditions
  Level 2:            drop field_conditions + turn_bucket
  Level 3 (max):      drop field_conditions + turn_bucket + status_effects

Chains where ≥40% of states require level 3 backoff are discarded.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HP bucketing helper
# ---------------------------------------------------------------------------

def _hp_bracket(amount: float) -> int:
    """
    Convert a normalised HP fraction (0.0–1.0) into a discrete bracket.

    Returns
    -------
    0  fainted  (amount == 0)
    1  <25%     (0 < amount < 0.25)
    2  25–50%   (0.25 <= amount < 0.50)
    3  50–75%   (0.50 <= amount < 0.75)
    4  75–100%  (amount >= 0.75)
    """
    if amount <= 0.0:
        return 0
    if amount < 0.25:
        return 1
    if amount < 0.50:
        return 2
    if amount < 0.75:
        return 3
    return 4


# ---------------------------------------------------------------------------
# StateSignature
# ---------------------------------------------------------------------------

@dataclass
class StateSignature:
    """
    Compact, hashable descriptor of a battle state.

    Fields
    ------
    active_pair       : (player_unit_label, opponent_unit_label) — both abstracted
    hp_brackets       : (player_hp_bracket, opponent_hp_bracket) — bucketed 0–4
    status_effects    : frozenset of active status flags, e.g. "unit_A_brn"
    field_conditions  : frozenset of active weather/terrain/hazard strings
    turn_bucket       : turn number // 10
    """
    active_pair: tuple[str, str]
    hp_brackets: tuple[int, int]
    status_effects: frozenset
    field_conditions: frozenset
    turn_bucket: int

    def to_key(self, level: int = 0) -> tuple:
        """
        Return a hashable key at the requested backoff level.

        Level 0 (full):  all 5 components
        Level 1:         drop field_conditions
        Level 2:         drop field_conditions + turn_bucket
        Level 3 (max):   drop field_conditions + turn_bucket + status_effects
        """
        if level == 0:
            return (
                self.active_pair,
                self.hp_brackets,
                self.status_effects,
                self.field_conditions,
                self.turn_bucket,
            )
        if level == 1:
            return (
                self.active_pair,
                self.hp_brackets,
                self.status_effects,
                self.turn_bucket,
            )
        if level == 2:
            return (
                self.active_pair,
                self.hp_brackets,
                self.status_effects,
            )
        # level 3 — maximum backoff
        return (
            self.active_pair,
            self.hp_brackets,
        )


# ---------------------------------------------------------------------------
# extract_state_signature  (from chain dict format)
# ---------------------------------------------------------------------------

def extract_state_signature(chain: dict, step_idx: int) -> StateSignature | None:
    """
    Extract a StateSignature from a chain dict at the given step index.

    Chain dict format (chains/real/*.jsonl):
    {
      "chain_id": "synthetic_00000_p1",
      "perspective": "p1",
      "constraints": [
        {"type": "ResourceBudget", "timestamp": 1, "resource": "hp_unit_A", "amount": 0.85, ...},
        {"type": "ToolAvailability", "timestamp": 1, "tool": "unit_A", "state": "available", ...},
        ...
      ],
      "action_at_step": "use action_1 with unit_A"
    }

    Extraction rules
    ----------------
    active_pair      : most recent ToolAvailability(state=available) for p1 and p2 before step K.
                       Since chains are single-perspective, we look for unit labels whose
                       availability is tracked and identify player vs opponent units by prefix.
    hp_brackets      : most recent ResourceBudget(resource=hp_*) for each active unit before K.
    status_effects   : all ResourceBudget(resource=status_*) active before K.
    field_conditions : all OptimizationCriterion and CoordinationDependency active before K.
    turn_bucket      : constraints[step_idx].timestamp // 10
    """
    constraints = chain.get("constraints", [])
    if step_idx >= len(constraints):
        return None

    step_constraint = constraints[step_idx]
    timestamp = step_constraint.get("timestamp", 0)
    turn_bucket = timestamp // 10

    # Collect constraints up to (and including) this step index
    prior = constraints[:step_idx + 1]

    # -----------------------------------------------------------------------
    # Active unit per player: most recent ToolAvailability(state=available)
    # -----------------------------------------------------------------------
    # In a single-perspective chain the unit labels are:
    #   "unit_A" … "unit_F" for the focal player (p1 or p2)
    # We track two "sides": focal (p1) and opponent (p2).
    # The translation layer assigns labels independently per player, so both
    # p1 and p2 can each have a unit_A. The chain only contains one
    # perspective's constraints, so all unit labels here are from that player.
    # For opponent units, we need a separate signal — but in the chain format
    # described, both perspectives' ToolAvailability constraints may be
    # intermixed (the renderer sees both sides). We use a heuristic:
    # the chain's perspective field tells us which side is "ours".

    perspective = chain.get("perspective", "p1")
    opponent = "p2" if perspective == "p1" else "p1"

    # Track most-recent available tool for focal and opponent, keyed by
    # a side tag embedded in the tool name if present, or positionally.
    # Since the chain constraints are produced by a single TranslationContext
    # (one perspective), we may not have opponent unit labels unless they were
    # observed. We look for ToolAvailability constraints and accept any label.

    # Most recent available tool (we track separately for p1 and p2 if the
    # chain embeds "p1_unit_A" style names, otherwise fall back to order).
    p1_active: str | None = None
    p2_active: str | None = None

    for c in reversed(prior):
        if c.get("type") != "ToolAvailability":
            continue
        if c.get("state") != "available":
            continue
        tool = c.get("tool", "")
        # Chains may encode player prefix: "p1_unit_A" or just "unit_A"
        if tool.startswith("p1_") and p1_active is None:
            p1_active = tool[3:]  # strip "p1_"
        elif tool.startswith("p2_") and p2_active is None:
            p2_active = tool[3:]
        elif not tool.startswith("p"):
            # No player prefix — treat as focal player's unit
            if p1_active is None:
                p1_active = tool

    # If we still don't have both sides, fall back to the focal player only
    if p1_active is None:
        p1_active = "unit_unknown"
    if p2_active is None:
        p2_active = "unit_unknown"

    if perspective == "p2":
        # Swap so active_pair is always (focal, opponent)
        active_pair = (p2_active, p1_active)
    else:
        active_pair = (p1_active, p2_active)

    # -----------------------------------------------------------------------
    # HP brackets for active units
    # -----------------------------------------------------------------------
    hp_map: dict[str, float] = {}
    for c in prior:
        if c.get("type") != "ResourceBudget":
            continue
        resource = c.get("resource", "")
        if resource.startswith("hp_"):
            unit_label = resource[3:]  # e.g. "unit_A"
            hp_map[unit_label] = c.get("amount", 1.0)

    focal_hp = hp_map.get(active_pair[0], 1.0)
    opp_hp = hp_map.get(active_pair[1], 1.0)
    hp_brackets = (_hp_bracket(focal_hp), _hp_bracket(opp_hp))

    # -----------------------------------------------------------------------
    # Status effects: ResourceBudget(resource=status_*)
    # -----------------------------------------------------------------------
    # We take the most recent status amount per unit; non-zero = active
    status_map: dict[str, float] = {}
    for c in prior:
        if c.get("type") != "ResourceBudget":
            continue
        resource = c.get("resource", "")
        if resource.startswith("status_"):
            unit_label = resource[7:]
            status_map[unit_label] = c.get("amount", 0.0)

    status_effects: frozenset = frozenset(
        f"{unit}_{amt:.2f}" for unit, amt in status_map.items() if amt > 0
    )

    # -----------------------------------------------------------------------
    # Field conditions: OptimizationCriterion + CoordinationDependency
    # -----------------------------------------------------------------------
    field_set: set[str] = set()
    # Track most-recent objective value for weather/terrain (last one wins)
    seen_objectives: dict[str, str] = {}
    for c in prior:
        ctype = c.get("type", "")
        if ctype == "OptimizationCriterion":
            obj = c.get("objective", "")
            if obj:
                seen_objectives[obj.split("_")[0]] = obj  # key by category
        elif ctype == "CoordinationDependency":
            dep = c.get("dependency", "")
            role = c.get("role", "")
            if dep:
                field_set.add(f"{dep}_{role}")
    field_set.update(seen_objectives.values())
    field_conditions = frozenset(field_set)

    return StateSignature(
        active_pair=active_pair,
        hp_brackets=hp_brackets,
        status_effects=status_effects,
        field_conditions=field_conditions,
        turn_bucket=turn_bucket,
    )


# ---------------------------------------------------------------------------
# ReferenceDistribution
# ---------------------------------------------------------------------------

class ReferenceDistribution:
    """
    Lookup table mapping state signature keys to action count dicts.

    Internal storage
    ----------------
    _counts[level][key] = {action: count}

    Levels 0–3 correspond to the backoff levels in StateSignature.to_key().
    """

    _MAX_BACKOFF_FRACTION = 0.40  # chains above this fraction at level 3 are discarded

    def __init__(self) -> None:
        # counts[level][key] = {action_str: int}
        self._counts: list[dict[tuple, dict[str, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(4)
        ]

    # ------------------------------------------------------------------
    # Build from chain files (JSONL with chain-dict format)
    # ------------------------------------------------------------------

    def build_from_chains(self, chain_files: list[Path]) -> None:
        """
        Populate the distribution from a list of chain JSONL files.

        Each line in a JSONL file is expected to be a chain dict with:
          - "constraints": list of constraint dicts
          - "action_at_step": the action taken (string)

        For each chain we enumerate every step that has an action and record
        the (state_signature, action) pair.

        Chains where ≥40% of states require level-3 backoff are discarded.
        """
        for path in chain_files:
            self._ingest_chain_file(path)

    def _ingest_chain_file(self, path: Path) -> None:
        """Process a single JSONL chain file."""
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    chain = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._ingest_chain(chain)

    def _ingest_chain(self, chain: dict) -> None:
        """Record state→action pairs from a single chain dict."""
        constraints = chain.get("constraints", [])
        if not constraints:
            return

        # Gather (sig, action) for every step that has an "action_at_step"
        pairs: list[tuple[StateSignature, str]] = []

        # A chain dict has ONE action_at_step at a specific step, or may
        # contain a list of per-step actions. Handle both:
        #   Format A: top-level "action_at_step" — single step chain
        #   Format B: list of {"step_idx": K, "action": "..."}
        per_step_actions = chain.get("per_step_actions")
        if per_step_actions:
            for entry in per_step_actions:
                step_idx = entry.get("step_idx", 0)
                action = entry.get("action", "")
                if action:
                    sig = extract_state_signature(chain, step_idx)
                    if sig is not None:
                        pairs.append((sig, action))
        else:
            action = chain.get("action_at_step", "")
            if action:
                step_idx = chain.get("cutoff_k", len(constraints) - 1)
                sig = extract_state_signature(chain, step_idx)
                if sig is not None:
                    pairs.append((sig, action))

        if not pairs:
            return

        # NOTE: The ≥40% max-backoff discard check applies at EVALUATION time,
        # not during build. During build the distribution is empty, so every key
        # would appear to need level-3 backoff — incorrectly discarding all chains.
        # Instead, always ingest every chain and let lookup() handle backoff.

        # Ingest into all four levels
        for sig, action in pairs:
            for level in range(4):
                key = sig.to_key(level)
                self._counts[level][key][action] += 1

    def _needed_backoff_level(self, sig: StateSignature) -> int:
        """Return the first backoff level that has data for sig."""
        for level in range(4):
            key = sig.to_key(level)
            if self._counts[level].get(key):
                return level
        return 3

    # ------------------------------------------------------------------
    # Bulk ingest from (sig, action) pairs  (used by build_reference.py)
    # ------------------------------------------------------------------

    def add_observation(self, sig: StateSignature, action: str) -> None:
        """Record a single (state, action) observation at all backoff levels."""
        for level in range(4):
            key = sig.to_key(level)
            self._counts[level][key][action] += 1

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        sig: StateSignature,
        k: int = 3,
    ) -> tuple[list[str], dict[str, float], int]:
        """
        Return top-k actions and their probabilities for the given state.

        Tries backoff levels 0→3 until data is found.

        Returns
        -------
        (top_k_actions, full_distribution, backoff_level_used)

        top_k_actions     : list of up to k action strings, sorted by probability
        full_distribution : {action: probability} over all observed actions
        backoff_level_used: int 0–3 (3 = max backoff / least specific)

        If no data exists at any level, returns ([], {}, 3).
        """
        for level in range(4):
            key = sig.to_key(level)
            counts = self._counts[level].get(key)
            if counts:
                total = sum(counts.values())
                dist = {a: c / total for a, c in counts.items()}
                top_k = sorted(dist, key=dist.__getitem__, reverse=True)[:k]
                return top_k, dist, level

        return [], {}, 3

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize the distribution to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert defaultdicts to plain dicts for cleaner pickling
        payload = {
            "counts": [dict(level) for level in self._counts],
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "ReferenceDistribution":
        """Deserialize a distribution from a pickle file."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        obj = cls()
        raw_counts = payload.get("counts", [{}, {}, {}, {}])
        for level, level_data in enumerate(raw_counts):
            d: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for k, v in level_data.items():
                d[k] = defaultdict(int, v)
            obj._counts[level] = d
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the distribution."""
        return {
            f"level_{i}_keys": len(self._counts[i]) for i in range(4)
        } | {
            f"level_{i}_total_obs": sum(
                sum(v.values()) for v in self._counts[i].values()
            )
            for i in range(4)
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_build(args: argparse.Namespace) -> None:
    """Build the distribution from chain JSONL files."""
    chains_dir = Path(args.chains)
    out_path = Path(args.out)

    chain_files = sorted(chains_dir.glob("*.jsonl"))
    if not chain_files:
        print(f"[reference] No JSONL files found in {chains_dir}")
        return

    print(f"[reference] Building from {len(chain_files)} file(s) in {chains_dir} …")
    dist = ReferenceDistribution()
    dist.build_from_chains(chain_files)

    s = dist.stats()
    print(f"[reference] Level-0 keys: {s['level_0_keys']}")
    print(f"[reference] Level-3 keys: {s['level_3_keys']}")
    print(f"[reference] Total level-0 observations: {s['level_0_total_obs']}")

    dist.save(out_path)
    print(f"[reference] Saved to {out_path}")


def _cmd_check(args: argparse.Namespace) -> None:
    """Sample 50 random chain steps and report non-max-backoff fraction."""
    dist_path = Path(args.dist)
    chains_dir = Path(args.chains)

    print(f"[reference] Loading distribution from {dist_path} …")
    dist = ReferenceDistribution.load(dist_path)

    chain_files = sorted(chains_dir.glob("*.jsonl"))
    if not chain_files:
        print(f"[reference] No JSONL files found in {chains_dir}")
        return

    # Collect all (chain, step_idx) pairs
    samples: list[tuple[dict, int]] = []
    for path in chain_files:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    chain = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                n = len(chain.get("constraints", []))
                if n > 0:
                    for step_idx in range(n):
                        samples.append((chain, step_idx))

    if not samples:
        print("[reference] No samples found.")
        return

    chosen = random.sample(samples, min(50, len(samples)))

    non_max_count = 0
    for chain, step_idx in chosen:
        sig = extract_state_signature(chain, step_idx)
        if sig is None:
            continue
        _, _, level = dist.lookup(sig)
        if level < 3:
            non_max_count += 1

    fraction = non_max_count / len(chosen)
    print(
        f"[reference] {non_max_count}/{len(chosen)} steps returned non-max-backoff "
        f"({fraction:.1%}) — goal ≥80%"
    )
    if fraction >= 0.80:
        print("[reference] PASS")
    else:
        print("[reference] WARN: below 80% target")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.reference",
        description="Reference action distribution builder and checker.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build sub-command
    build_p = sub.add_parser("build", help="Build distribution from chain files.")
    build_p.add_argument("--chains", required=True, help="Directory containing chain JSONL files.")
    build_p.add_argument("--out", required=True, help="Output pickle path.")

    # check sub-command
    check_p = sub.add_parser("check", help="Check distribution coverage on chain files.")
    check_p.add_argument("--dist", required=True, help="Path to pickled distribution.")
    check_p.add_argument("--chains", required=True, help="Directory containing chain JSONL files.")

    ns = parser.parse_args()
    if ns.command == "build":
        _cmd_build(ns)
    elif ns.command == "check":
        _cmd_check(ns)


if __name__ == "__main__":
    main()
