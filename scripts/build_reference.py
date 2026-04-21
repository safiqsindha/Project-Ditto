"""
scripts/build_reference.py

Build the reference action distribution from raw match data in data/raw/*.jsonl
OR from chain files in chains/real/*.jsonl (when Session 4 populates them).

Since chains/real/ is currently empty, the default mode processes data/raw/*.jsonl
directly using the Showdown log parser + translation layer to extract
(state_signature, action) pairs from each match.

Usage
-----
    # Primary: build from raw match logs (works now)
    python scripts/build_reference.py --raw data/raw/ --out data/reference_dist.pkl

    # Secondary: build from chain JSONL files (chains/real/ populated by Session 4)
    python scripts/build_reference.py --chains chains/real/ --out data/reference_dist.pkl

    # Check coverage of an existing distribution against chain files
    python scripts/build_reference.py --check --dist data/reference_dist.pkl --chains chains/real/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.normalize import normalize_action
from src.parser import constraint_events, filter_match, parse_showdown_log
from src.reference import (
    ReferenceDistribution,
    StateSignature,
    _hp_bracket,
    extract_state_signature,
)
from src.translation import (
    CoordinationDependency,
    OptimizationCriterion,
    ResourceBudget,
    TranslationContext,
    translate_event,
)


# ---------------------------------------------------------------------------
# Raw-log extraction helpers
# ---------------------------------------------------------------------------

def _extract_observations_from_match(
    match_data: dict,
    perspective: str,
    min_elo: int = 1700,
) -> list[tuple[StateSignature, str]]:
    """
    Extract (StateSignature, action_str) pairs from a raw match dict.

    The state is captured *before* each turn's moves so it reflects the
    information the player had when choosing their action.

    Parameters
    ----------
    match_data  : dict with keys "match_id", "log", "rating", "format", …
    perspective : "p1" or "p2"
    min_elo     : skip if the focal player's Elo is below this threshold

    Returns
    -------
    List of (StateSignature, action_str).
    """
    rating = match_data.get("rating", {})
    focal_elo = rating.get(perspective, 0)
    if focal_elo < min_elo:
        return []

    log = parse_showdown_log(
        match_data["log"], match_id=match_data.get("match_id", "unknown")
    )
    if not filter_match(log, min_elo=min_elo):
        return []

    events_list = constraint_events(log)
    ctx = TranslationContext(perspective=perspective)

    focal_player = perspective
    opp_player = "p2" if perspective == "p1" else "p1"

    # Running state
    hp_state: dict[tuple[str, str], float] = {}       # (player, unit_label) -> hp
    status_state: dict[str, float] = {}               # unit_label -> severity
    weather_obj: str | None = None
    terrain_obj: str | None = None
    hazards: set[str] = set()

    # Group events by turn number
    turn_events: dict[int, list] = defaultdict(list)
    for event in events_list:
        turn_events[event.turn].append(event)

    results: list[tuple[StateSignature, str]] = []

    for turn_num in sorted(turn_events.keys()):
        evts = turn_events[turn_num]

        # ------------------------------------------------------------------
        # Capture state BEFORE this turn's moves
        # ------------------------------------------------------------------
        focal_active = ctx.active_unit.get(focal_player)
        opp_active = ctx.active_unit.get(opp_player)

        if focal_active and opp_active:
            focal_hp = hp_state.get((focal_player, focal_active), 1.0)
            opp_hp = hp_state.get((opp_player, opp_active), 1.0)

            status_flags: set[str] = {
                f"{unit}_status"
                for unit, sev in status_state.items()
                if sev > 0
            }
            field_set: set[str] = set()
            if weather_obj:
                field_set.add(weather_obj)
            if terrain_obj:
                field_set.add(terrain_obj)
            field_set.update(hazards)

            sig: StateSignature | None = StateSignature(
                active_pair=(focal_active, opp_active),
                hp_brackets=(_hp_bracket(focal_hp), _hp_bracket(opp_hp)),
                status_effects=frozenset(status_flags),
                field_conditions=frozenset(field_set),
                turn_bucket=turn_num // 10,
            )
        else:
            sig = None

        # ------------------------------------------------------------------
        # Extract focal player's action for this turn
        # ------------------------------------------------------------------
        focal_action: str | None = None

        for evt in evts:
            if focal_action is not None:
                break

            if evt.type == "move" and evt.player == focal_player:
                unit_label = ctx.active_unit.get(focal_player)
                if unit_label:
                    move_name = evt.args[1].strip() if len(evt.args) > 1 else "unknown"
                    action_label = ctx.get_action_label(focal_player, unit_label, move_name)
                    focal_action = f"use {action_label} with {unit_label}"

            elif evt.type == "switch" and evt.player == focal_player:
                if evt.args:
                    m = re.match(r"^(p[12])[ab]?:\s*(.+)$", evt.args[0])
                    if m:
                        pokemon_name = m.group(2).strip()
                        incoming_label = ctx.get_unit_label(focal_player, pokemon_name)
                        focal_action = f"switch to {incoming_label}"

        if sig is not None and focal_action is not None:
            results.append((sig, normalize_action(focal_action)))

        # ------------------------------------------------------------------
        # Translate all events to update running state for next turn
        # ------------------------------------------------------------------
        for evt in evts:
            try:
                cs = translate_event(evt, ctx)
            except ValueError:
                return results  # too many distinct Pokémon; discard match
            for c in cs:
                if isinstance(c, ResourceBudget):
                    resource = c.resource
                    if resource.startswith("hp_"):
                        unit_label = resource[3:]
                        # Attribute to correct player via unit reverse map
                        attributed = False
                        for pl in ("p1", "p2"):
                            if unit_label in ctx._unit_rev.get(pl, {}):
                                hp_state[(pl, unit_label)] = c.amount
                                attributed = True
                                break
                        if not attributed:
                            for pl in ("p1", "p2"):
                                if ctx.active_unit.get(pl) == unit_label:
                                    hp_state[(pl, unit_label)] = c.amount
                                    break

                    elif resource.startswith("status_"):
                        unit_label = resource[7:]
                        status_state[unit_label] = c.amount

                elif isinstance(c, OptimizationCriterion):
                    obj = c.objective
                    if obj.startswith("weather_"):
                        weather_obj = None if obj == "weather_none" else obj
                    elif obj.startswith("field_"):
                        terrain_obj = obj

                elif isinstance(c, CoordinationDependency):
                    hazards.add(f"{c.dependency}_{c.role}")

    return results


def build_from_raw(
    raw_dir: Path,
    out_path: Path,
    min_elo: int = 1700,
    verbose: bool = True,
) -> ReferenceDistribution:
    """
    Process all JSONL files in raw_dir and build a ReferenceDistribution.

    Each line in each JSONL file is expected to be a JSON object with at least:
        "match_id", "log", "rating", "format"

    Returns the built ReferenceDistribution (also saved to out_path).
    """
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)

    jsonl_files = sorted(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {raw_dir}")

    dist = ReferenceDistribution()
    total_matches = 0
    total_obs = 0
    skipped = 0

    for jsonl_path in jsonl_files:
        if verbose:
            print(f"  Processing {jsonl_path.name} …", end=" ", flush=True)
        file_obs = 0

        with open(jsonl_path, encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    match_data = json.loads(raw_line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                total_matches += 1
                for perspective in ("p1", "p2"):
                    obs = _extract_observations_from_match(
                        match_data, perspective, min_elo=min_elo
                    )
                    for sig, action in obs:
                        dist.add_observation(sig, action)
                        file_obs += 1

        total_obs += file_obs
        if verbose:
            print(f"{file_obs} observations")

    if verbose:
        print(f"\n[build_reference] Processed {total_matches} matches ({skipped} skipped).")
        print(f"[build_reference] Total observations: {total_obs}")
        s = dist.stats()
        print(f"[build_reference] Level-0 unique states: {s['level_0_keys']}")
        print(f"[build_reference] Level-3 unique states: {s['level_3_keys']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dist.save(out_path)
    if verbose:
        print(f"[build_reference] Saved to {out_path}")

    return dist


# ---------------------------------------------------------------------------
# Chain-file build (Session 4 format)
# ---------------------------------------------------------------------------

def build_from_chains(chains_dir: Path, out_path: Path) -> None:
    """Build distribution from chain JSONL files (Session 4 format)."""
    chain_files = sorted(chains_dir.glob("*.jsonl"))
    print(f"Building reference distribution from {len(chain_files)} chain file(s) in {chains_dir} …")

    dist = ReferenceDistribution()
    dist.build_from_chains(chain_files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dist.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Stats: {dist.stats()}")


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def check(dist_path: Path, chains_dir: Path, n_sample: int = 50) -> None:
    """
    Sample 50 random chain steps and report what fraction return
    non-max-backoff results (goal: ≥80%).
    """
    dist = ReferenceDistribution.load(dist_path)
    chain_files = sorted(chains_dir.glob("*.jsonl"))
    if not chain_files:
        print(f"No JSONL files found in {chains_dir}")
        return

    # Collect (chain, step_idx) pairs
    samples: list[tuple[dict, int]] = []
    rng = random.Random(42)
    for cfile in chain_files:
        with open(cfile, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    chain = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                n = len(chain.get("constraints", []))
                for step_idx in range(n):
                    samples.append((chain, step_idx))

    chosen = rng.sample(samples, min(n_sample, len(samples)))
    non_max = 0
    total = 0
    for chain, step_idx in chosen:
        sig = extract_state_signature(chain, step_idx)
        if sig is None:
            continue
        _, _, level = dist.lookup(sig)
        total += 1
        if level < 3:
            non_max += 1

    pct = (non_max / total * 100) if total > 0 else 0.0
    print(f"Non-max-backoff rate: {non_max}/{total} = {pct:.1f}% (target ≥80%)")
    if pct >= 80:
        print("PASS")
    else:
        print("WARNING: below 80% target — reference distribution may be too sparse")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or check the reference action distribution."
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- build from raw logs (primary path while chains/real/ is empty) ----
    raw_p = sub.add_parser("build-raw", help="Build from data/raw/*.jsonl match files.")
    raw_p.add_argument(
        "--raw", type=Path, default=_PROJECT_ROOT / "data" / "raw",
        help="Directory of raw JSONL match files (default: data/raw).",
    )
    raw_p.add_argument(
        "--out", type=Path, default=_PROJECT_ROOT / "data" / "reference_dist.pkl",
        help="Output pickle path (default: data/reference_dist.pkl).",
    )
    raw_p.add_argument("--min-elo", type=int, default=1700)

    # -- build from chain files (Session 4+) --------------------------------
    chain_p = sub.add_parser("build", help="Build from chains/real/*.jsonl chain files.")
    chain_p.add_argument(
        "--chains", type=Path, default=_PROJECT_ROOT / "chains" / "real",
        help="Directory of chain JSONL files (default: chains/real).",
    )
    chain_p.add_argument(
        "--out", type=Path, default=_PROJECT_ROOT / "data" / "reference_dist.pkl",
        help="Output pickle path (default: data/reference_dist.pkl).",
    )

    # -- check coverage -----------------------------------------------------
    check_p = sub.add_parser("check", help="Check distribution coverage.")
    check_p.add_argument(
        "--dist", type=Path, default=_PROJECT_ROOT / "data" / "reference_dist.pkl",
    )
    check_p.add_argument(
        "--chains", type=Path, default=_PROJECT_ROOT / "chains" / "real",
    )
    check_p.add_argument("--n", type=int, default=50)

    args = parser.parse_args()

    if args.cmd == "build-raw":
        build_from_raw(args.raw, args.out, min_elo=args.min_elo, verbose=True)
    elif args.cmd == "build":
        build_from_chains(args.chains, args.out)
    elif args.cmd == "check":
        check(args.dist, args.chains, args.n)
    else:
        # Default: build from raw (most useful right now)
        print("[build_reference] No subcommand given — defaulting to build-raw.")
        raw_dir = _PROJECT_ROOT / "data" / "raw"
        out_path = _PROJECT_ROOT / "data" / "reference_dist.pkl"
        build_from_raw(raw_dir, out_path, verbose=True)


if __name__ == "__main__":
    main()
