"""
Session 4: End-to-end chain construction pipeline.

Reads raw match JSONL files from data/raw/, applies translation T,
asymmetric observability, validity filter, and shuffler to produce
evaluation-ready chain files.

Usage:
    python scripts/build_chains.py \
        --data data/raw/ \
        --out-real chains/real/ \
        --out-shuffled chains/shuffled/ \
        --seeds 42 1337 7919
"""

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filter import is_valid_chain
from src.observability import apply_asymmetric_observability_with_indices
from src.parser import constraint_events, filter_match, parse_showdown_log
from src.renderer import render_chain
from src.shuffler import shuffle_chain
from src.translation import TranslationContext, translate_event


def _serialize_constraint(c) -> dict:
    d = dataclasses.asdict(c)
    d["type"] = type(c).__name__
    return d



def _find_valid_window(constraints, min_len: int = 20, max_len: int = 40) -> tuple[list, int] | None:
    """
    Find the earliest contiguous window of length 20-40 that passes is_valid_chain.
    Returns (window, start_idx) or None.
    """
    n = len(constraints)
    for win_len in range(max_len, min_len - 1, -1):
        starts = [n // 4, n // 3, 10, 0, n // 2 - win_len // 2]
        for start in sorted(set(max(0, s) for s in starts)):
            end = start + win_len
            if end > n:
                continue
            window = constraints[start:end]
            if is_valid_chain(window):
                return window, start
    return None


def process_match(record: dict, perspectives: list[str], seeds: list[int]) -> tuple[list[dict], list[dict]]:
    """Convert one raw match record into real + shuffled chains."""
    real_chains: list[dict] = []
    shuffled_chains: list[dict] = []

    log = parse_showdown_log(record["log"], record["match_id"])
    if not filter_match(log):
        return [], []

    events = constraint_events(log)

    for perspective in perspectives:
        # Manual translate loop so we can record (p1_active, p2_active) after each
        # emitted constraint — needed for signature extraction (bug fix: opponent
        # units otherwise unrecoverable from chain tool names).
        ctx = TranslationContext(perspective=perspective)
        all_constraints = []
        active_pair_all: list[tuple[str | None, str | None]] = []
        try:
            for event in events:
                for c in translate_event(event, ctx):
                    all_constraints.append(c)
                    active_pair_all.append(
                        (ctx.active_unit.get("p1"), ctx.active_unit.get("p2"))
                    )
        except ValueError:
            continue

        # Asymmetric observability on full match — keep index alignment
        obs_all, kept_idx = apply_asymmetric_observability_with_indices(
            all_constraints, perspective
        )
        active_pair_obs = [active_pair_all[i] for i in kept_idx]

        # Extract a valid 20-40 constraint window
        found = _find_valid_window(obs_all)
        if found is None:
            continue
        window, start_idx = found
        window_active_pairs = active_pair_obs[start_idx:start_idx + len(window)]

        # Compute cutoff K (half-chain)
        k = len(window) // 2

        chain_id = f"{record['match_id']}_{perspective}"
        chain = {
            "chain_id": chain_id,
            "match_id": record["match_id"],
            "perspective": perspective,
            "constraints": window,
            "rendered": render_chain(window, perspective),
            "cutoff_k": k,
            "active_pair_by_step": window_active_pairs,
        }
        real_chains.append(chain)

        # Shuffled variants
        for seed in seeds:
            sh = shuffle_chain(chain, seed)
            sh["rendered"] = render_chain(sh["constraints"], perspective)
            shuffled_chains.append(sh)

    return real_chains, shuffled_chains


def save_chain(chain: dict, out_dir: Path):
    """Serialize one chain to a JSONL file (one chain per file)."""
    chain_id = chain["chain_id"]
    out_path = out_dir / f"{chain_id}.jsonl"
    serialized = dict(chain)
    serialized["constraints"] = [_serialize_constraint(c) for c in chain["constraints"]]
    with open(out_path, "w") as f:
        f.write(json.dumps(serialized) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-real", type=Path, default=Path("chains/real"))
    parser.add_argument("--out-shuffled", type=Path, default=Path("chains/shuffled"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7919])
    parser.add_argument("--perspectives", type=str, nargs="+", default=["p1", "p2"])
    args = parser.parse_args()

    args.out_real.mkdir(parents=True, exist_ok=True)
    args.out_shuffled.mkdir(parents=True, exist_ok=True)

    total_matches = 0
    total_real = 0
    total_shuffled = 0
    length_hist: Counter = Counter()
    type_freq: Counter = Counter()
    archetype_count: Counter = Counter()

    for data_file in sorted(args.data.glob("*.jsonl")):
        print(f"Processing {data_file.name} ...")
        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_matches += 1
                real_chains, shuffled_chains = process_match(record, args.perspectives, args.seeds)

                for chain in real_chains:
                    save_chain(chain, args.out_real)
                    total_real += 1
                    n = len(chain["constraints"])
                    if n <= 25:
                        length_hist["20-25"] += 1
                    elif n <= 32:
                        length_hist["26-32"] += 1
                    else:
                        length_hist["33-40"] += 1
                    for c in chain["constraints"]:
                        type_freq[type(c).__name__] += 1

                for chain in shuffled_chains:
                    save_chain(chain, args.out_shuffled)
                    total_shuffled += 1

    print(f"\n=== Build complete ===")
    print(f"Matches processed:  {total_matches}")
    print(f"Real chains:        {total_real}")
    print(f"Shuffled chains:    {total_shuffled}")
    print(f"Length histogram:   {dict(length_hist)}")
    print(f"Constraint types:   {dict(type_freq.most_common(10))}")

    if total_real < 200:
        print(f"\nWARNING: only {total_real} real chains — target is ≥200.")
        print("Consider lowering filter thresholds or generating more synthetic data.")


if __name__ == "__main__":
    main()
