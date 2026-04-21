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
from src.observability import apply_asymmetric_observability
from src.parser import constraint_events, filter_match, parse_showdown_log
from src.renderer import render_chain
from src.shuffler import shuffle_chain
from src.translation import translate_match


def _serialize_constraint(c) -> dict:
    d = dataclasses.asdict(c)
    d["type"] = type(c).__name__
    return d



def _find_valid_window(constraints, min_len: int = 20, max_len: int = 40) -> list | None:
    """
    Find the earliest contiguous window of length 20-40 that passes is_valid_chain.
    Tries windows at step=10, 20, 30 ... to spread across the match.
    Returns the window (list of constraints) or None if none found.
    """
    n = len(constraints)
    # Try different window lengths starting from max down to min
    for win_len in range(max_len, min_len - 1, -1):
        # Try starting positions: quarter-point, third-point, half-point, early
        starts = [n // 4, n // 3, 10, 0, n // 2 - win_len // 2]
        for start in sorted(set(max(0, s) for s in starts)):
            end = start + win_len
            if end > n:
                continue
            window = constraints[start:end]
            if is_valid_chain(window):
                return window
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
        # Translate full match — skip if Pokémon count exceeds 6-per-player limit
        try:
            all_constraints = translate_match(events, perspective)
        except ValueError:
            continue

        # Asymmetric observability on full match
        obs_all = apply_asymmetric_observability(all_constraints, perspective)

        # Extract a valid 20-40 constraint window
        window = _find_valid_window(obs_all)
        if window is None:
            continue

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
