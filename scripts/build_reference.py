"""
Session 2: Build reference action distribution.

Processes all real chains in chains/real/ to build a state-signature →
action distribution lookup table.  Serialized to data/reference_dist.pkl.

Usage:
    python scripts/build_reference.py --chains chains/real/ --out data/reference_dist.pkl
    python scripts/build_reference.py --check --dist data/reference_dist.pkl --chains chains/real/
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reference import ReferenceDistribution, extract_state_signature


def build(chains_dir: Path, out_path: Path) -> None:
    chain_files = sorted(chains_dir.glob("*.jsonl"))
    print(f"Building reference distribution from {len(chain_files)} chains ...")

    dist = ReferenceDistribution()
    dist.build_from_chains(chain_files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dist.save(out_path)
    print(f"Saved to {out_path}")
    print(f"Stats: {dist.stats()}")


def check(dist_path: Path, chains_dir: Path, n_sample: int = 50) -> None:
    dist = ReferenceDistribution.load(dist_path)
    chain_files = sorted(chains_dir.glob("*.jsonl"))
    rng = random.Random(42)
    sample_files = rng.sample(chain_files, min(n_sample, len(chain_files)))

    non_max_backoff = 0
    total = 0

    for cfile in sample_files:
        with open(cfile) as f:
            chain = json.loads(f.readline())
        cutoff_k = chain.get("cutoff_k", len(chain["constraints"]) // 2)
        sig = extract_state_signature(chain, cutoff_k)
        if sig is None:
            continue
        top_k, dist_result, backoff_level = dist.lookup(sig)
        total += 1
        if backoff_level < 3:
            non_max_backoff += 1

    pct = (non_max_backoff / total * 100) if total > 0 else 0
    print(f"Non-max-backoff rate: {non_max_backoff}/{total} = {pct:.1f}% (target ≥80%)")
    if pct >= 80:
        print("PASS")
    else:
        print("WARNING: below 80% target — reference distribution may be too sparse")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    build_p = sub.add_parser("build")
    build_p.add_argument("--chains", type=Path, default=Path("chains/real"))
    build_p.add_argument("--out", type=Path, default=Path("data/reference_dist.pkl"))

    check_p = sub.add_parser("check")
    check_p.add_argument("--dist", type=Path, default=Path("data/reference_dist.pkl"))
    check_p.add_argument("--chains", type=Path, default=Path("chains/real"))
    check_p.add_argument("--n", type=int, default=50)

    # Default (no subcommand): build
    args = parser.parse_args()
    if args.cmd == "build" or args.cmd is None:
        chains = getattr(args, "chains", Path("chains/real"))
        out = getattr(args, "out", Path("data/reference_dist.pkl"))
        build(chains, out)
    elif args.cmd == "check":
        check(args.dist, args.chains, args.n)


if __name__ == "__main__":
    main()
