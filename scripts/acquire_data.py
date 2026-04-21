"""
Session 1: Data acquisition script.

Run this script once to download and filter raw battle logs.

Usage:
    python scripts/acquire_data.py --source huggingface --out data/raw --target 2500
    python scripts/acquire_data.py --source pokechamp --dir /path/to/pokechamp/data --out data/raw

After running, data/raw/ contains one .jsonl file per 1000 matches, filtered to
Gen 9 OU singles with both players Elo >= 1700 and >= 15 turns.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import parse_showdown_log, parse_huggingface_replay, filter_match


def acquire_from_huggingface(out_dir: Path, target: int = 2500, hf_token: str | None = None):
    """Download from HolidayOugi/pokemon-showdown-replays on HuggingFace."""
    from datasets import load_dataset

    print(f"Loading HolidayOugi/pokemon-showdown-replays (streaming)...")
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token

    ds = load_dataset(
        "HolidayOugi/pokemon-showdown-replays",
        split="train",
        streaming=True,
        **kwargs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen = 0
    chunk = 0
    batch: list[dict] = []

    for record in ds:
        seen += 1
        log = parse_huggingface_replay(record)
        if log is None:
            continue
        if not filter_match(log):
            continue

        batch.append({
            "match_id": log.match_id,
            "format": log.format,
            "players": log.players,
            "rating": log.rating,
            "winner": log.winner,
            "log": record.get("log", ""),
        })
        kept += 1

        if len(batch) >= 1000:
            chunk += 1
            _flush_batch(batch, out_dir, chunk)
            batch = []
            print(f"  Kept {kept} / scanned {seen}")

        if kept >= target:
            break

    if batch:
        chunk += 1
        _flush_batch(batch, out_dir, chunk)

    print(f"Done. Kept {kept} matches from {seen} scanned.")
    return kept


def acquire_from_pokechamp(pokechamp_dir: Path, out_dir: Path, target: int = 2500):
    """Load from a local PokéChamp dataset directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen = 0
    chunk = 0
    batch: list[dict] = []

    for log_file in sorted(pokechamp_dir.rglob("*.log")):
        seen += 1
        try:
            log = parse_showdown_log(log_file.read_text(encoding="utf-8", errors="replace"), log_file.stem)
        except Exception:
            continue
        if not filter_match(log):
            continue

        batch.append({
            "match_id": log.match_id,
            "format": log.format,
            "players": log.players,
            "rating": log.rating,
            "winner": log.winner,
            "log": log_file.read_text(encoding="utf-8", errors="replace"),
        })
        kept += 1

        if len(batch) >= 1000:
            chunk += 1
            _flush_batch(batch, out_dir, chunk)
            batch = []
            print(f"  Kept {kept} / scanned {seen}")

        if kept >= target:
            break

    if batch:
        chunk += 1
        _flush_batch(batch, out_dir, chunk)

    print(f"Done. Kept {kept} matches from {seen} scanned.")
    return kept


def _flush_batch(batch: list[dict], out_dir: Path, chunk_idx: int):
    path = out_dir / f"matches_{chunk_idx:04d}.jsonl"
    with open(path, "w") as f:
        for record in batch:
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(batch)} records to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["huggingface", "pokechamp"], default="huggingface")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--target", type=int, default=2500, help="Target number of filtered matches")
    parser.add_argument("--pokechamp-dir", type=Path, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    if args.source == "huggingface":
        n = acquire_from_huggingface(args.out, args.target, args.hf_token)
    else:
        if not args.pokechamp_dir:
            print("--pokechamp-dir required for pokechamp source")
            sys.exit(1)
        n = acquire_from_pokechamp(args.pokechamp_dir, args.out, args.target)

    if n < 100:
        print("WARNING: fewer than 100 matches acquired. Pipeline may not produce enough chains.")
        sys.exit(1)


if __name__ == "__main__":
    main()
