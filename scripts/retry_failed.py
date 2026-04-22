"""
Retry batch evaluations whose raw result files have an ``error`` field or empty response.

Usage:
    python scripts/retry_failed.py --model sonnet --seed 42 \
        --chains chains/shuffled/ --results results/raw/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from dotenv import load_dotenv

from src.batch_runner import (
    MODELS,
    _build_request,
    _custom_id,
    _parse_custom_id,
    _poll_until_done,
    _write_result,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--chains", type=Path, required=True,
                   help="Directory of chain JSONL files to pull from.")
    p.add_argument("--results", type=Path, default=Path("results/raw"))
    p.add_argument("--poll-interval", type=float, default=30.0)
    args = p.parse_args()

    load_dotenv(override=True)

    # Find all failed raw results for (model, seed)
    failed_ids: list[str] = []
    for rf in sorted(args.results.glob(f"{args.model}_{args.seed}_*.json")):
        r = json.loads(rf.read_text())
        if r.get("error") or not r.get("response"):
            failed_ids.append(r["chain_id"])

    if not failed_ids:
        print(f"[retry] no failed results for model={args.model} seed={args.seed}")
        return

    print(f"[retry] found {len(failed_ids)} failed chains to re-run")

    # Build requests from the chain files
    chain_index: dict[str, Path] = {cf.stem: cf for cf in args.chains.glob("*.jsonl")}
    requests = []
    meta_by_cid: dict[str, dict] = {}
    missing: list[str] = []
    for cid in failed_ids:
        cf = chain_index.get(cid)
        if cf is None:
            missing.append(cid)
            continue
        req, meta = _build_request(cf, args.model, args.seed)
        requests.append(req)
        meta_by_cid[req["custom_id"]] = meta

    if missing:
        print(f"[retry] WARNING: {len(missing)} chain files not found under {args.chains}")
        print(f"        e.g. {missing[:3]}")

    print(f"[retry] submitting {len(requests)} requests ...")
    client = anthropic.Anthropic()
    created = client.messages.batches.create(requests=requests)
    batch_id = created.id
    print(f"[retry] batch_id={batch_id}  status={created.processing_status}")

    _poll_until_done(client, batch_id, args.poll_interval)

    outcome = {"succeeded": 0, "errored": 0, "other": 0}
    for r in client.messages.batches.results(batch_id):
        cid = r.custom_id
        mn, sd, chain_id = _parse_custom_id(cid)
        cutoff_k = meta_by_cid[cid]["cutoff_k"]

        if r.result.type == "succeeded":
            text = ""
            for block in r.result.message.content:
                if block.type == "text":
                    text = block.text.strip()
                    break
            _write_result(mn, sd, chain_id, cutoff_k, text, args.results)
            outcome["succeeded"] += 1
        elif r.result.type == "errored":
            err = getattr(r.result, "error", None)
            err_str = json.dumps(err.model_dump() if err else {"type": "unknown"})
            _write_result(mn, sd, chain_id, cutoff_k, "", args.results, error=err_str)
            outcome["errored"] += 1
        else:
            _write_result(mn, sd, chain_id, cutoff_k, "",
                          args.results, error=str(r.result.type))
            outcome["other"] += 1

    print(
        f"[retry] done. succeeded={outcome['succeeded']} "
        f"errored={outcome['errored']} other={outcome['other']}"
    )


if __name__ == "__main__":
    main()
