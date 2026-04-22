"""
Batch-mode evaluation runner — same contract as ``src.runner`` but submits
all chain evaluations as a single Message Batches API request (50% discount,
async up to 24h, typically minutes for small batches).

CLI mirrors ``src.runner`` so pilot → full is a drop-in swap:

    python -m src.batch_runner --model haiku --chains chains/real/ --seed 42

Output layout matches the sync runner exactly:
    results/raw/{model}_{seed}_{chain_id}.json
    results/blinded/{chain_id}_{cutoff_k}.json

Both are consumed unchanged by ``src.scorer``.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from dotenv import load_dotenv

from src.prompt_builder import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, cutoff_rendered
from src.runner import MODELS, _assert_real_data, _load_chain


def _count_steps(rendered: str) -> int:
    return len(re.findall(r"Step \d+", rendered))


def _custom_id(model_name: str, seed: int, chain_id: str) -> str:
    """Custom ID round-trips model/seed/chain_id through the batch.

    Anthropic allows [a-zA-Z0-9_-]{1,64}. Chain IDs like
    ``gen9ou-2257406016_p1_shuffled_42`` already fit; we just prefix.
    """
    return f"{model_name}__s{seed}__{chain_id}"


def _parse_custom_id(cid: str) -> tuple[str, int, str]:
    model_name, seed_part, chain_id = cid.split("__", 2)
    return model_name, int(seed_part[1:]), chain_id


def _build_request(
    chain_path: Path,
    model_name: str,
    seed: int,
    temperature: float = 0.0,
) -> tuple[Request, dict]:
    """Assemble one batch Request + the metadata needed to write results."""
    chain = _load_chain(chain_path)
    chain_id: str = chain["chain_id"]
    rendered: str = chain["rendered"]

    total_steps = _count_steps(rendered)
    cutoff_k = max(1, total_steps // 2)

    user_message = build_prompt(cutoff_rendered(rendered, cutoff_k), cutoff_k)
    model_id = MODELS[model_name]

    req = Request(
        custom_id=_custom_id(model_name, seed, chain_id),
        params=MessageCreateParamsNonStreaming(
            model=model_id,
            max_tokens=50,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ),
    )
    meta = {"chain_id": chain_id, "cutoff_k": cutoff_k}
    return req, meta


def _write_result(
    model_name: str,
    seed: int,
    chain_id: str,
    cutoff_k: int,
    response_text: str,
    output_dir: Path,
    error: str | None = None,
    temperature: float = 0.0,
) -> None:
    raw = {
        "chain_id": chain_id,
        "model": model_name,
        "seed": seed,
        "cutoff_k": cutoff_k,
        "response": response_text,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
    }
    if error is not None:
        raw["error"] = error

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_name}_{seed}_{chain_id}.json").write_text(
        json.dumps(raw, indent=2)
    )

    blinded_dir = output_dir.parent / "blinded"
    blinded_dir.mkdir(parents=True, exist_ok=True)
    (blinded_dir / f"{chain_id}_{cutoff_k}.json").write_text(
        json.dumps(
            {
                "chain_id": chain_id,
                "cutoff_k": cutoff_k,
                "response": response_text,
                "prompt_version": PROMPT_VERSION,
            },
            indent=2,
        )
    )


def _poll_until_done(
    client: anthropic.Anthropic,
    batch_id: str,
    poll_interval: float,
) -> anthropic.types.messages.MessageBatch:
    while True:
        b = client.messages.batches.retrieve(batch_id)
        counts = b.request_counts
        total = (
            counts.processing + counts.succeeded + counts.errored
            + counts.canceled + counts.expired
        )
        print(
            f"[batch] {b.processing_status}  "
            f"done={counts.succeeded + counts.errored + counts.canceled + counts.expired}/{total}  "
            f"(ok={counts.succeeded}  err={counts.errored}  "
            f"cancel={counts.canceled}  exp={counts.expired})"
        )
        if b.processing_status == "ended":
            return b
        time.sleep(poll_interval)


def run_batch(
    chains_dir: Path,
    model_name: str,
    seed: int,
    output_dir: Path = Path("results/raw"),
    n: int | None = None,
    allow_synthetic: bool = False,
    poll_interval: float = 30.0,
    batch_id: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Run one batch of evaluations. Returns per-outcome counts.

    If ``batch_id`` is given, skips submission and resumes by downloading
    results from that existing batch — useful if the network drops mid-poll.
    """
    load_dotenv(override=True)
    _assert_real_data(allow_synthetic=allow_synthetic)

    random.seed(seed)
    client = anthropic.Anthropic()

    chain_files = sorted(chains_dir.glob("*.jsonl"), key=lambda p: p.stem)
    if n is not None:
        chain_files = chain_files[:n]
    if not chain_files:
        print(f"[batch] no chain files found under {chains_dir}")
        return {"submitted": 0, "succeeded": 0, "errored": 0}

    # Metadata needed to turn results back into files
    meta_by_cid: dict[str, dict] = {}

    if batch_id is None:
        requests: list[Request] = []
        for cf in chain_files:
            req, meta = _build_request(cf, model_name, seed, temperature=temperature)
            requests.append(req)
            meta_by_cid[req["custom_id"]] = meta

        print(f"[batch] submitting {len(requests)} requests  model={model_name}  seed={seed}  temp={temperature} ...")
        created = client.messages.batches.create(requests=requests)
        batch_id = created.id
        print(f"[batch] batch_id={batch_id}  status={created.processing_status}")
        print(f"[batch] save this batch_id in case of interruption: --batch-id {batch_id}")
    else:
        # Resume: reconstruct metadata from current chain files
        for cf in chain_files:
            chain = _load_chain(cf)
            chain_id = chain["chain_id"]
            total_steps = _count_steps(chain["rendered"])
            cutoff_k = max(1, total_steps // 2)
            cid = _custom_id(model_name, seed, chain_id)
            meta_by_cid[cid] = {"chain_id": chain_id, "cutoff_k": cutoff_k}
        print(f"[batch] resuming from batch_id={batch_id} (metadata rebuilt from disk)")

    _poll_until_done(client, batch_id, poll_interval)

    # Stream results back and write one file per response.
    outcome = {"submitted": len(meta_by_cid), "succeeded": 0, "errored": 0, "other": 0}
    for r in client.messages.batches.results(batch_id):
        cid = r.custom_id
        meta = meta_by_cid.get(cid)
        if meta is None:
            print(f"[batch] WARNING: unknown custom_id {cid!r}, skipping")
            continue
        mn, sd, chain_id = _parse_custom_id(cid)

        if r.result.type == "succeeded":
            msg = r.result.message
            response_text = ""
            for block in msg.content:
                if block.type == "text":
                    response_text = block.text.strip()
                    break
            _write_result(mn, sd, chain_id, meta["cutoff_k"], response_text, output_dir, temperature=temperature)
            outcome["succeeded"] += 1
        elif r.result.type == "errored":
            err = getattr(r.result, "error", None)
            err_str = json.dumps(err.model_dump() if err else {"type": "unknown"})
            _write_result(mn, sd, chain_id, meta["cutoff_k"], "", output_dir, error=err_str, temperature=temperature)
            outcome["errored"] += 1
        else:
            _write_result(
                mn, sd, chain_id, meta["cutoff_k"], "",
                output_dir, error=f"{r.result.type}", temperature=temperature
            )
            outcome["other"] += 1

    print(
        f"[batch] done. submitted={outcome['submitted']} "
        f"succeeded={outcome['succeeded']} errored={outcome['errored']} other={outcome['other']}"
    )
    return outcome


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch evaluation runner (50% cheaper than sync).")
    parser.add_argument("--model", choices=list(MODELS), default="haiku")
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--n", type=int, default=None, help="Limit to the first N chains.")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--batch-id", type=str, default=None,
                        help="Resume from an existing batch_id instead of submitting.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0). Set >0 so different "
                             "eval seeds produce different responses (multi-seed variance).")
    args = parser.parse_args()

    run_batch(
        chains_dir=args.chains,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        n=args.n,
        allow_synthetic=args.allow_synthetic,
        poll_interval=args.poll_interval,
        batch_id=args.batch_id,
        temperature=args.temperature,
    )
