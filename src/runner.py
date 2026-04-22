"""
Evaluation runner for the Pokémon Showdown constraint-chain study.

Drives model evaluations against chain files, saves raw and blinded results,
and handles rate-limit errors with exponential backoff.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.prompt_builder import PROMPT_VERSION, SYSTEM_PROMPT, build_prompt, cutoff_rendered

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}

SEEDS: list[int] = [42, 1337, 7919]

# Exponential-backoff delays (seconds) for rate-limit retries.
_BACKOFF_DELAYS: list[float] = [2.0, 4.0, 8.0, 16.0]  # max 4 retries

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_chain(chain_path: Path) -> dict:
    """Read the first JSONL line from *chain_path* and return it as a dict."""
    with chain_path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
    return json.loads(first_line)


def _count_steps(rendered: str) -> int:
    """Return the number of 'Step N' blocks found in *rendered*."""
    import re

    return len(re.findall(r"Step \d+", rendered))


def _call_api_with_backoff(
    client: anthropic.Anthropic,
    model_id: str,
    user_message: str,
    temperature: float = 0.0,
) -> str:
    """Call the Messages API with exponential backoff on rate-limit errors.

    Retries up to 4 times with delays 2s, 4s, 8s, 16s.

    Args:
        client:       Initialised Anthropic client.
        model_id:     Exact model string to use.
        user_message: The user-turn content.
        temperature:  Sampling temperature. 0.0 for deterministic (pre-reg primary);
                      >0 introduces real stochasticity so different eval seeds produce
                      different responses (multi-seed variance study).

    Returns:
        The text content of the first TextBlock in the response.

    Raises:
        anthropic.RateLimitError: If all retries are exhausted.
        anthropic.APIError:       For any non-retryable API error.
    """
    last_exc: Exception | None = None

    for attempt, delay in enumerate([None] + _BACKOFF_DELAYS):
        if delay is not None:
            time.sleep(delay)

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=50,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            # Extract the first text block.
            for block in response.content:
                if block.type == "text":
                    return block.text.strip()
            return ""

        except anthropic.RateLimitError as exc:
            last_exc = exc
            retry_after = None
            try:
                retry_after = int(exc.response.headers.get("retry-after", ""))
            except (AttributeError, ValueError):
                pass

            remaining = len(_BACKOFF_DELAYS) - attempt
            if remaining <= 0:
                raise
            print(
                f"[runner] Rate-limited (attempt {attempt + 1}). "
                f"retry-after={retry_after}s. "
                f"Backing off {_BACKOFF_DELAYS[attempt]}s …"
            )

        except anthropic.APIStatusError as exc:
            # Retry on 5xx; re-raise on 4xx (except 429 handled above).
            if exc.status_code >= 500:
                last_exc = exc
                remaining = len(_BACKOFF_DELAYS) - attempt
                if remaining <= 0:
                    raise
                print(
                    f"[runner] Server error {exc.status_code} (attempt {attempt + 1}). "
                    f"Backing off {_BACKOFF_DELAYS[attempt]}s …"
                )
            else:
                raise

    # Should only be reached if all retries are exhausted without raising.
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_evaluation(
    chain_path: Path,
    model_name: str,
    seed: int,
    cutoff_k: int | None = None,
    output_dir: Path = Path("results/raw"),
    dry_run: bool = False,
    temperature: float = 0.0,
) -> dict:
    """Run one evaluation on one chain with one model + seed.

    Args:
        chain_path:  Path to a single-chain ``.jsonl`` file.
        model_name:  Key into ``MODELS`` — either ``"haiku"`` or ``"opus"``.
        seed:        Random seed used for this evaluation (metadata only;
                     temperature=0.0 so no actual randomness in the call).
        cutoff_k:    Step at which to cut the chain. ``None`` → auto (half of
                     chain length, rounded down).
        output_dir:  Directory for raw result JSON files.
        dry_run:     If ``True``, print the assembled prompt and return a
                     result dict without calling the API.

    Returns:
        A dict with keys: chain_id, model, seed, cutoff_k, response,
        prompt_version.
    """
    load_dotenv(override=True)

    import os
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "[runner] ERROR: ANTHROPIC_API_KEY is not set.\n"
            "  Add it to .env (see .env.example) or export it before running.\n"
            "  Use --dry-run to test without an API key."
        )
        raise SystemExit(1)

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Choose from: {list(MODELS)}"
        )
    model_id = MODELS[model_name]

    # ------------------------------------------------------------------
    # Load chain
    # ------------------------------------------------------------------
    chain = _load_chain(chain_path)
    chain_id: str = chain["chain_id"]
    rendered: str = chain["rendered"]

    # ------------------------------------------------------------------
    # Determine cutoff step
    # ------------------------------------------------------------------
    total_steps = _count_steps(rendered)
    if cutoff_k is None:
        cutoff_k = max(1, total_steps // 2)

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    truncated = cutoff_rendered(rendered, cutoff_k)
    user_message = build_prompt(truncated, cutoff_k)

    if dry_run:
        print("=" * 72)
        print(f"[dry_run] chain_id={chain_id}  model={model_name}  seed={seed}")
        print(f"[dry_run] cutoff_k={cutoff_k} / {total_steps} steps")
        print("-" * 72)
        print("SYSTEM:\n", SYSTEM_PROMPT)
        print("-" * 72)
        print("USER:\n", user_message)
        print("=" * 72)
        return {
            "chain_id": chain_id,
            "model": model_name,
            "seed": seed,
            "cutoff_k": cutoff_k,
            "response": None,
            "prompt_version": PROMPT_VERSION,
        }

    # ------------------------------------------------------------------
    # Call the API
    # ------------------------------------------------------------------
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response_text = _call_api_with_backoff(client, model_id, user_message, temperature=temperature)

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result = {
        "chain_id": chain_id,
        "model": model_name,
        "seed": seed,
        "cutoff_k": cutoff_k,
        "response": response_text,
        "prompt_version": PROMPT_VERSION,
        "temperature": temperature,
    }

    # ------------------------------------------------------------------
    # Persist raw result (includes all metadata)
    # ------------------------------------------------------------------
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{model_name}_{seed}_{chain_id}.json"
    with raw_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # ------------------------------------------------------------------
    # Persist blinded result (scorer-facing; no model/seed/condition)
    # ------------------------------------------------------------------
    blinded_dir = raw_dir.parent / "blinded"
    blinded_dir.mkdir(parents=True, exist_ok=True)
    blinded = {
        "chain_id": chain_id,
        "cutoff_k": cutoff_k,
        "response": response_text,
        "prompt_version": PROMPT_VERSION,
    }
    blinded_path = blinded_dir / f"{chain_id}_{cutoff_k}.json"
    with blinded_path.open("w", encoding="utf-8") as fh:
        json.dump(blinded, fh, indent=2)

    return result


def _assert_real_data(allow_synthetic: bool = False) -> None:
    """Raise SystemExit if data/raw/SOURCE.txt is missing or marks synthetic data."""
    source_file = Path("data/raw/SOURCE.txt")
    if not source_file.exists():
        if allow_synthetic:
            print("[runner] WARNING: data/raw/SOURCE.txt not found (--allow-synthetic bypasses check).")
            return
        print(
            "[runner] ERROR: data/raw/SOURCE.txt not found.\n"
            "  Create this file with the data source before running evaluations.\n"
            "  Use --allow-synthetic to bypass if you intentionally use synthetic data."
        )
        raise SystemExit(1)

    content = source_file.read_text().strip().lower()
    if "synthetic" in content and not allow_synthetic:
        print(
            f"[runner] ERROR: data/raw/SOURCE.txt indicates synthetic data:\n"
            f"  {source_file.read_text().strip()}\n"
            "  Evaluations must use real data. Pass --allow-synthetic to override."
        )
        raise SystemExit(1)


def run_all(
    chains_dir: Path,
    model_name: str,
    seed: int,
    output_dir: Path = Path("results/raw"),
    dry_run: bool = False,
    n: int | None = None,
    allow_synthetic: bool = False,
    temperature: float = 0.0,
) -> list[dict]:
    """Run evaluations for every chain in *chains_dir* with the given model.

    Chains are processed in deterministic order (sorted by chain_id parsed
    from the filename).  ``random.seed(seed)`` is set before any processing.

    Args:
        chains_dir:  Directory containing ``.jsonl`` chain files.
        model_name:  Key into ``MODELS``.
        seed:        Seed for ``random`` (and metadata).
        output_dir:  Raw result directory.
        dry_run:     Pass through to :func:`run_evaluation`.
        n:           If given, limit to the first *n* chains.

    Returns:
        List of result dicts, one per chain processed.
    """
    if not dry_run:
        _assert_real_data(allow_synthetic=allow_synthetic)

    random.seed(seed)

    chain_files = sorted(chains_dir.glob("*.jsonl"), key=lambda p: p.stem)
    if n is not None:
        chain_files = chain_files[:n]

    results = []
    for cf in chain_files:
        result = run_evaluation(
            chain_path=cf,
            model_name=model_name,
            seed=seed,
            output_dir=output_dir,
            dry_run=dry_run,
            temperature=temperature,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Pokémon Showdown constraint-chain evaluations."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="haiku",
        help="Model to evaluate with (default: haiku).",
    )
    parser.add_argument(
        "--chains",
        type=Path,
        required=True,
        help="Path to a directory containing .jsonl chain files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the API.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to the first N chains (useful for smoke-testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw"),
        help="Directory to write raw results into (default: results/raw).",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow evaluations on synthetic data (skips SOURCE.txt check).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0, deterministic pre-reg primary). "
             "Set >0 so different eval seeds produce different responses "
             "(multi-seed variance study).",
    )

    args = parser.parse_args()

    results = run_all(
        chains_dir=args.chains,
        model_name=args.model,
        seed=args.seed,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        n=args.n,
        allow_synthetic=args.allow_synthetic,
        temperature=args.temperature,
    )

    print(f"\n[runner] Completed {len(results)} evaluations.")
    if results:
        sample = results[0]
        print(
            f"[runner] Sample → chain_id={sample['chain_id']}  "
            f"model={sample['model']}  cutoff_k={sample['cutoff_k']}  "
            f"response={sample['response']!r}"
        )
