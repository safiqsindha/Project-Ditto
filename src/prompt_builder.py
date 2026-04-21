"""
Prompt builder for the Pokémon Showdown constraint-chain evaluation.

FIXED versioned template — must not change after Session 5 implementation.
"""

PROMPT_VERSION = "v1.0"

SYSTEM_PROMPT = """You are reasoning about an adaptive pipeline that operates under
a sequence of changing constraints. At each step, new constraints
may appear, existing constraints may change, and resources may
be depleted. Your job is to propose the correct adaptation at
each step, given the full prior history.

At each step you receive a partially-observable state: you see
what happened to YOUR units but only limited information about
opposing units until they are revealed through action.

Constraints carry forward unless explicitly superseded. Treat
"tool UNAVAILABLE" as persistent unless a later event makes it
available again."""


def cutoff_rendered(rendered: str, k: int) -> str:
    """Return only the first k steps of a rendered chain string.

    Steps in the rendered format start with "Step N" (where N is a positive
    integer).  The function splits on those boundaries and returns the
    re-joined prefix for the first *k* steps.

    Args:
        rendered: Full rendered chain string.
        k:        Number of steps to keep (1-indexed).

    Returns:
        The rendered text containing only steps 1 through k.
        If k <= 0 or the string contains no recognisable step headers, an
        empty string is returned.
    """
    import re

    if k <= 0:
        return ""

    # Split on every "Step N" boundary (keeping the delimiter via a
    # capturing group so we can reconstruct the text faithfully).
    parts = re.split(r"(Step \d+)", rendered)

    # parts has the form:
    #   [preamble, "Step 1", body1, "Step 2", body2, ...]
    # where preamble may be an empty string.

    # Collect (header, body) pairs.
    steps: list[tuple[str, str]] = []
    i = 0
    # Skip any leading non-step content.
    while i < len(parts) and not re.fullmatch(r"Step \d+", parts[i]):
        i += 1

    while i + 1 < len(parts):
        header = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if re.fullmatch(r"Step \d+", header):
            steps.append((header, body))
            i += 2
        else:
            i += 1

    if not steps:
        return ""

    selected = steps[:k]
    return "".join(header + body for header, body in selected)


def build_prompt(rendered_steps: str, cutoff_k: int) -> str:
    """Build the user-facing prompt given a rendered chain up to step K.

    The system prompt is kept separate; this function returns only the
    *user* message.

    Args:
        rendered_steps: Rendered chain text already truncated to step K
                        (i.e. the output of ``cutoff_rendered(rendered, k)``
                        or the equivalent pre-truncated string).
        cutoff_k:       The step number K at which the chain is cut off.
                        Used to phrase the question correctly.

    Returns:
        The user message string.
    """
    return (
        f"{rendered_steps.rstrip()}\n"
        "\n"
        "---\n"
        "\n"
        f"Given the state above, propose the next action for your side at\n"
        f"step {cutoff_k + 1}. Output only the action label (e.g., "
        '"use action_2 with\n'
        'unit_A" or "switch to unit_C"). No explanation.'
    )
