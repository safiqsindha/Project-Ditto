"""
Action string normalisation for consistent scorer matching.

Applied at both reference-build time (to focal_action) and at eval time
(to model responses) so that trivial formatting differences don't count
as mismatches.
"""

from __future__ import annotations

import re


def normalize_action(s: str) -> str:
    """
    Normalise an action string to a canonical lower-case form.

    Rules applied in order
    ----------------------
    1. Strip leading/trailing whitespace.
    2. Lowercase.
    3. Remove punctuation characters other than underscores and spaces.
    4. Collapse multiple consecutive spaces to one.
    5. Strip again.

    Examples
    --------
    "Use action_2 with unit_A"  → "use action_2 with unit_a"
    "switch to unit_C."         → "switch to unit_c"
    "  USE ACTION_1 WITH UNIT_B " → "use action_1 with unit_b"
    """
    s = s.strip().lower()
    # Normalise all whitespace variants (tabs, newlines) to plain spaces first
    s = re.sub(r"\s+", " ", s)
    # Remove all characters except alphanumerics, underscores, and spaces
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    # Collapse any double-spaces produced by removal above
    s = re.sub(r" +", " ", s).strip()
    return s
