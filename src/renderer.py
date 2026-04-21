"""
Renderer: converts abstract Constraint objects into human-readable, domain-abstracted English.

NO Pokémon vocabulary is permitted in any output string.
"""

from __future__ import annotations

import re

from src.translation import (
    Constraint,
    CoordinationDependency,
    InformationState,
    OptimizationCriterion,
    ResourceBudget,
    SubGoalTransition,
    ToolAvailability,
)


# ---------------------------------------------------------------------------
# Per-type renderers
# ---------------------------------------------------------------------------

def _render_resource_budget(c: ResourceBudget) -> str:
    pct = f"{c.amount * 100:.1f}%"
    parts = [f"ResourceBudget: {c.resource} at {pct}"]
    if c.decay != "none":
        parts.append(f"(decay={c.decay})")
    if c.recover_in is not None:
        parts.append(f"(recover_in={c.recover_in} turns)")
    return " ".join(parts)


def _render_tool_availability(c: ToolAvailability) -> str:
    state_str = c.state.upper()
    if c.state == "unavailable" and c.recover_in is None:
        return f"ToolAvailability: {c.tool} is now {state_str} (permanent)"
    elif c.state == "unavailable":
        return f"ToolAvailability: {c.tool} is now {state_str} (recover_in={c.recover_in})"
    else:
        return f"ToolAvailability: {c.tool} is now {state_str}"


def _render_subgoal_transition(c: SubGoalTransition) -> str:
    return (
        f"SubGoalTransition: phase shifted from '{c.from_phase}' to '{c.to_phase}'"
        f" (trigger={c.trigger})"
    )


def _render_information_state(c: InformationState) -> str:
    added = ", ".join(c.observable_added) if c.observable_added else "none"
    removed = ", ".join(c.observable_removed) if c.observable_removed else "none"
    return (
        f"InformationState: added=[{added}] removed=[{removed}]"
        f" uncertainty={c.uncertainty:.2f}"
    )


def _render_coordination_dependency(c: CoordinationDependency) -> str:
    return (
        f"CoordinationDependency: {c.role} depends on {c.dependency}"
        f" -> expected_action={c.expected_action}"
    )


def _render_optimization_criterion(c: OptimizationCriterion) -> str:
    return (
        f"OptimizationCriterion: objective={c.objective}"
        f" weight_shift={c.weight_shift}"
    )


_RENDERERS = {
    ResourceBudget:          _render_resource_budget,
    ToolAvailability:        _render_tool_availability,
    SubGoalTransition:       _render_subgoal_transition,
    InformationState:        _render_information_state,
    CoordinationDependency:  _render_coordination_dependency,
    OptimizationCriterion:   _render_optimization_criterion,
}


def render_constraint(c: Constraint) -> str:
    """Render a single constraint as a one-line abstract English description."""
    renderer = _RENDERERS.get(type(c))
    if renderer is None:
        return f"UnknownConstraint: {c!r}"
    return renderer(c)


# ---------------------------------------------------------------------------
# Chain renderer
# ---------------------------------------------------------------------------

def render_chain(constraints: list[Constraint], perspective: str) -> str:
    """
    Render a list of constraints as numbered steps.

    Parameters
    ----------
    constraints : ordered list of Constraint objects
    perspective : "p1" or "p2" (included in header for traceability)

    Returns
    -------
    Multi-line string, one step per constraint.
    """
    if not constraints:
        return f"# Constraint chain (perspective={perspective})\n(no constraints)\n"

    lines: list[str] = [f"# Constraint chain (perspective={perspective})"]
    step = 1
    for c in constraints:
        turn = getattr(c, "timestamp", 0)
        body = render_constraint(c)
        lines.append(f"Step {step} (turn={turn}):")
        lines.append(f"  {body}")
        step += 1

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Leakage checker
# ---------------------------------------------------------------------------

def check_pokemon_leakage(rendered: str, pokemon_names: set[str]) -> list[str]:
    """
    Return any Pokémon names found in the rendered output.

    A match is case-insensitive whole-word search so that partial
    substrings (e.g. "sand" inside "sandstorm") are not flagged as
    Pokémon names unless the full name matches.

    Parameters
    ----------
    rendered       : string produced by render_chain()
    pokemon_names  : set of known Pokémon species names to check for

    Returns
    -------
    List of leaking names (empty list means no leakage).
    """
    leaked: list[str] = []
    for name in pokemon_names:
        # Whole-word, case-insensitive search
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, rendered, flags=re.IGNORECASE):
            leaked.append(name)
    return leaked
