"""Deterministic gate between the language model and the solver (spec section 08).

Nothing a model says reaches the optimizer without passing through `enforce`.
The input is treated as hostile: it may not be a list, entries may not be
dicts, numbers may be strings, booleans, NaN or infinity. The output is always
exactly one well-formed DirectiveInterpretation per operator note, in order.

The failure policy is "safe no_op": an entry that cannot be validated is
replaced by a no_op rather than repaired by guesswork, because a wrong hard
constraint can make the LP infeasible or silently cost money, while a no_op
only forgoes one directive. Every replacement is reported in the returned
problems list so the interpreter can re-ask the model with specifics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .schemas import (
    DIRECTIVE_TYPES,
    REQUIRED_ADJUSTMENT_KEYS,
    TOLERANCE,
    Battery,
    DirectiveInterpretation,
)

MAX_EXPLANATION_CHARS = 300

UNVALIDATED_EXPLANATION = (
    "This note's interpretation could not be validated, so it was treated as a "
    "no_op and no adjustment was applied."
)

_DEFAULT_EXPLANATIONS: dict[str, str] = {
    "solar_reduction": "Usable solar is reduced during the stated hours.",
    "minimum_battery_reserve": "A minimum battery reserve is required during the stated hours.",
    "no_charge_window": "Battery charging is unavailable during the stated hours.",
    "no_discharge_window": "Battery discharge is not allowed during the stated hours.",
    "max_grid_window": "Grid import is capped during the stated hours.",
    "no_op": "This note does not affect today's energy schedule.",
}


class _Invalid(Exception):
    """Internal: carries the human-readable reason an entry was rejected."""


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; True must not pass for hour 1.
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _number(adjustment: Mapping[str, Any], key: str) -> float:
    number = _finite(adjustment.get(key))
    if number is None:
        raise _Invalid(f"{key} must be a finite number, got {adjustment.get(key)!r}")
    return number


def _hours(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise _Invalid(f"hours must be a list of integers 0..23, got {type(value).__name__}")
    for hour in value:
        if not _is_int(hour) or not 0 <= hour <= 23:
            raise _Invalid(f"hours contains {hour!r}, which is not an integer 0..23")
    if not value:
        raise _Invalid("hours is empty")
    return sorted(set(value))


def _explanation(value: Any, directive_type: str) -> str:
    text = " ".join(value.split()) if isinstance(value, str) else ""
    if not text:
        text = _DEFAULT_EXPLANATIONS[directive_type]
    if len(text) > MAX_EXPLANATION_CHARS:
        text = text[: MAX_EXPLANATION_CHARS - 3].rstrip() + "..."
    return text


def _safe_no_op(note_index: int) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=UNVALIDATED_EXPLANATION,
    )


def _check(item: Mapping[str, Any], note_index: int, capacity_kwh: float) -> DirectiveInterpretation:
    """Validate one entry whose note_index is already known to be good."""
    flagged = item.get("_problem")
    if flagged:
        raise _Invalid(str(flagged)[:200])

    directive_type = item.get("directive_type")
    if not isinstance(directive_type, str) or directive_type not in DIRECTIVE_TYPES:
        raise _Invalid(f"directive_type {directive_type!r} is not one of {list(DIRECTIVE_TYPES)}")

    if directive_type == "no_op":
        # applies/adjustment are forced, not checked: no_op has one legal shape.
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=_explanation(item.get("explanation"), "no_op"),
        )

    adjustment = item.get("structured_adjustment")
    if not isinstance(adjustment, Mapping):
        raise _Invalid(f"{directive_type} needs a structured_adjustment object")
    required = REQUIRED_ADJUSTMENT_KEYS[directive_type]
    if set(adjustment.keys()) != set(required):
        raise _Invalid(
            f"structured_adjustment keys {sorted(map(str, adjustment.keys()))} "
            f"must be exactly {sorted(required)}"
        )

    clean: dict[str, Any] = {"hours": _hours(adjustment["hours"])}

    if directive_type == "solar_reduction":
        factor = _number(adjustment, "factor")
        if not 0.0 <= factor <= 1.0:
            raise _Invalid(f"factor {factor} is outside 0..1")
        clean["factor"] = round(factor, 6)

    elif directive_type == "minimum_battery_reserve":
        reserve = _number(adjustment, "minimum_energy_kwh")
        if reserve < 0.0:
            raise _Invalid(f"minimum_energy_kwh {reserve} is negative")
        if reserve > capacity_kwh + TOLERANCE:
            raise _Invalid(
                f"minimum_energy_kwh {reserve} exceeds battery capacity {capacity_kwh} kWh"
            )
        # Within tolerance of full: clamp, or the solver sees floor > capacity.
        clean["minimum_energy_kwh"] = round(min(reserve, capacity_kwh), 6)

    elif directive_type == "max_grid_window":
        cap = _number(adjustment, "max_grid_kwh")
        if cap < 0.0:
            raise _Invalid(f"max_grid_kwh {cap} is negative")
        clean["max_grid_kwh"] = round(cap, 6)

    return DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=clean,
        explanation=_explanation(item.get("explanation"), directive_type),
    )


def _enforce(
    items: Any, n_notes: int, capacity_kwh: float
) -> tuple[list[DirectiveInterpretation], list[str]]:
    problems: list[str] = []
    accepted: dict[int, DirectiveInterpretation] = {}
    rejected: set[int] = set()

    if not isinstance(items, (list, tuple)):
        problems.append(f"interpretation must be a list of items, got {type(items).__name__}")
        items = []

    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            problems.append(f"item {position}: not an object ({type(item).__name__})")
            continue

        note_index = item.get("note_index")
        if not _is_int(note_index) or not 0 <= note_index < n_notes:
            problems.append(
                f"item {position}: note_index {note_index!r} is not an integer in 0..{n_notes - 1}"
            )
            continue

        if note_index in accepted:
            problems.append(f"note {note_index}: duplicate entry ignored (first valid one kept)")
            continue

        try:
            accepted[note_index] = _check(item, note_index, capacity_kwh)
        except _Invalid as invalid:
            problems.append(f"note {note_index}: {invalid}")
            rejected.add(note_index)

    result: list[DirectiveInterpretation] = []
    for note_index in range(n_notes):
        entry = accepted.get(note_index)
        if entry is None:
            if note_index not in rejected:
                problems.append(f"note {note_index}: no entry was returned for this note")
            entry = _safe_no_op(note_index)
        result.append(entry)
    return result, problems


def enforce(
    items: list[dict], n_notes: int, battery: Battery
) -> tuple[list[DirectiveInterpretation], list[str]]:
    """Validate raw interpretation items. Returns (one entry per note, problems).

    Never raises. Anything that cannot be validated becomes a safe no_op and a
    "note <i>: <reason>" line in the problems list.
    """
    try:
        count = max(0, int(n_notes))
    except (TypeError, ValueError, OverflowError):
        count = 0
    try:
        capacity_kwh = _finite(getattr(battery, "capacity_kwh", None))
        if capacity_kwh is None:
            # No trustworthy capacity: only a zero reserve can be proven safe.
            capacity_kwh = 0.0
        return _enforce(items, count, capacity_kwh)
    except Exception as exc:  # noqa: BLE001 - the gate itself must never fail open
        return (
            [_safe_no_op(i) for i in range(count)],
            [f"guardrails internal error ({type(exc).__name__}); every note treated as no_op"],
        )
