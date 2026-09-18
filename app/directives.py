"""Turn validated directives into the per-hour arrays the model works with.

Input is the guardrailed interpretation list — by the time anything reaches
here, types are known, hours are clean integers 0..23, and numbers are in
range. This module only does arithmetic.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .schemas import Constraints, OptimizeRequest


def _rows(directives: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in directives:
        if isinstance(d, Mapping):
            out.append(dict(d))
        elif hasattr(d, "model_dump"):
            out.append(d.model_dump())
    return out


_VALUE_KEY = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "max_grid_window": "max_grid_kwh",
}


def _value_for(kind: Any, adjustment: Mapping[str, Any]) -> float:
    """The one number a directive carries (0.0 for the window-only types)."""
    key = _VALUE_KEY.get(kind)
    if key is None:
        return 0.0
    raw = adjustment[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(key)
    return float(raw)


def build_constraints(
    request: OptimizeRequest,
    directives: Iterable[Any],
    *,
    apply_solar: bool = True,
    apply_reserve: bool = True,
    apply_windows: bool = True,
    apply_grid_cap: bool = True,
) -> Constraints:
    """Fold every applicable directive into five 24-long arrays.

    The `apply_*` switches exist for main.py's relaxation ladder: when the LP
    is infeasible, the most likely cause is a hallucinated hard directive, so
    we re-solve with one class of directive dropped at a time.
    """
    hours = request.by_hour()
    battery = request.battery

    effective_solar = [float(h.solar_kwh) for h in hours]
    floor = [float(battery.minimum_energy_kwh)] * 24
    charge_allowed = [True] * 24
    discharge_allowed = [True] * 24
    grid_cap: list[float | None] = [None] * 24

    for row in _rows(directives):
        kind = row.get("directive_type")
        adjustment = row.get("structured_adjustment")
        if kind == "no_op" or row.get("applies") is False or not isinstance(adjustment, Mapping):
            continue
        # bool is an int in Python; True must not quietly become hour 1.
        listed = sorted(
            {
                h
                for h in adjustment.get("hours") or []
                if isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23
            }
        )
        try:
            value = _value_for(kind, adjustment)
        except (KeyError, TypeError, ValueError):
            # Guardrails run first, so this should be unreachable; a row that
            # slips through is skipped rather than turned into a 500.
            continue

        if kind == "solar_reduction" and apply_solar:
            factor = value
            for h in listed:
                # Overlapping reductions multiply. That is never looser than
                # "lowest factor wins", so the plan stays valid whichever rule
                # the judge applies.
                effective_solar[h] = round(effective_solar[h] * factor, 6)

        elif kind == "minimum_battery_reserve" and apply_reserve:
            reserve = value
            for h in listed:
                # Highest reserve wins.
                floor[h] = max(floor[h], reserve)

        elif kind == "no_charge_window" and apply_windows:
            for h in listed:
                charge_allowed[h] = False

        elif kind == "no_discharge_window" and apply_windows:
            for h in listed:
                discharge_allowed[h] = False

        elif kind == "max_grid_window" and apply_grid_cap:
            cap = value
            for h in listed:
                # Lowest cap wins.
                grid_cap[h] = cap if grid_cap[h] is None else min(float(grid_cap[h]), cap)

    return Constraints(
        effective_solar=effective_solar,
        floor=floor,
        charge_allowed=charge_allowed,
        discharge_allowed=discharge_allowed,
        grid_cap=grid_cap,
    )
