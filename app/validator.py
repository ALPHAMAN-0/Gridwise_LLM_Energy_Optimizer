"""Independent replay of any 24-hour plan against the GridWise rules.

This module deliberately imports nothing from `optimizer.py`. It is a second
implementation of the same contract, written from the Problem Statement rather
than from the solver, so that a bug in the solver cannot hide behind a matching
bug in the checker. The judge does exactly this to our output; we do it first.

Every failure is reported, not just the first, so one bad run tells you
everything that is wrong with a plan.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .schemas import TOLERANCE, Constraints, HourPlan, OptimizeRequest

_ACTIONS = ("charge", "discharge", "idle")
_NUMERIC_FIELDS = (
    "grid_kwh",
    "solar_used_kwh",
    "battery_kwh",
    "battery_energy_after_kwh",
)


def _as_row(entry: Any) -> dict[str, Any]:
    """Accept HourPlan objects, plain dicts, or anything dict-like."""
    if isinstance(entry, HourPlan):
        return entry.model_dump()
    if isinstance(entry, Mapping):
        return dict(entry)
    return {}


def _finite(value: Any) -> float | None:
    """Return value as a finite float, or None if it is not usable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def validate(
    plan: Iterable[Any],
    request: OptimizeRequest,
    constraints: Constraints,
    tol: float = TOLERANCE,
) -> tuple[bool, list[str]]:
    """Replay `plan` hour by hour. Returns (is_valid, [every failure]).

    `tol` defaults to the judge's published tolerance; tests pass something far
    tighter, because the official package may be stricter than the statement.
    """
    errors: list[str] = []
    rows = [_as_row(entry) for entry in plan]

    # -- shape -------------------------------------------------------------
    if len(rows) != 24:
        errors.append(f"hourly_plan must have exactly 24 entries, found {len(rows)}")

    by_hour: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        hour = row.get("hour")
        if not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23:
            errors.append(f"entry {position}: hour {hour!r} is not an integer 0..23")
            continue
        if hour in by_hour:
            errors.append(f"entry {position}: hour {hour} appears more than once")
            continue
        by_hour[hour] = row

    missing = [h for h in range(24) if h not in by_hour]
    if missing:
        errors.append(f"hourly_plan is missing hour(s) {missing}")

    # -- per-hour replay ---------------------------------------------------
    hours = request.by_hour()
    battery = request.battery
    energy = float(battery.initial_energy_kwh)
    # Second chain, advanced only by battery_kwh. The reported chain above can
    # hide a small per-hour error that a judge recursing from the initial level
    # would accumulate, so both are checked.
    chain = float(battery.initial_energy_kwh)
    chain_ok = True

    for h in range(24):
        row = by_hour.get(h)
        if row is None:
            continue
        where = f"hour {h}"

        values: dict[str, float] = {}
        for field in _NUMERIC_FIELDS:
            value = _finite(row.get(field))
            if value is None:
                errors.append(f"{where}: {field} is missing or not a finite number")
            else:
                if value < -tol:
                    errors.append(f"{where}: {field} is negative ({value})")
                values[field] = value

        action = row.get("battery_action")
        if action not in _ACTIONS:
            errors.append(f"{where}: battery_action {action!r} is not one of {_ACTIONS}")
            action = None

        if len(values) != len(_NUMERIC_FIELDS) or action is None:
            # Cannot replay this hour; the battery recursion is now unreliable,
            # so stop advancing energy and keep collecting shape errors.
            chain_ok = False
            continue

        grid = values["grid_kwh"]
        solar_used = values["solar_used_kwh"]
        magnitude = values["battery_kwh"]
        energy_after = values["battery_energy_after_kwh"]

        charge = magnitude if action == "charge" else 0.0
        discharge = magnitude if action == "discharge" else 0.0

        if action == "idle" and abs(magnitude) > tol:
            errors.append(f"{where}: battery_kwh must be 0 when idle, got {magnitude}")

        # Energy balance: supply in == demand out.
        demand = float(hours[h].demand_kwh)
        supply = grid + solar_used + discharge
        draw = demand + charge
        if abs(supply - draw) > tol:
            errors.append(
                f"{where}: energy balance broken — grid+solar+discharge={supply:.4f} "
                f"but demand+charge={draw:.4f}"
            )

        # Solar cannot exceed what is actually available after directives.
        available = float(constraints.effective_solar[h])
        if solar_used > available + tol:
            errors.append(
                f"{where}: solar_used_kwh {solar_used:.4f} exceeds effective solar "
                f"{available:.4f}"
            )

        # Battery state transition.
        expected = energy + charge - discharge
        if abs(energy_after - expected) > tol:
            errors.append(
                f"{where}: battery_energy_after_kwh {energy_after:.4f} does not follow "
                f"from {energy:.4f} {'+' if charge else '-'} {magnitude:.4f} "
                f"(expected {expected:.4f})"
            )
        energy = energy_after

        chain += charge - discharge
        if chain_ok and abs(energy_after - chain) > tol:
            errors.append(
                f"{where}: battery_energy_after_kwh {energy_after:.4f} drifts from the level "
                f"implied by the battery actions since hour 0 ({chain:.4f})"
            )
            chain_ok = False  # report the first divergence only

        # Battery bounds, with any active reserve raising the floor.
        floor = float(constraints.floor[h])
        if energy < floor - tol:
            errors.append(
                f"{where}: battery energy {energy:.4f} is below the required floor {floor:.4f}"
            )
        if energy > float(battery.capacity_kwh) + tol:
            errors.append(
                f"{where}: battery energy {energy:.4f} exceeds capacity "
                f"{float(battery.capacity_kwh):.4f}"
            )

        # Hourly rate limits.
        if charge > float(battery.max_charge_kwh_per_hour) + tol:
            errors.append(
                f"{where}: charge {charge:.4f} exceeds max_charge_kwh_per_hour "
                f"{float(battery.max_charge_kwh_per_hour):.4f}"
            )
        if discharge > float(battery.max_discharge_kwh_per_hour) + tol:
            errors.append(
                f"{where}: discharge {discharge:.4f} exceeds max_discharge_kwh_per_hour "
                f"{float(battery.max_discharge_kwh_per_hour):.4f}"
            )

        # Operator directives.
        if charge > tol and not constraints.charge_allowed[h]:
            errors.append(f"{where}: charged {charge:.4f} inside a no_charge_window")
        if discharge > tol and not constraints.discharge_allowed[h]:
            errors.append(
                f"{where}: discharged {discharge:.4f} inside a no_discharge_window"
            )
        cap = constraints.grid_cap[h]
        if cap is not None and grid > float(cap) + tol:
            errors.append(
                f"{where}: grid_kwh {grid:.4f} exceeds max_grid_window cap {float(cap):.4f}"
            )

    # -- end-of-day neutrality --------------------------------------------
    if len(by_hour) == 24:
        initial = float(battery.initial_energy_kwh)
        final = _finite(by_hour[23].get("battery_energy_after_kwh"))
        if final is None:
            errors.append("hour 23: battery_energy_after_kwh is missing or not finite")
        elif abs(final - initial) > tol:
            errors.append(
                f"end-of-day neutrality broken: battery ends at {final:.4f} "
                f"but started at {initial:.4f}"
            )

    return (not errors), errors


def totals(plan: Iterable[Any], request: OptimizeRequest) -> tuple[float, float, float]:
    """Recompute (total_grid_kwh, total_cost_bdt, peak_grid_kwh) from the plan.

    The judge derives these from the returned hourly_plan, never from a solver
    objective, so we report exactly what the plan implies.
    """
    hours = {h.hour: h for h in request.hours}
    total_grid = 0.0
    total_cost = 0.0
    peak = 0.0
    for entry in plan:
        row = _as_row(entry)
        grid = _finite(row.get("grid_kwh")) or 0.0
        hour = row.get("hour")
        tariff = float(hours[hour].tariff_bdt_per_kwh) if hour in hours else 0.0
        total_grid += grid
        total_cost += grid * tariff
        peak = max(peak, grid)
    return round(total_grid, 4), round(total_cost, 4), round(peak, 4)
