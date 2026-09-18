"""The always-valid plan: meet demand from solar first, then the grid.

No optimizer, no solver, no failure mode. The battery never moves, so
end-of-day neutrality holds by construction rather than by constraint.

This is the last tier of main.py's fallback ladder. It is not cost-optimal and
it cannot honour a max_grid_window cap, so earlier tiers should almost always
win; this exists so that the service always has *something* valid to return.
"""

from __future__ import annotations

from .schemas import HourPlan, OptimizeRequest


def build_fallback(
    request: OptimizeRequest, effective_solar: list[float]
) -> list[HourPlan]:
    """Grid-and-solar-only plan for all 24 hours.

    `effective_solar` is passed in rather than read off the request so the plan
    still respects any solar_reduction directive the interpreter found.
    """
    hours = request.by_hour()
    resting = round(float(request.battery.initial_energy_kwh), 4)

    plan: list[HourPlan] = []
    for h, entry in enumerate(hours):
        solar_used = min(float(entry.demand_kwh), max(0.0, float(effective_solar[h])))
        plan.append(
            HourPlan(
                hour=entry.hour,
                grid_kwh=round(float(entry.demand_kwh) - solar_used, 4),
                solar_used_kwh=round(solar_used, 4),
                battery_action="idle",
                battery_kwh=0.0,
                battery_energy_after_kwh=resting,
            )
        )
    return plan
