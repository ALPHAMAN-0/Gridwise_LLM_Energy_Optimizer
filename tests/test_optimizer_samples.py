"""The LP must match the reference optimum and survive inputs the public pack never shows.

Every public input is an integer and the LP is a network problem, so public
solutions sit exactly on the lattice. Rounding and order-of-operations bugs are
invisible there, which is why the fuzz test perturbs the data with decimals.
"""

from __future__ import annotations

import copy
import random

import pytest

from app import optimizer
from app.directives import build_constraints
from app.fallback import build_fallback
from app.schemas import OptimizeRequest
from app.validator import totals, validate

from .conftest import load_cases

CASES = load_cases()
STRICT_TOL = 1e-4


def _available_solvers() -> list[str]:
    found = []
    for name in ("cbc", "highs"):
        try:
            optimizer._make_solver(name)
            found.append(name)
        except Exception:  # noqa: BLE001
            pass
    return found


SOLVERS = _available_solvers()


def _solve(case_input: dict, directives: list[dict], elastic: bool = False):
    request = OptimizeRequest.model_validate(case_input)
    constraints = build_constraints(request, directives)
    return request, constraints, optimizer.solve(request, constraints, elastic=elastic)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_matches_reference_cost(case, solver, monkeypatch):
    monkeypatch.setattr(optimizer, "_preferred", solver)
    expected = case["expected_output"]
    request, constraints, result = _solve(case["input"], expected["directive_interpretation"])

    assert result.status == "optimal"
    ok, errors = validate(result.plan, request, constraints, tol=STRICT_TOL)
    assert ok, errors
    _, cost, _ = totals(result.plan, request)
    assert cost == pytest.approx(expected["total_cost_bdt"], abs=0.01)
    # One action per hour: the LP's simultaneous charge/discharge must be netted away.
    assert all((p.battery_kwh == 0) == (p.battery_action == "idle") for p in result.plan)


def test_warm_up_picks_a_solver():
    assert optimizer.warm_up() is True
    assert optimizer.solver_name() in ("cbc", "highs", "coin")


@pytest.mark.parametrize("seed", range(12))
def test_decimal_fuzz_stays_valid_and_beats_fallback(seed):
    rng = random.Random(seed)
    case = copy.deepcopy(CASES[seed % len(CASES)])
    data = case["input"]
    for hour in data["hours"]:
        hour["demand_kwh"] = round(hour["demand_kwh"] * rng.uniform(0.7, 1.3), 7)
        hour["solar_kwh"] = round(hour["solar_kwh"] * rng.uniform(0.0, 1.6), 7)
        hour["tariff_bdt_per_kwh"] = round(rng.choice([0.0, 5.5, rng.uniform(3, 35)]), 5)
    battery = data["battery"]
    shape = seed % 4
    if shape == 1:  # a battery that cannot move
        battery["max_charge_kwh_per_hour"] = 0
        battery["max_discharge_kwh_per_hour"] = 0
    elif shape == 2:  # no usable headroom at all
        battery["capacity_kwh"] = battery["initial_energy_kwh"] = battery["minimum_energy_kwh"] = 80.125
    elif shape == 3:
        battery["capacity_kwh"] = round(battery["capacity_kwh"] * 1.37, 5)
        battery["initial_energy_kwh"] = round(battery["minimum_energy_kwh"] + 0.12345, 5)

    # No directives: the perturbed data may not be feasible under the original ones.
    request, constraints, result = _solve(data, [])
    assert result.status == "optimal"
    ok, errors = validate(result.plan, request, constraints, tol=STRICT_TOL)
    assert ok, errors

    _, cost, _ = totals(result.plan, request)
    _, fallback_cost, _ = totals(build_fallback(request, constraints.effective_solar), request)
    assert cost <= fallback_cost + 0.01


def test_zero_valued_directives_are_binding():
    """factor 0.0, a 0 kWh cap and hour 0 are all falsy; none may be dropped."""
    data = copy.deepcopy(CASES[0]["input"])
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [11, 12], "factor": 0.0}, "explanation": ""},
        {"note_index": 1, "applies": True, "directive_type": "no_discharge_window",
         "structured_adjustment": {"hours": [0, 1]}, "explanation": ""},
    ]
    request, constraints, result = _solve(data, directives)
    assert constraints.effective_solar[11] == 0 and constraints.effective_solar[12] == 0
    assert constraints.discharge_allowed[0] is False
    assert result.status == "optimal"
    assert result.plan[11].solar_used_kwh == 0 and result.plan[12].solar_used_kwh == 0
    assert result.plan[0].battery_action != "discharge"
    ok, errors = validate(result.plan, request, constraints, tol=STRICT_TOL)
    assert ok, errors


def test_infeasible_strict_model_gets_minimal_violation_plan():
    """An over-strict cap must not end in a plan that ignores the cap everywhere."""
    data = copy.deepcopy(CASES[4]["input"])  # SAMPLE-05: feeder cap 155 on 18-20
    too_strict = [
        {"note_index": 0, "applies": True, "directive_type": "max_grid_window",
         "structured_adjustment": {"hours": [17, 18, 19, 20, 21, 22], "max_grid_kwh": 100},
         "explanation": ""}
    ]
    request, constraints, strict = _solve(data, too_strict)
    assert strict.status == "infeasible"

    _, _, relaxed = _solve(data, too_strict, elastic=True)
    assert relaxed.status == "optimal"
    physical = build_constraints(request, too_strict, apply_reserve=False, apply_grid_cap=False)
    ok, errors = validate(relaxed.plan, request, physical, tol=STRICT_TOL)
    assert ok, errors
    # Still honours the true (looser) cap of 155 on the true hours.
    assert all(relaxed.plan[h].grid_kwh <= 155 + 0.01 for h in (18, 19, 20))


def test_overlapping_solar_reductions_never_loosen():
    data = copy.deepcopy(CASES[0]["input"])
    both = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [12], "factor": 0.5}, "explanation": ""},
        {"note_index": 1, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [12], "factor": 0.4}, "explanation": ""},
    ]
    request = OptimizeRequest.model_validate(data)
    constraints = build_constraints(request, both)
    assert constraints.effective_solar[12] <= 180 * 0.4 + 1e-9
