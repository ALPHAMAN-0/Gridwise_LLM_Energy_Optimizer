"""HTTP contract: status codes, schema, and "never crash, never leak".

The language model is always stubbed here; nothing in this file touches the network.
"""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from app import main, pipeline
from app.directives import build_constraints
from app.schemas import DirectiveInterpretation, OptimizeRequest
from app.validator import validate

from .conftest import load_cases

CASES = load_cases()

RESPONSE_KEYS = [
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
]
PLAN_KEYS = {
    "hour",
    "grid_kwh",
    "solar_used_kwh",
    "battery_action",
    "battery_kwh",
    "battery_energy_after_kwh",
}
INTERP_KEYS = {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}


@pytest.fixture()
def client():
    with TestClient(main.app, raise_server_exceptions=False) as c:
        yield c


def _stub_interpreter(monkeypatch, entries: list[dict]) -> None:
    async def fake(_request):
        return [DirectiveInterpretation(**e) for e in entries]

    monkeypatch.setattr(pipeline, "interpret", fake)


def _body(case: dict) -> bytes:
    return json.dumps(case["input"]).encode()


# -- happy path ---------------------------------------------------------------


def test_health_and_root(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_full_response_matches_reference(client, monkeypatch, case):
    expected = case["expected_output"]
    _stub_interpreter(monkeypatch, expected["directive_interpretation"])

    response = client.post("/optimize-energy", content=_body(case), headers={"content-type": "application/json"})
    assert response.status_code == 200
    data = response.json()

    assert list(data.keys()) == RESPONSE_KEYS
    assert data["scenario_id"] == case["input"]["scenario_id"]
    assert [p["hour"] for p in data["hourly_plan"]] == list(range(24))
    assert all(set(p.keys()) == PLAN_KEYS for p in data["hourly_plan"])
    assert all(set(d.keys()) == INTERP_KEYS for d in data["directive_interpretation"])
    assert [d["note_index"] for d in data["directive_interpretation"]] == list(range(len(case["input"]["operator_notes"])))
    for entry in data["directive_interpretation"]:
        if entry["directive_type"] == "no_op":
            assert entry["applies"] is False and entry["structured_adjustment"] is None
        else:
            assert entry["applies"] is True

    assert data["total_cost_bdt"] == pytest.approx(expected["total_cost_bdt"], abs=0.01)
    assert data["total_grid_kwh"] == pytest.approx(sum(p["grid_kwh"] for p in data["hourly_plan"]), abs=0.01)
    assert data["peak_grid_kwh"] == pytest.approx(max(p["grid_kwh"] for p in data["hourly_plan"]), abs=0.01)

    # Replay exactly as the judge does: against the ground-truth directives.
    request = OptimizeRequest.model_validate(case["input"])
    truth = build_constraints(request, expected["directive_interpretation"])
    ok, errors = validate(data["hourly_plan"], request, truth, tol=1e-4)
    assert ok, errors


@pytest.mark.parametrize("headers", [{"content-type": "text/plain"}, {"content-type": "application/x-www-form-urlencoded"}, {}])
def test_valid_json_is_accepted_whatever_the_content_type(client, monkeypatch, headers):
    _stub_interpreter(monkeypatch, CASES[1]["expected_output"]["directive_interpretation"])
    response = client.post("/optimize-energy", content=_body(CASES[1]), headers=headers)
    assert response.status_code == 200, response.text


def test_extra_fields_and_unordered_hours_are_tolerated(client, monkeypatch):
    _stub_interpreter(monkeypatch, CASES[1]["expected_output"]["directive_interpretation"])
    data = copy.deepcopy(CASES[1]["input"])
    data["hours"].reverse()
    data["unexpected"] = {"anything": 1}
    response = client.post("/optimize-energy", json=data)
    assert response.status_code == 200
    assert [p["hour"] for p in response.json()["hourly_plan"]] == list(range(24))


# -- bad input ----------------------------------------------------------------


def _mutated(mutate) -> bytes:
    data = copy.deepcopy(CASES[0]["input"])
    mutate(data)
    return json.dumps(data).encode()


BAD_BODIES = {
    "malformed json": b'{"scenario_id": "X", ',
    "empty body": b"",
    "json array": b"[]",
    "json null": b"null",
    "nan": _body(CASES[0]).replace(b'"demand_kwh": 90', b'"demand_kwh": NaN', 1),
    "four notes": _mutated(lambda d: d.__setitem__("operator_notes", ["a", "b", "c", "d"])),
    "no notes": _mutated(lambda d: d.__setitem__("operator_notes", [])),
    "blank note": _mutated(lambda d: d.__setitem__("operator_notes", ["   "])),
    "note not a string": _mutated(lambda d: d.__setitem__("operator_notes", [7])),
    "23 hours": _mutated(lambda d: d["hours"].pop()),
    "duplicate hour": _mutated(lambda d: d["hours"][1].__setitem__("hour", 0)),
    "hour out of range": _mutated(lambda d: d["hours"][0].__setitem__("hour", 24)),
    "negative demand": _mutated(lambda d: d["hours"][0].__setitem__("demand_kwh", -1)),
    "string number": _mutated(lambda d: d["hours"][0].__setitem__("demand_kwh", "12.5")),
    "bool number": _mutated(lambda d: d["battery"].__setitem__("capacity_kwh", True)),
    "missing battery": _mutated(lambda d: d.pop("battery")),
    "missing scenario_id": _mutated(lambda d: d.pop("scenario_id")),
}


@pytest.mark.parametrize("name", list(BAD_BODIES))
def test_structurally_invalid_requests_get_400(client, name):
    response = client.post("/optimize-energy", content=BAD_BODIES[name], headers={"content-type": "application/json"})
    assert response.status_code == 400, (name, response.text)
    body = response.json()
    assert body["error"] == "invalid_request" and isinstance(body["detail"], str)
    assert "Traceback" not in response.text


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["battery"].__setitem__("initial_energy_kwh", 9999),
        lambda d: d["battery"].__setitem__("initial_energy_kwh", 1),
        lambda d: d["battery"].__setitem__("minimum_energy_kwh", 9999),
    ],
    ids=["initial above capacity", "initial below minimum", "minimum above capacity"],
)
def test_impossible_battery_state_gets_422(client, mutate):
    response = client.post("/optimize-energy", content=_mutated(mutate))
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "unprocessable_scenario"


# -- failure handling ---------------------------------------------------------


def test_interpreter_crash_still_returns_a_valid_plan(client, monkeypatch):
    async def boom(_request):
        raise RuntimeError("provider exploded with sk-SECRET-VALUE")

    monkeypatch.setattr(pipeline, "interpret", boom)
    case = CASES[5]
    response = client.post("/optimize-energy", content=_body(case))
    assert response.status_code == 200
    data = response.json()
    assert "SECRET" not in response.text
    assert [d["directive_type"] for d in data["directive_interpretation"]] == ["no_op"] * len(case["input"]["operator_notes"])
    assert all(d["applies"] is False and d["structured_adjustment"] is None for d in data["directive_interpretation"])

    request = OptimizeRequest.model_validate(case["input"])
    ok, errors = validate(data["hourly_plan"], request, build_constraints(request, []), tol=1e-4)
    assert ok, errors


def test_over_strict_interpretation_never_ignores_the_directive_entirely(client, monkeypatch):
    """A hallucinated, infeasible cap must not collapse into a plan with no cap at all."""
    case = CASES[4]  # true cap: 155 kWh on hours 18-20
    _stub_interpreter(
        monkeypatch,
        [
            {
                "note_index": 0,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {"hours": [17, 18, 19, 20, 21, 22], "max_grid_kwh": 100},
                "explanation": "too strict on purpose",
            }
        ],
    )
    response = client.post("/optimize-energy", content=_body(case))
    assert response.status_code == 200
    data = response.json()
    assert data["directive_interpretation"][0]["directive_type"] == "max_grid_window"  # still reported

    request = OptimizeRequest.model_validate(case["input"])
    truth = build_constraints(request, case["expected_output"]["directive_interpretation"])
    ok, errors = validate(data["hourly_plan"], request, truth)
    assert ok, errors


def test_unhandled_error_is_a_controlled_500(client, monkeypatch):
    async def boom(_request):
        raise RuntimeError("token=sk-SECRET-VALUE")

    monkeypatch.setattr(pipeline, "run", boom)
    response = client.post("/optimize-energy", content=_body(CASES[0]))
    assert response.status_code == 500
    assert response.json() == {"error": "internal_error"}
    assert "SECRET" not in response.text and "Traceback" not in response.text
