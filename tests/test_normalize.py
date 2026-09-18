"""normalize_item: the arithmetic the model is not trusted with."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.interpreter import FEW_SHOT_EXAMPLES, normalize_item
from app.guardrails import enforce
from app.schemas import REQUIRED_ADJUSTMENT_KEYS, Battery

FINAL_KEYS = {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}


def battery(capacity: float = 200.0) -> Battery:
    return Battery(
        capacity_kwh=capacity,
        initial_energy_kwh=capacity / 2,
        minimum_energy_kwh=20,
        max_charge_kwh_per_hour=50,
        max_discharge_kwh_per_hour=50,
    )


def item(directive_type: str, windows=((13, 15),), **fields) -> dict:
    base = {
        "note_index": 0,
        "directive_type": directive_type,
        "windows": [{"start_hour": s, "end_hour": e} for s, e in windows],
        "solar_percent": None,
        "solar_percent_is": None,
        "reserve_value": None,
        "reserve_unit": None,
        "max_grid_kwh": None,
        "explanation": "because",
    }
    base.update(fields)
    return base


# -- windows ---------------------------------------------------------------


def test_window_end_is_exclusive():
    out = normalize_item(item("no_charge_window", [(13, 15)]), battery())
    assert set(out) == FINAL_KEYS
    assert out["structured_adjustment"] == {"hours": [13, 14]}
    assert out["applies"] is True


def test_window_wraps_past_midnight():
    out = normalize_item(item("no_discharge_window", [(22, 2)]), battery())
    assert out["structured_adjustment"] == {"hours": [0, 1, 22, 23]}


def test_window_to_midnight_and_all_day():
    assert normalize_item(item("no_charge_window", [(18, 24)]), battery())[
        "structured_adjustment"
    ] == {"hours": [18, 19, 20, 21, 22, 23]}
    # An end of 0 is midnight too: 22..0 wraps with nothing after midnight.
    assert normalize_item(item("no_charge_window", [(22, 0)]), battery())[
        "structured_adjustment"
    ] == {"hours": [22, 23]}
    assert normalize_item(item("no_charge_window", [(0, 24)]), battery())[
        "structured_adjustment"
    ] == {"hours": list(range(24))}


def test_multiple_windows_are_unioned_sorted_unique():
    out = normalize_item(item("no_charge_window", [(14, 16), (2, 4), (15, 17)]), battery())
    assert out["structured_adjustment"] == {"hours": [2, 3, 14, 15, 16]}


def test_integral_float_and_string_hours_are_accepted():
    raw = item("no_charge_window")
    raw["windows"] = [{"start_hour": 9.0, "end_hour": "11"}]
    assert normalize_item(raw, battery())["structured_adjustment"] == {"hours": [9, 10]}


@pytest.mark.parametrize(
    "windows",
    [
        [],
        None,
        "13-15",
        [{"start_hour": 5, "end_hour": 5}],
        [{"start_hour": -1, "end_hour": 3}],
        [{"start_hour": 3, "end_hour": 25}],
        [{"start_hour": 3}],
        [{"start_hour": True, "end_hour": 4}],
        [{"start_hour": 2.5, "end_hour": 4}],
        ["13..15"],
    ],
)
def test_unusable_windows_are_flagged(windows):
    raw = item("no_charge_window")
    raw["windows"] = windows
    out = normalize_item(raw, battery())
    assert "_problem" in out
    assert out["structured_adjustment"] is None
    assert out["applies"] is False


# -- solar -----------------------------------------------------------------


def test_eighty_percent_reduction_is_exactly_point_two():
    out = normalize_item(
        item("solar_reduction", [(11, 14)], solar_percent=80, solar_percent_is="reduction"),
        battery(),
    )
    assert out["structured_adjustment"] == {"hours": [11, 12, 13], "factor": 0.2}
    assert "_problem" not in out


def test_twenty_five_percent_remaining():
    out = normalize_item(
        item("solar_reduction", [(12, 14)], solar_percent=25, solar_percent_is="remaining"),
        battery(),
    )
    assert out["structured_adjustment"] == {"hours": [12, 13], "factor": 0.25}


def test_solar_offline_and_untouched_extremes():
    off = normalize_item(
        item("solar_reduction", solar_percent=0, solar_percent_is="remaining"), battery()
    )
    assert off["structured_adjustment"]["factor"] == 0.0
    full_loss = normalize_item(
        item("solar_reduction", solar_percent=100, solar_percent_is="reduction"), battery()
    )
    assert full_loss["structured_adjustment"]["factor"] == 0.0


def test_a_third_rounds_to_six_places():
    out = normalize_item(
        item("solar_reduction", solar_percent=33.333333, solar_percent_is="remaining"), battery()
    )
    assert out["structured_adjustment"]["factor"] == 0.333333


def test_bare_fraction_is_read_as_a_share():
    out = normalize_item(
        item("solar_reduction", solar_percent=0.2, solar_percent_is="remaining"), battery()
    )
    assert out["structured_adjustment"]["factor"] == 0.2


@pytest.mark.parametrize(
    "fields",
    [
        {"solar_percent": None, "solar_percent_is": "remaining"},
        {"solar_percent": 40, "solar_percent_is": None},
        {"solar_percent": 40, "solar_percent_is": "sideways"},
        {"solar_percent": 140, "solar_percent_is": "reduction"},
        {"solar_percent": -5, "solar_percent_is": "reduction"},
        {"solar_percent": float("nan"), "solar_percent_is": "remaining"},
        {"solar_percent": True, "solar_percent_is": "remaining"},
    ],
)
def test_solar_missing_pieces_are_flagged(fields):
    out = normalize_item(item("solar_reduction", **fields), battery())
    assert "_problem" in out and out["structured_adjustment"] is None


# -- reserve ---------------------------------------------------------------


def test_fifty_percent_of_two_hundred_is_one_hundred():
    out = normalize_item(
        item(
            "minimum_battery_reserve",
            [(18, 21)],
            reserve_value=50,
            reserve_unit="percent_of_capacity",
        ),
        battery(200),
    )
    assert out["structured_adjustment"] == {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0}


def test_reserve_in_kwh_passes_through():
    out = normalize_item(
        item("minimum_battery_reserve", [(18, 22)], reserve_value=90, reserve_unit="kwh"),
        battery(250),
    )
    assert out["structured_adjustment"] == {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 90.0}


def test_percent_reserve_uses_this_battery():
    out = normalize_item(
        item("minimum_battery_reserve", reserve_value=75, reserve_unit="percent_of_capacity"),
        battery(180),
    )
    assert out["structured_adjustment"]["minimum_energy_kwh"] == 135.0


@pytest.mark.parametrize(
    "fields",
    [
        {"reserve_value": None, "reserve_unit": "kwh"},
        {"reserve_value": 90, "reserve_unit": None},
        {"reserve_value": 90, "reserve_unit": "joules"},
        {"reserve_value": -1, "reserve_unit": "kwh"},
        {"reserve_value": 150, "reserve_unit": "percent_of_capacity"},
        {"reserve_value": float("inf"), "reserve_unit": "kwh"},
    ],
)
def test_reserve_missing_pieces_are_flagged(fields):
    out = normalize_item(item("minimum_battery_reserve", **fields), battery())
    assert "_problem" in out and out["structured_adjustment"] is None


# -- grid cap, no_op, junk ---------------------------------------------------


def test_max_grid_window():
    out = normalize_item(item("max_grid_window", [(18, 21)], max_grid_kwh=155), battery())
    assert out["structured_adjustment"] == {"hours": [18, 19, 20], "max_grid_kwh": 155.0}


@pytest.mark.parametrize("cap", [None, -3, "lots", float("nan")])
def test_max_grid_missing_or_bad_cap_is_flagged(cap):
    out = normalize_item(item("max_grid_window", max_grid_kwh=cap), battery())
    assert "_problem" in out


def test_no_op_has_no_adjustment_and_ignores_stray_fields():
    out = normalize_item(item("no_op", [(1, 5)], solar_percent=50, max_grid_kwh=10), battery())
    assert out == {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "because",
    }


def test_unknown_type_and_non_dict_are_flagged():
    assert "_problem" in normalize_item(item("generator_test"), battery())
    assert "_problem" in normalize_item("nonsense", battery())  # type: ignore[arg-type]
    assert "_problem" in normalize_item(None, battery())  # type: ignore[arg-type]


def test_empty_explanation_gets_a_factual_default():
    out = normalize_item(item("max_grid_window", [(18, 21)], max_grid_kwh=155, explanation=" "), battery())
    assert "155" in out["explanation"] and "18-20" in out["explanation"]


def test_exact_key_sets_for_every_type():
    samples = {
        "solar_reduction": item("solar_reduction", solar_percent=50, solar_percent_is="remaining"),
        "minimum_battery_reserve": item(
            "minimum_battery_reserve", reserve_value=60, reserve_unit="kwh"
        ),
        "no_charge_window": item("no_charge_window"),
        "no_discharge_window": item("no_discharge_window"),
        "max_grid_window": item("max_grid_window", max_grid_kwh=100),
    }
    for directive_type, raw in samples.items():
        out = normalize_item(raw, battery())
        assert set(out) == FINAL_KEYS
        assert set(out["structured_adjustment"]) == set(REQUIRED_ADJUSTMENT_KEYS[directive_type])


def test_normalize_is_pure():
    raw = item("solar_reduction", solar_percent=80, solar_percent_is="reduction")
    snapshot = repr(raw)
    first = normalize_item(raw, battery())
    second = normalize_item(raw, battery())
    assert first == second and repr(raw) == snapshot


# -- the prompt's own examples must survive the pipeline ---------------------


def test_every_few_shot_example_normalizes_and_validates():
    expected = {
        0: [
            ("solar_reduction", {"hours": [9, 10, 11], "factor": 0.65}),
            ("no_op", None),
            ("no_charge_window", {"hours": [13, 14]}),
        ],
        1: [
            ("solar_reduction", {"hours": [13, 14, 15], "factor": 0.4}),
            ("minimum_battery_reserve", {"hours": [0, 1, 2, 3, 22, 23], "minimum_energy_kwh": 65.0}),
            ("no_op", None),
        ],
        2: [
            ("minimum_battery_reserve", {"hours": [17, 18, 19], "minimum_energy_kwh": 135.0}),
            ("max_grid_window", {"hours": [18, 19, 20, 21, 22, 23], "max_grid_kwh": 140.0}),
            ("no_discharge_window", {"hours": [0, 1, 2, 3, 4, 5, 6]}),
        ],
        3: [("no_op", None), ("no_op", None)],
    }
    for number, (capacity, notes, items) in enumerate(FEW_SHOT_EXAMPLES):
        bat = battery(capacity)
        result, problems = enforce([normalize_item(i, bat) for i in items], len(notes), bat)
        assert problems == []
        assert [(r.directive_type, r.structured_adjustment) for r in result] == expected[number]
