"""guardrails.enforce: nothing unvalidated reaches the solver, and it never raises."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.guardrails import MAX_EXPLANATION_CHARS, UNVALIDATED_EXPLANATION, enforce
from app.schemas import Battery, DirectiveInterpretation

BATTERY = Battery(
    capacity_kwh=200,
    initial_energy_kwh=100,
    minimum_energy_kwh=30,
    max_charge_kwh_per_hour=50,
    max_discharge_kwh_per_hour=50,
)


def entry(note_index, directive_type, adjustment=None, explanation="why", **extra) -> dict:
    return {
        "note_index": note_index,
        "applies": directive_type != "no_op",
        "directive_type": directive_type,
        "structured_adjustment": adjustment,
        "explanation": explanation,
        **extra,
    }


def assert_safe_no_op(item: DirectiveInterpretation, note_index: int) -> None:
    assert item.note_index == note_index
    assert item.applies is False
    assert item.directive_type == "no_op"
    assert item.structured_adjustment is None
    assert item.explanation == UNVALIDATED_EXPLANATION


def only(adjustment_or_item, directive_type=None):
    """Run a single-note enforce and return (entry, problems)."""
    item = (
        adjustment_or_item
        if directive_type is None
        else entry(0, directive_type, adjustment_or_item)
    )
    result, problems = enforce([item], 1, BATTERY)
    assert len(result) == 1
    return result[0], problems


# -- happy path ------------------------------------------------------------


def test_valid_entries_of_every_type_pass_unchanged():
    items = [
        entry(0, "solar_reduction", {"hours": [11, 12, 13], "factor": 0.2}),
        entry(1, "minimum_battery_reserve", {"hours": [18, 19], "minimum_energy_kwh": 100}),
        entry(2, "no_op"),
    ]
    result, problems = enforce(items, 3, BATTERY)
    assert problems == []
    assert [r.note_index for r in result] == [0, 1, 2]
    assert all(isinstance(r, DirectiveInterpretation) for r in result)
    assert result[0].structured_adjustment == {"hours": [11, 12, 13], "factor": 0.2}
    assert result[0].applies is True
    assert result[1].structured_adjustment == {"hours": [18, 19], "minimum_energy_kwh": 100.0}
    assert result[2].applies is False and result[2].structured_adjustment is None

    for kind, adjustment in [
        ("no_charge_window", {"hours": [2, 3, 4]}),
        ("no_discharge_window", {"hours": [17, 18]}),
        ("max_grid_window", {"hours": [19, 20], "max_grid_kwh": 180}),
    ]:
        item, problems = only(adjustment, kind)
        assert problems == [] and item.directive_type == kind and item.applies is True


def test_output_is_ordered_by_note_index_regardless_of_input_order():
    items = [entry(2, "no_op"), entry(0, "no_charge_window", {"hours": [1]}), entry(1, "no_op")]
    result, problems = enforce(items, 3, BATTERY)
    assert problems == []
    assert [r.note_index for r in result] == [0, 1, 2]
    assert result[0].directive_type == "no_charge_window"


def test_hours_are_sorted_and_deduped_silently():
    item, problems = only({"hours": [14, 13, 14, 2]}, "no_charge_window")
    assert problems == []
    assert item.structured_adjustment == {"hours": [2, 13, 14]}


# -- garbage in --------------------------------------------------------------


@pytest.mark.parametrize("garbage", [None, "items", 42, {"items": []}, 3.5, True])
def test_non_list_input_becomes_all_no_op(garbage):
    result, problems = enforce(garbage, 2, BATTERY)
    assert len(result) == 2
    for i, item in enumerate(result):
        assert_safe_no_op(item, i)
    assert problems and "must be a list" in problems[0]
    assert "note 0: no entry was returned for this note" in problems
    assert "note 1: no entry was returned for this note" in problems


def test_non_dict_entries_are_reported_and_skipped():
    result, problems = enforce(["x", None, 7, [1], entry(0, "no_op")], 1, BATTERY)
    assert result[0].directive_type == "no_op" and result[0].explanation == "why"
    assert len(problems) == 4 and all(p.startswith("item ") for p in problems)


def test_empty_list_reports_every_note_missing():
    result, problems = enforce([], 3, BATTERY)
    assert [r.note_index for r in result] == [0, 1, 2]
    assert len(problems) == 3


def test_zero_notes_returns_nothing():
    assert enforce([entry(0, "no_op")], 0, BATTERY)[0] == []


# -- note_index --------------------------------------------------------------


@pytest.mark.parametrize("bad_index", [-1, 1, 99, "0", 0.0, None, True, False, [0]])
def test_bad_note_index_is_rejected(bad_index):
    result, problems = enforce([entry(bad_index, "no_op")], 1, BATTERY)
    assert_safe_no_op(result[0], 0)
    assert any("note_index" in p for p in problems)
    assert "note 0: no entry was returned for this note" in problems


def test_duplicate_note_first_valid_occurrence_wins():
    items = [
        entry(0, "no_charge_window", {"hours": [2, 3]}),
        entry(0, "no_discharge_window", {"hours": [9]}),
    ]
    result, problems = enforce(items, 1, BATTERY)
    assert result[0].directive_type == "no_charge_window"
    assert problems == ["note 0: duplicate entry ignored (first valid one kept)"]


def test_valid_duplicate_replaces_an_earlier_invalid_one():
    items = [
        entry(0, "solar_reduction", {"hours": [9], "factor": 1.2}),
        entry(0, "solar_reduction", {"hours": [9], "factor": 0.5}),
    ]
    result, problems = enforce(items, 1, BATTERY)
    assert result[0].structured_adjustment == {"hours": [9], "factor": 0.5}
    assert len(problems) == 1 and "factor" in problems[0]


def test_missing_note_becomes_safe_no_op():
    result, problems = enforce([entry(0, "no_op"), entry(2, "no_op")], 3, BATTERY)
    assert_safe_no_op(result[1], 1)
    assert problems == ["note 1: no entry was returned for this note"]


# -- directive_type and adjustment shape ---------------------------------------


@pytest.mark.parametrize("bad_type", ["generator_test", "", None, 5, "NO_OP", ["no_op"]])
def test_unknown_directive_type(bad_type):
    item, problems = only(entry(0, bad_type, {"hours": [1]}))
    assert_safe_no_op(item, 0)
    assert problems and problems[0].startswith("note 0: directive_type")


@pytest.mark.parametrize(
    "kind, adjustment",
    [
        ("solar_reduction", {"hours": [1]}),
        ("solar_reduction", {"hours": [1], "factor": 0.5, "extra": 1}),
        ("no_charge_window", {"hours": [1], "factor": 0.5}),
        ("max_grid_window", {"hours": [1], "max_grid": 100}),
        ("minimum_battery_reserve", {"hours": [1], "minimum_energy": 50}),
        ("no_discharge_window", {}),
        ("no_discharge_window", None),
        ("no_discharge_window", [1, 2]),
        ("no_discharge_window", "hours 1-2"),
    ],
)
def test_adjustment_key_set_must_be_exact(kind, adjustment):
    item, problems = only(adjustment, kind)
    assert_safe_no_op(item, 0)
    assert len(problems) == 1 and problems[0].startswith("note 0:")


@pytest.mark.parametrize(
    "hours",
    [[], [24], [-1], [True, 2], [False], [1.0], ["3"], [None], "12", None, 5, [1, 2, 99]],
)
def test_bad_hours_are_rejected(hours):
    item, problems = only({"hours": hours}, "no_charge_window")
    assert_safe_no_op(item, 0)
    assert len(problems) == 1 and "hours" in problems[0]


# -- numeric ranges ----------------------------------------------------------


@pytest.mark.parametrize(
    "factor", [1.2, -0.1, float("nan"), float("inf"), "0.5", None, True, [0.5]]
)
def test_bad_factor_is_rejected(factor):
    item, problems = only({"hours": [10], "factor": factor}, "solar_reduction")
    assert_safe_no_op(item, 0)
    assert "factor" in problems[0]


@pytest.mark.parametrize("factor", [0, 1, 0.0, 1.0, 0.333333])
def test_factor_bounds_are_inclusive(factor):
    item, problems = only({"hours": [10], "factor": factor}, "solar_reduction")
    assert problems == [] and item.structured_adjustment["factor"] == float(factor)


@pytest.mark.parametrize(
    "reserve", [200.5, 1e9, -1, float("nan"), float("-inf"), "100", None, True]
)
def test_bad_reserve_is_rejected(reserve):
    item, problems = only({"hours": [18], "minimum_energy_kwh": reserve}, "minimum_battery_reserve")
    assert_safe_no_op(item, 0)
    assert "minimum_energy_kwh" in problems[0]


def test_reserve_above_capacity_names_the_capacity():
    _, problems = only({"hours": [18], "minimum_energy_kwh": 260}, "minimum_battery_reserve")
    assert "exceeds battery capacity" in problems[0]


def test_reserve_within_tolerance_of_capacity_is_clamped():
    item, problems = only(
        {"hours": [18], "minimum_energy_kwh": 200.005}, "minimum_battery_reserve"
    )
    assert problems == []
    assert item.structured_adjustment["minimum_energy_kwh"] == 200.0


@pytest.mark.parametrize("cap", [-0.01, float("nan"), float("inf"), "155", None, False])
def test_bad_grid_cap_is_rejected(cap):
    item, problems = only({"hours": [18], "max_grid_kwh": cap}, "max_grid_window")
    assert_safe_no_op(item, 0)
    assert "max_grid_kwh" in problems[0]


def test_zero_grid_cap_is_allowed():
    item, problems = only({"hours": [18], "max_grid_kwh": 0}, "max_grid_window")
    assert problems == [] and item.structured_adjustment["max_grid_kwh"] == 0.0


# -- forced semantics ----------------------------------------------------------


def test_applies_is_forced_from_the_type():
    lying = entry(0, "no_charge_window", {"hours": [3]})
    lying["applies"] = False
    assert only(lying)[0].applies is True

    noisy = entry(0, "no_op", {"hours": [3]})
    noisy["applies"] = True
    item, problems = only(noisy)
    assert problems == []
    assert item.applies is False and item.structured_adjustment is None


def test_item_carrying_a_problem_flag_is_invalid():
    flagged = entry(0, "no_charge_window", {"hours": [3]}, _problem="window was unusable")
    item, problems = only(flagged)
    assert_safe_no_op(item, 0)
    assert problems == ["note 0: window was unusable"]


@pytest.mark.parametrize("explanation", [None, "", "   ", 17, ["a"], {"a": 1}])
def test_explanation_is_coerced_to_non_empty_text(explanation):
    item, problems = only(entry(0, "no_charge_window", {"hours": [3]}, explanation=explanation))
    assert problems == []
    assert isinstance(item.explanation, str) and item.explanation.strip()


def test_explanation_is_capped_and_whitespace_collapsed():
    item, _ = only(entry(0, "no_op", explanation="word  \n " * 200))
    assert len(item.explanation) <= MAX_EXPLANATION_CHARS
    assert "\n" not in item.explanation and "  " not in item.explanation


# -- never raises --------------------------------------------------------------


def test_never_raises_on_hostile_input():
    class Exploding(dict):
        def get(self, *_a, **_k):
            raise RuntimeError("boom")

    result, problems = enforce([Exploding()], 2, BATTERY)
    assert len(result) == 2 and problems
    for i, item in enumerate(result):
        assert_safe_no_op(item, i)

    result, problems = enforce([entry(0, "no_op")], 1, None)  # type: ignore[arg-type]
    assert len(result) == 1

    result, _ = enforce([], "three", BATTERY)  # type: ignore[arg-type]
    assert result == []


def test_mixed_batch_keeps_the_good_and_neutralises_the_bad():
    items = [
        entry(0, "solar_reduction", {"hours": [11, 12], "factor": 0.25}),
        entry(1, "minimum_battery_reserve", {"hours": [18], "minimum_energy_kwh": 999}),
        entry(2, "max_grid_window", {"hours": [True], "max_grid_kwh": 150}),
    ]
    result, problems = enforce(items, 3, BATTERY)
    assert result[0].applies is True
    assert_safe_no_op(result[1], 1)
    assert_safe_no_op(result[2], 2)
    assert [p.split(":")[0] for p in problems] == ["note 1", "note 2"]
