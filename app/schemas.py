"""Pydantic v2 models for the GridWise optimize-energy contract.

Field names here are load-bearing: the judge harness matches them exactly
against the Problem Statement (sections 07 and 10). Do not rename anything.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Tolerance the judge uses for float comparisons (kWh and BDT alike).
TOLERANCE = 0.01

BatteryAction = Literal["charge", "discharge", "idle"]

DIRECTIVE_TYPES: tuple[str, ...] = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)

# Exact structured_adjustment key sets, per Problem Statement section 04.
REQUIRED_ADJUSTMENT_KEYS: dict[str, frozenset[str]] = {
    "solar_reduction": frozenset({"hours", "factor"}),
    "minimum_battery_reserve": frozenset({"hours", "minimum_energy_kwh"}),
    "no_charge_window": frozenset({"hours"}),
    "no_discharge_window": frozenset({"hours"}),
    "max_grid_window": frozenset({"hours", "max_grid_kwh"}),
}


# `allow_inf_nan=False` makes NaN/Infinity a schema failure, which main.py
# turns into a 400 rather than letting it poison the solver.
_STRICT = ConfigDict(extra="ignore", allow_inf_nan=False)


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------


class HourEntry(BaseModel):
    model_config = _STRICT

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float


class Battery(BaseModel):
    model_config = _STRICT

    capacity_kwh: float = Field(ge=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class OptimizeRequest(BaseModel):
    model_config = _STRICT

    scenario_id: str
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourEntry] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("hours")
    @classmethod
    def _hours_cover_0_to_23(cls, v: list[HourEntry]) -> list[HourEntry]:
        seen = sorted(h.hour for h in v)
        if seen != list(range(24)):
            raise ValueError("hours must contain each hour 0 through 23 exactly once")
        return v

    def by_hour(self) -> list[HourEntry]:
        """Hours sorted 0..23, so downstream code can index positionally."""
        return sorted(self.hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourPlan(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float

    @property
    def charge(self) -> float:
        return self.battery_kwh if self.battery_action == "charge" else 0.0

    @property
    def discharge(self) -> float:
        return self.battery_kwh if self.battery_action == "discharge" else 0.0


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# --------------------------------------------------------------------------
# Constraints handed to the optimizer and the validator
# --------------------------------------------------------------------------


class Constraints(BaseModel):
    """Per-hour arrays derived from the validated directives.

    The optimizer builds a plan from these; the validator independently
    replays a plan against them. Both read the same object, but neither
    imports the other.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    effective_solar: list[float] = Field(min_length=24, max_length=24)
    floor: list[float] = Field(min_length=24, max_length=24)
    charge_allowed: list[bool] = Field(min_length=24, max_length=24)
    discharge_allowed: list[bool] = Field(min_length=24, max_length=24)
    grid_cap: list[float | None] = Field(min_length=24, max_length=24)
