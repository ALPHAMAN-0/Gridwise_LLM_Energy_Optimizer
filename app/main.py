"""GridWise API service.

Checkpoint A: /health plus a fallback-only /optimize-energy, so the contract
and the validator can be exercised end to end before the LLM and the solver
land. Steps 4-8 replace the interpretation and planning stages in place.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .directives import build_constraints
from .fallback import build_fallback
from .schemas import (
    DirectiveInterpretation,
    OptimizeRequest,
    OptimizeResponse,
)
from .validator import totals, validate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise Energy Optimizer",
    description="LLM-assisted campus energy scheduling for BUP CSE Fest 2026.",
    version="0.1.0",
)


# --------------------------------------------------------------------------
# Error handling: 400 for bad input, never a stack trace
# --------------------------------------------------------------------------


def _brief(exc: RequestValidationError) -> str:
    """One short human line. Never echo the request body back."""
    first = (exc.errors() or [{}])[0]
    location = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
    return f"{location}: {first.get('msg', 'invalid value')}"[:200]


@app.exception_handler(RequestValidationError)
async def _on_bad_request(_request: Request, exc: RequestValidationError):
    # FastAPI raises this for malformed JSON as well as schema failures, so a
    # single handler covers both cases the spec calls out as 400.
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "detail": _brief(exc)},
    )


@app.exception_handler(Exception)
async def _on_unhandled(_request: Request, _exc: Exception):
    # Last-resort net. The route body has its own fallback path, so this
    # should never fire; it exists only so a bug cannot leak a traceback.
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: OptimizeRequest) -> OptimizeResponse:
    # Step 6 replaces this with the LLM interpreter; until then every note is
    # a no_op, which is the same safe answer the service falls back to when
    # the model is unreachable.
    interpretation = [
        DirectiveInterpretation(
            note_index=i,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="Interpreter not yet wired; treated as not affecting today's schedule.",
        )
        for i in range(len(request.operator_notes))
    ]

    constraints = build_constraints(request, interpretation)

    # Step 4 puts the LP in front of this; the fallback stays as the last tier.
    plan = build_fallback(request, constraints.effective_solar)

    ok, errors = validate(plan, request, constraints)
    if not ok:
        log.warning("plan failed validation with %d error(s): %s", len(errors), errors[:3])

    total_grid, total_cost, peak_grid = totals(plan, request)

    return OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=(
            "Solar is used first each hour and the grid covers the remainder; "
            "the battery holds its starting charge, so the day ends where it began."
        ),
    )
