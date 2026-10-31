"""GridWise API service: routes, error mapping and startup only.

All behaviour lives in pipeline.py. This module's job is the HTTP contract:
exact endpoint names, 400/422/500 without stack traces, and a /health that
answers in constant time no matter what the model or the solver is doing.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import llm, optimizer, pipeline
from .config import get_settings
from .schemas import SEMANTIC_ERROR, OptimizeRequest, OptimizeResponse

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx logs every request line at INFO. The key travels in a header, not the
# URL, but there is no reason to let a transport library write to our logs.
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("gridwise")

# The judge fails a request at 30 s. The interpreter has its own, shorter budget;
# this outer limit only exists so that a hang anywhere still yields a valid 200.
_REQUEST_DEADLINE_S = 25.0


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Nothing here may raise or touch the network: uvicorn binds the port only
    # after startup finishes, and /health must answer within 60 s of boot even
    # with no API key and no working solver.
    try:
        await llm.startup()
    except Exception as exc:  # noqa: BLE001
        log.error("llm client startup failed: %s", type(exc).__name__)
    try:
        await asyncio.to_thread(optimizer.warm_up)
    except Exception as exc:  # noqa: BLE001
        log.error("solver warm-up crashed: %s", type(exc).__name__)
    if not settings.gemini_api_keys:
        log.warning("no GEMINI_API_KEYS configured: every note will be reported as no_op")
    yield
    try:
        await llm.shutdown()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(
    title="GridWise Energy Optimizer",
    description="LLM-assisted campus energy scheduling for BUP CSE Fest 2026.",
    version="1.0.0",
    lifespan=_lifespan,
)


# --------------------------------------------------------------------------
# Error handling: 400 / 422 / 500, never a stack trace, never the request body
# --------------------------------------------------------------------------


def _brief(errors: list[dict]) -> str:
    """One short human line. Never echo the request body back."""
    first = (errors or [{}])[0]
    location = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "body"
    return f"{location}: {first.get('msg', 'invalid value')}"[:200]


def _invalid(errors: list[dict]) -> JSONResponse:
    # Well-formed JSON describing a scenario that can never have a valid plan is
    # the spec's optional 422; everything structural is 400.
    semantic = bool(errors) and all(e.get("type") == SEMANTIC_ERROR for e in errors)
    return JSONResponse(
        status_code=422 if semantic else 400,
        content={
            "error": "unprocessable_scenario" if semantic else "invalid_request",
            "detail": _brief(errors),
        },
    )


@app.exception_handler(RequestValidationError)
async def _on_bad_request(_request: Request, exc: RequestValidationError):
    return _invalid(list(exc.errors()))


@app.exception_handler(Exception)
async def _on_unhandled(_request: Request, _exc: Exception):
    # Last-resort net. pipeline.run has its own fallbacks, so this should never
    # fire; it exists only so a bug cannot leak a traceback.
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error"})


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "gridwise", "endpoints": "GET /health, POST /optimize-energy"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(raw: Request):
    # The body is parsed here rather than by FastAPI so the route does not care
    # about Content-Type: a harness that posts valid JSON as text/plain, or with
    # no header at all, would otherwise be rejected before we ever saw it.
    body = await raw.body()
    try:
        request = OptimizeRequest.model_validate_json(body)
    except ValidationError as exc:
        return _invalid(exc.errors(include_url=False, include_input=False, include_context=False))

    try:
        return await asyncio.wait_for(pipeline.run(request), timeout=_REQUEST_DEADLINE_S)
    except asyncio.TimeoutError:
        log.error("scenario=%s overran %.0fs; returning emergency plan", request.scenario_id, _REQUEST_DEADLINE_S)
        return pipeline.emergency(request)
