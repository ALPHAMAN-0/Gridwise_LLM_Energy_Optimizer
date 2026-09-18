"""Gemini REST client: one JSON-in, JSON-out call with a model/key cascade.

Why hand-rolled REST instead of an SDK: the only thing this service needs from
the model is a single structured-output call, and owning the HTTP layer lets us
enforce the two properties the judge actually punishes - a hard wall-clock
budget (a hung upstream must never hang /optimize-energy) and graceful
degradation across models and keys when the free tier answers 429 or 503.

Secrets discipline: the API key travels only in the `x-goog-api-key` header,
never in the URL, and nothing in this module logs a URL, a header, a body or
the text of an httpx exception (which can embed the request URL). Log lines
carry the model name, the key's INDEX, the HTTP status, elapsed milliseconds
and the exception class name - nothing else.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import time
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger("gridwise.llm")

# httpx logs every request line (including the URL) at INFO. The URL holds no
# key, but the rule for this module is "no URLs in logs", so quieten it.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Three notes produce roughly 400 output tokens; the headroom is for "low"
# thinking tokens, which count against the same cap on thinking models.
_MAX_OUTPUT_TOKENS = 2048
_CONNECT_TIMEOUT_S = 3.0
# A call that has less than this left cannot plausibly finish; do not start it.
_MIN_CALL_S = 1.0

_BLOCKED_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "MALFORMED_FUNCTION_CALL",
        "OTHER",
    }
)


class LLMError(Exception):
    """Every model/key combination failed, or no call could be started."""


class _Skip(Exception):
    """Internal: this attempt failed; carries where the cascade should go next."""

    def __init__(self, reason: str, *, next_model: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.next_model = next_model


_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None

# Models that rejected thinkingConfig / answered 404. Learned at runtime so the
# wasted round trip is paid once per process, not once per request.
_no_thinking: set[str] = set()
_missing_models: set[str] = set()

# Rotates the first key tried so concurrent requests spread across keys.
_rotation = itertools.count()


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def _new_client() -> httpx.AsyncClient:
    settings = get_settings()
    pool = max(4, settings.llm_max_concurrency * 2)
    return httpx.AsyncClient(
        timeout=httpx.Timeout(settings.llm_call_timeout_s, connect=_CONNECT_TIMEOUT_S),
        limits=httpx.Limits(max_connections=pool, max_keepalive_connections=pool),
        headers={"content-type": "application/json"},
        follow_redirects=False,
    )


def _get_client() -> httpx.AsyncClient:
    """The shared client, created lazily and re-created if the loop changed.

    An AsyncClient is bound to the event loop that first used it. Production
    has one loop for the life of the process; tests run many, so a client from
    a dead loop is dropped rather than reused.
    """
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = _new_client()
        _client_loop = loop
    return _client


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(get_settings().llm_max_concurrency)
        _semaphore_loop = loop
    return _semaphore


async def startup() -> None:
    """Create the shared client up front so the first request does not pay for it."""
    _get_client()
    _get_semaphore()


async def shutdown() -> None:
    global _client, _client_loop
    client, _client, _client_loop = _client, None, None
    if client is not None and not client.is_closed:
        try:
            await client.aclose()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.debug("client close failed: %s", type(exc).__name__)


# --------------------------------------------------------------------------
# Request / response shaping
# --------------------------------------------------------------------------


def _thinking_config(model: str, level: str) -> dict[str, Any] | None:
    if not level or model in _no_thinking:
        return None
    if model.startswith("gemini-2."):
        # The 2.x family predates thinkingLevel and takes a token budget; zero
        # is the closest match to "low" and is what flash-lite does by default.
        # Pro variants cannot disable thinking, so they get no config at all.
        return None if "pro" in model else {"thinkingBudget": 0}
    return {"thinkingLevel": level}


def _body(system: str, user: str, schema: dict, thinking: dict[str, Any] | None) -> dict[str, Any]:
    generation: dict[str, Any] = {
        "temperature": 0,
        "maxOutputTokens": _MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "responseSchema": schema,
    }
    if thinking is not None:
        generation["thinkingConfig"] = thinking
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": generation,
    }


def _error_message(response: httpx.Response) -> str:
    """The API's error message, lower-cased, for classification only (never logged)."""
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return str(payload["error"].get("message") or "").lower()
    return ""


def _extract(payload: Any) -> dict:
    """Pull the JSON object out of a generateContent response, or raise _Skip."""
    if not isinstance(payload, dict):
        raise _Skip("response_not_object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise _Skip("no_candidates")
    candidate = candidates[0]
    if str(candidate.get("finishReason") or "").upper() in _BLOCKED_FINISH_REASONS:
        raise _Skip("blocked_finish_reason")

    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise _Skip("no_parts")
    text = "".join(
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    ).strip()
    if not text:
        raise _Skip("empty_text")

    # JSON mode should never fence its output, but a fence costs nothing to strip.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        raise _Skip("invalid_json") from None
    if not isinstance(parsed, dict):
        raise _Skip("json_not_object")
    return parsed


async def _attempt(
    client: httpx.AsyncClient,
    *,
    model: str,
    key: str,
    key_index: int,
    system: str,
    user: str,
    schema: dict,
    timeout_s: float,
    allow_thinking_retry: bool = True,
) -> dict:
    settings = get_settings()
    thinking = _thinking_config(model, settings.gemini_thinking_level)
    started = time.monotonic()

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        # wait_for bounds the WHOLE exchange. httpx timeouts are per socket
        # operation, so a response that trickles in would otherwise outlive them.
        response = await asyncio.wait_for(
            client.post(
                _ENDPOINT.format(model=model),
                headers={"x-goog-api-key": key},
                json=_body(system, user, schema, thinking),
                timeout=httpx.Timeout(timeout_s, connect=min(_CONNECT_TIMEOUT_S, timeout_s)),
            ),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        log.warning(
            "model=%s key=%d timeout after %d ms (%s)",
            model, key_index, elapsed_ms(), type(exc).__name__,
        )
        raise _Skip("timeout") from None
    except Exception as exc:  # noqa: BLE001 - any transport failure means "try the next one"
        log.warning(
            "model=%s key=%d transport error after %d ms (%s)",
            model, key_index, elapsed_ms(), type(exc).__name__,
        )
        raise _Skip("transport_error") from None

    status = response.status_code
    if status != 200:
        log.warning("model=%s key=%d status=%d after %d ms", model, key_index, status, elapsed_ms())
        if status == 400 and thinking is not None and "thinking" in _error_message(response):
            _no_thinking.add(model)
            remaining = timeout_s - (time.monotonic() - started)
            if allow_thinking_retry and remaining >= _MIN_CALL_S:
                return await _attempt(
                    client,
                    model=model,
                    key=key,
                    key_index=key_index,
                    system=system,
                    user=user,
                    schema=schema,
                    timeout_s=remaining,
                    allow_thinking_retry=False,
                )
            raise _Skip("thinking_unsupported")
        if status == 404:
            # Unknown model: no other key will find it either.
            _missing_models.add(model)
            raise _Skip("model_not_found", next_model=True)
        raise _Skip(f"http_{status}")

    try:
        payload = response.json()
    except ValueError:
        log.warning("model=%s key=%d status=200 unreadable body after %d ms", model, key_index, elapsed_ms())
        raise _Skip("unreadable_body") from None

    try:
        result = _extract(payload)
    except _Skip as skip:
        log.warning(
            "model=%s key=%d status=200 unusable (%s) after %d ms",
            model, key_index, skip.reason, elapsed_ms(),
        )
        raise
    log.info("model=%s key=%d status=200 ok in %d ms", model, key_index, elapsed_ms())
    return result


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _call_timeout(per_call_s: float, left_s: float, *, last_attempt: bool) -> float:
    if last_attempt and math.isfinite(left_s):
        return left_s
    return min(per_call_s, left_s)


async def generate_json(
    system: str,
    user: str,
    schema: dict,
    *,
    deadline: float | None = None,
) -> dict:
    """Return the model's JSON object, trying every model and key in turn.

    `deadline` is a `time.monotonic()` value. No call is started that could not
    finish before it, and each call's timeout is clamped to what remains.
    Raises LLMError when nothing usable came back.
    """
    settings = get_settings()
    keys = settings.gemini_api_keys
    if not keys:
        raise LLMError("no Gemini API key configured")
    if not settings.gemini_models:
        raise LLMError("no Gemini model configured")

    def remaining() -> float:
        return float("inf") if deadline is None else deadline - time.monotonic()

    # Skip models already known to 404 - unless that would leave nothing to try.
    models = [m for m in settings.gemini_models if m not in _missing_models]
    models = models or list(settings.gemini_models)

    first_key = next(_rotation) % len(keys)
    semaphore = _get_semaphore()

    if remaining() < _MIN_CALL_S:
        raise LLMError("no time budget left for a model call")
    try:
        # Queueing behind other requests spends the same budget as the call.
        await asyncio.wait_for(
            semaphore.acquire(), timeout=None if deadline is None else max(0.0, remaining())
        )
    except asyncio.TimeoutError:
        raise LLMError("timed out waiting for a model slot") from None

    last_reason = "no attempt made"
    try:
        client = _get_client()
        for position, model in enumerate(models):
            # Nothing comes after the last model, so holding back budget for a
            # later attempt would only turn a slow answer into no answer.
            is_last_model = position == len(models) - 1
            for offset in range(len(keys)):
                left = remaining()
                if left < _MIN_CALL_S:
                    raise LLMError(f"time budget exhausted (last failure: {last_reason})")
                key_index = (first_key + offset) % len(keys)
                try:
                    return await _attempt(
                        client,
                        model=model,
                        key=keys[key_index],
                        key_index=key_index,
                        system=system,
                        user=user,
                        schema=schema,
                        timeout_s=_call_timeout(
                            settings.llm_call_timeout_s,
                            left,
                            last_attempt=is_last_model and offset == len(keys) - 1,
                        ),
                    )
                except _Skip as skip:
                    last_reason = f"{model}: {skip.reason}"
                    if skip.next_model:
                        break
    finally:
        semaphore.release()

    raise LLMError(f"all models and keys failed (last failure: {last_reason})")
