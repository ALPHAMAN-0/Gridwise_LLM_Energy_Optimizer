"""Runtime settings, read once from the environment.

Everything tunable lives here so no other module touches `os.environ`. The
loader is deliberately forgiving: a missing .env, a quoted key, a stray space
or a non-numeric timeout must never stop the service from starting, because a
service that is up with defaults still scores on /health and on the solver,
while one that crashed at import scores nothing.

Secrets are only ever held in the Settings object. Nothing here logs them.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Explicit path: load_dotenv() with no argument walks the call stack to find
# the file, which crashes under some launchers (frozen apps, `python -c`).
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)
DEFAULT_THINKING_LEVEL = "low"
DEFAULT_LLM_CALL_TIMEOUT_S = 7.0
DEFAULT_LLM_TOTAL_BUDGET_S = 14.0
DEFAULT_LLM_MAX_CONCURRENCY = 4
DEFAULT_CACHE_SIZE = 512
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"

_QUOTES = "\"'` \t\r\n"


@dataclass(frozen=True)
class Settings:
    gemini_api_keys: tuple[str, ...]
    gemini_models: tuple[str, ...]
    gemini_thinking_level: str
    llm_call_timeout_s: float
    llm_total_budget_s: float
    llm_max_concurrency: int
    cache_size: int
    port: int
    log_level: str

    def __repr__(self) -> str:
        # The default dataclass repr would print the keys into any log line or
        # traceback that happens to include a Settings object.
        return (
            f"Settings(gemini_api_keys=<{len(self.gemini_api_keys)} configured>, "
            f"gemini_models={self.gemini_models!r}, "
            f"gemini_thinking_level={self.gemini_thinking_level!r}, "
            f"llm_call_timeout_s={self.llm_call_timeout_s}, "
            f"llm_total_budget_s={self.llm_total_budget_s}, "
            f"llm_max_concurrency={self.llm_max_concurrency}, "
            f"cache_size={self.cache_size}, port={self.port}, "
            f"log_level={self.log_level!r})"
        )


def _clean(value: str | None) -> str:
    """Strip whitespace and any quotes a hand-edited .env left around a value."""
    return (value or "").strip(_QUOTES)


def _split(value: str | None) -> tuple[str, ...]:
    """Comma list -> cleaned, de-duplicated tuple, original order kept."""
    seen: dict[str, None] = {}
    for part in _clean(value).split(","):
        item = _clean(part)
        if item:
            seen.setdefault(item, None)
    return tuple(seen)


def _float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(_clean(os.environ.get(name)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value < minimum:
        return default
    return value


def _int(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(float(_clean(os.environ.get(name))))
    except (TypeError, ValueError, OverflowError):
        return default
    return value if value >= minimum else default


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        # override=False: a real environment variable (the deploy platform's
        # secret store) always beats the developer's local file.
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    except Exception:  # noqa: BLE001 - a broken .env must not stop startup
        pass


def _build() -> Settings:
    _load_dotenv()

    keys = _split(
        ",".join(
            [os.environ.get("GEMINI_API_KEYS") or "", os.environ.get("GEMINI_API_KEY") or ""]
        )
    )
    models = _split(os.environ.get("GEMINI_MODELS")) or DEFAULT_GEMINI_MODELS

    # Unset -> default; set but empty -> thinkingConfig is not sent at all.
    raw_thinking = os.environ.get("GEMINI_THINKING_LEVEL")
    thinking = DEFAULT_THINKING_LEVEL if raw_thinking is None else _clean(raw_thinking).lower()

    level = _clean(os.environ.get("LOG_LEVEL")).upper() or DEFAULT_LOG_LEVEL
    if not isinstance(logging.getLevelName(level), int):
        level = DEFAULT_LOG_LEVEL

    port = _int("PORT", DEFAULT_PORT, minimum=1)
    if port > 65535:
        port = DEFAULT_PORT

    return Settings(
        gemini_api_keys=keys,
        gemini_models=models,
        gemini_thinking_level=thinking,
        llm_call_timeout_s=_float("LLM_CALL_TIMEOUT_S", DEFAULT_LLM_CALL_TIMEOUT_S, minimum=0.5),
        llm_total_budget_s=_float("LLM_TOTAL_BUDGET_S", DEFAULT_LLM_TOTAL_BUDGET_S, minimum=0.5),
        llm_max_concurrency=_int("LLM_MAX_CONCURRENCY", DEFAULT_LLM_MAX_CONCURRENCY, minimum=1),
        cache_size=_int("CACHE_SIZE", DEFAULT_CACHE_SIZE, minimum=0),
        port=port,
        log_level=level,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Tests call `get_settings.cache_clear()` to reload."""
    return _build()
