"""interpret(): stubbed model, cache, re-ask and failure paths. No network.

Also covers the two modules interpret() stands on - the Gemini cascade in
app.llm (driven through httpx.MockTransport) and the env parsing in app.config.
pytest-asyncio is not installed, so coroutines are driven with asyncio.run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import logging
import time

import httpx
import pytest

from app import config, interpreter, llm
from app.config import Settings
from app.guardrails import UNVALIDATED_EXPLANATION
from app.interpreter import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    UNAVAILABLE_EXPLANATION,
    cache_clear,
    interpret,
)
from app.llm import LLMError
from app.schemas import DIRECTIVE_TYPES, OptimizeRequest

FAKE_KEYS = ("test-key-AAAA", "test-key-BBBB")


def settings(**overrides) -> Settings:
    base = dict(
        gemini_api_keys=FAKE_KEYS,
        gemini_models=("model-a", "model-b"),
        gemini_thinking_level="low",
        llm_call_timeout_s=7.0,
        llm_total_budget_s=14.0,
        llm_max_concurrency=4,
        cache_size=8,
        port=8000,
        log_level="INFO",
    )
    base.update(overrides)
    return Settings(**base)


def request(notes: list[str], capacity: float = 200.0) -> OptimizeRequest:
    return OptimizeRequest(
        scenario_id="t-1",
        operator_notes=notes,
        hours=[
            {"hour": h, "demand_kwh": 100, "solar_kwh": 20, "tariff_bdt_per_kwh": 8}
            for h in range(24)
        ],
        battery={
            "capacity_kwh": capacity,
            "initial_energy_kwh": capacity / 2,
            "minimum_energy_kwh": 20,
            "max_charge_kwh_per_hour": 50,
            "max_discharge_kwh_per_hour": 50,
        },
    )


def raw_item(note_index, directive_type, windows=(), **fields) -> dict:
    base = {
        "note_index": note_index,
        "directive_type": directive_type,
        "windows": [{"start_hour": s, "end_hour": e} for s, e in windows],
        "solar_percent": None,
        "solar_percent_is": None,
        "reserve_value": None,
        "reserve_unit": None,
        "max_grid_kwh": None,
        "explanation": f"reading of note {note_index}",
    }
    base.update(fields)
    return base


class StubLLM:
    """Stands in for llm.generate_json; replays queued answers and records calls."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: list[dict] = []

    async def __call__(self, system, user, schema, *, deadline=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "deadline": deadline})
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(interpreter, "get_settings", lambda: settings())
    monkeypatch.setattr(llm, "get_settings", lambda: settings())
    cache_clear()
    yield
    cache_clear()


def use(monkeypatch, stub: StubLLM) -> StubLLM:
    monkeypatch.setattr(llm, "generate_json", stub)
    return stub


# --------------------------------------------------------------------------
# interpret
# --------------------------------------------------------------------------


def test_happy_path_normalizes_and_orders(monkeypatch):
    stub = use(
        monkeypatch,
        StubLLM(
            {
                "items": [
                    raw_item(0, "solar_reduction", [(11, 14)], solar_percent=80, solar_percent_is="reduction"),
                    raw_item(1, "no_op"),
                    raw_item(
                        2, "minimum_battery_reserve", [(18, 21)],
                        reserve_value=50, reserve_unit="percent_of_capacity",
                    ),
                ]
            }
        ),
    )
    notes = ["solar down 80% 11-2", "club notices tomorrow", "keep half the battery 6-9 pm"]
    result = asyncio.run(interpret(request(notes, capacity=200)))

    assert [r.note_index for r in result] == [0, 1, 2]
    assert result[0].structured_adjustment == {"hours": [11, 12, 13], "factor": 0.2}
    assert result[0].applies is True
    assert result[1].directive_type == "no_op" and result[1].structured_adjustment is None
    assert result[2].structured_adjustment == {"hours": [18, 19, 20], "minimum_energy_kwh": 100.0}

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["system"] is SYSTEM_PROMPT and call["schema"] is RESPONSE_SCHEMA
    assert 13.0 < call["deadline"] - time.monotonic() <= 14.0
    payload = json.loads(call["user"].split("\n", 1)[1])
    # Capacity is withheld on purpose so the model cannot pre-multiply a percentage.
    assert "battery_capacity_kwh" not in payload
    assert [n["note_index"] for n in payload["notes"]] == [0, 1, 2]
    assert [n["text"] for n in payload["notes"]] == notes


def test_patching_the_interpreters_own_reference_also_works(monkeypatch):
    stub = StubLLM({"items": [raw_item(0, "no_charge_window", [(2, 5)])]})
    monkeypatch.setattr(interpreter, "generate_json", stub)
    result = asyncio.run(interpret(request(["charger isolated 2-5 am"])))
    assert result[0].structured_adjustment == {"hours": [2, 3, 4]}
    assert len(stub.calls) == 1


def test_llm_error_gives_all_no_op_and_is_not_cached(monkeypatch):
    stub = use(monkeypatch, StubLLM(LLMError("everything is down")))
    req = request(["a", "b"])
    result = asyncio.run(interpret(req))
    assert [r.note_index for r in result] == [0, 1]
    for entry in result:
        assert entry.applies is False and entry.directive_type == "no_op"
        assert entry.structured_adjustment is None
        assert entry.explanation == UNAVAILABLE_EXPLANATION
    assert len(stub.calls) == 1

    # Not cached: once the model recovers the same request is interpreted for real.
    stub.answers = [{"items": [raw_item(0, "no_op"), raw_item(1, "no_charge_window", [(1, 2)])]}]
    again = asyncio.run(interpret(req))
    assert len(stub.calls) == 2
    assert again[1].directive_type == "no_charge_window"


@pytest.mark.parametrize("failure", [RuntimeError("bug"), ValueError("bad"), KeyError("x")])
def test_any_exception_is_swallowed(monkeypatch, failure):
    use(monkeypatch, StubLLM(failure))
    result = asyncio.run(interpret(request(["a", "b", "c"])))
    assert len(result) == 3
    assert all(r.explanation == UNAVAILABLE_EXPLANATION for r in result)


@pytest.mark.parametrize("junk", [None, [], "text", {"items": "nope"}, {"wrong": []}, {"items": [1, 2]}])
def test_junk_model_output_never_raises(monkeypatch, junk):
    stub = use(monkeypatch, StubLLM(junk))
    result = asyncio.run(interpret(request(["a", "b"])))
    assert [r.note_index for r in result] == [0, 1]
    assert all(r.directive_type == "no_op" and r.applies is False for r in result)
    assert len(stub.calls) == 2  # it did try a re-ask


def test_cache_hit_does_not_call_the_model_again(monkeypatch):
    stub = use(monkeypatch, StubLLM({"items": [raw_item(0, "max_grid_window", [(18, 21)], max_grid_kwh=155)]}))
    req = request(["feeder limit 155 from 6 to 9 pm"])
    first = asyncio.run(interpret(req))
    second = asyncio.run(interpret(req))
    assert len(stub.calls) == 1
    assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    # Callers own what they get back; mutating it must not poison the cache.
    second[0].structured_adjustment["hours"].append(99)
    third = asyncio.run(interpret(req))
    assert third[0].structured_adjustment == {"hours": [18, 19, 20], "max_grid_kwh": 155.0}
    assert len(stub.calls) == 1


def test_cache_key_includes_notes_and_capacity(monkeypatch):
    answer = {
        "items": [
            raw_item(0, "minimum_battery_reserve", [(18, 20)], reserve_value=50, reserve_unit="percent_of_capacity")
        ]
    }
    stub = use(monkeypatch, StubLLM(answer))
    small = asyncio.run(interpret(request(["keep half"], capacity=200)))
    large = asyncio.run(interpret(request(["keep half"], capacity=300)))
    asyncio.run(interpret(request(["keep half, please"], capacity=300)))
    assert len(stub.calls) == 3
    assert small[0].structured_adjustment["minimum_energy_kwh"] == 100.0
    assert large[0].structured_adjustment["minimum_energy_kwh"] == 150.0


def test_cache_is_lru_capped_and_clearable(monkeypatch):
    monkeypatch.setattr(interpreter, "get_settings", lambda: settings(cache_size=2))
    stub = use(monkeypatch, StubLLM({"items": [raw_item(0, "no_op")]}))
    for text in ["one", "two", "three"]:
        asyncio.run(interpret(request([text])))
    assert len(stub.calls) == 3
    asyncio.run(interpret(request(["three"])))  # still cached
    assert len(stub.calls) == 3
    asyncio.run(interpret(request(["one"])))  # evicted
    assert len(stub.calls) == 4
    cache_clear()
    asyncio.run(interpret(request(["one"])))
    assert len(stub.calls) == 5


def test_reask_repairs_a_bad_first_answer(monkeypatch):
    bad = {"items": [raw_item(0, "solar_reduction", [(10, 12)], solar_percent=None), raw_item(1, "no_op")]}
    good = {
        "items": [
            raw_item(0, "solar_reduction", [(10, 12)], solar_percent=50, solar_percent_is="remaining"),
            raw_item(1, "no_op"),
        ]
    }
    stub = use(monkeypatch, StubLLM(bad, good))
    req = request(["about half the solar 10 to noon", "library hours next week"])
    result = asyncio.run(interpret(req))

    assert len(stub.calls) == 2
    assert result[0].structured_adjustment == {"hours": [10, 11], "factor": 0.5}
    assert stub.calls[1]["user"].startswith(stub.calls[0]["user"])
    assert "note 0: solar_reduction needs solar_percent" in stub.calls[1]["user"]
    assert stub.calls[1]["deadline"] == stub.calls[0]["deadline"]  # one shared budget

    asyncio.run(interpret(req))
    assert len(stub.calls) == 2  # the repaired answer was cached


def test_reask_merges_per_note_when_both_answers_are_flawed(monkeypatch):
    first = {
        "items": [
            raw_item(0, "no_charge_window", [(14, 16)]),
            raw_item(1, "max_grid_window", [(18, 21)]),  # cap missing
        ]
    }
    second = {
        "items": [
            raw_item(0, "no_charge_window", [(5, 5)]),  # re-ask broke note 0
            raw_item(1, "max_grid_window", [(18, 21)], max_grid_kwh=150),
        ]
    }
    stub = use(monkeypatch, StubLLM(first, second))
    result = asyncio.run(interpret(request(["a", "b"])))
    assert result[0].structured_adjustment == {"hours": [14, 15]}
    assert result[1].structured_adjustment == {"hours": [18, 19, 20], "max_grid_kwh": 150.0}
    assert len(stub.calls) == 2


def test_failed_reask_keeps_the_first_answer_uncached(monkeypatch):
    first = {"items": [raw_item(0, "no_charge_window", [(14, 16)]), raw_item(1, "max_grid_window", [(18, 21)])]}
    stub = use(monkeypatch, StubLLM(first, LLMError("quota")))
    req = request(["a", "b"])
    result = asyncio.run(interpret(req))
    assert result[0].structured_adjustment == {"hours": [14, 15]}
    assert result[1].directive_type == "no_op" and result[1].explanation == UNVALIDATED_EXPLANATION
    assert len(stub.calls) == 2

    stub.answers = [first]
    asyncio.run(interpret(req))
    assert len(stub.calls) >= 3  # nothing was cached


def test_no_reask_when_the_budget_is_nearly_spent(monkeypatch):
    monkeypatch.setattr(interpreter, "get_settings", lambda: settings(llm_total_budget_s=4.0))
    stub = use(monkeypatch, StubLLM({"items": [raw_item(0, "max_grid_window", [(18, 21)])]}))
    result = asyncio.run(interpret(request(["a"])))
    assert len(stub.calls) == 1
    assert result[0].directive_type == "no_op"


def test_mislabelled_indices_are_aligned_by_position(monkeypatch):
    use(
        monkeypatch,
        StubLLM({"items": [raw_item(1, "no_charge_window", [(2, 4)]), raw_item(2, "no_op")]}),
    )
    result = asyncio.run(interpret(request(["a", "b"])))
    assert result[0].directive_type == "no_charge_window"
    # The second item (labelled 2 by the model) lands on note 1, not on a safe no_op.
    assert result[1].note_index == 1 and result[1].directive_type == "no_op"
    assert result[1].explanation == "reading of note 2"


def test_reserve_above_capacity_is_neutralised(monkeypatch):
    stub = use(
        monkeypatch,
        StubLLM({"items": [raw_item(0, "minimum_battery_reserve", [(18, 20)], reserve_value=500, reserve_unit="kwh")]}),
    )
    result = asyncio.run(interpret(request(["keep 500 kWh"], capacity=200)))
    assert result[0].directive_type == "no_op" and result[0].applies is False
    assert len(stub.calls) == 2


# --------------------------------------------------------------------------
# prompt and schema
# --------------------------------------------------------------------------


def test_schema_uses_the_gemini_dialect():
    item = RESPONSE_SCHEMA["properties"]["items"]["items"]
    assert RESPONSE_SCHEMA["type"] == "OBJECT" and item["type"] == "OBJECT"
    assert item["properties"]["directive_type"]["enum"] == list(DIRECTIVE_TYPES)
    assert set(item["propertyOrdering"]) == set(item["properties"])
    for name in ("solar_percent", "solar_percent_is", "reserve_value", "reserve_unit", "max_grid_kwh"):
        assert item["properties"][name]["nullable"] is True

    def walk(node):
        if isinstance(node, dict):
            if "type" in node and isinstance(node["type"], str):
                assert node["type"] in {"OBJECT", "ARRAY", "STRING", "INTEGER", "NUMBER"}
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(RESPONSE_SCHEMA)
    json.dumps(RESPONSE_SCHEMA)


def test_prompt_names_every_type_and_embeds_valid_examples():
    for directive_type in DIRECTIVE_TYPES:
        assert directive_type in SYSTEM_PROMPT
    outputs = [line for line in SYSTEM_PROMPT.splitlines() if line.startswith('{"items"')]
    assert len(outputs) == len(interpreter.FEW_SHOT_EXAMPLES)
    shown = [item for line in outputs for item in json.loads(line)["items"]]
    assert 8 <= len(shown) <= 12
    assert {item["directive_type"] for item in shown} == set(DIRECTIVE_TYPES)


# --------------------------------------------------------------------------
# llm cascade (httpx.MockTransport - nothing leaves the process)
# --------------------------------------------------------------------------


def gemini_ok(obj) -> httpx.Response:
    body = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(obj)}]}}]}
    return httpx.Response(200, json=body)


def gemini_error(status: int, message: str = "nope") -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}})


class Upstream:
    def __init__(self, handler):
        self.handler = handler
        self.seen: list[dict] = []

    def __call__(self, http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        record = {
            "model": http_request.url.path.split("/models/")[1].split(":")[0],
            "key": http_request.headers.get("x-goog-api-key"),
            "url": str(http_request.url),
            "body": body,
            "thinking": body["generationConfig"].get("thinkingConfig"),
        }
        self.seen.append(record)
        return self.handler(record)


@pytest.fixture
def upstream(monkeypatch):
    def install(handler, **overrides) -> Upstream:
        up = Upstream(handler)
        monkeypatch.setattr(llm, "get_settings", lambda: settings(**overrides))
        monkeypatch.setattr(llm, "_new_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(up)))
        monkeypatch.setattr(llm, "_client", None)
        monkeypatch.setattr(llm, "_semaphore", None)
        monkeypatch.setattr(llm, "_no_thinking", set())
        monkeypatch.setattr(llm, "_missing_models", set())
        return up

    return install


def call_llm(**kwargs):
    async def run():
        try:
            return await llm.generate_json("SYS", "USER", {"type": "OBJECT"}, **kwargs)
        finally:
            await llm.shutdown()

    return asyncio.run(run())


def test_llm_request_shape_and_key_stays_out_of_the_url(upstream):
    up = upstream(lambda r: gemini_ok({"items": []}))
    assert call_llm() == {"items": []}
    sent = up.seen[0]
    assert sent["url"] == "https://generativelanguage.googleapis.com/v1beta/models/model-a:generateContent"
    assert sent["key"] in FAKE_KEYS
    assert all(key not in sent["url"] for key in FAKE_KEYS)
    assert sent["body"]["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert sent["body"]["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
    config_sent = sent["body"]["generationConfig"]
    assert config_sent["temperature"] == 0
    assert config_sent["responseMimeType"] == "application/json"
    assert config_sent["responseSchema"] == {"type": "OBJECT"}
    assert config_sent["thinkingConfig"] == {"thinkingLevel": "low"}


def test_llm_no_keys_raises_immediately(upstream):
    up = upstream(lambda r: gemini_ok({}), gemini_api_keys=())
    with pytest.raises(LLMError):
        call_llm()
    assert up.seen == []


@pytest.mark.parametrize("status", [429, 500, 503])
def test_llm_retryable_status_moves_to_next_key_then_next_model(upstream, status):
    up = upstream(lambda r: gemini_error(status) if r["model"] == "model-a" else gemini_ok({"ok": 1}))
    assert call_llm() == {"ok": 1}
    assert [s["model"] for s in up.seen] == ["model-a", "model-a", "model-b"]
    assert {s["key"] for s in up.seen[:2]} == set(FAKE_KEYS)


def test_llm_404_skips_straight_to_the_next_model_and_is_remembered(upstream):
    up = upstream(lambda r: gemini_error(404) if r["model"] == "model-a" else gemini_ok({"ok": 1}))

    async def twice():
        try:
            await llm.generate_json("S", "U", {})
            await llm.generate_json("S", "U", {})
        finally:
            await llm.shutdown()

    asyncio.run(twice())
    assert [s["model"] for s in up.seen] == ["model-a", "model-b", "model-b"]


def test_llm_thinking_rejection_retries_same_model_without_it(upstream):
    def handler(record):
        if record["thinking"] is not None:
            return gemini_error(400, "Thinking level is not supported for this model.")
        return gemini_ok({"ok": 1})

    up = upstream(handler)

    async def twice():
        try:
            first = await llm.generate_json("S", "U", {})
            await llm.generate_json("S", "U", {})
            return first
        finally:
            await llm.shutdown()

    assert asyncio.run(twice()) == {"ok": 1}
    assert [(s["model"], s["thinking"] is not None) for s in up.seen] == [
        ("model-a", True),
        ("model-a", False),
        ("model-a", False),  # remembered for the second call
    ]
    assert up.seen[0]["key"] == up.seen[1]["key"]


def test_llm_empty_thinking_level_omits_the_config(upstream):
    up = upstream(lambda r: gemini_ok({}), gemini_thinking_level="")
    call_llm()
    assert up.seen[0]["thinking"] is None


def test_llm_unusable_bodies_fall_through(upstream):
    responses = iter(
        [
            httpx.Response(200, json={"candidates": []}),
            httpx.Response(200, json={"candidates": [{"finishReason": "SAFETY", "content": {"parts": [{"text": "{}"}]}}]}),
            httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}),
            httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "[1, 2]"}]}}]}),
        ]
    )
    up = upstream(lambda r: next(responses))
    with pytest.raises(LLMError):
        call_llm()
    assert len(up.seen) == 4


def test_llm_concatenates_parts_and_skips_thoughts(upstream):
    body = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "pondering", "thought": True},
                        {"text": '{"items": '},
                        {"text": "[]}"},
                    ]
                },
            }
        ]
    }
    upstream(lambda r: httpx.Response(200, json=body))
    assert call_llm() == {"items": []}


def test_llm_timeouts_and_transport_errors_cascade(upstream):
    def handler(record):
        if record["model"] == "model-a":
            raise httpx.ReadTimeout("hung") if record["key"] == FAKE_KEYS[0] else httpx.ConnectError("down")
        return gemini_ok({"ok": 1})

    up = upstream(handler)
    assert call_llm() == {"ok": 1}
    assert len(up.seen) == 3


def test_llm_never_starts_a_call_past_the_deadline(upstream):
    up = upstream(lambda r: gemini_ok({}))
    with pytest.raises(LLMError):
        call_llm(deadline=time.monotonic() + 0.2)
    assert up.seen == []


def test_llm_hung_upstream_is_cut_off_at_the_deadline(monkeypatch):
    async def hang(_request):
        await asyncio.sleep(30)

    class Hanging(httpx.AsyncBaseTransport):
        async def handle_async_request(self, http_request):
            await hang(http_request)

    monkeypatch.setattr(llm, "get_settings", lambda: settings(gemini_api_keys=("k",), gemini_models=("m",)))
    monkeypatch.setattr(llm, "_new_client", lambda: httpx.AsyncClient(transport=Hanging()))
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_semaphore", None)

    started = time.monotonic()
    with pytest.raises(LLMError):
        call_llm(deadline=time.monotonic() + 1.3)
    assert time.monotonic() - started < 3.0


def test_llm_keys_rotate_across_calls(upstream):
    up = upstream(lambda r: gemini_ok({}))

    async def four():
        try:
            for _ in range(4):
                await llm.generate_json("S", "U", {})
        finally:
            await llm.shutdown()

    asyncio.run(four())
    assert {s["key"] for s in up.seen} == set(FAKE_KEYS)


def test_llm_logs_never_contain_a_key_or_url(upstream, caplog):
    upstream(lambda r: gemini_error(503, "model is currently experiencing high demand"))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMError) as raised:
            call_llm()
    text = caplog.text + str(raised.value)
    assert "status=503" in caplog.text and "key=0" in caplog.text and "key=1" in caplog.text
    for secret in (*FAKE_KEYS, "googleapis", "x-goog-api-key", "high demand"):
        assert secret not in text


def test_llm_works_without_startup_and_survives_a_new_event_loop(upstream):
    up = upstream(lambda r: gemini_ok({"n": 1}))
    assert asyncio.run(llm.generate_json("S", "U", {})) == {"n": 1}
    assert asyncio.run(llm.generate_json("S", "U", {})) == {"n": 1}  # fresh loop, old client dropped
    assert len(up.seen) == 2


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

ENV_NAMES = (
    "GEMINI_API_KEYS", "GEMINI_API_KEY", "GEMINI_MODELS", "GEMINI_THINKING_LEVEL",
    "LLM_CALL_TIMEOUT_S", "LLM_TOTAL_BUDGET_S", "LLM_MAX_CONCURRENCY", "CACHE_SIZE",
    "PORT", "LOG_LEVEL",
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_load_dotenv", lambda: None)  # never touch the real .env
    return monkeypatch


def test_config_defaults_with_nothing_configured(clean_env):
    built = config._build()
    assert built.gemini_api_keys == ()
    assert built.gemini_models == config.DEFAULT_GEMINI_MODELS
    assert built.gemini_thinking_level == "low"
    assert (built.llm_call_timeout_s, built.llm_total_budget_s) == (7.0, 18.0)
    assert (built.llm_max_concurrency, built.cache_size, built.port) == (4, 512, 8000)
    assert built.log_level == "INFO"


def test_config_merges_and_cleans_keys(clean_env):
    clean_env.setenv("GEMINI_API_KEYS", ' "k1" , k2,, k1 ,')
    clean_env.setenv("GEMINI_API_KEY", " 'k3' ")
    assert config._build().gemini_api_keys == ("k1", "k2", "k3")


def test_config_single_quoted_key_only(clean_env):
    clean_env.setenv("GEMINI_API_KEY", '"only-key"')
    assert config._build().gemini_api_keys == ("only-key",)


def test_config_bad_numbers_fall_back(clean_env):
    for name, value in {
        "LLM_CALL_TIMEOUT_S": "fast", "LLM_TOTAL_BUDGET_S": "nan", "LLM_MAX_CONCURRENCY": "0",
        "CACHE_SIZE": "-4", "PORT": "99999", "LOG_LEVEL": "chatty",
    }.items():
        clean_env.setenv(name, value)
    built = config._build()
    assert (built.llm_call_timeout_s, built.llm_total_budget_s) == (7.0, 18.0)
    assert (built.llm_max_concurrency, built.cache_size, built.port) == (4, 512, 8000)
    assert built.log_level == "INFO"


def test_config_overrides_and_empty_thinking_level(clean_env):
    clean_env.setenv("GEMINI_MODELS", " m1, m2 ,m1")
    clean_env.setenv("GEMINI_THINKING_LEVEL", "")
    clean_env.setenv("LLM_CALL_TIMEOUT_S", "5.5")
    clean_env.setenv("LOG_LEVEL", "debug")
    built = config._build()
    assert built.gemini_models == ("m1", "m2")
    assert built.gemini_thinking_level == ""
    assert built.llm_call_timeout_s == 5.5
    assert built.log_level == "DEBUG"


def test_config_is_frozen_cached_and_hides_keys(clean_env):
    clean_env.setenv("GEMINI_API_KEY", "super-secret-value")
    built = config._build()
    assert "super-secret-value" not in repr(built) and "super-secret-value" not in str(built)
    with pytest.raises(Exception):
        built.port = 1  # type: ignore[misc]

    # The cache is process-wide: clear it on the way out so the fake key set
    # above cannot leak into whatever test runs next.
    config.get_settings.cache_clear()
    try:
        assert config.get_settings() is config.get_settings()
        assert config.get_settings().gemini_api_keys == ("super-secret-value",)
    finally:
        config.get_settings.cache_clear()
