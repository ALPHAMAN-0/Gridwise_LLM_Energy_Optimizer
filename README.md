# GridWise LLM Energy Optimizer

GridWise is a FastAPI service for the BUP CSE Fest 2026 preliminary round. It takes a 24-hour campus
forecast (demand, solar, tariff), a battery description, and one to three free-text operator notes.
A Google Gemini model reads the notes and extracts the operational directives they contain; deterministic
code normalizes and guards that output; a linear program then produces the minimum-cost 24-hour
grid/solar/battery schedule that honours every directive; and an independent validator replays the plan
before it is returned. The service always answers with a valid plan, even when the LLM is unreachable.

| | |
|---|---|
| Live base URL | `<PUBLIC_BASE_URL>` |
| Health check | `<PUBLIC_BASE_URL>/health` |
| Optimize endpoint | `POST <PUBLIC_BASE_URL>/optimize-energy` |
| Source | https://github.com/ALPHAMAN-0/Gridwise_LLM_Energy_Optimizer |
| Container image | `ghcr.io/alphaman-0/gridwise-llm-energy-optimizer:v1.0.0` |

Measured latency: <to be filled from scripts/run_samples.py>

Contents: [Architecture](#architecture) | [Why this split](#why-this-split) | [Guardrails](#guardrails) |
[Optimization method](#optimization-method) | [Reliability](#reliability) | [Quickstart](#quickstart-local) |
[Docker](#run-with-docker) | [Configuration](#configuration) | [API reference](#api-reference) |
[Testing](#testing) | [Deployment](#deployment-notes) | [Secrets](#secret-handling) |
[Limitations](#known-limitations-and-assumptions) | [Credits](#credits)

---

## Architecture

A fixed pipeline, not an autonomous agent: every request walks the same stages in the same order, and
every stage has a deterministic exit.

```
POST /optimize-energy
        |
        v
+----------------------------------+
| 1. Request validation            |  schemas.py      Pydantic v2: 24 unique hours, 1-3 notes,
|                                  |                  no NaN/Infinity          -> 400 / 422 on failure
+----------------------------------+
        |
        v
+----------------------------------+
| 2. LLM interpreter               |  interpreter.py  ONE Gemini call for ALL notes, JSON-schema
|    (one call, all notes)         |  llm.py          structured output, temperature 0
+----------------------------------+
        |   intermediate form: clock window, percent + remaining/reduction, value + unit
        v
+----------------------------------+
| 3. Deterministic normalization   |  interpreter.py  end-exclusive hour expansion, factor arithmetic,
|                                  |                  percent-of-capacity -> kWh
+----------------------------------+
        |
        v
+----------------------------------+
| 4. Guardrails                    |  guardrails.py   type whitelist, ranges, exact key sets;
|                                  |                  one re-ask, then demote that note to no_op
+----------------------------------+
        |
        v
+----------------------------------+
| 5. Constraint arrays             |  directives.py   effective_solar, floor, charge_allowed,
|                                  |                  discharge_allowed, grid_cap (24 values each)
+----------------------------------+
        |
        v
+----------------------------------+
| 6. LP optimizer                  |  optimizer.py    PuLP + bundled CBC, in-process HiGHS fallback
+----------------------------------+
        |
        v
+----------------------------------+
| 7. Independent replay validator  |  validator.py    second implementation of the rules, imports
|                                  |                  nothing from the optimizer
+----------------------------------+
        |   only if the strict LP is infeasible
        v
+----------------------------------+
| 8. Relaxation ladder             |  pipeline.py     elastic LP: penalized slack on grid caps and
|                                  |  optimizer.py    reserves, physical rules stay hard, replayed again
+----------------------------------+
        |   only if no solver tier produced a plan
        v
+----------------------------------+
| 9. Always-valid fallback plan    |  fallback.py     solar first, then grid, battery idle
+----------------------------------+
        |
        v
HTTP 200  (totals recomputed from the returned hourly_plan, never from the solver objective)
```

### File map

| File | Responsibility |
|---|---|
| `app/schemas.py` | Pydantic v2 request/response models, directive type whitelist, exact `structured_adjustment` key sets, judge tolerance (0.01). |
| `app/config.py` | Reads every environment variable once (keys, model cascade, timeouts, cache size, port, log level). |
| `app/llm.py` | Gemini REST client (`generateContent`) over httpx: structured output, model cascade, key rotation, per-call timeout, global budget. |
| `app/interpreter.py` | Prompt + response schema, the single LLM call for all notes, deterministic normalization of the intermediate form, the one re-ask, and the in-memory cache of validated interpretations. |
| `app/guardrails.py` | Validates every interpretation entry; triggers one re-ask; demotes anything still invalid to `no_op`. |
| `app/directives.py` | Folds the validated directives into five 24-long constraint arrays. Pure arithmetic. |
| `app/optimizer.py` | Builds and solves the LP, picks the solver, post-processes the solution into `hourly_plan` rows. |
| `app/validator.py` | Independent hour-by-hour replay of any plan against the rules; recomputes the three totals. |
| `app/fallback.py` | The plan that cannot fail: solar first, grid for the rest, battery never moves. |
| `app/pipeline.py` | Orchestrates stages 2-9: the planning tiers (`lp` -> `elastic` -> `fallback`), totals, and `plan_summary`. |
| `app/main.py` | Routes, startup, and error mapping only (`/health`, `/optimize-energy`, 400/422/500, the 25 s request deadline). |

**The LLM is in the interpretation path.** Every directive in `directive_interpretation` originates from
what the Gemini model extracted from the note text. There is no keyword matcher, regex classifier, or
lookup table of known note wordings deciding the directive type or its numbers. Code only *normalizes*
what the model extracted (clock times to hour lists, percentages to factors, percentages of capacity to
kWh) and *rejects* output that is structurally impossible.

---

## Why this split

LLMs are good at reading paraphrased language and bad at small exact arithmetic. So the model is asked
for an **intermediate form** that stays close to the words in the note, and code does the arithmetic.

| The note says | The LLM extracts | Code computes |
|---|---|---|
| "from 1 PM to 3 PM" | start 13:00, end 15:00 | `hours: [13, 14]` (start inclusive, end exclusive) |
| "an 80% reduction in rooftop solar" | percent 80, kind `reduction` | `factor: 0.2` (fraction remaining) |
| "treated as roughly 25% of the forecast" | percent 25, kind `remaining` | `factor: 0.25` |
| "keep at least 50% of the battery capacity" (capacity 200 kWh) | value 50, unit percent of capacity | `minimum_energy_kwh: 100` |
| "must not exceed 155 kWh in any hour" | value 155, unit kWh | `max_grid_kwh: 155` |
| "the library is extending book-return hours next week" | not about today's schedule | `no_op`, `applies: false`, `structured_adjustment: null` |

This removes the two classic error classes of LLM-only interpretation:

1. **Off-by-one windows.** Models frequently include the end hour (`[13, 14, 15]`). The model never
   writes an hour list here; it reports the two clock times and code expands them end-exclusive.
2. **Inverted or mis-scaled numbers.** "80% reduction" becoming `factor: 0.8`, or "50% of capacity"
   becoming `50` kWh. The model reports the number and what kind of number it is; code applies
   `1 - p/100`, `p/100`, or `p/100 * capacity_kwh`.

All notes go in **one** call: fewer round trips inside the 30 s limit, and the model sees distractor
notes next to real ones, which makes `no_op` decisions more stable.

---

## Guardrails

Applied to every entry after normalization, before anything reaches the optimizer:

- `directive_type` must be one of the six allowed values: `solar_reduction`, `minimum_battery_reserve`,
  `no_charge_window`, `no_discharge_window`, `max_grid_window`, `no_op`. Nothing else is ever emitted.
- Exactly one entry per operator note, in `note_index` order `0..N-1`. An out-of-range index is
  discarded, a duplicate is ignored (first valid entry wins), and a missing note gets a `no_op`.
- `hours` is a non-empty list of integers in `0..23` (booleans and numeric strings are rejected),
  emitted unique and ascending.
- `factor` is finite and within `[0, 1]`.
- `minimum_energy_kwh` is finite and within `[0, capacity_kwh]`.
- `max_grid_kwh` is finite and non-negative.
- `applies` / null semantics are forced, not trusted: `no_op` always has `applies: false` and
  `structured_adjustment: null`; every other type always has `applies: true` and a non-null adjustment.
- `explanation` is always a non-empty single-line string, at most 300 characters.
- The model output is treated as hostile input: it may not be a list, entries may not be objects, and
  numbers may be strings, `NaN`, or `Infinity`. The gate itself never raises.
- Exact key sets per type, no extras and none missing:

| `directive_type` | `structured_adjustment` keys |
|---|---|
| `solar_reduction` | `hours`, `factor` |
| `minimum_battery_reserve` | `hours`, `minimum_energy_kwh` |
| `no_charge_window` | `hours` |
| `no_discharge_window` | `hours` |
| `max_grid_window` | `hours`, `max_grid_kwh` |
| `no_op` | `null` |

**On invalid model output:** the interpreter re-asks the model once, telling it what was wrong. If an
entry is still invalid, *that note alone* is demoted to `no_op` with an explanation saying so. The
request never crashes and the service never invents a directive type.

---

## Optimization method

A linear program over 24 hours, 5 continuous variables per hour (120 variables).

**Variables** (per hour `h`): `grid[h]`, `solar_used[h]`, `charge[h]`, `discharge[h]`, `energy[h]`
(battery energy at the end of hour `h`).

**Constraints**

| Rule | Form |
|---|---|
| Energy balance | `grid[h] + solar_used[h] + discharge[h] = demand[h] + charge[h]` |
| Battery state recursion | `energy[h] = energy[h-1] + charge[h] - discharge[h]`, with `energy[-1] = initial_energy_kwh` |
| Battery bounds and floors | `floor[h] <= energy[h] <= capacity_kwh`, where `floor[h] = max(minimum_energy_kwh, any active reserve)` |
| Solar availability | `0 <= solar_used[h] <= effective_solar[h]` (`solar_kwh[h] * factor` inside a reduction window) |
| Rate limits | `0 <= charge[h] <= max_charge_kwh_per_hour`, `0 <= discharge[h] <= max_discharge_kwh_per_hour` |
| Windows | `charge[h] = 0` in a `no_charge_window`; `discharge[h] = 0` in a `no_discharge_window` |
| Grid caps | `0 <= grid[h] <= max_grid_kwh` in a `max_grid_window` |
| End-of-day neutrality | `energy[23] = initial_energy_kwh` |

**Objective:** minimize `sum over h of grid[h] * tariff_bdt_per_kwh[h]`. A tie-break term of `1e-6` per
kWh of battery throughput prefers the schedule that cycles the battery least; its effect on cost is
orders of magnitude below the 0.01 tolerance.

**Solver:** PuLP 3.3.2 with its bundled CBC binary. If CBC cannot run (it is a subprocess and needs a
writable temp directory), the same model is solved in-process by HiGHS (`highspy`). The working solver
is picked once, by a warm-up solve at startup, so the first real request does not pay for solver
discovery. Each solve has a 5 s time limit.

**Post-solve**

1. Only the *net* battery flow per hour is taken from the solver. Simultaneous charge and discharge in
   one hour (a cost-neutral LP degeneracy) is netted into a single `charge`, `discharge`, or `idle`
   action with a non-negative `battery_kwh`.
2. Values are rounded to 6 decimal places. Any rounding residue on end-of-day neutrality (at most
   `1e-3` kWh) is absorbed by the last active battery hour.
3. Battery energy, `solar_used_kwh`, and `grid_kwh` are then derived in closed form from that flow:
   `grid = demand + charge - discharge - solar_used`. The balance equation and the battery recursion
   therefore hold exactly in the emitted numbers instead of approximately across independently rounded
   solver variables.
4. The plan is replayed by `validator.py`, which shares no code with the optimizer.
5. `total_grid_kwh`, `total_cost_bdt`, and `peak_grid_kwh` are recomputed from the returned
   `hourly_plan`, which is exactly what the judge does.

**When the strict model is infeasible** the optimizer solves an *elastic* version: grid caps and reserves
above the battery's own `minimum_energy_kwh` get slack variables priced far above any tariff, so the
plan violates as few kWh as possible. Physical limits (balance, capacity, base minimum, rates, windows,
neutrality) stay hard, and the elastic plan is replayed against them before it is returned. Directives
are relaxed by the smallest amount, never silently dropped: an under-constrained plan is what fails a
ground-truth replay, while an over-constrained one only costs a little money.

**Result:** the LP reproduces the reference optimal `total_cost_bdt` exactly on all 10 public sample
cases. The hour-by-hour schedule may differ from the reference where several schedules share the same
optimal cost; the sample pack states that equivalent optimal schedules are accepted.

---

## Reliability

| Situation | Behaviour |
|---|---|
| Judge limit is 30 s per request | LLM work is capped at **14 s total** (`LLM_TOTAL_BUDGET_S`) and **7 s per call** (`LLM_CALL_TIMEOUT_S`), leaving more than half the limit for solving, validation, and network. |
| Model overloaded, 429, 5xx, or timeout | Model cascade (`GEMINI_MODELS`, in order) crossed with key rotation (`GEMINI_API_KEYS`), all inside the global budget. |
| Same scenario sent again | In-memory cache of *validated* interpretations (`CACHE_SIZE` entries): no second LLM call. Failed interpretations are never cached. |
| LLM fully down, or no key configured | **HTTP 200**. Every note is reported as `no_op` with an explanation saying the interpreter was unavailable, and a valid optimized plan is still returned. |
| Model returns invalid structure | One re-ask, then per-note demotion to `no_op` (see [Guardrails](#guardrails)). |
| LP infeasible (for example a hallucinated impossible directive) | Relaxation ladder: strict LP -> elastic LP with penalized slack on grid caps and reserves (fewest kWh violated) -> fallback plan. The reported interpretation is unchanged. |
| CBC cannot execute in the container | HiGHS solves the same model in-process. The solver is chosen by a warm-up solve at startup and the result is logged. |
| Every solver tier fails | Always-valid fallback plan: solar first, grid for the remainder, battery idle, so neutrality holds by construction. |
| Anything hangs | An outer 25 s request deadline returns the fallback plan with HTTP 200 before the judge's 30 s limit. |
| Malformed JSON or structurally invalid input | **400** `invalid_request` with a one-line reason. The request body is never echoed back. |
| A battery state that can never be valid (`minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh` does not hold) | **422** `unprocessable_scenario` with a one-line reason. |
| Unexpected bug | **500** `{"error": "internal_error"}`. No stack trace leaves the process. |
| Startup | Nothing at startup touches the network, and a missing key or dead solver cannot stop `/health` from answering. |

---

## Quickstart (local)

Requires Python 3.13 and a Gemini API key from https://aistudio.google.com/apikey (the free tier works).

```bash
git clone https://github.com/ALPHAMAN-0/Gridwise_LLM_Energy_Optimizer.git
cd Gridwise_LLM_Energy_Optimizer

python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and set GEMINI_API_KEYS=<your-key>

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second terminal, from the repository root:

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

```bash
python -c "import json; print(json.dumps(json.load(open('ProblemSet/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'))['cases'][0]['input']))" > sample.json

curl -s -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample.json
```

Change `['cases'][0]` to any index `0..9` for the other public samples. Interactive API docs are served
at http://localhost:8000/docs.

---

## Run with Docker

The published image is built and smoke-tested by GitHub Actions: `linux/amd64`, runs as a non-root user,
binds `0.0.0.0`, honours `PORT` (default `8000`), and has no secrets baked in.

```bash
docker pull ghcr.io/alphaman-0/gridwise-llm-energy-optimizer:v1.0.0

docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEYS=<your-key> \
  ghcr.io/alphaman-0/gridwise-llm-energy-optimizer:v1.0.0
```

Then use the same two `curl` commands as in the quickstart.

Build it yourself:

```bash
docker build --platform linux/amd64 -t gridwise-llm-energy-optimizer:local .

docker run --rm -p 8000:8000 --env-file .env gridwise-llm-energy-optimizer:local
```

Docker's `--env-file` parser is stricter than python-dotenv: it keeps quotes as part of the value and
rejects spaces around `=`. Write `NAME=value` with no quotes and no spaces when using it, or pass
`-e GEMINI_API_KEYS=<your-key>` instead.

A different port: `docker run --rm -e PORT=9000 -p 9000:9000 -e GEMINI_API_KEYS=<your-key> ghcr.io/alphaman-0/gridwise-llm-energy-optimizer:v1.0.0`

The base image is Debian slim on purpose: the CBC binary bundled with PuLP is linked against glibc and
does not run on Alpine. The Dockerfile runs a CBC smoke solve at build time, so a broken solver fails
the build instead of the first request.

---

## Configuration

All configuration is by environment variable, read once in `app/config.py`. Locally, `.env` in the
repository root is loaded with python-dotenv (quotes and spaces around `=` are tolerated). A real
environment variable always wins over `.env`. In Docker and on Render, set real environment variables.
An unparsable or out-of-range value falls back to its default instead of stopping startup.

| Name | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEYS` | none | Comma-separated list of Gemini API keys. Preferred. Keys are rotated on 429, 5xx, and timeout. Keys from the same Google Cloud project share one quota, so extra keys only help if they come from different projects. One key is enough. |
| `GEMINI_API_KEY` | none | Single-key name, also honoured. If both are set the lists are merged and de-duplicated. |
| `GEMINI_MODELS` | `gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-3.1-flash-lite` | Comma-separated Gemini model ids, tried left to right when a model errors or is rate limited. |
| `GEMINI_THINKING_LEVEL` | `low` | Thinking level requested from the model. Set to an empty string to send no thinking config at all. |
| `LLM_CALL_TIMEOUT_S` | `7` | Timeout in seconds for a single Gemini HTTP call. |
| `LLM_TOTAL_BUDGET_S` | `18` | Wall-clock budget in seconds for all LLM attempts within one request (cascade, rotation, and re-ask combined). |
| `LLM_MAX_CONCURRENCY` | `4` | Maximum simultaneous outbound LLM calls across all in-flight requests. |
| `CACHE_SIZE` | `512` | Maximum entries in the in-memory cache of validated interpretations. |
| `PORT` | `8000` | Port the server binds inside the container. Render injects its own value. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

**With no key set**, the service still starts and `/health` returns ok. `POST /optimize-energy` returns
HTTP 200 with a valid optimized plan, and every note is reported as `no_op` with an explanation that the
interpreter was unavailable. Useful for checking the solver path offline; not useful for scoring.

---

## API reference

### `GET /health`

Returns `200` with `{"status":"ok"}`. No dependencies are touched, so it is safe for keep-alive pings.

### `POST /optimize-energy`

`Content-Type: application/json`

**Request body**

| Field | Type | Rules |
|---|---|---|
| `scenario_id` | string | Echoed back unchanged. |
| `operator_notes` | string[] | 1 to 3 free-text notes. |
| `hours` | object[] | Exactly 24 entries, each hour `0..23` exactly once (any order). |
| `hours[].hour` | integer | `0..23`. |
| `hours[].demand_kwh` | number | `>= 0`, finite. |
| `hours[].solar_kwh` | number | `>= 0`, finite. Forecast before any directive. |
| `hours[].tariff_bdt_per_kwh` | number | Finite. |
| `battery.capacity_kwh` | number | `>= 0`. |
| `battery.initial_energy_kwh` | number | `>= 0`. Also the required energy at the end of hour 23. |
| `battery.minimum_energy_kwh` | number | `>= 0`. Floor at every hour. |
| `battery.max_charge_kwh_per_hour` | number | `>= 0`. |
| `battery.max_discharge_kwh_per_hour` | number | `>= 0`. |

Unknown extra fields are ignored. `NaN` and `Infinity` are rejected.

**Response body (200)**

| Field | Type | Meaning |
|---|---|---|
| `scenario_id` | string | Same as the request. |
| `directive_interpretation` | object[] | One entry per note, in note order. |
| `directive_interpretation[].note_index` | integer | `0..N-1`. |
| `directive_interpretation[].applies` | boolean | `false` only for `no_op`. |
| `directive_interpretation[].directive_type` | string | One of the six allowed types. |
| `directive_interpretation[].structured_adjustment` | object or null | Exact key set for the type (see [Guardrails](#guardrails)); `null` for `no_op`. |
| `directive_interpretation[].explanation` | string | One sentence, human readable. |
| `hourly_plan` | object[] | Exactly 24 entries, hours `0..23` ascending. |
| `hourly_plan[].hour` | integer | `0..23`. |
| `hourly_plan[].grid_kwh` | number | Grid import this hour, `>= 0`. |
| `hourly_plan[].solar_used_kwh` | number | Solar consumed this hour, never above effective solar. |
| `hourly_plan[].battery_action` | string | `charge`, `discharge`, or `idle`. |
| `hourly_plan[].battery_kwh` | number | Non-negative magnitude of the action; `0` when idle. |
| `hourly_plan[].battery_energy_after_kwh` | number | Battery energy at the end of the hour. |
| `total_grid_kwh` | number | Sum of `grid_kwh`. |
| `total_cost_bdt` | number | Sum of `grid_kwh * tariff_bdt_per_kwh`. |
| `peak_grid_kwh` | number | Maximum hourly `grid_kwh`. |
| `plan_summary` | string | Short description of what the plan does and which directives shaped it. |

**Status codes**

| Code | When | Body |
|---|---|---|
| `200` | Always, for a well-formed request. This includes LLM outage, infeasible directives, solver fallback, and the 25 s deadline. | The response above. |
| `400` | Malformed JSON, missing or mistyped fields, not 24 unique hours, 0 or more than 3 notes, a blank note, negative or non-finite numbers. | `{"error": "invalid_request", "detail": "<one line>"}` |
| `422` | Well-formed input whose battery can never be valid: `minimum_energy_kwh <= initial_energy_kwh <= capacity_kwh` does not hold. | `{"error": "unprocessable_scenario", "detail": "<one line>"}` |
| `500` | Unexpected internal error. | `{"error": "internal_error"}` |

The body is parsed as JSON regardless of the `Content-Type` header, so a harness that posts valid JSON
as `text/plain` is still served. `detail` names the offending field and never echoes the request body.

**Example response** (public sample 1, trimmed to 3 of the 24 hours; `explanation` and `plan_summary`
wording varies)

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": { "hours": [12, 13], "factor": 0.25 },
      "explanation": "Solar availability is reduced to 25% during the panel-cleaning window."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    { "hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0 },
    { "hour": 1, "grid_kwh": 45.0, "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 40.0, "battery_energy_after_kwh": 70.0 },
    { "hour": 2, "grid_kwh": 130.0, "solar_used_kwh": 0.0, "battery_action": "charge", "battery_kwh": 50.0, "battery_energy_after_kwh": 120.0 }
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Uses the reduced midday solar, ignores the unrelated note, shifts battery energy toward higher-tariff hours, and restores the initial battery level."
}
```

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The test suite needs no API key and no network; the language model is always stubbed. It checks:

- **Solver** (`tests/test_optimizer_samples.py`): the LP matches the reference optimal cost on all 10
  public samples under both CBC and HiGHS; decimal-perturbed (fuzzed) inputs still produce plans that
  pass the independent validator and beat the fallback plan; zero-valued and overlapping directives
  bind correctly; an infeasible strict model yields the minimal-violation elastic plan.
- **HTTP contract** (`tests/test_api.py`): `/health`; the full response for every public sample; valid
  JSON accepted whatever the `Content-Type`; extra fields and unordered hours tolerated; structurally
  invalid bodies get 400; an impossible battery state gets 422; an interpreter crash still returns a
  valid plan with HTTP 200; an unhandled error is a controlled 500 with no traceback.
- `tests/data/paraphrases.json` holds 24 hand-written paraphrased notes (24-hour clock, "PV" synonyms,
  "cut by 60%" versus "60% of") with their expected type, hours, and value.

End-to-end against a running server (this path does call Gemini, so the server needs a key):

```bash
python scripts/run_samples.py --url http://localhost:8000
python scripts/run_samples.py --url <PUBLIC_BASE_URL> --wait 90
```

Useful flags: `--wait SECONDS` polls `/health` through a cold start first, `--only SAMPLE-03` runs one
case, `-v` prints every error instead of the first, `--min-ratio 0.999` also fails a case on cost. HTTP
is standard-library only; the replay imports `app.validator`, so run it from the repository root with
the project interpreter. Exit code is `0` only if `/health` and every case passed.

This is the judge's procedure run by us first. It posts each public `input`, then prints one line per case:

```
PASS  SAMPLE-01   interp=ok  valid=ok  totals=ok  ratio=1.0000    <latency> ms
```

| Column | Meaning |
|---|---|
| `interp` | `directive_interpretation` matches the reference: `note_index`, `directive_type`, `applies`, exact key set, exact `hours`, numeric values within 0.01. `explanation` only has to be a non-empty string. |
| `valid` | The returned `hourly_plan` passes the independent replay under the **ground-truth** directives, not the ones the service claimed. A wrong interpretation therefore shows up here as a rule violation, exactly as it would for the judge. |
| `totals` | The reported `total_grid_kwh`, `total_cost_bdt`, and `peak_grid_kwh` equal what the plan implies, within 0.01. |
| `ratio` | Reference optimal cost divided by our cost. `1.0000` is optimal; lower is worse; above 1 with `valid=ok` triggers a warning that a constraint was probably missed. |
| last column | Wall-clock latency of the request in milliseconds, followed by the first error if there is one. |

The summary block then gives `/health` status, passed and failed counts, `interp`/`valid`/`totals`
counts out of 10, mean/min/max cost ratio, and latency `p50`, `p95`, and `max`. That last line is the
source for the `Measured latency` placeholder at the top of this file.

---

## Deployment notes

Hosted on Render: Docker runtime, free plan. The repository ships a Blueprint, `render.yaml`, that
declares the whole service (Docker runtime, free plan, `/health` health check, auto-deploy, and every
non-secret environment variable).

1. Render dashboard -> **New** -> **Blueprint** -> select this repository. Render reads `render.yaml`.
2. When prompted, enter the value of `GEMINI_API_KEYS`. It is declared `sync: false`, so the value lives
   only in the Render dashboard and never in the repository.
3. Apply, wait for the build, then verify: `curl -s <PUBLIC_BASE_URL>/health`.

Without the Blueprint: **New** -> **Web Service**, then either connect the repository (Render builds
the `Dockerfile`) or choose **Existing image** and enter
`ghcr.io/alphaman-0/gridwise-llm-energy-optimizer:v1.0.0`. Pick runtime **Docker**, instance type
**Free**, health check path `/health`, and add `GEMINI_API_KEYS` as a secret. Do not set `PORT`; Render
injects it and the container honours it.

**Image pipeline.** `.github/workflows/image.yml` runs on every push to `main` and on `v*` tags. It
builds for `linux/amd64`, starts the container **with no API key**, waits for `/health`, posts public
sample 1 and asserts HTTP 200 with a 24-entry plan, and only then pushes `:v1.0.0`, `:latest`, and
`:sha-<short>` to GHCR. A broken image never replaces a published tag.

**Keep-alive.** Free Render instances sleep after a period without traffic, and a cold start can exceed
the judge's time limit. An external uptime monitor sends `GET <PUBLIC_BASE_URL>/health` every 5 minutes
to keep the instance warm. `/health` touches neither the LLM nor the solver, so the pings cost no quota.

**Redeploy.** Repository-backed service: push to `main` (auto-deploy), or **Manual Deploy -> Deploy
latest commit**. Image-backed service: push a new tag, update the image reference in the service
settings, and deploy. Changing an environment variable triggers a redeploy on its own.

The process runs a single uvicorn worker on purpose: the interpretation cache and the warmed-up solver
live in process memory, and the free instance has 512 MB.

---

## Secret handling

- The Gemini key is sent only in the `x-goog-api-key` request header. It is never placed in a URL or
  query string, so it cannot appear in access logs, proxies, or exception messages that include a URL.
- Keys are never logged. LLM log lines carry only the model name, the key's *index*, the HTTP status,
  elapsed milliseconds, and the exception class name; never a URL, header, body, or httpx exception
  text. The settings object masks keys in its `repr`, and `httpx` request logging is silenced.
- Error responses never include upstream response bodies, the request body, or stack traces.
- `.env` is gitignored (`.env`, `.env.*`, with `.env.example` explicitly allowed).
- `.dockerignore` excludes `.env`, and the Dockerfile copies only `requirements.txt` and `app/`, so a
  local `.env` cannot end up in an image layer. Secrets reach the container only as runtime
  environment variables.
- `.env.example` contains variable names only, no values.
- On Render the key is stored as a secret environment variable in the dashboard, not in the repository.

---

## Known limitations and assumptions

- **Overlapping directives** on the same hour resolve to the most restrictive value: lowest `factor`,
  highest `minimum_energy_kwh`, lowest `max_grid_kwh`. Window directives are a union.
- **Whole-hour windows.** Start inclusive, end exclusive: "1 PM to 3 PM" is `[13, 14]`. The contract has
  no way to express a part of an hour.
- **Only the six directive types exist.** A note asking for anything else (for example a demand change)
  is reported as `no_op`, because no allowed type represents it.
- **Free-tier quota.** A Gemini free-tier key has per-minute and per-day limits. Rotation across several
  keys and models softens this; when the budget runs out the service degrades to `no_op` interpretations
  with a valid plan rather than failing.
- **Infeasible directives.** If a hallucinated or genuinely impossible directive makes the LP infeasible
  (for example a grid cap below unavoidable demand), the relaxation ladder relaxes grid caps and
  reserves by the fewest kWh possible; `plan_summary` says so. Charge/discharge windows, solar
  reductions, and every physical rule stay hard. The response still reports the interpretation exactly
  as extracted, so the interpretation score is not sacrificed to obtain a plan.
- **The last-resort fallback plan ignores `max_grid_window` and reserves above the starting level.** It
  applies solar reductions and keeps the battery idle, which satisfies charge/discharge windows and
  neutrality by construction, but it cannot lower grid import or raise stored energy. It is reached only
  if no solver works or the 25 s deadline fires.
- **Battery model** follows the problem statement: no round-trip losses, no self-discharge, one action
  per hour.
- **The cache is per process and in memory.** It is empty after a restart and is not shared between
  instances. The deployment runs one worker, so this is sufficient.
- **Determinism.** Temperature is 0 and the output is schema-constrained, but a hosted model is not
  guaranteed to be bit-for-bit repeatable. Structured fields are stable in practice; `explanation`
  wording can vary.

---

## Credits

- [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) - web framework and ASGI server
- [Pydantic](https://docs.pydantic.dev/) - request and response validation
- [httpx](https://www.python-httpx.org/) - HTTP client for the Gemini REST API
- [PuLP](https://coin-or.github.io/pulp/) with the bundled [CBC](https://github.com/coin-or/Cbc) solver - LP modelling and solving
- [HiGHS](https://highs.dev/) via `highspy` - in-process fallback solver
- [Google Gemini API](https://ai.google.dev/) - operator note interpretation
- [python-dotenv](https://github.com/theskumar/python-dotenv) - local `.env` loading
- [pytest](https://docs.pytest.org/) - tests

An AI coding assistant was used for implementation support. The architecture, the interpretation and
optimization logic, and all design decisions are the team's own.

Built for BUP CSE Fest 2026, GridWise LLM preliminary round.
