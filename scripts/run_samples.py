"""Score a running GridWise service against the public sample pack.

This is the judge's procedure, run by us first: POST each case, compare the
directive interpretation with the reference, then replay the returned schedule
under the GROUND-TRUTH directives (not the ones the service claimed), because
that is how a wrong interpretation actually costs points: the plan it produces
is checked against constraints it never saw.

HTTP is stdlib-only (urllib) so the script can be pointed at a deployed URL
from any machine. The replay reuses the repo's own `app.validator`, so run it
with the project interpreter:

    .venv/bin/python scripts/run_samples.py --url http://localhost:8000
    .venv/bin/python scripts/run_samples.py --url https://x.onrender.com --wait 90
    .venv/bin/python scripts/run_samples.py --url http://localhost:8000 --only SAMPLE-03 -v

Exit code is 1 if /health or any case failed, 0 otherwise.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = REPO_ROOT / "ProblemSet" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

# Same tolerance the judge applies to kWh and BDT comparisons.
TOLERANCE = 0.01

_RESPONSE_FIELDS = (
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
)
_KNOWN_ENDPOINTS = ("/optimize-energy", "/health")


# --------------------------------------------------------------------------
# Replay: the repo's own validator, imported lazily so a missing dependency
# becomes a reported failure instead of a crash at import time.
# --------------------------------------------------------------------------


@dataclass
class Replay:
    request_model: Any = None
    build_constraints: Any = None
    validate: Any = None
    totals: Any = None
    error: str | None = None


def load_replay() -> Replay:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from app.directives import build_constraints
        from app.schemas import OptimizeRequest
        from app.validator import totals, validate
    except Exception as exc:  # noqa: BLE001 - any import failure is reported the same way
        return Replay(
            error=f"replay unavailable ({type(exc).__name__}: {exc}); "
            "run with the project interpreter, e.g. .venv/bin/python"
        )
    return Replay(OptimizeRequest, build_constraints, validate, totals)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


@dataclass
class HttpResult:
    status: int | None
    body: bytes
    elapsed_ms: float
    error: str | None = None


def http_call(url: str, payload: Any | None, timeout: float) -> HttpResult:
    """One GET (payload None) or JSON POST. Never raises."""
    data = None
    headers = {"Accept": "application/json", "User-Agent": "gridwise-run-samples/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return HttpResult(response.status, body, _since(started))
    except urllib.error.HTTPError as exc:
        # A 4xx/5xx is still a response; keep the body for the error line.
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001
            body = b""
        return HttpResult(exc.code, body, _since(started))
    except urllib.error.URLError as exc:
        return HttpResult(None, b"", _since(started), f"connection error: {exc.reason}")
    except TimeoutError:
        return HttpResult(None, b"", _since(started), f"timed out after {timeout:g}s")
    except (http.client.HTTPException, OSError, ValueError) as exc:
        return HttpResult(None, b"", _since(started), f"{type(exc).__name__}: {exc}")


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def parse_json(body: bytes) -> tuple[Any, str | None]:
    try:
        return json.loads(body.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError):
        preview = body[:120].decode("utf-8", errors="replace").replace("\n", " ")
        return None, f"body is not JSON: {preview!r}"


def normalise_base(url: str) -> str:
    base = url.strip().rstrip("/")
    for endpoint in _KNOWN_ENDPOINTS:
        if base.endswith(endpoint):
            base = base[: -len(endpoint)]
    return base.rstrip("/")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def compare_interpretation(returned: Any, expected: list[dict[str, Any]]) -> list[str]:
    """Every difference from the reference semantics. Explanation wording is ignored."""
    if not isinstance(returned, list):
        return [f"directive_interpretation is {type(returned).__name__}, expected a list"]
    errors: list[str] = []
    if len(returned) != len(expected):
        errors.append(f"directive_interpretation has {len(returned)} entries, expected {len(expected)}")

    for position, want in enumerate(expected):
        if position >= len(returned):
            break
        got = returned[position]
        where = f"note {position}"
        if not isinstance(got, dict):
            errors.append(f"{where}: entry is not an object")
            continue

        if got.get("note_index") != want["note_index"] or isinstance(got.get("note_index"), bool):
            errors.append(f"{where}: note_index {got.get('note_index')!r} != {want['note_index']}")
        if got.get("directive_type") != want["directive_type"]:
            errors.append(
                f"{where}: directive_type {got.get('directive_type')!r} != {want['directive_type']!r}"
            )
        if got.get("applies") is not want["applies"]:
            errors.append(f"{where}: applies {got.get('applies')!r} != {want['applies']!r}")
        if not isinstance(got.get("explanation"), str) or not got["explanation"].strip():
            errors.append(f"{where}: explanation must be a non-empty string")

        errors.extend(
            _compare_adjustment(where, got.get("structured_adjustment"), want["structured_adjustment"])
        )
    return errors


def _compare_adjustment(where: str, got: Any, want: dict[str, Any] | None) -> list[str]:
    if want is None:
        return [] if got is None else [f"{where}: structured_adjustment must be null for no_op"]
    if not isinstance(got, dict):
        return [f"{where}: structured_adjustment is {got!r}, expected {want}"]

    errors: list[str] = []
    if set(got) != set(want):
        errors.append(f"{where}: adjustment keys {sorted(got)} != {sorted(want)}")

    for key, want_value in want.items():
        if key not in got:
            continue
        got_value = got[key]
        if key == "hours":
            clean = isinstance(got_value, list) and all(
                isinstance(h, int) and not isinstance(h, bool) for h in got_value
            )
            if not clean:
                errors.append(f"{where}: hours {got_value!r} is not a list of integers")
            elif got_value != want_value:
                errors.append(f"{where}: hours {got_value} != {want_value}")
            continue
        number = _number(got_value)
        if number is None:
            errors.append(f"{where}: {key} {got_value!r} is not a finite number")
        elif abs(number - float(want_value)) > TOLERANCE:
            errors.append(f"{where}: {key} {number:g} != {float(want_value):g}")
    return errors


def check_shape(response: dict[str, Any], scenario_id: str) -> list[str]:
    errors = [f"response is missing {name}" for name in _RESPONSE_FIELDS if name not in response]
    if "scenario_id" in response and response["scenario_id"] != scenario_id:
        errors.append(f"scenario_id {response['scenario_id']!r} does not echo {scenario_id!r}")
    summary = response.get("plan_summary")
    if "plan_summary" in response and (not isinstance(summary, str) or not summary.strip()):
        errors.append("plan_summary must be a non-empty string")
    return errors


# --------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    http_ok: bool = False
    interp_ok: bool = False
    valid_ok: bool = False
    totals_ok: bool = False
    ratio: float | None = None
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    min_ratio_ok: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.http_ok
            and self.interp_ok
            and self.valid_ok
            and self.totals_ok
            and self.min_ratio_ok
            and not self.errors
        )


def run_case(
    case: dict[str, Any], base: str, timeout: float, replay: Replay, min_ratio: float | None
) -> CaseResult:
    result = CaseResult(case_id=str(case.get("id", "?")))
    payload = case.get("input")
    expected = case.get("expected_output")
    if not isinstance(payload, dict) or not isinstance(expected, dict):
        result.errors.append("case is missing input or expected_output")
        return result

    reply = http_call(f"{base}/optimize-energy", payload, timeout)
    result.latency_ms = reply.elapsed_ms
    if reply.error:
        result.errors.append(reply.error)
        return result
    if reply.status != 200:
        preview = reply.body[:160].decode("utf-8", errors="replace").replace("\n", " ")
        result.errors.append(f"HTTP {reply.status}: {preview}")
        return result

    response, problem = parse_json(reply.body)
    if problem is None and not isinstance(response, dict):
        problem = f"body is a JSON {type(response).__name__}, expected an object"
    if problem:
        result.errors.append(problem)
        return result
    result.http_ok = True

    result.errors.extend(check_shape(response, str(payload.get("scenario_id"))))

    interp_errors = compare_interpretation(
        response.get("directive_interpretation"), expected["directive_interpretation"]
    )
    result.interp_ok = not interp_errors
    result.errors.extend(interp_errors)

    recomputed_cost = _replay(result, response, payload, expected, replay)

    # Judge scoring is reference cost over our cost: 1.0 is optimal, lower is worse.
    ours = recomputed_cost if recomputed_cost is not None else _number(response.get("total_cost_bdt"))
    reference = _number(expected.get("total_cost_bdt"))
    if ours is not None and reference is not None:
        if abs(ours) > 1e-9:
            result.ratio = reference / ours
        elif abs(reference) <= 1e-9:
            result.ratio = 1.0
    if min_ratio is not None and (result.ratio is None or result.ratio < min_ratio):
        result.min_ratio_ok = False
        shown = "n/a" if result.ratio is None else f"{result.ratio:.4f}"
        result.errors.append(f"cost ratio {shown} is below --min-ratio {min_ratio:g}")
    return result


def _replay(
    result: CaseResult,
    response: dict[str, Any],
    payload: dict[str, Any],
    expected: dict[str, Any],
    replay: Replay,
) -> float | None:
    """Validate the returned plan under ground-truth directives; return its recomputed cost."""
    if replay.error:
        result.errors.append(replay.error)
        return None
    plan = response.get("hourly_plan")
    if not isinstance(plan, list):
        result.errors.append(f"hourly_plan is {type(plan).__name__}, expected a list")
        return None

    try:
        request = replay.request_model.model_validate(payload)
        constraints = replay.build_constraints(request, expected["directive_interpretation"])
        valid, plan_errors = replay.validate(plan, request, constraints)
        total_grid, total_cost, peak_grid = replay.totals(plan, request)
    except Exception as exc:  # noqa: BLE001 - a malformed plan must not crash the run
        result.errors.append(f"replay crashed: {type(exc).__name__}: {exc}")
        return None

    result.valid_ok = bool(valid)
    result.errors.extend(f"plan: {message}" for message in plan_errors)

    totals_errors: list[str] = []
    for name, recomputed in (
        ("total_grid_kwh", total_grid),
        ("total_cost_bdt", total_cost),
        ("peak_grid_kwh", peak_grid),
    ):
        reported = _number(response.get(name))
        if reported is None:
            totals_errors.append(f"{name} {response.get(name)!r} is not a finite number")
        elif abs(reported - recomputed) > TOLERANCE:
            totals_errors.append(f"{name} reported {reported:g} but the plan implies {recomputed:g}")
    result.totals_ok = not totals_errors
    result.errors.extend(totals_errors)
    return float(total_cost)


# --------------------------------------------------------------------------
# Health, reporting, entry point
# --------------------------------------------------------------------------


def check_health(base: str, timeout: float, wait: float) -> tuple[bool, bool, str]:
    """Returns (ok, reachable, detail). `wait` polls through a cold start."""
    deadline = time.monotonic() + max(wait, 0.0)
    while True:
        reply = http_call(f"{base}/health", None, timeout)
        if reply.error is None:
            body, problem = parse_json(reply.body)
            if reply.status == 200 and problem is None and isinstance(body, dict) and body.get("status") == "ok":
                return True, True, f"ok ({reply.elapsed_ms:.0f} ms)"
            detail = problem or f"HTTP {reply.status}, body {body!r}"
        else:
            detail = reply.error
        if time.monotonic() >= deadline:
            return False, reply.error is None, detail
        time.sleep(2.0)


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile; safe for a single value."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _flag(ok: bool) -> str:
    return "ok " if ok else "BAD"


def print_case(result: CaseResult, verbose: bool) -> None:
    ratio = "   n/a" if result.ratio is None else f"{result.ratio:6.4f}"
    line = (
        f"{'PASS' if result.passed else 'FAIL'}  {result.case_id:<10}  "
        f"interp={_flag(result.interp_ok)} valid={_flag(result.valid_ok)} "
        f"totals={_flag(result.totals_ok)} ratio={ratio}  {result.latency_ms:7.0f} ms"
    )
    if result.errors:
        line += f"  {result.errors[0]}"
        if len(result.errors) > 1 and not verbose:
            line += f"  (+{len(result.errors) - 1} more, use -v)"
    print(line)
    if verbose:
        for message in result.errors[1:]:
            print(f"      {message}")


def print_summary(results: list[CaseResult], health_ok: bool, health_detail: str) -> None:
    count = len(results)
    passed = sum(r.passed for r in results)
    print("-" * 78)
    print(f"health: {'ok' if health_ok else 'FAILED'} - {health_detail}")
    print(f"cases: {count}  passed: {passed}  failed: {count - passed}")
    print(
        f"interp ok: {sum(r.interp_ok for r in results)}/{count}  "
        f"valid ok: {sum(r.valid_ok for r in results)}/{count}  "
        f"totals ok: {sum(r.totals_ok for r in results)}/{count}"
    )
    ratios = [r.ratio for r in results if r.ratio is not None]
    if ratios:
        print(
            f"cost ratio (reference / ours): mean {sum(ratios) / len(ratios):.4f}  "
            f"min {min(ratios):.4f}  max {max(ratios):.4f}  (n={len(ratios)})"
        )
        # Cheaper than the optimal reference while "valid" means our validator
        # is looser than the judge's, or the interpretation dropped a constraint.
        beaten = [r.case_id for r in results if r.ratio is not None and r.ratio > 1.0 + 1e-4 and r.valid_ok]
        if beaten:
            print(f"warning: cheaper than the optimal reference (check the rules) in: {', '.join(beaten)}")
    latencies = [r.latency_ms for r in results if r.http_ok]
    if latencies:
        print(
            f"latency ms: p50 {percentile(latencies, 0.50):.0f}  "
            f"p95 {percentile(latencies, 0.95):.0f}  max {max(latencies):.0f}"
        )


def load_cases(path: Path, only: list[str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        pack = json.load(handle)
    cases = pack.get("cases") if isinstance(pack, dict) else pack
    if not isinstance(cases, list):
        raise ValueError("case file has no 'cases' list")
    cases = [c for c in cases if isinstance(c, dict)]
    wanted = {name.strip().upper() for chunk in only for name in chunk.split(",") if name.strip()}
    if wanted:
        cases = [c for c in cases if str(c.get("id", "")).upper() in wanted]
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POST the public GridWise sample cases to a running service and score the replies."
    )
    parser.add_argument("--url", default="http://localhost:8000", help="service base URL (default: %(default)s)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="path to the sample case JSON")
    parser.add_argument(
        "--only", action="append", default=[], metavar="ID",
        help="run only this case id; repeat or comma-separate for several (e.g. SAMPLE-03)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds (default: 30)")
    parser.add_argument(
        "--wait", type=float, default=0.0, metavar="SECONDS",
        help="keep polling /health for up to this long first (cold-starting free instances)",
    )
    parser.add_argument(
        "--min-ratio", type=float, default=None, metavar="R",
        help="also fail a case whose cost ratio (reference / ours) is below R, e.g. 0.999",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print every error, not just the first")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = normalise_base(args.url)

    try:
        cases = load_cases(args.cases, args.only)
    except (OSError, ValueError) as exc:
        print(f"cannot load cases from {args.cases}: {exc}", file=sys.stderr)
        return 1
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 1

    replay = load_replay()
    if replay.error:
        print(f"warning: {replay.error}", file=sys.stderr)

    print(f"target: {base}  cases: {len(cases)}  timeout: {args.timeout:g}s")
    health_ok, reachable, health_detail = check_health(base, args.timeout, args.wait)
    print(f"{'PASS' if health_ok else 'FAIL'}  GET /health  {health_detail}")
    if not reachable:
        # Nothing is listening; ten identical connection errors add nothing.
        print("service unreachable, not running cases", file=sys.stderr)
        return 1

    results: list[CaseResult] = []
    for case in cases:
        result = run_case(case, base, args.timeout, replay, args.min_ratio)
        results.append(result)
        print_case(result, args.verbose)

    print_summary(results, health_ok, health_detail)
    return 0 if health_ok and all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
