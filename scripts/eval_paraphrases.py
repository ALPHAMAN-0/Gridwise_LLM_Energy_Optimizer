"""Live interpretation check against hidden-style paraphrases (uses the real model).

    python scripts/eval_paraphrases.py [--file tests/data/paraphrases.json] [--batch 3]

Calls app.interpreter.interpret directly, the same code path the API uses, so it
measures the prompt, the normaliser and the guardrails together. Notes are sent
in batches of up to three, exactly like a real request. Needs GEMINI_API_KEYS
(or GEMINI_API_KEY) in the environment or in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.interpreter import cache_clear, interpret  # noqa: E402
from app.schemas import OptimizeRequest  # noqa: E402

_VALUE_KEY = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "max_grid_window": "max_grid_kwh",
}


def _request(notes: list[str], capacity: float) -> OptimizeRequest:
    return OptimizeRequest.model_validate(
        {
            "scenario_id": "EVAL",
            "operator_notes": notes,
            "hours": [
                {"hour": h, "demand_kwh": 100, "solar_kwh": 50, "tariff_bdt_per_kwh": 10}
                for h in range(24)
            ],
            "battery": {
                "capacity_kwh": capacity,
                "initial_energy_kwh": capacity / 2,
                "minimum_energy_kwh": 0,
                "max_charge_kwh_per_hour": 50,
                "max_discharge_kwh_per_hour": 50,
            },
        }
    )


def _verdict(expected: dict, got) -> str | None:
    """None when correct, otherwise a short description of the mismatch."""
    if got.directive_type != expected["expected_type"]:
        return f"type {got.directive_type} != {expected['expected_type']}"
    if expected["expected_type"] == "no_op":
        ok = got.applies is False and got.structured_adjustment is None
        return None if ok else "no_op with applies/adjustment set"
    adjustment = got.structured_adjustment or {}
    if adjustment.get("hours") != expected["expected_hours"]:
        return f"hours {adjustment.get('hours')} != {expected['expected_hours']}"
    key = _VALUE_KEY.get(expected["expected_type"])
    if key is not None:
        value = adjustment.get(key)
        if value is None or abs(float(value) - float(expected["expected_value"])) > 0.01:
            return f"{key} {value} != {expected['expected_value']}"
    return None


async def _main(path: Path, batch: int) -> int:
    items = json.loads(path.read_text(encoding="utf-8"))
    cache_clear()

    # Group by capacity so each synthetic request carries one battery.
    groups: dict[float, list[dict]] = {}
    for item in items:
        groups.setdefault(float(item["battery_capacity_kwh"]), []).append(item)

    failures, latencies = [], []
    for capacity, group in groups.items():
        for start in range(0, len(group), batch):
            chunk = group[start : start + batch]
            began = time.perf_counter()
            result = await interpret(_request([c["note"] for c in chunk], capacity))
            latencies.append((time.perf_counter() - began) * 1000)
            for expected, got in zip(chunk, result):
                problem = _verdict(expected, got)
                print(f"{'PASS' if problem is None else 'FAIL'}  {expected['expected_type']:24s} {expected['note'][:90]}")
                if problem is not None:
                    print(f"      -> {problem}   | explanation: {got.explanation[:140]}")
                    failures.append(expected["note"])

    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    print(f"\n{len(items) - len(failures)}/{len(items)} correct | calls={len(latencies)} "
          f"p50={latencies[len(latencies) // 2]:.0f}ms p95={p95:.0f}ms")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=ROOT / "tests" / "data" / "paraphrases.json")
    parser.add_argument("--batch", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.file, args.batch)))
