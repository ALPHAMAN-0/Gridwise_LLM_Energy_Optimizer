"""Shared fixtures: the organisers' public sample pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES_PATH = ROOT / "ProblemSet" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def load_cases() -> list[dict]:
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="session")
def cases() -> list[dict]:
    return load_cases()
