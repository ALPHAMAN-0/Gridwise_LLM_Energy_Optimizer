"""Degraded-mode note reader: regular expressions instead of a language model.

The language model is the interpreter. This module exists only for the moment
every model and key has failed (the free tier answers 429 after ~15 requests a
minute), when the alternative is "every note is a no_op", which forgoes every
directive in the request. A conservative rule-based reading beats that.

Conservative means: a directive is emitted only when clear evidence of ONE
type, a parsable clock window and any number the type needs are all present.
Anything else is a no_op, because a wrong type puts a wrong hard constraint on
the schedule while a no_op only forgoes one directive.

`read_notes` returns the same INTERMEDIATE shape the model returns (clock
windows as written, a percentage plus remaining/reduction, a reserve plus its
unit), so its output goes through `interpreter.normalize_item` and
`guardrails.enforce` exactly like model output. No arithmetic happens here.
"""

from __future__ import annotations

import re
from typing import Any

_SUFFIX = "(rule-based reading; language model unavailable)"
_MAX_NOTE_CHARS = 1200  # bounds regex work on a hostile body

Window = tuple[int, int]

# -- relevance ---------------------------------------------------------------

_OTHER_DAY = re.compile(
    r"\b(?:tomorrow|yesterday|next\s+(?:week|month|year|term|semester)|last\s+(?:week|month|night|year)"
    r"|weekend|(?:mon|tues|wednes|thurs|fri|satur|sun)day)s?\b"
)
_CANCELLED = re.compile(
    r"\b(?:lifted|cancell?ed|no\s+longer|rescinded|withdrawn|revoked|postponed|called\s+off)\b"
)
# Equipment the schedule does not model but whose notes borrow its vocabulary
# ("EV charging bays closed", "generator load test").
_UNMODELLED = re.compile(
    r"\b(?:evs?|electric\s+vehicles?|vehicles?|car\s+park|parking|e-?bikes?|scooters?|phones?|laptops?"
    r"|diesel|generators?|gensets?|hvac|chillers?|air[\s-]?con\w*|lifts?|elevators?|pumps?)\b"
)

# -- directive evidence ------------------------------------------------------

_SOLAR = re.compile(r"\b(?:solar|pv|photovoltaic|panels?|array|rooftop\s+generation)\b")
_BATTERY = re.compile(r"\b(?:battery|batteries|storage|stored\s+energy|state\s+of\s+charge|soc|bess|in\s+reserve)\b")
_GRID = re.compile(r"\b(?:grid|feeder|transformer|substation|utility|mains|import\w*|intake)\b")

_RESERVE_CUE = re.compile(
    r"\b(?:keep|kept|hold|held|retain\w*|maintain\w*|preserve\w*|reserve[ds]?|remain\w*|stay\w*|sits?"
    r"|at\s+least|no\s+less\s+than|not\s+less\s+than|minimum|above|or\s+(?:better|more|higher)"
    r"|below|under)\b"
)
_CAP_CUE = re.compile(
    r"\b(?:exceed\w*|at\s+or\s+(?:below|under)|limit\w*|cap(?:s|ped|ping)?|no\s+more\s+than"
    r"|not\s+more\s+than|below|under|maximum|max|restrict\w*|ceiling|up\s+to|can\s+only)\b"
)
_BLOCKED = (
    r"(?:isolated|unavailable|disabled|locked\s+out|offline|out\s+of\s+service|prohibited|suspended"
    r"|inhibited|switched\s+off|not\s+(?:be\s+)?(?:permitted|allowed|available))"
)
_NEGATION = (
    r"(?:do\s+not|don't|must\s+not|may\s+not|cannot|can't|should\s+not|shall\s+not|never|no"
    r"|nothing\s+may\s+be)"
)
# \bcharg... cannot match inside "discharg...": there is no word boundary there.
_NO_CHARGE = re.compile(
    rf"\b(?:charger|rectifier|charging|charge\s+(?:circuit|path))\b[^.;]{{0,40}}?\b{_BLOCKED}"
    rf"|\b{_NEGATION}\s+(?:battery\s+)?(?:charg(?:e|ing)|recharg\w+|top(?:ping)?[\s-]?up)\b"
)
_NO_DISCHARGE = re.compile(
    rf"\b{_NEGATION}\s+(?:battery\s+)?discharg\w+"
    rf"|\bdischarg\w+\b[^.;]{{0,40}}?\b{_BLOCKED}"
    rf"|\b{_NEGATION}\s+(?:be\s+)?draw\w*\s+(?:from|on|down)\s+(?:the\s+)?(?:battery|batteries|storage)"
    rf"|\b(?:battery|inverter|storage)\s+(?:output|export)\b[^.;]{{0,40}}?\b{_BLOCKED}"
    r"|\bhold\s+the\s+charge\b"
)
_SOLAR_OFF = re.compile(
    r"\b(?:offline|no\s+(?:solar|output|generation|pv)|zero\s+(?:solar|output|generation)|fully\s+shaded"
    r"|switched\s+off|disconnected|out\s+of\s+service)\b"
)

# -- numbers -----------------------------------------------------------------

_QUANTITY = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mwh|megawatt[\s-]?hours?|kwh|kilowatt[\s-]?hours?|kw|kilowatts?)\b"
)
_FRACTIONS = {"tenth": 10.0, "fifth": 20.0, "quarter": 25.0, "third": 100.0 / 3.0, "half": 50.0}
_COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4}
_SHARE = re.compile(
    r"(?<![\d.])(?P<num>\d+(?:\.\d+)?)\s*(?:%|percent\b|per\s+cent\b)"
    rf"|\b(?:(?P<count>{'|'.join(_COUNTS)})[\s-]+)?(?P<frac>{'|'.join(_FRACTIONS)})s?\b"
    r"|\b(?P<halved>halved?|halves)\b"
)
_HEDGE = r"(?:(?:about|around|roughly|approximately|nearly|almost|only|just|some|an?)\s+)*"
# The preposition decides: "down BY 30%" is lost, "down TO 30%" is what is left.
_LOST_BEFORE = re.compile(
    rf"\b(?:by|lose|loses|losing|lost|loss\s+of|drop\s+of|cut\s+of|reduction\s+of|down|sheds?"
    rf"|reduced|cuts?|lowered|decreased|drops?|dropped|falls?|fell)\s+{_HEDGE}$"
)
_LEFT_BEFORE = re.compile(
    rf"\b(?:to|at|as|only|just|leave|leaves|leaving|manage|manages|deliver\w*|produc\w+)\s+{_HEDGE}$"
)
_LOST_AFTER = re.compile(
    r"\s*(?:\w+\s+){0,3}?(?:reduction|drop|cut|loss|less|lower|decrease|derat\w+|offline|shaded|unavailable)\b"
)
_LEFT_AFTER = re.compile(
    r"\s*(?:of\s+(?:the\s+|its\s+|their\s+|our\s+)?(?:forecast\w*|normal|usual|expected|rated|predicted)"
    r"|usable|available|remain\w*|output|capacity)\b"
)
_OF_CAPACITY_AFTER = re.compile(
    r"\s*(?:(?:of\s+)?(?:the\s+|its\s+|total\s+|rated\s+|usable\s+)*(?:battery|capacity|storage|charge)"
    r"|full\b|charged?\b|capacity\b|soc\b|state\s+of\s+charge)"
)
_CHARGE_LEVEL_BEFORE = re.compile(r"\b(?:battery|storage|charge|soc)\b[^.;%]{0,30}$")

# -- clock times -------------------------------------------------------------

_WORD_HOURS = {
    word: hour
    for hour, word in enumerate(
        "one two three four five six seven eight nine ten eleven twelve".split(), start=1
    )
}
_FIXED_WORDS = {"noon": 12, "midday": 12, "midnight": 0}
# A bare number followed by one of these is a quantity, never an hour.
_NOT_A_UNIT = r"(?!\s*(?:%|percent|per\s+cent|kwh|kw\b|mwh|kilowatt|megawatt|degrees|°))"


def _atom(tag: str) -> str:
    """Regex for one clock time; group names carry `tag` so two fit in one pattern."""
    words = "|".join([*_WORD_HOURS, *_FIXED_WORDS])
    return (
        rf"(?:(?<![\d.:])(?P<h{tag}>\d{{1,2}})(?::(?P<m{tag}>[0-5]\d)(?:\s*hrs?\b)?)?(?:\s*(?P<ap{tag}>[ap]m)\b)?"
        rf"(?![\da-z]|[.:]\d)|\b(?P<w{tag}>{words})\b)(?:\s+o'?clock)?"
        rf"(?:\s+(?:in\s+the\s+|this\s+)?(?P<part{tag}>morning|afternoon|evening|tonight)\b)?{_NOT_A_UNIT}"
    )


_SLOTS = re.compile(
    r"\bhours?\s+\d{1,2}(?:\s*(?:,|and|&)\s*(?:hours?\s+)?\d{1,2})*\b"
    r"(?!\s*(?::|am\b|pm\b|to\b|until\b|till\b|through\b|thru\b|-)\s*\d?)"
)
_RANGE = re.compile(
    rf"(?:\b(?P<lead>from|between)\s+)?{_atom('a')}\s*(?P<conn>to|until|till|through|thru|up\s+to|-|and)\s*"
    rf"{_atom('b')}"
)
_OPEN = re.compile(
    rf"(?:\b(?P<kind>after|beyond|from|before|until|till|up\s+to|prior\s+to)\s+)?{_atom('a')}"
    r"(?P<onward>\s+(?:onwards?|on|forward)\b|\s+(?:to|until|till)\s+(?:the\s+)?(?:end|close)\s+of\s+(?:the\s+)?day)?"
)
_ALL_DAY = re.compile(
    r"\ball[\s-]day\b|\b(?:whole|entire|full)\s+day\b|\bthroughout\s+(?:today|the\s+day)\b"
    r"|\ball\s+of\s+today\b|\baround\s+the\s+clock\b|\buntil\s+further\s+notice\b"
)

Clock = tuple[int, "int | None", "str | None", bool]  # hour, minute, meridiem, already 24-hour


def _clock(match: re.Match[str], tag: str) -> Clock | None:
    word, part = match.group(f"w{tag}"), match.group(f"part{tag}")
    meridiem = match.group(f"ap{tag}") or (("am" if part == "morning" else "pm") if part else None)
    if word in _FIXED_WORDS:
        return _FIXED_WORDS[word], None, None, True
    if word:
        return _WORD_HOURS[word], None, meridiem, False
    hour, minute = int(match.group(f"h{tag}")), match.group(f"m{tag}")
    if hour > 24 or (meridiem and not 1 <= hour <= 12):
        return None
    return hour, int(minute) if minute else None, meridiem, meridiem is None and (hour > 12 or hour == 0)


def _is_bare(clock: Clock) -> bool:
    _hour, minute, meridiem, fixed = clock
    return minute is None and meridiem is None and not fixed


def _to24(hour: int, meridiem: str) -> int:
    return hour % 12 + (12 if meridiem == "pm" else 0)


def _readings(clock: Clock, partner_meridiem: str | None) -> list[int]:
    """Possible 24-hour values, most likely first."""
    hour, minute, meridiem, fixed = clock
    if fixed:
        return [hour]
    if meridiem:
        return [_to24(hour, meridiem)]
    if partner_meridiem:  # "from 6 to 9 PM": try the partner's half of the day first
        other = "am" if partner_meridiem == "pm" else "pm"
        return [_to24(hour, partner_meridiem), _to24(hour, other)]
    if minute is not None:
        return [hour]  # "09:00" with no AM/PM anywhere is a 24-hour clock
    # Crews work and panels produce in daylight: 7-12 is morning, 1-6 is afternoon.
    daytime = hour if 7 <= hour <= 12 else hour + 12
    return [daytime, (daytime + 12) % 24]


def _as_end(clock: Clock, partner_meridiem: str | None) -> list[int]:
    # Minutes round the end up so the whole stated period is covered; midnight ends at 24.
    return [(hour + (1 if clock[1] else 0)) or 24 for hour in _readings(clock, partner_meridiem)]


def _read_range(match: re.Match[str]) -> list[Window]:
    first, second = _clock(match, "a"), _clock(match, "b")
    if first is None or second is None:
        return []
    lead, conn = match.group("lead"), match.group("conn")
    if (conn == "and" and lead != "between") or (_is_bare(first) and _is_bare(second) and not lead):
        return []
    starts, ends = _readings(first, second[2]), _as_end(second, first[2])
    ordered = [(start, end) for start in starts for end in ends if start < end]
    # No ordered reading means the range crosses midnight; normalize_item wraps it.
    start, end = ordered[0] if ordered else (starts[-1], ends[-1])
    return [(start, end)] if start != end and start <= 23 and end <= 24 else []


def _read_open(match: re.Match[str]) -> list[Window]:
    clock, kind = _clock(match, "a"), match.group("kind")
    if clock is None or _is_bare(clock):
        return []
    if kind in ("after", "beyond") or (match.group("onward") and kind in (None, "from")):
        start = _readings(clock, None)[0]
        return [(start, 24)] if start <= 23 else []
    if kind and kind != "from":
        end = _as_end(clock, None)[0]
        return [(0, end)] if end <= 24 else []
    return []


def _read_slots(match: re.Match[str]) -> list[Window]:
    """"hours 18 and 19" names whole hour slots, so the window ends one past the last."""
    slots = sorted({int(number) for number in re.findall(r"\d+", match.group(0))})
    if slots[-1] > 23:
        return []
    if slots == list(range(slots[0], slots[-1] + 1)):
        return [(slots[0], slots[-1] + 1)]
    return [(slot, slot + 1) for slot in slots]


def _windows(text: str) -> list[Window]:
    found: list[Window] = []
    for pattern, reader in ((_SLOTS, _read_slots), (_RANGE, _read_range), (_OPEN, _read_open)):
        for match in list(pattern.finditer(text)):
            windows = reader(match)
            found.extend(window for window in windows if window not in found)
            if windows:
                # Blank what was read (same length, so later spans still line up) so that
                # "until 8 PM" inside a range is not re-read as an open-ended window.
                text = text[: match.start()] + " " * len(match.group(0)) + text[match.end() :]
    if not found and _ALL_DAY.search(text):
        found.append((0, 24))
    return found


# -- amounts -----------------------------------------------------------------


def _share_value(match: re.Match[str]) -> float:
    if match.group("num"):
        return float(match.group("num"))
    if match.group("halved"):
        return 50.0
    return round(_COUNTS.get(match.group("count") or "a", 1) * _FRACTIONS[match.group("frac")], 6)


def _around(text: str, match: re.Match[str]) -> tuple[str, str]:
    return text[max(0, match.start() - 40) : match.start()], text[match.end() : match.end() + 40]


def _solar_amount(text: str) -> tuple[float, str] | None:
    shares = [match for match in _SHARE.finditer(text) if _share_value(match) <= 100.0]
    for match in shares:
        before, after = _around(text, match)
        if match.group("halved"):
            return 50.0, "remaining"
        # Words before the figure ("by", "to") outrank words after it ("of the forecast").
        for hit, mode in (
            (_LOST_BEFORE.search(before), "reduction"),
            (_LEFT_BEFORE.search(before), "remaining"),
            (_LOST_AFTER.match(after), "reduction"),
            (_LEFT_AFTER.match(after), "remaining"),
        ):
            if hit:
                return _share_value(match), mode
    # A share with no clear direction is not guessed, even if "offline" also appears.
    return (0.0, "remaining") if not shares and _SOLAR_OFF.search(text) else None


def _single_quantity(text: str, *, allow_power: bool) -> float | None:
    """The one kWh figure in the note; several different figures is ambiguous."""
    values = set()
    for number, unit in _QUANTITY.findall(text):
        if unit in ("kw", "kilowatt", "kilowatts") and not allow_power:
            continue
        values.add(float(number) * (1000.0 if unit.startswith("m") else 1.0))
    return values.pop() if len(values) == 1 else None


def _figure_owner(text: str) -> str | None:
    """Which subject the note's kWh figure belongs to: the nearest one before it, else the first after.

    "The battery must not drop below 120 kWh in case the feeder trips" has a
    battery, a grid word, a limit word and a figure; only position tells the
    reserve from the grid cap. A figure that follows "solar" is neither.
    """
    figure = _QUANTITY.search(text)
    if figure is None:
        return None
    before, after = text[: figure.start()], text[figure.end() :]
    subjects = {"battery": _BATTERY, "grid": _GRID, "solar": _SOLAR}
    last = {name: max((m.start() for m in rx.finditer(before)), default=-1) for name, rx in subjects.items()}
    if max(last.values()) >= 0:
        return max(last, key=last.__getitem__)
    first = {name: min((m.start() for m in rx.finditer(after)), default=len(text)) for name, rx in subjects.items()}
    return min(first, key=first.__getitem__)


def _reserve_amount(text: str) -> tuple[float, str] | None:
    energy = _single_quantity(text, allow_power=False)
    if energy is not None:
        return energy, "kwh"
    for match in _SHARE.finditer(text):
        before, after = _around(text, match)
        tied_to_battery = _OF_CAPACITY_AFTER.match(after) or _CHARGE_LEVEL_BEFORE.search(before)
        if not match.group("halved") and tied_to_battery and _share_value(match) <= 100.0:
            return _share_value(match), "percent_of_capacity"
    return None


# -- one note ----------------------------------------------------------------


def _normalise(note: str) -> str:
    text = note[:_MAX_NOTE_CHARS].lower()
    text = re.sub("[\u2010-\u2015\u2212]", "-", text).replace("\u2019", "'")  # typographic dashes, apostrophe
    text = re.sub(r"(?<![a-z])([ap])\.\s?m\b\.?", r"\1m", text)  # "p.m." -> "pm"
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # "1,200 kWh"
    # "0900-1200 hrs" -> "09:00-12:00": the unit marks the last time, which then marks the first.
    text = re.sub(r"\b([01]\d|2[0-4])([0-5]\d)(?=\s*(?:hrs?|hours|h)\b)", r"\1:\2", text)
    return re.sub(r"\b([01]\d|2[0-4])([0-5]\d)(?=\s*(?:-|to|until|till|through|and)\s*\d{1,2}:\d\d)", r"\1:\2", text)


def _candidates(text: str) -> dict[str, dict[str, Any]]:
    """Every directive type the note gives full evidence for, with its numbers."""
    found: dict[str, dict[str, Any]] = {}
    solar = _solar_amount(text) if _SOLAR.search(text) else None
    if solar:
        found["solar_reduction"] = {"solar_percent": solar[0], "solar_percent_is": solar[1]}
    reserve = _reserve_amount(text) if _BATTERY.search(text) and _RESERVE_CUE.search(text) else None
    cap = _single_quantity(text, allow_power=True) if _GRID.search(text) and _CAP_CUE.search(text) else None
    owner = _figure_owner(text)
    if reserve and reserve[1] == "kwh" and owner != "battery":
        reserve = None
    if owner != "grid":
        cap = None
    if reserve:
        found["minimum_battery_reserve"] = {"reserve_value": reserve[0], "reserve_unit": reserve[1]}
    if cap is not None:
        found["max_grid_window"] = {"max_grid_kwh": cap}
    if _NO_CHARGE.search(text):
        found["no_charge_window"] = {}
    # "Keep at least X in the battery, do not discharge below it" is a reserve.
    if _NO_DISCHARGE.search(text) and not reserve:
        found["no_discharge_window"] = {}
    return found


def _item(index: int, directive_type: str, reason: str, windows: list[Window], numbers: dict) -> dict:
    fields = ("solar_percent", "solar_percent_is", "reserve_value", "reserve_unit", "max_grid_kwh")
    return {
        "note_index": index,
        "directive_type": directive_type,
        "windows": [{"start_hour": start, "end_hour": end} for start, end in windows],
        **{field: numbers.get(field) for field in fields},
        "explanation": f"{reason} {_SUFFIX}",
    }


def _no_op(index: int, reason: str) -> dict:
    return _item(index, "no_op", reason, [], {})


_EXPLANATIONS = {
    "solar_reduction": "Forecast solar is limited ({solar_percent:g}% {solar_percent_is}) during {span}",
    "minimum_battery_reserve": "The battery must hold at least {reserve_value:g} {unit} during {span}",
    "no_charge_window": "The battery must not be charged during {span}",
    "no_discharge_window": "The battery must not be discharged during {span}",
    "max_grid_window": "Grid import is capped at {max_grid_kwh:g} kWh per hour during {span}",
}


def _describe(directive_type: str, numbers: dict[str, Any], windows: list[Window]) -> str:
    span = " and ".join(f"{start:02d}:00-{end:02d}:00" for start, end in windows)
    unit = "kWh" if numbers.get("reserve_unit") == "kwh" else "% of capacity"
    return _EXPLANATIONS[directive_type].format(**numbers, span=span, unit=unit)


def _read_one(index: int, note: Any) -> dict:
    if not isinstance(note, str) or not note.strip():
        return _no_op(index, "The note is empty or not text")
    text = _normalise(note)
    if _OTHER_DAY.search(text):
        return _no_op(index, "The note is about another day, not today's schedule")
    if _CANCELLED.search(text):
        return _no_op(index, "The note reports a restriction that no longer applies")
    if _UNMODELLED.search(text):
        return _no_op(index, "The note concerns equipment outside the solar, battery and grid schedule")
    found = _candidates(text)
    if len(found) != 1:
        problem = "No single" if found else "No"
        return _no_op(index, f"{problem} solar, battery or grid constraint could be identified with confidence")
    (directive_type, numbers), = found.items()
    windows = _windows(text)
    if not windows:
        return _no_op(index, "A constraint was mentioned but no usable time window could be read")
    return _item(index, directive_type, _describe(directive_type, numbers, windows), windows, numbers)


def read_notes(notes: list[str]) -> list[dict]:
    """One INTERMEDIATE item per note (see RESPONSE_SCHEMA in app/interpreter.py).

    Never raises. Unknown or unsure -> directive_type "no_op", windows [],
    numbers None.
    """
    try:
        listed = list(notes)
    except TypeError:
        return []
    items: list[dict] = []
    for index, note in enumerate(listed):
        try:
            items.append(_read_one(index, note))
        except Exception:  # noqa: BLE001 - the last resort must not be what breaks the request
            items.append(_no_op(index, "The note could not be read"))
    return items
