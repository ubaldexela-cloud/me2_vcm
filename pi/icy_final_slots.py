"""Extract one validated parameter from a transcript.

Safe Mix chooses the intent. This module only reads the value for that intent.
It does not guess, reuse an earlier value, or assign a default.
"""
from __future__ import annotations

import re

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_DURATION_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}
_FAMILIES = {
    "SET_TIMER": set(_DURATION_UNITS) | {"timer", "countdown"},
    "SET_ALARM": {"alarm", "am", "pm", "oclock"},
    "SET_TEMPERATURE": {"degree", "degrees", "temperature", "celsius", "centigrade"},
    "LIGHT_DIM": {"percent", "percentage", "brightness", "dim", "dimness"},
}
_TIMER_LIMIT = 604800


def extract_parameter(intent, transcript):
    """Return (payload, error, slot_text). A payload is present only when error is None."""
    if intent not in _FAMILIES:
        return None, "This command does not take a spoken value.", ""
    text = _normalize(transcript)
    if not text:
        return None, "No spoken value was recognized.", ""
    family_error = _family_conflict(intent, text)
    if family_error:
        return None, family_error, ""
    if intent == "SET_TIMER":
        return _duration(text)
    if intent == "SET_ALARM":
        return _clock(text)
    if intent == "SET_TEMPERATURE":
        return _temperature(text)
    return _brightness(text)


def _normalize(transcript):
    text = str(transcript or "").lower()
    text = re.sub(r"(?<![a-z0-9])-(\d)", r"minus \1", text)
    text = text.replace("-", " ")
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = re.sub(r"\ba\s+m\b", "am", text)
    text = re.sub(r"\bp\s+m\b", "pm", text)
    text = text.replace("o'clock", "oclock").replace("o’clock", "oclock")
    text = re.sub(r"[^a-z0-9:%.']+", " ", text)
    tokens = []
    raw = text.split()
    index = 0
    while index < len(raw):
        word = raw[index]
        if word in {"a", "an"} and index + 1 < len(raw) and raw[index + 1] in _DURATION_UNITS:
            tokens.append("1")
            index += 1
            continue
        if word in _TENS:
            value = _TENS[word]
            if index + 1 < len(raw) and raw[index + 1] in _ONES and _ONES[raw[index + 1]] < 10:
                value += _ONES[raw[index + 1]]
                index += 2
                tokens.append(str(value))
                continue
            tokens.append(str(value))
            index += 1
            continue
        if word in _ONES:
            tokens.append(str(_ONES[word]))
            index += 1
            continue
        tokens.append(word)
        index += 1
    return " ".join(tokens)


def _family_conflict(intent, text):
    words = set(re.findall(r"[a-z']+", text))
    hits = [name for name, vocab in _FAMILIES.items() if words & vocab]
    if len(hits) > 1:
        return "The spoken value is ambiguous for the recognized command."
    if len(hits) == 1 and hits[0] != intent:
        return "The spoken value does not match the recognized command."
    return None


def _numbers(text):
    values = []
    for match in re.finditer(r"(minus\s+)?(\d+(?:\.\d+)?)", text):
        value = float(match.group(2))
        if match.group(1):
            value = -value
        values.append(value)
    return values


def _duration(text):
    if re.search(r"minus\s+\d", text):
        return None, "The timer duration must be a whole number of seconds.", ""
    found = []
    for number, unit in re.findall(
        r"(\d+(?:\.\d+)?) (seconds|second|secs|sec|minutes|minute|mins|min|hours|hour|hrs|hr)",
        text,
    ):
        seconds = float(number) * _DURATION_UNITS[unit]
        if not seconds.is_integer():
            return None, "The timer duration must be a whole number of seconds.", ""
        found.append(int(seconds))
    if not found:
        return None, "No timer duration was recognized. Say a number of seconds, minutes, or hours.", ""
    if len(set(found)) != 1:
        return None, "More than one timer duration was recognized.", ""
    seconds = found[0]
    if not 1 <= seconds <= _TIMER_LIMIT:
        return None, "Timer must be between 1 second and 7 days.", ""
    return {"seconds": seconds}, None, f"{seconds} seconds"


def _clock(text):
    found = []
    patterns = (
        re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\b"),
        re.compile(r"\b(\d{1,2})\s+(\d{2})\s*(am|pm)\b"),
        re.compile(r"\b(\d{1,2})\s*(am|pm)\b"),
        re.compile(r"\b(\d{1,2}):(\d{2})\b"),
    )
    consumed = text
    for pattern in patterns[:3]:
        for match in pattern.finditer(consumed):
            groups = match.groups()
            if len(groups) == 3 and groups[2] in {"am", "pm"} and pattern is patterns[1]:
                hour, minute, meridiem = int(groups[0]), int(groups[1]), groups[2]
            elif len(groups) == 3:
                hour, minute, meridiem = int(groups[0]), int(groups[1]), groups[2]
            else:
                hour, minute, meridiem = int(groups[0]), 0, groups[1]
            clock = _to_hhmm(hour, minute, meridiem)
            if clock is None:
                return None, "The alarm time is not a valid clock time.", ""
            found.append(clock)
        consumed = pattern.sub(" ", consumed)
    if not found:
        for match in patterns[3].finditer(consumed):
            hour, minute = int(match.group(1)), int(match.group(2))
            if 1 <= hour <= 12:
                return None, "The alarm time needs AM or PM.", ""
            clock = _to_hhmm(hour, minute, None)
            if clock is None:
                return None, "The alarm time is not a valid clock time.", ""
            found.append(clock)
    if not found:
        return None, "No alarm time was recognized. Say the hour with AM or PM.", ""
    if len(set(found)) != 1:
        return None, "More than one alarm time was recognized.", ""
    return {"time": found[0]}, None, found[0]


def _to_hhmm(hour, minute, meridiem):
    if not 0 <= minute <= 59:
        return None
    if meridiem is None:
        if not 0 <= hour <= 23 or 1 <= hour <= 12:
            return None
        return f"{hour:02d}:{minute:02d}"
    if not 1 <= hour <= 12:
        return None
    if hour == 12:
        hour = 0
    if meridiem == "pm":
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def _temperature(text):
    values = []
    for value in _numbers(text):
        if value in values:
            continue
        values.append(value)
    if not values:
        return None, "No temperature was recognized.", ""
    if len(values) != 1:
        return None, "More than one temperature was recognized.", ""
    value = values[0]
    if not 0 <= value <= 50:
        return None, "Temperature must be between 0 and 50 degrees.", ""
    return {"temperature": value}, None, f"{value:g} degrees"


def _brightness(text):
    values = []
    for value in _numbers(text):
        if not float(value).is_integer():
            return None, "Brightness must be a whole percent from 0 to 100.", ""
        whole = int(value)
        if whole not in values:
            values.append(whole)
    if not values:
        return None, "No brightness percentage was recognized.", ""
    if len(values) != 1:
        return None, "More than one brightness percentage was recognized.", ""
    value = values[0]
    if not 0 <= value <= 100:
        return None, "Brightness must be between 0 and 100 percent.", ""
    return {"brightness": value}, None, f"{value} percent"
