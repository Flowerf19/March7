"""Malformed timestamps must not abort recall-list formatting."""
from __future__ import annotations

from datetime import datetime

import pytest

from twin.shared.memory.vn_time import VN_TZ
from twin.shared.tools.modules.memory.memory_format import (
    days_ago_display,
    format_memories,
    memory_metadata,
)

FIXED_NOW = datetime(2026, 7, 3, 12, 0, tzinfo=VN_TZ)
GOOD_TS = datetime(2026, 7, 1, 9, 0, tzinfo=VN_TZ).timestamp()
GOOD_DISPLAY = "2 ngày trước — 01/07"

BAD_NON_NONE = [
    float("inf"),
    float("-inf"),
    float("nan"),
    "1e999",
    1e300,
    -1e300,
    "not-a-date",
]


@pytest.mark.parametrize("bad", [*BAD_NON_NONE, None])
def test_days_ago_display_bad_created_at_returns_none(bad):
    assert days_ago_display({"created_at": bad}, now=FIXED_NOW) is None


@pytest.mark.parametrize("bad", BAD_NON_NONE)
def test_days_ago_display_bad_period_end_returns_none(bad):
    # Original policy: non-None period_end never falls back to created_at.
    memory = {"period_end": bad, "created_at": GOOD_TS}
    assert days_ago_display(memory, now=FIXED_NOW) is None


@pytest.mark.parametrize("bad", BAD_NON_NONE)
def test_memory_metadata_bad_created_at_falls_back_to_raw(bad):
    metadata = memory_metadata({"created_at": bad})
    assert f"created_at={bad}" in metadata


def test_memory_metadata_none_created_at_omits_field():
    assert memory_metadata({"created_at": None}) == []
    assert memory_metadata({}) == []


def test_days_ago_display_valid_finite():
    assert days_ago_display({"created_at": GOOD_TS}, now=FIXED_NOW) == GOOD_DISPLAY


def test_memory_metadata_valid_finite_and_datetime():
    finite = memory_metadata({"created_at": GOOD_TS})
    assert len(finite) == 1
    assert finite[0].startswith("created_at=2026-")
    assert "01T" in finite[0]
    dt = datetime(2026, 7, 1, 9, 0, tzinfo=VN_TZ)
    assert memory_metadata({"created_at": dt}) == [f"created_at={dt.isoformat()}"]


def test_format_memories_bad_period_end_keeps_both_texts():
    memories = [
        {"summary": "synthetic bad beta", "period_end": float("inf")},
        {"summary": "synthetic good alpha", "period_end": GOOD_TS},
    ]
    result = format_memories(memories, now=FIXED_NOW)
    assert "Tìm thấy 2 ký ức" in result
    assert "synthetic bad beta" in result
    assert "synthetic good alpha" in result
    assert GOOD_DISPLAY in result


def test_format_memories_bad_created_at_keeps_both_texts():
    memories = [
        {"summary": "synthetic bad beta", "created_at": "1e999"},
        {"summary": "synthetic good alpha", "created_at": GOOD_TS},
    ]
    result = format_memories(memories, now=FIXED_NOW)
    assert "Tìm thấy 2 ký ức" in result
    assert "synthetic bad beta" in result
    assert "synthetic good alpha" in result
    assert GOOD_DISPLAY in result
    assert "created_at=1e999" in result
