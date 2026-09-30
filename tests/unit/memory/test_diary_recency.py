"""Focused get_recent recency-selection regressions (#33 GLOBAL policy).

"Recent" is period_end with a created_at fallback for legacy docs missing
period_end — and the candidates must be selected per partition BEFORE the
final limit: top-n by period_end plus top-n missing-period_end by created_at,
merged, deduped, re-sorted. A single SORTBY period_end + LIMIT n window lets
n current-schema rows with any period_end evict a newer legacy doc the
Python re-sort can never recover.
"""
from __future__ import annotations

from typing import Any

import pytest

from twin.shared.memory.diary import TimelineSummaryStore


def _reply(*docs: tuple[str, dict[str, Any]]) -> list[Any]:
    """RESP2 flat-list FT.SEARCH reply for (summary_id, {field: value})."""
    out: list[Any] = [len(docs)]
    for sid, fields in docs:
        flat: list[bytes] = []
        for key, value in fields.items():
            flat += [str(key).encode(), str(value).encode()]
        out += [f"timeline:summary:{sid}".encode(), flat]
    return out


class BranchRedis:
    """FT.SEARCH stub routing canned replies by SORTBY field (modern vs
    legacy partition branch) and recording every call."""

    def __init__(
        self,
        modern_reply: Any = None,
        legacy_reply: Any = None,
        *,
        fail_on: set[str] | None = None,
        failure: Exception | None = None,
    ):
        self.modern_reply = modern_reply if modern_reply is not None else [0]
        self.legacy_reply = legacy_reply if legacy_reply is not None else [0]
        self.fail_on = fail_on or set()
        self.failure = failure or Exception("boom")
        self.calls: list[tuple] = []

    async def execute_command(self, *args):
        assert args[0] == "FT.SEARCH"
        self.calls.append(args)
        argv = list(args)
        sort_field = argv[argv.index("SORTBY") + 1]
        if sort_field in self.fail_on:
            raise self.failure
        return self.modern_reply if sort_field == "period_end" else self.legacy_reply


def _sort_of(call: tuple) -> str:
    argv = list(call)
    return argv[argv.index("SORTBY") + 1]


@pytest.mark.asyncio
async def test_limit_one_returns_newer_legacy_over_ancient_period_end():
    """Headline bug: current-schema period_end=100/created_at=9999 vs legacy
    created_at=900 with limit=1 MUST return the legacy doc (recency 900 > 100).
    The old single-window query returned the current doc."""
    modern = _reply(("cur", {"summary": "c", "created_at": 9999.0, "period_end": 100.0}))
    legacy = _reply(("leg", {"summary": "l", "created_at": 900.0}))
    redis = BranchRedis(modern, legacy)
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=1)

    assert [d["summary_id"] for d in results] == ["leg"]


@pytest.mark.asyncio
async def test_enough_current_rows_cannot_evict_newer_legacy():
    """Three current rows (period_end 100/90/80) exceed limit=2, but the
    legacy doc at created_at=900 still ranks global #1 — the legacy branch
    windows independently so current-row volume cannot bury it."""
    modern = _reply(
        ("c1", {"summary": "c1", "created_at": 9999.0, "period_end": 100.0}),
        ("c2", {"summary": "c2", "created_at": 9998.0, "period_end": 90.0}),
        ("c3", {"summary": "c3", "created_at": 9997.0, "period_end": 80.0}),
    )
    legacy = _reply(
        ("leg-new", {"summary": "l", "created_at": 900.0}),
        ("leg-old", {"summary": "o", "created_at": 50.0}),
    )
    redis = BranchRedis(modern, legacy)
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=2)

    assert [d["summary_id"] for d in results] == ["leg-new", "c1"]


@pytest.mark.asyncio
async def test_mixed_merged_summary_orders_by_content_time():
    """An old-created summary merged today (new period_end) interleaves with
    legacy docs purely by content time, across both partitions."""
    modern = _reply(
        ("merged", {"summary": "m", "created_at": 1000.0, "period_end": 9500.0}),
        ("cur-old", {"summary": "c", "created_at": 1500.0, "period_end": 2000.0}),
    )
    legacy = _reply(
        ("leg-new", {"summary": "l", "created_at": 9000.0}),
        ("leg-old", {"summary": "o", "created_at": 500.0}),
    )
    redis = BranchRedis(modern, legacy)
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=10)

    assert [d["summary_id"] for d in results] == [
        "merged", "leg-new", "cur-old", "leg-old",
    ]


@pytest.mark.asyncio
async def test_window_bounds_apply_per_branch_with_escaped_user():
    """since/until filter modern by period_end and legacy by created_at; the
    user tag (escaped) scopes both branches."""
    redis = BranchRedis()
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    await store.get_recent("u-1.x", limit=5, since_ts=1000.0, until_ts=2000.0)

    assert len(redis.calls) == 2
    modern, legacy = redis.calls  # modern branch runs first
    assert _sort_of(modern) == "period_end"
    assert _sort_of(legacy) == "created_at"
    assert modern[2] == "(@user_id:{u\\-1\\.x} @period_end:[1000.0 2000.0])"
    assert legacy[2] == (
        "(@user_id:{u\\-1\\.x} -@period_end:[-inf +inf] "
        "@created_at:[1000.0 2000.0])"
    )
    assert "DIALECT" in modern and "DIALECT" in legacy


@pytest.mark.asyncio
async def test_open_ended_since_only_window():
    redis = BranchRedis()
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    await store.get_recent("u1", limit=5, since_ts=1000.0)

    assert len(redis.calls) == 2
    modern, legacy = redis.calls
    assert "@period_end:[1000.0 +inf]" in modern[2]
    assert "@created_at:[1000.0 +inf]" in legacy[2]


@pytest.mark.asyncio
async def test_only_legacy_docs():
    redis = BranchRedis(
        modern_reply=[0],
        legacy_reply=_reply(
            ("l1", {"summary": "a", "created_at": 300.0}),
            ("l2", {"summary": "b", "created_at": 700.0}),
        ),
    )
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=10)

    assert [d["summary_id"] for d in results] == ["l2", "l1"]


@pytest.mark.asyncio
async def test_only_current_docs():
    redis = BranchRedis(
        modern_reply=_reply(
            ("c1", {"summary": "a", "created_at": 50.0, "period_end": 300.0}),
            ("c2", {"summary": "b", "created_at": 60.0, "period_end": 700.0}),
        ),
        legacy_reply=[0],
    )
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=10)

    assert [d["summary_id"] for d in results] == ["c2", "c1"]


@pytest.mark.asyncio
async def test_no_results_either_branch():
    redis = BranchRedis(modern_reply=[0], legacy_reply=[0])
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    assert await store.get_recent("u1", limit=10) == []


@pytest.mark.asyncio
async def test_duplicate_candidates_deduped_first_wins():
    """The same doc surfacing on both branches (defensive: partitions are
    disjoint, but never trust the index blindly) collapses to one hit."""
    modern = _reply(("same", {"summary": "m", "created_at": 100.0, "period_end": 500.0}))
    legacy = _reply(("same", {"summary": "l", "created_at": 100.0}))
    redis = BranchRedis(modern, legacy)
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=10)

    assert [d["summary_id"] for d in results] == ["same"]


@pytest.mark.asyncio
async def test_branch_error_fails_closed_to_empty():
    """A non-schema Redis error on either branch logs and returns [] — a
    half-merged list would silently violate the GLOBAL recency guarantee."""
    modern = _reply(("c1", {"summary": "c", "created_at": 1.0, "period_end": 9.0}))
    legacy = _reply(("l1", {"summary": "l", "created_at": 8.0}))

    store = TimelineSummaryStore(
        redis_client=BranchRedis(modern, legacy, fail_on={"period_end"}),
        embedding_dim=8,
    )
    assert await store.get_recent("u1", limit=10) == []

    store = TimelineSummaryStore(
        redis_client=BranchRedis(modern, legacy, fail_on={"created_at"}),
        embedding_dim=8,
    )
    assert await store.get_recent("u1", limit=10) == []


class LegacyIndexRedis:
    """Index without diary fields: any period_end term (query or SORTBY)
    raises unknown-field; created_at-only queries succeed."""

    def __init__(self, reply: Any):
        self.reply = reply
        self.calls: list[tuple] = []

    async def execute_command(self, *args):
        assert args[0] == "FT.SEARCH"
        self.calls.append(args)
        argv = list(args)
        if "period_end" in args[2] or argv[argv.index("SORTBY") + 1] == "period_end":
            raise Exception("Property `period_end` not loaded nor in schema")
        return self.reply


@pytest.mark.asyncio
async def test_legacy_index_fallback_queries_created_at_only():
    """On a no-diary-field index the fallback must reference no period_end
    term at all (unknown-field raises instead of matching nothing) and
    keeps the exact tag-only unfiltered shape with no DIALECT."""
    redis = LegacyIndexRedis(_reply(("r1", {"summary": "x", "created_at": 5.0})))
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=10)

    assert [d["summary_id"] for d in results] == ["r1"]
    assert len(redis.calls) == 2  # modern attempt, then the fallback
    fallback = redis.calls[-1]
    assert fallback[2] == "@user_id:{u1}"
    assert "period_end" not in fallback[2]
    assert _sort_of(fallback) == "created_at"
    assert "DIALECT" not in fallback


@pytest.mark.asyncio
async def test_legacy_index_fallback_window_degrades_to_created_at():
    redis = LegacyIndexRedis(_reply(("r1", {"summary": "x", "created_at": 1500.0})))
    store = TimelineSummaryStore(redis_client=redis, embedding_dim=8)

    results = await store.get_recent("u1", limit=10, since_ts=1000.0, until_ts=2000.0)

    assert [d["summary_id"] for d in results] == ["r1"]
    fallback = redis.calls[-1]
    assert fallback[2] == "(@user_id:{u1} @created_at:[1000.0 2000.0])"
    assert _sort_of(fallback) == "created_at"
