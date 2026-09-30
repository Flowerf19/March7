"""Regression tests for the durable store_summary idempotency interface
(#6, diary side): OPTIONAL idempotency_key with a fully compatible API.

Rules under test: a claimed marker is honored only while its HASH still
EXISTS; marker + HASH + TTL commit atomically under WATCH/CAS for both
merge and append; markers bind scope+topic+source batch; no in-memory
state (a fresh store over the same Redis dedups).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from twin.shared.config.settings import Config
from twin.shared.memory.diary import TimelineSummaryStore
from twin.shared.memory.diary.idempotency import marker_key
from unit.memory.redis_tx_fake import TxFakeRedis

DIM = 8


class FakeEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    async def get_embedding(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.5] * DIM


def _ts(hour: int) -> float:
    return datetime(2026, 7, 2, hour, tzinfo=timezone.utc).timestamp()


def _append_store(fake: TxFakeRedis) -> TimelineSummaryStore:
    # No embedding_service -> append-only path (no merge lookup noise).
    return TimelineSummaryStore(redis_client=fake, embedding_dim=DIM)


@pytest.mark.asyncio
async def test_identical_retry_returns_original_id_without_duplicate():
    fake = TxFakeRedis(dim=DIM)
    store = _append_store(fake)

    kwargs = dict(
        user_id="u1", summary="Chiều fix bug auth.", embedding=[0.1] * DIM,
        topic="work", importance=3, period_start=_ts(4), period_end=_ts(5),
        source_entry_ids=["e1", "e2"], idempotency_key="batch-7",
    )
    sid1 = await store.store_summary(**kwargs)
    # Fresh store instance over the same Redis: dedup is durable, not RAM.
    sid2 = await _append_store(fake).store_summary(**kwargs)

    assert sid1 == sid2
    docs = fake.summary_hashes()
    assert list(docs) == [f"timeline:summary:{sid1}"]
    # Exactly one atomic commit batch; the retry was read-only.
    assert len(fake.exec_batches) == 1
    kinds = [op[0] for op in fake.exec_batches[0]]
    assert kinds.count("hset") == 1 and kinds.count("set") == 1
    assert kinds.count("expire") == 2  # HASH TTL + marker TTL together
    marker = marker_key("u1", "work", ["e1", "e2"], "batch-7")
    assert fake.strings[marker] == sid1.encode()
    assert fake.ttls[marker] == fake.ttls[f"timeline:summary:{sid1}"]


@pytest.mark.asyncio
async def test_marker_binds_scope_topic_and_source_batch():
    """Same caller key string, different scope/topic/batch -> distinct
    markers and distinct docs (no cross-batch aliasing)."""
    fake = TxFakeRedis(dim=DIM)
    store = _append_store(fake)

    base = dict(
        summary="T.", embedding=[0.1] * DIM, importance=3,
        period_start=_ts(4), period_end=_ts(5), idempotency_key="k",
    )
    a = await store.store_summary(user_id="u1", topic="work", source_entry_ids=["e1"], **base)
    b = await store.store_summary(user_id="u1", topic="play", source_entry_ids=["e1"], **base)
    c = await store.store_summary(user_id="u1", topic="work", source_entry_ids=["e2"], **base)
    d = await store.store_summary(user_id="u2", topic="work", source_entry_ids=["e1"], **base)

    assert len({a, b, c, d}) == 4
    assert len(fake.summary_hashes()) == 4
    assert len(fake.strings) == 4  # four distinct marker keys


@pytest.mark.asyncio
async def test_stale_marker_restores_instead_of_false_ack():
    """Marker claims a HASH that no longer EXISTS (TTL race/eviction) ->
    retry re-stores a fresh doc and re-points the marker; never returns
    an id for an absent doc."""
    fake = TxFakeRedis(dim=DIM)
    store = _append_store(fake)
    kwargs = dict(
        user_id="u1", summary="T.", embedding=[0.1] * DIM,
        topic="work", importance=3, period_start=_ts(4), period_end=_ts(5),
        source_entry_ids=["e1"], idempotency_key="k9",
    )
    sid1 = await store.store_summary(**kwargs)
    del fake.hashes[f"timeline:summary:{sid1}"]  # doc gone, marker remains

    sid2 = await store.store_summary(**kwargs)

    assert sid2 != sid1
    assert f"timeline:summary:{sid2}" in fake.hashes
    marker = marker_key("u1", "work", ["e1"], "k9")
    assert fake.strings[marker] == sid2.encode()


@pytest.mark.asyncio
async def test_idempotent_merge_marks_atomically_and_retry_is_clean(monkeypatch):
    """First attempt merges into the seeded doc with the marker committed
    in the SAME EXEC batch; the retry returns the seed id without
    appending the same text a second time."""
    monkeypatch.setattr(Config, "T2_MERGE_MIN_COSINE", 0.60)
    monkeypatch.setattr(Config, "T2_MERGE_MAX_CHARS", 1500)
    fake = TxFakeRedis(dim=DIM)
    store = TimelineSummaryStore(
        redis_client=fake, embedding_dim=DIM, embedding_service=FakeEmbedder(),
    )
    seed_id = await store.store_summary(
        user_id="u1", summary="Sáng họp.", embedding=[0.1] * DIM, topic="work",
        importance=3, period_start=_ts(2), period_end=_ts(3),
        source_entry_ids=["e0"],
    )
    fake.merge_target = f"timeline:summary:{seed_id}"

    kwargs = dict(
        user_id="u1", summary="Chiều fix bug.", embedding=[0.1] * DIM,
        topic="work", importance=3, period_start=_ts(4), period_end=_ts(5),
        source_entry_ids=["e1"], idempotency_key="m1",
    )
    sid1 = await store.store_summary(**kwargs)
    assert sid1 == seed_id
    merge_batches = [b for b in fake.exec_batches if any(op[0] == "set" for op in b)]
    assert len(merge_batches) == 1
    kinds = [op[0] for op in merge_batches[0]]
    assert kinds.count("hset") == 1 and kinds.count("set") == 1  # one atomic EXEC

    sid2 = await store.store_summary(**kwargs)
    assert sid2 == seed_id
    doc = fake.summary_hashes()[f"timeline:summary:{seed_id}"]
    assert doc["summary"].count("Chiều fix bug.") == 1  # merged exactly once
    assert set(json.loads(doc["source_entry_ids"])) == {"e0", "e1"}


@pytest.mark.asyncio
async def test_concurrent_identical_appends_converge_on_one_doc():
    """Two identical idempotent appends racing WATCH(marker) converge:
    one commits, the other reports the winner — single HASH either way."""
    import asyncio

    fake = TxFakeRedis(dim=DIM)
    store = _append_store(fake)
    kwargs = dict(
        user_id="u1", summary="T.", embedding=[0.1] * DIM,
        topic="work", importance=3, period_start=_ts(4), period_end=_ts(5),
        source_entry_ids=["e1"], idempotency_key="race",
    )
    sid1, sid2 = await asyncio.gather(
        store.store_summary(**kwargs), store.store_summary(**kwargs),
    )
    assert sid1 == sid2
    assert len(fake.summary_hashes()) == 1


@pytest.mark.asyncio
async def test_race_loser_inside_commit_reports_winner():
    """Deterministic post-EXEC WATCH abort: a rival claims the marker
    between our GET and EXEC -> append_doc returns the rival's id after
    verifying its HASH exists."""
    fake = TxFakeRedis(dim=DIM)
    store = _append_store(fake)
    rival_key = "timeline:summary:rival"
    await fake.hset(rival_key, mapping={"summary": "R."})

    async def rival_commits():
        marker = marker_key("u1", "work", ["e1"], "late-race")
        await fake.set(marker, "rival")
        fake.pre_execute_hooks.clear()  # fire once

    fake.pre_execute_hooks.append(rival_commits)
    sid = await store.store_summary(
        user_id="u1", summary="T.", embedding=[0.1] * DIM,
        topic="work", importance=3, period_start=_ts(4), period_end=_ts(5),
        source_entry_ids=["e1"], idempotency_key="late-race",
    )
    assert sid == "rival"
    assert len(fake.summary_hashes()) == 1  # only the rival HASH, no dup


@pytest.mark.asyncio
async def test_idempotent_write_fails_closed_without_transactions():
    """A client without pipeline() cannot honor atomicity -> RuntimeError,
    never a split marker/HASH write."""

    class PlainRedis:
        async def execute_command(self, *args):
            return [0]

        async def hset(self, key, mapping):
            pass  # pragma: no cover

        async def expire(self, key, seconds):
            pass  # pragma: no cover

    store = TimelineSummaryStore(redis_client=PlainRedis(), embedding_dim=DIM)
    with pytest.raises(RuntimeError, match="transactional"):
        await store.store_summary(
            user_id="u1", summary="T.", embedding=[0.1] * DIM,
            idempotency_key="k",
        )


@pytest.mark.asyncio
async def test_blank_or_nonstr_key_rejected():
    fake = TxFakeRedis(dim=DIM)
    store = _append_store(fake)
    for bad in ("", "   ", 123, ["k"]):
        with pytest.raises(ValueError, match="idempotency_key"):
            await store.store_summary(
                user_id="u1", summary="T.", embedding=[0.1] * DIM,
                idempotency_key=bad,  # type: ignore[arg-type]
            )
    assert fake.summary_hashes() == {} and fake.strings == {}
