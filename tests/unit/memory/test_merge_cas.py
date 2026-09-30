"""Regression tests for concurrent diary merge (#8): read -> await
re-embed -> write commits through WATCH/CAS with bounded retry, then a
safe append fallback. Both workers' texts and provenance must survive;
success is never reported for a lost update.

Deterministic via TxFakeRedis (in-memory WATCH semantics) plus an
embedder barrier forcing both workers to read the same candidate
version before either commits. The real-Redis twin lives in
tests/integration/test_diary_merge_redis.py.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from twin.shared.config.settings import Config
from twin.shared.memory.diary import TimelineSummaryStore
from twin.shared.memory.diary import merge as merge_module
from twin.shared.memory.diary.writer import DiaryWriter, MergeConflict
from unit.memory.redis_tx_fake import TxFakeRedis

DIM = 8


class BarrierEmbedder:
    """First two get_embedding calls rendezvous (both workers read the
    same candidate version before either commits); later calls (merge
    retries) pass through immediately."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls = 0
        self.texts: list[str] = []
        self._barrier = asyncio.Barrier(2)

    async def get_embedding(self, text: str) -> list[float]:
        self.calls += 1
        self.texts.append(text)
        if self.calls <= 2:
            await self._barrier.wait()
        return [0.5] * self.dim


def _ts(year: int, month: int, day: int, hour: int) -> float:
    return datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp()


def _make_store(fake: TxFakeRedis, embedder) -> TimelineSummaryStore:
    return TimelineSummaryStore(
        redis_client=fake, embedding_dim=DIM, embedding_service=embedder,
    )


async def _seed(fake: TxFakeRedis, store: TimelineSummaryStore) -> str:
    """Seed one same-day doc; point the live candidate lookup at it."""
    sid = await store.store_summary(
        user_id="u1",
        summary="Sáng họp dự án X.",
        embedding=[0.1] * DIM,
        topic="work",
        topic_display="Công việc",
        importance=3,
        period_start=_ts(2026, 7, 2, 2),
        period_end=_ts(2026, 7, 2, 3),
        source_entry_ids=["e0"],
    )
    fake.merge_target = f"timeline:summary:{sid}"
    return sid


@pytest.mark.asyncio
async def test_concurrent_merge_both_updates_survive_via_retry(monkeypatch):
    """Both workers read v0, one commits v1, the loser conflicts, retries
    from a fresh read, and merges — one doc holds both texts + all ids."""
    monkeypatch.setattr(Config, "T2_MERGE_MIN_COSINE", 0.60)
    monkeypatch.setattr(Config, "T2_MERGE_MAX_CHARS", 1500)
    fake = TxFakeRedis(dim=DIM)
    embedder = BarrierEmbedder()
    store = _make_store(fake, embedder)
    seed_id = await _seed(fake, store)

    async def worker(text: str, entry: str, hour: int) -> str:
        return await store.store_summary(
            user_id="u1",
            summary=text,
            embedding=[0.1] * DIM,
            topic="work",
            importance=3,
            period_start=_ts(2026, 7, 2, hour),
            period_end=_ts(2026, 7, 2, hour + 1),
            source_entry_ids=[entry],
        )

    sid_a, sid_b = await asyncio.gather(
        worker("Trưa review PR auth.", "eA", 4),
        worker("Chiều fix xong bug auth.", "eB", 6),
    )

    # Both merged into the seeded doc (loser via retry), same id thrice.
    assert sid_a == sid_b == seed_id
    docs = fake.summary_hashes()
    assert list(docs) == [f"timeline:summary:{seed_id}"]
    doc = docs[f"timeline:summary:{seed_id}"]
    assert "Trưa review PR auth." in doc["summary"]
    assert "Chiều fix xong bug auth." in doc["summary"]
    assert "Sáng họp dự án X." in doc["summary"]
    assert set(json.loads(doc["source_entry_ids"])) == {"e0", "eA", "eB"}
    assert doc["merge_version"] == "2"  # two winning commits on top of v0
    assert doc["topic"] == "work"
    # The retry re-embedded the fresh merged text (3 embed calls total).
    assert embedder.calls == 3


@pytest.mark.asyncio
async def test_concurrent_merge_conflict_budget_exhausted_appends_safely(monkeypatch):
    """Attempts=1: the loser cannot retry, so it appends a fresh doc with
    its own text — winner merged, loser appended, nothing lost."""
    monkeypatch.setattr(Config, "T2_MERGE_MIN_COSINE", 0.60)
    monkeypatch.setattr(Config, "T2_MERGE_MAX_CHARS", 1500)
    monkeypatch.setattr(merge_module, "MERGE_MAX_ATTEMPTS", 1)
    fake = TxFakeRedis(dim=DIM)
    store = _make_store(fake, BarrierEmbedder())
    seed_id = await _seed(fake, store)

    async def worker(text: str, entry: str, hour: int) -> str:
        return await store.store_summary(
            user_id="u1",
            summary=text,
            embedding=[0.1] * DIM,
            topic="work",
            importance=3,
            period_start=_ts(2026, 7, 2, hour),
            period_end=_ts(2026, 7, 2, hour + 1),
            source_entry_ids=[entry],
        )

    sid_a, sid_b = await asyncio.gather(
        worker("Trưa review PR auth.", "eA", 4),
        worker("Chiều fix xong bug auth.", "eB", 6),
    )

    assert sid_a != sid_b
    docs = fake.summary_hashes()
    assert len(docs) == 2
    assert f"timeline:summary:{seed_id}" in docs
    texts = [d["summary"] for d in docs.values()]
    assert any("Trưa review PR auth." in t for t in texts)
    assert any("Chiều fix xong bug auth." in t for t in texts)
    all_ids = set()
    for d in docs.values():
        all_ids.update(json.loads(d["source_entry_ids"]))
    assert {"e0", "eA", "eB"} <= all_ids
    # Each reported id resolves to a doc holding that worker's own text.
    for sid, text in ((sid_a, "Trưa review PR auth."), (sid_b, "Chiều fix xong bug auth.")):
        assert text in docs[f"timeline:summary:{sid}"]["summary"]


@pytest.mark.asyncio
async def test_commit_merge_rejects_stale_version():
    """Writer-level: a commit against a moved merge_version raises
    MergeConflict and leaves the winner's HASH untouched."""
    fake = TxFakeRedis(dim=DIM)
    writer = DiaryWriter(fake, prefix="timeline:summary")
    key = "timeline:summary:doc"
    await fake.hset(key, mapping={"summary": "v0", "merge_version": 0})

    # Rival wins first.
    await writer.commit_merge(
        key, "doc", 0, {"summary": "v0\nrival"}, 90 * 86400,
    )
    assert (await fake.hgetall(key))[b"summary"] == b"v0\nrival"

    # Stale commit (still expecting v0) must not overwrite.
    with pytest.raises(MergeConflict):
        await writer.commit_merge(
            key, "doc", 0, {"summary": "v0\nstale"}, 90 * 86400,
        )
    assert (await fake.hgetall(key))[b"summary"] == b"v0\nrival"
    assert (await fake.hgetall(key))[b"merge_version"] == b"1"


@pytest.mark.asyncio
async def test_commit_merge_legacy_client_writes_directly():
    """Non-transactional doubles (no pipeline): non-idempotent merges keep
    the legacy direct-write behavior instead of crashing."""

    class PlainRedis:
        def __init__(self):
            self.hsets: list[dict] = []
            self.expires: list[tuple] = []

        async def hset(self, key, mapping):
            self.hsets.append({"key": key, "mapping": mapping})

        async def expire(self, key, seconds):
            self.expires.append((key, seconds))

    plain = PlainRedis()
    writer = DiaryWriter(plain, prefix="timeline:summary")
    assert writer.supports_transactions() is False
    out = await writer.commit_merge("k", "sid", 0, {"summary": "m"}, 100)
    assert out is None
    assert plain.hsets[0]["mapping"]["summary"] == "m"
    assert plain.expires == [("k", 100)]


def test_normalize_topic():
    assert merge_module.normalize_topic(" Work ") == "work"
    assert merge_module.normalize_topic("WORK") == "work"
    assert merge_module.normalize_topic(None) == ""
    assert merge_module.normalize_topic("") == ""
