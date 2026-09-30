"""Real-Redis twin of the diary-merge CAS regression (#8).

Runs ONLY against a disposable Redis Stack provided by the operator:
requires MARCH7_TEST_REDIS=1 plus a reachable RediSearch-capable server
(default 127.0.0.1:6379, override MARCH7_TEST_REDIS_URL), otherwise
skips. Cleans up every key and the test index it creates.

Reference disposable setup (never the live march7-redis):
  docker run -d --name t2test --network none redis-stack...
  docker run --rm --network container:t2test <test-runner> ...
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration

from twin.shared.config.settings import Config
from twin.shared.memory.diary import TimelineSummaryStore

DIM = 8
TEST_USER = "900001"


def _redis_addr() -> tuple[str, int]:
    url = os.environ.get("MARCH7_TEST_REDIS_URL", "redis://127.0.0.1:6379/0")
    host_port = url.split("://", 1)[1].split("/", 1)[0]
    host, _, port = host_port.partition(":")
    return host or "127.0.0.1", int(port or 6379)


def _needs_disposable_redis():
    if os.environ.get("MARCH7_TEST_REDIS") != "1":
        pytest.skip("MARCH7_TEST_REDIS!=1 (needs a disposable Redis Stack)")
    host, port = _redis_addr()
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        pytest.skip(f"no Redis at {host}:{port}")


class BarrierEmbedder:
    """First two re-embed calls rendezvous so both workers commit from the
    same candidate version; retries pass through."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls = 0
        self._barrier = asyncio.Barrier(2)

    async def get_embedding(self, text: str) -> list[float]:
        self.calls += 1
        if self.calls <= 2:
            await asyncio.wait_for(self._barrier.wait(), timeout=30)
        return [0.5] * self.dim


def _ts(hour: int) -> float:
    return datetime(2026, 7, 2, hour, tzinfo=timezone.utc).timestamp()


async def _user_summary_keys(client, user: str) -> list[bytes]:
    keys = [k async for k in client.scan_iter(match=b"timeline:summary:*", count=200)]
    return [k for k in keys if await client.hget(k, "user_id") == user.encode()]


@pytest.mark.asyncio
async def test_concurrent_merge_real_redis_both_updates_survive(monkeypatch):
    _needs_disposable_redis()
    monkeypatch.setattr(Config, "T2_MERGE_MIN_COSINE", 0.60)
    monkeypatch.setattr(Config, "T2_MERGE_MAX_CHARS", 1500)
    import redis.asyncio as aioredis

    host, port = _redis_addr()
    client = aioredis.Redis(host=host, port=port, db=0, decode_responses=False)
    try:
        try:
            await client.execute_command("FT._LIST")
        except Exception:
            pytest.skip("server has no RediSearch module")
        store = TimelineSummaryStore(
            redis_client=client, embedding_dim=DIM,
            embedding_service=BarrierEmbedder(),
        )
        await store.initialize()

        seed_id = await store.store_summary(
            user_id=TEST_USER, summary="Sáng họp dự án X.",
            embedding=[0.1] * DIM, topic="work", importance=3,
            period_start=_ts(2), period_end=_ts(3), source_entry_ids=["e0"],
        )

        async def worker(text: str, entry: str, hour: int) -> str:
            return await store.store_summary(
                user_id=TEST_USER, summary=text, embedding=[0.1] * DIM,
                topic="work", importance=3,
                period_start=_ts(hour), period_end=_ts(hour + 1),
                source_entry_ids=[entry],
            )

        sid_a, sid_b = await asyncio.wait_for(asyncio.gather(
            worker("Trưa review PR auth.", "eA", 4),
            worker("Chiều fix xong bug auth.", "eB", 6),
        ), timeout=60)

        assert sid_a == sid_b == seed_id
        raw = await client.hgetall(f"timeline:summary:{seed_id}")
        text = raw[b"summary"].decode()
        assert "Trưa review PR auth." in text
        assert "Chiều fix xong bug auth." in text
        assert set(json.loads(raw[b"source_entry_ids"].decode())) == {"e0", "eA", "eB"}
        assert raw[b"merge_version"] == b"2"
    finally:
        try:
            user_keys = await _user_summary_keys(client, TEST_USER)
            if user_keys:
                await client.delete(*user_keys)
            markers = [k async for k in client.scan_iter(match=b"timeline:idem:*", count=200)]
            if markers:
                await client.delete(*markers)
            try:
                await client.execute_command("FT.DROPINDEX", "timeline_summaries")
            except Exception:
                pass
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_idempotent_retry_real_redis_no_duplicate_or_false_ack():
    """Same idempotency_key twice -> one HASH, same id; doc deleted ->
    retry re-stores instead of acknowledging the absent doc."""
    _needs_disposable_redis()
    import redis.asyncio as aioredis

    from twin.shared.memory.diary.idempotency import marker_key as _marker_key

    host, port = _redis_addr()
    client = aioredis.Redis(host=host, port=port, db=0, decode_responses=False)
    try:
        try:
            await client.execute_command("FT._LIST")
        except Exception:
            pytest.skip("server has no RediSearch module")
        store = TimelineSummaryStore(redis_client=client, embedding_dim=DIM)
        await store.initialize()

        kwargs = dict(
            user_id=TEST_USER, summary="Chiều fix bug.", embedding=[0.1] * DIM,
            topic="work", importance=3, period_start=_ts(4), period_end=_ts(5),
            source_entry_ids=["e1"], idempotency_key="real-m1",
        )
        sid1 = await store.store_summary(**kwargs)
        sid2 = await store.store_summary(**kwargs)
        assert sid1 == sid2
        assert len(await _user_summary_keys(client, TEST_USER)) == 1
        marker = _marker_key(TEST_USER, "work", ["e1"], "real-m1")
        assert (await client.get(marker)) == sid1.encode()
        assert await client.ttl(marker) > 0

        await client.delete(f"timeline:summary:{sid1}".encode())
        sid3 = await store.store_summary(**kwargs)
        assert sid3 != sid1
        assert (await client.get(marker)) == sid3.encode()
    finally:
        try:
            user_keys = await _user_summary_keys(client, TEST_USER)
            if user_keys:
                await client.delete(*user_keys)
            markers = [k async for k in client.scan_iter(match=b"timeline:idem:*", count=200)]
            if markers:
                await client.delete(*markers)
            try:
                await client.execute_command("FT.DROPINDEX", "timeline_summaries")
            except Exception:
                pass
        finally:
            await client.aclose()
