"""Real-Redis T1 Lua regression (observe/trim/clear via actual EVAL).

Opt-in only: MARCH7_TEST_REDIS=1 + reachable disposable Redis Stack
(default 127.0.0.1:6379, override MARCH7_TEST_REDIS_URL), else skip.
Unique scope_ids per test; cleanup deletes ONLY own T1 keys (targeted
DEL/UNLINK via scan of own prefix). Never FLUSHALL/FLUSHDB, never T2 keys.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration

from twin.shared.config.settings import Config
from twin.shared.memory.active import ActiveMemory, FastPathDetector
from twin.shared.memory.active.models import ActiveEntry
from twin.shared.memory.active.store import ActiveStore
from twin.shared.memory.vn_time import vn_day_str

SCOPE = "user"


def _redis_addr() -> tuple[str, int]:
    url = os.environ.get("MARCH7_TEST_REDIS_URL", "redis://127.0.0.1:6379/0")
    host_port = url.split("://", 1)[1].split("/", 1)[0]
    host, _, port = host_port.partition(":")
    return host or "127.0.0.1", int(port or 6379)


def _needs_redis():
    if os.environ.get("MARCH7_TEST_REDIS") != "1":
        pytest.skip("MARCH7_TEST_REDIS!=1 (needs disposable Redis Stack)")
    host, port = _redis_addr()
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError:
        pytest.skip(f"no Redis at {host}:{port}")


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _client():
    import redis.asyncio as aioredis

    host, port = _redis_addr()
    return aioredis.Redis(host=host, port=port, db=0, decode_responses=False)


async def _cleanup(client, scope: str, scope_id: str) -> None:
    idx = f"active_index:{scope}:{scope_id}".encode()
    st = f"active_state:{scope}:{scope_id}".encode()
    summ = f"active_summarized:{scope}:{scope_id}".encode()
    prefix = f"active:{scope}:{scope_id}:".encode()
    arch = f"t1:archive:{scope}:{scope_id}:".encode()
    try:
        docs = [k async for k in client.scan_iter(match=prefix + b"*", count=1000)]
        if docs:
            for i in range(0, len(docs), 1000):
                await client.delete(*docs[i:i + 1000])
        archs = [k async for k in client.scan_iter(match=arch + b"*", count=200)]
        if archs:
            await client.delete(*archs)
        await client.delete(idx, st, summ)
    except Exception:
        pass


async def _require_json(client) -> None:
    probe = f"__t1_probe__:{uuid.uuid4().hex}"
    try:
        await client.execute_command("JSON.SET", probe, "$", '{"a":1}')
    except Exception:
        pytest.skip("server has no RedisJSON module")
    finally:
        try:
            await client.delete(probe)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_t1_observe_100_cross_client_real_eval():
    """2 indep clients x50 concurrent observes: 100 uniq docs, exact tokens/ts."""
    _needs_redis()
    a, b = _client(), _client()
    sid = _uid("t1obs")
    try:
        await _require_json(a)
        m1 = ActiveMemory(store=ActiveStore(a), detector=FastPathDetector(),
                          token_counter=lambda _: 10)
        m2 = ActiveMemory(store=ActiveStore(b), detector=FastPathDetector(),
                          token_counter=lambda _: 10)
        out = await asyncio.wait_for(asyncio.gather(
            *[m1.observe(SCOPE, sid, "user", f"a-{i}") for i in range(50)],
            *[m2.observe(SCOPE, sid, "user", f"b-{i}") for i in range(50)],
        ), timeout=60)
        assert len({e.entry_id for e in out}) == 100
        state = await m1.store.get_state(SCOPE, sid)
        assert state["unsummarized_tokens"] == 1000
        assert state["last_entry_ts"] is not None
        hi = max(e.created_at.timestamp() for e in out)
        assert abs(state["last_entry_ts"] - hi) < 1e-6
        assert await a.zcard(f"active_index:{SCOPE}:{sid}") == 100
        assert len(await m1.get_context(SCOPE, sid, limit=200)) == 100
        raw = await a.execute_command(
            "JSON.GET", f"active:{SCOPE}:{sid}:{out[0].entry_id}")
        doc = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        assert doc["tokens"] == 10 and doc["entry_id"] == out[0].entry_id
        # Lua max() must not regress: older-dated entry after newer max.
        old = ActiveEntry(scope=SCOPE, scope_id=sid, role="user",
                          content="old-after-new", tokens=10,
                          created_at=datetime.fromtimestamp(hi - 3600, tz=timezone.utc))
        await m1.store.observe_entry(old)
        state2 = await m1.store.get_state(SCOPE, sid)
        assert abs(state2["last_entry_ts"] - hi) < 1e-6
        assert state2["unsummarized_tokens"] == 1010
        assert await a.zcard(f"active_index:{SCOPE}:{sid}") == 101
    finally:
        try:
            await _cleanup(a, SCOPE, sid)
        finally:
            await a.aclose()
            await b.aclose()


@pytest.mark.asyncio
async def test_t1_trim_racing_observe_real_eval(monkeypatch):
    """Racing trim+observe: new/unsummarized tokens + max ts survive; retry
    idempotent via tail marker; archive grouped by VN day."""
    _needs_redis()
    monkeypatch.setattr(Config, "T1_ARCHIVE_ENABLED", True)
    monkeypatch.setattr(Config, "T1_ARCHIVE_TTL_DAYS", 90)
    client = _client()
    sid = _uid("t1trim")
    try:
        await _require_json(client)
        store = ActiveStore(client)
        mem = ActiveMemory(store=store, detector=FastPathDetector(),
                           token_counter=lambda _: 10)
        base = datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc).timestamp()
        ids: list[str] = []
        for i in range(8):
            # First 3 on VN 07-02, rest on VN 07-03 (span midnight).
            ts = base + (i * 3600 if i < 3 else 7200 + i * 3600)
            e = ActiveEntry(scope=SCOPE, scope_id=sid, role="user",
                            content=f"m-{i}", tokens=10,
                            created_at=datetime.fromtimestamp(ts, tz=timezone.utc))
            await store.observe_entry(e)
            ids.append(e.entry_id)
        summarized = ids[:6]
        _, r0, r1 = await asyncio.wait_for(asyncio.gather(
            mem.trim(SCOPE, sid, summarized, keep_recent=5),
            mem.observe(SCOPE, sid, "user", "race-0"),
            mem.observe(SCOPE, sid, "user", "race-1"),
        ), timeout=60)
        state = await store.get_state(SCOPE, sid)
        # 6 summarized subtracted once (deleted or tail-marked); 2 old + 2 new remain.
        assert state["unsummarized_tokens"] == 40
        hi = max(r0.created_at.timestamp(), r1.created_at.timestamp())
        assert abs(state["last_entry_ts"] - hi) < 1e-6
        idx_key = f"active_index:{SCOPE}:{sid}"
        members = {m.decode() for m in await client.zrange(idx_key, 0, -1)}
        # Either linearization: raced-before-Lua -> 5 survivors, after -> 7.
        assert len(members) in (5, 7)
        assert {r0.entry_id, r1.entry_id, ids[5], ids[6], ids[7]} <= members
        assert set(ids[:3]).isdisjoint(members)
        # Idempotent retry: same trim changes nothing.
        await mem.trim(SCOPE, sid, summarized, keep_recent=5)
        assert (await store.get_state(SCOPE, sid))["unsummarized_tokens"] == 40
        assert {m.decode() for m in await client.zrange(idx_key, 0, -1)} == members
        # Archive grouping: deleted ids across both VN days, TTL refreshed.
        days = {vn_day_str(base), vn_day_str(base + 7200 + 7 * 3600)}
        assert len(days) == 2
        got: list[str] = []
        for day in days:
            key = f"t1:archive:{SCOPE}:{sid}:{day}"
            vals = await client.lrange(key, 0, -1)
            assert await client.ttl(key.encode()) > 0
            got += [json.loads(v.decode())["entry_id"] for v in vals]
        assert sorted(got) == sorted(ids[:3])
    finally:
        try:
            await _cleanup(client, SCOPE, sid)
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_t1_clear_over_10k_real_eval_unrelated_survives():
    """Clear removes >10k indexed docs + orphan via Lua SCAN; other scope intact."""
    _needs_redis()
    client = _client()
    main, other = _uid("t1clr"), _uid("t1keep")
    try:
        await _require_json(client)
        store = ActiveStore(client)
        ostore = ActiveStore(client)
        keep_ids = []
        for i in range(3):
            e = ActiveEntry(scope=SCOPE, scope_id=other, role="user",
                            content=f"k-{i}", tokens=5)
            await ostore.observe_entry(e)
            keep_ids.append(e.entry_id)
        keep_state = await ostore.get_state(SCOPE, other)
        n = 10_500
        idx = f"active_index:{SCOPE}:{main}"
        st = f"active_state:{SCOPE}:{main}"
        summ = f"active_summarized:{SCOPE}:{main}"
        prefix = f"active:{SCOPE}:{main}:"
        for lo in range(0, n, 1000):
            pipe = client.pipeline(transaction=False)
            batch = {f"synth-{i:06d}": float(1000 + i)
                     for i in range(lo, min(lo + 1000, n))}
            for eid, score in batch.items():
                pipe.execute_command("JSON.SET", prefix + eid, "$", '{"tokens": 2}')
            pipe.zadd(idx, batch)
            await pipe.execute()
        await client.hset(st, mapping={"unsummarized_tokens": str(n * 2),
                                       "last_entry_ts": str(float(1000 + n))})
        await client.sadd(summ, "synth-000001", "synth-000002")
        await client.execute_command(
            "JSON.SET", prefix + "orphan-legacy", "$", '{"tokens": 9}')
        assert await client.zcard(idx) == n
        await store.clear_scope(SCOPE, main)
        assert await client.exists(idx) == 0
        assert await client.exists(st) == 0
        assert await client.exists(summ) == 0
        leftovers = [k async for k in client.scan_iter(
            match=(prefix + "*").encode(), count=1000)]
        assert leftovers == []
        assert await store.get_state(SCOPE, main) == {
            "unsummarized_tokens": 0, "last_entry_ts": None, "recent_catalogs": []}
        # Unrelated scope untouched (no global deletes).
        assert [e.entry_id for e in await ostore.list_entries(SCOPE, other)] == keep_ids
        assert await ostore.get_state(SCOPE, other) == keep_state
    finally:
        try:
            await _cleanup(client, SCOPE, main)
            await _cleanup(client, SCOPE, other)
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_t1_clear_observe_concurrent_consistent_real_eval():
    """Concurrent clear+observes: survivors index/state/docs agree, no orphans."""
    _needs_redis()
    client = _client()
    sid, other = _uid("t1race"), _uid("t1racekeep")
    try:
        await _require_json(client)
        store = ActiveStore(client)
        mem = ActiveMemory(store=store, detector=FastPathDetector(),
                           token_counter=lambda _: 5)
        for i in range(3):
            await mem.observe(SCOPE, sid, "user", f"old-{i}")
        keep = await mem.observe(SCOPE, other, "user", "keep")
        await asyncio.wait_for(asyncio.gather(
            store.clear_scope(SCOPE, sid),
            *[mem.observe(SCOPE, sid, "user", f"new-{i}") for i in range(10)],
        ), timeout=60)
        idx = f"active_index:{SCOPE}:{sid}"
        prefix = f"active:{SCOPE}:{sid}:"
        members = {m.decode() if isinstance(m, bytes) else m
                   for m in await client.zrange(idx, 0, -1)}
        docs = [k async for k in client.scan_iter(
            match=(prefix + "*").encode(), count=1000)]
        doc_ids = {k.decode().split(":")[-1] for k in docs}
        assert members == doc_ids  # no orphans either direction
        total = 0
        for eid in members:
            raw = await client.execute_command("JSON.GET", prefix + eid)
            total += int(json.loads(raw.decode())["tokens"])
        state = await store.get_state(SCOPE, sid)
        assert state["unsummarized_tokens"] == total
        if members:
            scores = await client.zrange(idx, 0, -1, withscores=True)
            assert abs(state["last_entry_ts"] - max(s for _, s in scores)) < 1e-6
        else:
            assert state["last_entry_ts"] is None
        assert (await store.get_entry(SCOPE, other, keep.entry_id)) is not None
    finally:
        try:
            await _cleanup(client, SCOPE, sid)
            await _cleanup(client, SCOPE, other)
        finally:
            await client.aclose()
