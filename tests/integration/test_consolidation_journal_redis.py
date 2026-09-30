"""Real-Redis consolidation-journal regression (actual Lua EVAL semantics).

Opt-in only: MARCH7_TEST_REDIS=1 + reachable disposable Redis Stack
(default 127.0.0.1:6379, override MARCH7_TEST_REDIS_URL), else skip.
Unique scope_ids per test; cleanup deletes ONLY own consol:* keys
(targeted SCAN of own scope + DEL). Never FLUSHALL/FLUSHDB, never T2 keys.

Covers what fakes cannot prove: atomic all-or-none claim under real EVAL,
concurrent same-batch single-winner, disjoint progress, matching-only
release, pending-no-expiry vs completion-TTL, and the caller claim/ack/
clear lifecycle including races.
"""
from __future__ import annotations

import os
import socket
import uuid

import pytest

pytestmark = pytest.mark.integration

from twin.shared.memory.consolidation_journal import (
    PLAN_CACHE_TTL_SECONDS,
    CallerJournal,
    ReceiverJournal,
    batch_id_of_plan_key,
    build_plan_cache_key,
    caller_pending_key,
)


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


def _uid() -> str:
    return f"j-{uuid.uuid4().hex[:12]}"


def _client():
    import redis.asyncio as aioredis

    host, port = _redis_addr()
    return aioredis.Redis(host=host, port=port, db=0, decode_responses=False)


async def _cleanup(client, scope_id: str) -> None:
    patterns = [f"consol:own:user:{scope_id}:*",
                f"consol:batch:user:{scope_id}:*",
                f"consol:plan:user:{scope_id}:*"]
    try:
        for pattern in patterns:
            keys = [k async for k in client.scan_iter(match=pattern, count=500)]
            if keys:
                await client.delete(*keys)
        await client.delete(caller_pending_key("user", scope_id))
    except Exception:
        pass


def _entries(*ids: str) -> list[dict]:
    return [{"entry_id": i, "role": "user", "content": f"body {i}"} for i in ids]


async def _cleanup_guarded(client, scope_id: str) -> None:
    patterns = [f"active:user:{scope_id}:*", f"t1:archive:user:{scope_id}:*",
                f"consol:own:user:{scope_id}:*", f"consol:batch:user:{scope_id}:*",
                f"consol:plan:user:{scope_id}:*"]
    singles = [f"active_index:user:{scope_id}", f"active_state:user:{scope_id}",
               f"active_summarized:user:{scope_id}", caller_pending_key("user", scope_id)]
    try:
        for pattern in patterns:
            keys = [k async for k in client.scan_iter(match=pattern, count=500)]
            if keys:
                await client.delete(*keys)
        for k in singles:
            try:
                await client.delete(k)
            except Exception:
                pass
    except Exception:
        pass


@pytest.mark.asyncio
async def test_receiver_claim_all_or_none_on_conflict():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        a = _entries("e1", "e2")
        got = await journal.claim_new("user", sid, a, "plan-A")
        assert got.status == "claimed"

        # Overlapping different batch: rejected, and NOTHING of it persists
        # (e3 owner absent, B plan absent) while A's witnesses stay intact.
        b = _entries("e2", "e3")
        denied = await journal.claim_new("user", sid, b, "plan-B")
        assert denied.status == "conflict"
        assert await client.get(f"consol:own:user:{sid}:e3") is None
        assert await client.get(build_plan_cache_key("user", sid, b)) is None
        assert await client.get(f"consol:own:user:{sid}:e1") is not None
        assert await client.get(f"consol:own:user:{sid}:e2") is not None
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_receiver_concurrent_same_batch_single_winner():
    import asyncio

    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        entries = _entries("e1", "e2")
        ra, rb = await asyncio.gather(
            journal.claim_new("user", sid, [dict(e) for e in entries], "plan-A"),
            journal.claim_new("user", sid, [dict(e) for e in entries], "plan-B"),
        )
        assert sorted([ra.status, rb.status]) == ["adopted", "claimed"]
        assert ra.payload == rb.payload
        assert ra.payload in ("plan-A", "plan-B")
        key = build_plan_cache_key("user", sid, entries)
        assert (await client.get(key)) in (b"plan-A", b"plan-B")
        assert await client.ttl(key) == -1  # pending: no expiry
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_receiver_disjoint_batches_both_claimed():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        assert (await journal.claim_new("user", sid, _entries("e1"), "p1")).status == "claimed"
        assert (await journal.claim_new("user", sid, _entries("e2"), "p2")).status == "claimed"
        o1 = await client.get(f"consol:own:user:{sid}:e1")
        o2 = await client.get(f"consol:own:user:{sid}:e2")
        assert o1 is not None and o2 is not None and o1 != o2
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_receiver_release_matching_only_and_completion_ttl():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        a = _entries("e1", "e2")
        b = _entries("e3",)
        ca = await journal.claim_new("user", sid, a, "plan-A")
        assert ca.status == "claimed"
        assert (await journal.claim_new("user", sid, b, "plan-B")).status == "claimed"
        ka = build_plan_cache_key("user", sid, a)
        ba = batch_id_of_plan_key(ka)

        rel = await journal.release("user", sid, ["e1", "e2"], ba, ka, ca.generation)
        assert rel.ok and rel.released == 2
        assert await client.get(f"consol:own:user:{sid}:e1") is None
        assert await client.get(f"consol:own:user:{sid}:e2") is None
        # Foreign batch untouched.
        assert await client.get(f"consol:own:user:{sid}:e3") is not None
        # Completed plan: fresh TTL starting now; batch record dropped.
        ttl = await client.ttl(ka)
        assert PLAN_CACHE_TTL_SECONDS - 60 <= ttl <= PLAN_CACHE_TTL_SECONDS
        assert await client.get(f"consol:batch:user:{sid}:{ba}") is None
        # Idempotent retry: success, nothing more released.
        again = await journal.release("user", sid, ["e1", "e2"], ba, ka, ca.generation)
        assert again.ok and again.released == 0
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_receiver_probe_exact_conflict_fresh():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        a = _entries("e1", "e2")
        assert (await journal.claim_new("user", sid, a, "plan-A")).status == "claimed"
        exact = await journal.probe("user", sid, [dict(e) for e in a])
        assert exact.status == "exact"
        assert exact.plan_key == build_plan_cache_key("user", sid, a)
        grown = await journal.probe("user", sid, _entries("e1", "e2", "e3"))
        assert grown.status == "conflict"
        fresh = await journal.probe("user", sid, _entries("e9"))
        assert fresh.status == "fresh"
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_caller_claim_ack_clear_lifecycle():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        from twin.shared.memory.consolidation_journal import build_caller_pending

        journal = CallerJournal(client)
        assert (await journal.load("user", sid)).status == "missing"
        pending = build_caller_pending("user", sid, _entries("e1", "e2"))
        status, _ = await journal.claim("user", sid, pending)
        assert status == "claimed"
        status2, existing = await journal.claim(
            "user", sid, build_caller_pending("user", sid, _entries("e9")))
        assert status2 == "exists"  # concurrent rival converges, never replaces
        assert existing["entry_ids"] == ["e1", "e2"]

        acked, rec1 = await journal.mark_acknowledged(
            "user", sid, pending, ["e1", "e2"],
            {"status": "ok", "entry_ids": ["e1", "e2"],
             "topics_failed": 0, "has_meaningful_content": True,
             "topics_stored": 1, "receiver_generation": "a" * 32})
        assert acked == "acked" and isinstance(rec1, dict)
        assert (await journal.load("user", sid)).status == "acknowledged"
        again, rec2 = await journal.mark_acknowledged(
            "user", sid, pending, ["e1", "e2"],
            {"status": "ok", "entry_ids": ["e1", "e2"],
             "topics_failed": 0, "has_meaningful_content": True,
             "topics_stored": 1, "receiver_generation": "a" * 32})
        assert again == "already" and isinstance(rec2, dict)
        assert rec2["trim_ids"] == ["e1", "e2"]
        assert await client.ttl(caller_pending_key("user", sid)) == -1

        bad, _ = await journal.clear("user", sid, ["e1"], pending["caller_nonce"])
        assert bad == "mismatch"
        cleared, _ = await journal.clear("user", sid, ["e1", "e2"], pending["caller_nonce"])
        assert cleared == "cleared"
        assert (await journal.load("user", sid)).status == "missing"
        cleared2, _ = await journal.clear("user", sid, ["e1", "e2"], pending["caller_nonce"])
        assert cleared2 == "cleared"  # idempotent
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_caller_clear_refuses_unacknowledged():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        from twin.shared.memory.consolidation_journal import build_caller_pending

        journal = CallerJournal(client)
        pending = build_caller_pending("user", sid, _entries("e1"))
        assert (await journal.claim("user", sid, pending))[0] == "claimed"
        status, _ = await journal.clear("user", sid, ["e1"])
        assert status == "not_acked"
        assert (await journal.load("user", sid)).status == "pending"
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_caller_concurrent_claim_single_pending():
    import asyncio

    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        from twin.shared.memory.consolidation_journal import build_caller_pending

        journal = CallerJournal(client)
        pa = build_caller_pending("user", sid, _entries("e1", "e2"))
        pb = build_caller_pending("user", sid, _entries("e3"))
        ra, rb = await asyncio.gather(
            journal.claim("user", sid, pa),
            journal.claim("user", sid, pb),
        )
        assert sorted([ra[0], rb[0]]) == ["claimed", "exists"]
        loaded = await journal.load("user", sid)
        assert loaded.status == "pending"
        assert loaded.record is not None
        assert loaded.record["entry_ids"] in (["e1", "e2"], ["e3"])
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_receiver_adopt_persists_ttl_plan():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        entries = _entries("e1")
        key = build_plan_cache_key("user", sid, entries)
        c1 = await journal.claim_new("user", sid, entries, "plan-gen1")
        assert c1.status == "claimed"
        ba = batch_id_of_plan_key(key)
        rel = await journal.release("user", sid, ["e1"], ba, key, c1.generation)
        assert rel.ok
        assert await client.ttl(key) > 0
        got = await journal.claim_new("user", sid, entries, "plan-gen2")
        assert got.status == "adopted"
        assert got.payload == "plan-gen1"
        assert await client.ttl(key) == -1
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_caller_ack_winner_and_cas():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        from twin.shared.memory.consolidation_journal import build_caller_pending
        journal = CallerJournal(client)
        pending = build_caller_pending("user", sid, _entries("e1", "e2"))
        assert (await journal.claim("user", sid, pending))[0] == "claimed"
        st1, rec1 = await journal.mark_acknowledged(
            "user", sid, pending, ["e1", "e2"],
            {"status": "ok", "entry_ids": ["e1", "e2"],
             "topics_failed": 0, "has_meaningful_content": True,
             "topics_stored": 1, "winner": 1,
             "receiver_generation": "b" * 32})
        assert st1 == "acked" and rec1["ack"]["winner"] == 1
        st2, rec2 = await journal.mark_acknowledged(
            "user", sid, pending, ["e1", "e2"],
            {"status": "ok", "entry_ids": ["e1", "e2"],
             "topics_failed": 0, "has_meaningful_content": True,
             "topics_stored": 1, "winner": 2,
             "receiver_generation": "b" * 32})
        assert st2 == "already" and rec2["ack"]["winner"] == 1
        drifted = dict(pending, fingerprints=["a" * 64, "b" * 64])
        st3, _ = await journal.mark_acknowledged(
            "user", sid, drifted, ["e1", "e2"],
            {"status": "ok", "entry_ids": ["e1", "e2"],
             "topics_failed": 0, "has_meaningful_content": True,
             "topics_stored": 1, "receiver_generation": "b" * 32})
        assert st3 == "mismatch"
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_real_privacy_no_raw_in_journals():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        from twin.shared.memory.consolidation_journal import build_caller_pending
        needle = "SYNTHETIC-REAL-NEEDLE-7h2k9"
        entries = [{"entry_id": "e1", "role": "user", "content": f"hello {needle}"}]
        cj, rj = CallerJournal(client), ReceiverJournal(client)
        pend = build_caller_pending("user", sid, entries)
        assert (await cj.claim("user", sid, pend))[0] == "claimed"
        assert (await rj.claim_new("user", sid, entries, "plan")).status == "claimed"
        keys = [k async for k in client.scan_iter(match=f"consol:*:user:{sid}:*", count=500)]
        keys.append(caller_pending_key("user", sid).encode() if isinstance(caller_pending_key("user", sid), str) else caller_pending_key("user", sid))
        for k in keys:
            v = await client.get(k)
            if v is not None:
                assert needle.encode() not in (v if isinstance(v, bytes) else str(v).encode())
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_real_generation_stale_release_keeps_new():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        entries = _entries("e1", "e2")
        key = build_plan_cache_key("user", sid, entries)
        bid = batch_id_of_plan_key(key)
        c1 = await journal.claim_new("user", sid, entries, "plan-g1")
        assert c1.status == "claimed"
        assert (await journal.release("user", sid, ["e1", "e2"], bid, key, c1.generation)).ok
        await client.delete(key)
        c2 = await journal.claim_new("user", sid, entries, "plan-g2")
        assert c2.status == "claimed" and c2.generation != c1.generation
        stale = await journal.release("user", sid, ["e1", "e2"], bid, key, c1.generation)
        assert not stale.ok
        o1 = await client.get(f"consol:own:user:{sid}:e1")
        assert o1 is not None and o1.decode().endswith(c2.generation)
        assert await client.ttl(key) == -1
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_real_release_foreign_no_partial_no_ttl():
    _needs_redis()
    sid = _uid()
    client = _client()
    try:
        journal = ReceiverJournal(client)
        a = _entries("e1", "e2")
        ca = await journal.claim_new("user", sid, a, "plan-A")
        assert ca.status == "claimed"
        ka = build_plan_cache_key("user", sid, a)
        ba = batch_id_of_plan_key(ka)
        await client.set(f"consol:own:user:{sid}:e2", "foreign:" + "f" * 32)
        rel = await journal.release("user", sid, ["e1", "e2"], ba, ka, ca.generation)
        assert not rel.ok
        assert await client.get(f"consol:own:user:{sid}:e1") is not None
        assert await client.get(f"consol:own:user:{sid}:e2") is not None
        assert await client.ttl(ka) == -1
    finally:
        await _cleanup(client, sid)
        await client.aclose()


@pytest.mark.asyncio
async def test_real_guarded_reset_restore_rejects():
    import asyncio as _aio
    from datetime import datetime, timezone as _tz
    from twin.shared.memory.active.detector import FastPathDetector as _Det
    from twin.shared.memory.active.models import ActiveEntry as _Ent
    from twin.shared.memory.active.service import ActiveMemory as _Mem
    from twin.shared.memory.active.store import ActiveStore as _Store
    from twin.shared.memory.consolidation_coordinator import ConsolidationCoordinator as _Co
    from twin.shared.memory.consolidation_journal import build_caller_pending as _mkpend
    _needs_redis()
    sid = _uid()
    t1_client, rcv_client = _client(), _client()
    try:
        store, mem = _Store(t1_client), None
        mem = _Mem(store=store, detector=_Det(), token_counter=lambda _: 10)
        cj, rj = CallerJournal(t1_client), ReceiverJournal(rcv_client)
        old = _Ent(entry_id="e1", scope="user", scope_id=sid, role="user", content="OLDFACT-A", author_id="u1", author_name="U", message_id="m-e1-0", created_at=datetime(2026, 7, 1, 3, 0, tzinfo=_tz.utc), tokens=10)
        await store.observe_entry(old)
        ship = [{"entry_id": "e1", "message_id": old.message_id, "role": "user", "content": "OLDFACT-A", "author_id": "u1", "author_name": "U", "timestamp": old.created_at.isoformat()}]
        pend_old = _mkpend("user", sid, ship)
        assert (await cj.claim("user", sid, pend_old))[0] == "claimed"
        c0 = await rj.claim_new("user", sid, ship, "plan-old")
        assert c0.status == "claimed"
        ack = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0, "has_meaningful_content": True, "topics_stored": 1, "receiver_generation": c0.generation}
        st, stored = await cj.mark_acknowledged("user", sid, pend_old, ["e1"], dict(ack))
        assert st == "acked" and isinstance(stored, dict)
        co = _Co(mem, lambda: None, lambda: None, get_caller_redis=lambda: t1_client, get_receiver_redis=lambda: rcv_client)
        entered, release = _aio.Event(), _aio.Event()
        orig_eval = t1_client.eval
        async def _barrier(script, numkeys, *ka):
            s = script.decode() if isinstance(script, bytes) else str(script)
            if "T1_TRIM_GUARDED_V1" in s:
                entered.set()
                await release.wait()
            return await orig_eval(script, numkeys, *ka)
        t1_client.eval = _barrier  # type: ignore[method-assign]
        async def _fin():
            return await co._finish_acknowledged("user", sid, dict(stored))
        async def _int():
            await entered.wait()
            await mem.reset_scope("user", sid)
            new = _Ent(entry_id="e1", scope="user", scope_id=sid, role="user", content="NEWFACT-Z restored", author_id="u1", author_name="U", message_id="m-e1-0", created_at=datetime(2026, 7, 1, 3, 0, tzinfo=_tz.utc), tokens=10)
            await store.observe_entry(new)
            for i in range(1, 7):
                await store.observe_entry(_Ent(entry_id=f"en{i}", scope="user", scope_id=sid, role="user", content=f"fill {i}", author_id="u1", author_name="U", message_id=f"m-en{i}", created_at=datetime(2026, 7, 1, 3, i, tzinfo=_tz.utc), tokens=10))
            pend_new = _mkpend("user", sid, [{"entry_id": "e1", "message_id": new.message_id, "role": "user", "content": new.content, "author_id": "u1", "author_name": "U", "timestamp": new.created_at.isoformat()}])
            assert (await cj.claim("user", sid, pend_new))[0] == "claimed"
            release.set()
        fout, _ = await _aio.gather(_fin(), _int())
        t1_client.eval = orig_eval  # type: ignore[method-assign]
        assert fout.get("status") == "failed"
        cur = await store.get_entry("user", sid, "e1")
        assert cur is not None and cur.content == "NEWFACT-Z restored"
        assert (await cj.load("user", sid)).status == "pending"
    finally:
        await _cleanup_guarded(t1_client, sid)
        await _cleanup_guarded(rcv_client, sid)
        await t1_client.aclose()
        await rcv_client.aclose()


@pytest.mark.asyncio
async def test_real_guarded_same_id_save_rejects():
    import asyncio as _aio
    from datetime import datetime, timezone as _tz
    from twin.shared.memory.active.detector import FastPathDetector as _Det
    from twin.shared.memory.active.models import ActiveEntry as _Ent
    from twin.shared.memory.active.service import ActiveMemory as _Mem
    from twin.shared.memory.active.store import ActiveStore as _Store
    from twin.shared.memory.consolidation_coordinator import ConsolidationCoordinator as _Co
    from twin.shared.memory.consolidation_journal import build_caller_pending as _mkpend
    _needs_redis()
    sid = _uid()
    t1_client, rcv_client = _client(), _client()
    try:
        store = _Store(t1_client)
        mem = _Mem(store=store, detector=_Det(), token_counter=lambda _: 10)
        cj, rj = CallerJournal(t1_client), ReceiverJournal(rcv_client)
        e1 = _Ent(entry_id="e1", scope="user", scope_id=sid, role="user", content="orig", author_id="u1", author_name="U", message_id="m1", created_at=datetime(2026, 7, 1, 3, 0, tzinfo=_tz.utc), tokens=10)
        await store.observe_entry(e1)
        ship = [{"entry_id": "e1", "message_id": "m1", "role": "user", "content": "orig", "author_id": "u1", "author_name": "U", "timestamp": e1.created_at.isoformat()}]
        pend = _mkpend("user", sid, ship)
        assert (await cj.claim("user", sid, pend))[0] == "claimed"
        c0 = await rj.claim_new("user", sid, ship, "plan")
        assert c0.status == "claimed"
        ack = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0, "has_meaningful_content": True, "topics_stored": 1, "receiver_generation": c0.generation}
        st, stored = await cj.mark_acknowledged("user", sid, pend, ["e1"], dict(ack))
        assert st == "acked"
        co = _Co(mem, lambda: None, lambda: None, get_caller_redis=lambda: t1_client, get_receiver_redis=lambda: rcv_client)
        entered, release = _aio.Event(), _aio.Event()
        orig_eval = t1_client.eval
        async def _barrier(script, numkeys, *ka):
            s = script.decode() if isinstance(script, bytes) else str(script)
            if "T1_TRIM_GUARDED_V1" in s:
                entered.set()
                await release.wait()
            return await orig_eval(script, numkeys, *ka)
        t1_client.eval = _barrier  # type: ignore[method-assign]
        async def _fin():
            return await co._finish_acknowledged("user", sid, dict(stored))
        async def _int():
            await entered.wait()
            cur = await store.get_entry("user", sid, "e1")
            assert cur is not None
            cur.content = "MUTATED race"
            await store.save(cur)
            release.set()
        fout, _ = await _aio.gather(_fin(), _int())
        t1_client.eval = orig_eval  # type: ignore[method-assign]
        assert fout.get("status") == "failed"
        assert (await cj.load("user", sid)).status == "acknowledged"
        assert (await store.get_entry("user", sid, "e1")).content == "MUTATED race"
    finally:
        await _cleanup_guarded(t1_client, sid)
        await _cleanup_guarded(rcv_client, sid)
        await t1_client.aclose()
        await rcv_client.aclose()


@pytest.mark.asyncio
async def test_real_guarded_concurrent_subtract_once():
    import asyncio as _aio
    from datetime import datetime, timezone as _tz
    from twin.shared.memory.active.detector import FastPathDetector as _Det
    from twin.shared.memory.active.models import ActiveEntry as _Ent
    from twin.shared.memory.active.service import ActiveMemory as _Mem
    from twin.shared.memory.active.store import ActiveStore as _Store
    from twin.shared.memory.consolidation_journal import build_caller_pending as _mkpend
    _needs_redis()
    sid = _uid()
    t1_client, rcv_client = _client(), _client()
    try:
        store = _Store(t1_client)
        mem = _Mem(store=store, detector=_Det(), token_counter=lambda _: 10)
        cj, rj = CallerJournal(t1_client), ReceiverJournal(rcv_client)
        batch = [_Ent(entry_id="e1", scope="user", scope_id=sid, role="user", content="A", author_id="u1", author_name="U", message_id="m1", created_at=datetime(2026, 7, 1, 3, 0, tzinfo=_tz.utc), tokens=10), _Ent(entry_id="e2", scope="user", scope_id=sid, role="user", content="B", author_id="u1", author_name="U", message_id="m2", created_at=datetime(2026, 7, 1, 3, 1, tzinfo=_tz.utc), tokens=10)]
        for e in batch:
            await store.observe_entry(e)
        for i in range(5):
            await store.observe_entry(_Ent(entry_id=f"f{i}", scope="user", scope_id=sid, role="user", content=f"fill {i}", author_id="u1", author_name="U", message_id=f"mf{i}", created_at=datetime(2026, 7, 1, 3, 2 + i, tzinfo=_tz.utc), tokens=10))
        ship = [{"entry_id": e.entry_id, "message_id": e.message_id, "role": "user", "content": e.content, "author_id": "u1", "author_name": "U", "timestamp": e.created_at.isoformat()} for e in batch]
        pend = _mkpend("user", sid, ship)
        assert (await cj.claim("user", sid, pend))[0] == "claimed"
        c0 = await rj.claim_new("user", sid, ship, "plan")
        assert c0.status == "claimed"
        ack = {"status": "ok", "entry_ids": ["e1", "e2"], "topics_failed": 0, "has_meaningful_content": True, "topics_stored": 1, "receiver_generation": c0.generation}
        st, stored = await cj.mark_acknowledged("user", sid, pend, ["e1", "e2"], dict(ack))
        assert st == "acked"
        from twin.shared.memory.active.consolidation_trim import guarded_trim_batch
        r1, r2 = await _aio.gather(guarded_trim_batch(store, "user", sid, dict(stored), 5), guarded_trim_batch(store, "user", sid, dict(stored), 5))
        assert sorted([r1["status"], r2["status"]]) in (["already", "trimmed"], ["failed", "trimmed"])
        # Exactly-once: 70-20=50 (failed second still means single subtract).
        assert (await store.get_state("user", sid))["unsummarized_tokens"] == 50
        assert await store.get_entry("user", sid, "e1") is None
    finally:
        await _cleanup_guarded(t1_client, sid)
        await _cleanup_guarded(rcv_client, sid)
        await t1_client.aclose()
        await rcv_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["-9223372036854775807", "9223372036854775808"])
async def test_real_guarded_corrupt_counter_no_partial(bad):
    from datetime import datetime, timezone as _tz
    from twin.shared.memory.active.detector import FastPathDetector as _Det
    from twin.shared.memory.active.models import ActiveEntry as _Ent
    from twin.shared.memory.active.service import ActiveMemory as _Mem
    from twin.shared.memory.active.store import ActiveStore as _Store
    from twin.shared.memory.consolidation_journal import build_caller_pending as _mkpend
    _needs_redis()
    sid = _uid()
    t1_client, rcv_client = _client(), _client()
    try:
        store = _Store(t1_client)
        mem = _Mem(store=store, detector=_Det(), token_counter=lambda _: 1)
        cj, rj = CallerJournal(t1_client), ReceiverJournal(rcv_client)
        e1 = _Ent(entry_id="e1", scope="user", scope_id=sid, role="user", content="A", author_id="u1", author_name="U", message_id="m1", created_at=datetime(2026, 7, 1, 3, 0, tzinfo=_tz.utc), tokens=1)
        e2 = _Ent(entry_id="e2", scope="user", scope_id=sid, role="user", content="B", author_id="u1", author_name="U", message_id="m2", created_at=datetime(2026, 7, 1, 3, 1, tzinfo=_tz.utc), tokens=1)
        await store.observe_entry(e1)
        await store.observe_entry(e2)
        ship = [{"entry_id": e.entry_id, "message_id": e.message_id, "role": "user", "content": e.content, "author_id": "u1", "author_name": "U", "timestamp": e.created_at.isoformat()} for e in (e1, e2)]
        pend = _mkpend("user", sid, ship)
        assert (await cj.claim("user", sid, pend))[0] == "claimed"
        c0 = await rj.claim_new("user", sid, ship, "plan")
        assert c0.status == "claimed"
        ack = {"status": "ok", "entry_ids": ["e1", "e2"], "topics_failed": 0, "has_meaningful_content": True, "topics_stored": 1, "receiver_generation": c0.generation}
        st, stored = await cj.mark_acknowledged("user", sid, pend, ["e1", "e2"], dict(ack))
        assert st == "acked"
        index_key = _Store._index_key("user", sid)  # noqa: SLF001
        state_key = _Store._state_key("user", sid)  # noqa: SLF001
        summ_key = _Store._summarized_key("user", sid)  # noqa: SLF001
        caller_key = caller_pending_key("user", sid)
        await t1_client.hset(state_key, mapping={"unsummarized_tokens": bad})
        before_e1 = await t1_client.execute_command("JSON.GET", f"active:user:{sid}:e1")
        before_e2 = await t1_client.execute_command("JSON.GET", f"active:user:{sid}:e2")
        before_index = await t1_client.zrange(index_key, 0, -1)
        before_summ = await t1_client.smembers(summ_key)
        before_counter = await t1_client.hget(state_key, "unsummarized_tokens")
        before_caller = await t1_client.get(caller_key)
        from twin.shared.memory.active.consolidation_trim import guarded_trim_batch
        out = await guarded_trim_batch(store, "user", sid, dict(stored), 0)
        assert out.get("status") == "failed"
        assert out.get("reason") == "wrongtype"
        assert out.get("detail") == "counter"
        assert await store.get_entry("user", sid, "e1") is not None
        assert await store.get_entry("user", sid, "e2") is not None
        assert await t1_client.execute_command("JSON.GET", f"active:user:{sid}:e1") == before_e1
        assert await t1_client.execute_command("JSON.GET", f"active:user:{sid}:e2") == before_e2
        assert await t1_client.zrange(index_key, 0, -1) == before_index
        assert await t1_client.smembers(summ_key) == before_summ
        assert await t1_client.hget(state_key, "unsummarized_tokens") == before_counter
        assert await t1_client.get(caller_key) == before_caller
        assert (await cj.load("user", sid)).status == "acknowledged"
    finally:
        await _cleanup_guarded(t1_client, sid)
        await _cleanup_guarded(rcv_client, sid)
        await t1_client.aclose()
        await rcv_client.aclose()
