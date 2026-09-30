"""Guarded consolidation trim regressions (TOCTOU + gen + wrongtype).

Actual ActiveMemory/ActiveStore + FakeRedis (real Lua-mirror EVAL, no network).
Barrier via asyncio.Events (no sleeps). Each test uses isolated fakes.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from unit.memory.active_test import FakeRedis as T1FakeRedis
from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.active import ActiveEntry, ActiveMemory, FastPathDetector
from twin.shared.memory.active.store import ActiveStore
from twin.shared.memory.consolidation_coordinator import ConsolidationCoordinator
from twin.shared.memory.consolidation_journal import (
    CallerJournal,
    ReceiverJournal,
    batch_id_of_plan_key,
    build_caller_pending,
    build_plan_cache_key,
    caller_pending_key,
)


def _mkentry(eid, scope, sid, content, ts_min, **kw):
    return ActiveEntry(
        entry_id=eid, scope=scope, scope_id=sid, role=kw.get("role", "user"),
        content=content, author_id=kw.get("author_id", "u1"),
        author_name=kw.get("author_name", "U"),
        message_id=kw.get("message_id", f"m-{eid}-{ts_min}"),
        created_at=kw.get("created_at") or datetime(2026, 7, 1, 3, ts_min, tzinfo=timezone.utc),
        tokens=10,
    )


def _ship(entries: list[ActiveEntry]) -> list[dict]:
    return [{
        "entry_id": e.entry_id, "message_id": e.message_id, "role": e.role,
        "content": e.content, "author_id": e.author_id,
        "author_name": e.author_name, "timestamp": e.created_at.isoformat(),
    } for e in entries]


def _real_t1(redis=None):
    r = redis or T1FakeRedis()
    mem = ActiveMemory(store=ActiveStore(r), detector=FastPathDetector(),
                       token_counter=lambda _: 10)
    return mem, r


class BarrierRedis(T1FakeRedis):
    """Pause once at guarded EVAL for deterministic race (events, no sleep)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.pause_on_guarded = False
        self.guarded_seen = 0

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        if self.pause_on_guarded and "T1_TRIM_GUARDED_V1" in script:
            self.guarded_seen += 1
            self.entered.set()
            await self.release.wait()
        return await super().eval(script, numkeys, *keys_and_args)


async def _setup_acked(mem, t1_redis, rcv_redis, scope, sid, entries):
    cj = CallerJournal(t1_redis)
    rj = ReceiverJournal(rcv_redis)
    for e in entries:
        await mem.store.observe_entry(e)
    ship = _ship(entries)
    pend = build_caller_pending(scope, sid, ship)
    assert (await cj.claim(scope, sid, pend))[0] == "claimed"
    c0 = await rj.claim_new(scope, sid, ship, "plan-old")
    assert c0.status == "claimed"
    ack = {"status": "ok", "entry_ids": [e.entry_id for e in entries],
           "topics_failed": 0, "has_meaningful_content": True,
           "topics_stored": 1, "receiver_generation": c0.generation}
    st, stored = await cj.mark_acknowledged(
        scope, sid, pend, [e.entry_id for e in entries], dict(ack))
    assert st == "acked"
    assert isinstance(stored, dict)
    return cj, rj, pend, stored, c0


# ------------------------------------------------- reset/restore TOCTOU


@pytest.mark.asyncio
async def test_reset_restore_old_finisher_rejects_new_survives():
    scope, sid = "user", "guard1"
    t1_redis = BarrierRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    old_e1 = _mkentry("e1", scope, sid, "OLDFACT-A original body", 0)
    cj, rj, pend_old, stored, c0 = await _setup_acked(
        mem, t1_redis, rcv_redis, scope, sid, [old_e1])
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    t1_redis.pause_on_guarded = True

    async def finisher():
        return await co._finish_acknowledged(scope, sid, dict(stored))

    async def interferer():
        await t1_redis.entered.wait()
        await mem.reset_scope(scope, sid)
        new_e1 = _mkentry("e1", scope, sid, "NEWFACT-Z restored NEVER summarized", 0)
        await mem.store.observe_entry(new_e1)
        for i in range(1, 7):
            await mem.store.observe_entry(
                _mkentry(f"en{i}", scope, sid, f"newer filler {i}", i))
        new_ship = _ship([new_e1])
        pend_new = build_caller_pending(scope, sid, new_ship)
        st2, _ = await cj.claim(scope, sid, pend_new)
        assert st2 == "claimed"
        assert pend_new["caller_nonce"] != pend_old["caller_nonce"]
        t1_redis.release.set()
        return pend_new

    fout, pend_new = await asyncio.gather(finisher(), interferer())
    assert fout.get("status") == "failed"
    assert fout.get("ack_durable") is True
    # Guard rejected stale snapshots/nonce; NEW e1 survives unsummarized.
    cur = await mem.store.get_entry(scope, sid, "e1")
    assert cur is not None and cur.content.startswith("NEWFACT-Z")
    all_ids = [e.entry_id for e in await mem.store.list_entries(scope, sid, limit=50)]
    assert all_ids == ["e1", "en1", "en2", "en3", "en4", "en5", "en6"]
    assert "e1" not in await mem.store.list_summarized_ids(scope, sid)
    state = await mem.store.get_state(scope, sid)
    assert state["unsummarized_tokens"] == 70
    loaded = await cj.load(scope, sid)
    assert loaded.status == "pending"
    assert loaded.record is not None
    assert loaded.record["caller_nonce"] == pend_new["caller_nonce"]
    # Old receiver owners intact (release never ran for stale).
    assert rcv_redis._get_str(f"consol:own:{scope}:{sid}:e1") is not None


@pytest.mark.asyncio
async def test_missing_guarded_capability_fails_closed_without_trim():
    # Old-style T1 without new API must fail closed (not fallback to trim).
    scope, sid = "user", "guard1b"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "body", 0)
    _, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1])

    class OldT1:
        def __init__(self, real):
            self._real = real
            self.store = real.store
        async def get_entries_by_ids(self, s, si, ids):
            return await self._real.get_entries_by_ids(s, si, ids)
        async def trim(self, s, si, ids, keep_recent=None):
            raise AssertionError("ordinary trim must not be called")
        async def list_unsummarized_entries(self, s, si, limit=200):
            return await self._real.list_unsummarized_entries(s, si, limit=limit)

    co = ConsolidationCoordinator(
        OldT1(mem), lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    out = await co._finish_acknowledged(scope, sid, dict(stored))
    assert out.get("status") == "failed"
    assert "guarded" in str(out.get("detail", "")).lower()
    assert await mem.store.get_entry(scope, sid, "e1") is not None


# ------------------------------------------------- same-ID save race


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "role", "author", "timestamp", "message_id"])
async def test_same_id_save_mutation_rejects_no_partial(field):
    scope, sid = f"user", f"guard2-{field}"
    t1_redis = BarrierRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "original body", 0)
    e2 = _mkentry("e2", scope, sid, "second body", 1)
    cj, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1, e2])
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    before_state = await mem.store.get_state(scope, sid)
    before_docs = dict(t1_redis.docs)
    t1_redis.pause_on_guarded = True

    async def finisher():
        return await co._finish_acknowledged(scope, sid, dict(stored))

    async def interferer():
        await t1_redis.entered.wait()
        cur = await mem.store.get_entry(scope, sid, "e1")
        assert cur is not None
        if field == "content":
            cur.content = "MUTATED content race"
        elif field == "role":
            cur.role = "assistant"
        elif field == "author":
            cur.author_id = "mallory"
            cur.author_name = "Mallory"
        elif field == "timestamp":
            cur.created_at = datetime(2026, 7, 2, 3, 0, tzinfo=timezone.utc)
        elif field == "message_id":
            cur.message_id = "m-race-999"
        await mem.store.save(cur)
        t1_redis.release.set()

    fout, _ = await asyncio.gather(finisher(), interferer())
    assert fout.get("status") == "failed"
    assert fout.get("ack_durable") is True
    # No partial: both docs present, index intact, no markers, counter same.
    assert await mem.store.get_entry(scope, sid, "e1") is not None
    assert await mem.store.get_entry(scope, sid, "e2") is not None
    assert (await mem.store.get_state(scope, sid))["unsummarized_tokens"] == before_state["unsummarized_tokens"]
    assert await mem.store.list_summarized_ids(scope, sid) == set()
    # Caller still acknowledged (not trimmed).
    assert (await cj.load(scope, sid)).status == "acknowledged"
    # Mutated doc preserved (not deleted), second untouched.
    assert t1_redis.docs[f"active:{scope}:{sid}:e2"] == before_docs[f"active:{scope}:{sid}:e2"]


# ------------------------------------------------- concurrent finishers


@pytest.mark.asyncio
async def test_concurrent_finishers_subtract_once():
    scope, sid = "user", "guard3"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    batch = [_mkentry("e1", scope, sid, "A", 0), _mkentry("e2", scope, sid, "B", 1)]
    fillers = [_mkentry(f"f{i}", scope, sid, f"fill {i}", 2 + i) for i in range(5)]
    cj, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, batch)
    for f in fillers:
        await mem.store.observe_entry(f)
    assert (await mem.store.get_state(scope, sid))["unsummarized_tokens"] == 70
    # Direct guarded concurrency: one trimmed, one already, single subtract.
    from twin.shared.memory.active.consolidation_trim import guarded_trim_batch
    r1, r2 = await asyncio.gather(
        guarded_trim_batch(mem.store, scope, sid, dict(stored), 5),
        guarded_trim_batch(mem.store, scope, sid, dict(stored), 5),
    )
    assert sorted([r1["status"], r2["status"]]) == ["already", "trimmed"]
    assert (await mem.store.get_state(scope, sid))["unsummarized_tokens"] == 50
    assert await mem.store.get_entry(scope, sid, "e1") is None
    assert await mem.store.get_entry(scope, sid, "e2") is None
    assert (await cj.load(scope, sid)).status == "trimmed"


@pytest.mark.asyncio
async def test_stale_ack_after_trim_plus_restore_no_retrim():
    scope, sid = "user", "guard3b"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "orig", 0)
    cj, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1])
    from twin.shared.memory.active.consolidation_trim import guarded_trim_batch
    first = await guarded_trim_batch(mem.store, scope, sid, dict(stored), 5)
    assert first["status"] == "trimmed"
    # Same-ID restore via save while caller stays trimmed.
    restored = _mkentry("e1", scope, sid, "RESTORED after trim", 0)
    await mem.store.save(restored)
    # Stale in-memory ack (acknowledged copy) must converge idempotently
    # without deleting the restored entry.
    again = await guarded_trim_batch(mem.store, scope, sid, dict(stored), 5)
    assert again["status"] == "already"
    cur = await mem.store.get_entry(scope, sid, "e1")
    assert cur is not None and cur.content == "RESTORED after trim"


# ------------------------------------------------- lost response + release


@pytest.mark.asyncio
async def test_lost_eval_response_retry_without_llm_or_raw():
    scope, sid = "user", "guard4"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    e2 = _mkentry("e2", scope, sid, "B", 1)
    cj, rj, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1, e2])
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    orig_eval = t1_redis.eval
    fired = {"n": 0}

    async def flaky(script, numkeys, *ka):
        if "T1_TRIM_GUARDED_V1" in script and fired["n"] == 0:
            fired["n"] += 1
            res = await orig_eval(script, numkeys, *ka)
            assert res[0] == 1  # server actually trimmed
            raise OSError("lost response after commit")
        return await orig_eval(script, numkeys, *ka)

    t1_redis.eval = flaky  # type: ignore[method-assign]
    first = await co._finish_acknowledged(scope, sid, dict(stored))
    assert first["status"] == "failed" and first["error"] == "trim_failed"
    assert first["ack_durable"] is True
    assert (await cj.load(scope, sid)).status == "trimmed"
    # Retry via coordinator loads trimmed; must not touch raw T1 or LLM.
    t1_redis.eval = orig_eval  # type: ignore[method-assign]
    raw_calls = {"n": 0}
    orig_get = mem.store.get_entry
    orig_by_ids = mem.store.get_entries_by_ids

    async def _no_raw(*a, **k):
        raw_calls["n"] += 1
        raise AssertionError("retry must not read raw T1")

    mem.store.get_entry = _no_raw  # type: ignore[method-assign]
    mem.store.get_entries_by_ids = _no_raw  # type: ignore[method-assign]
    loaded = await cj.load(scope, sid)
    assert loaded.status == "trimmed" and loaded.record is not None
    second = await co._finish_acknowledged(scope, sid, dict(loaded.record))
    assert second.get("status") == "ok"
    assert raw_calls["n"] == 0
    assert (await cj.load(scope, sid)).status == "missing"
    mem.store.get_entry = orig_get  # type: ignore[method-assign]
    mem.store.get_entries_by_ids = orig_by_ids  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_release_failure_retains_trimmed_foreign_gen2_untouched():
    scope, sid = "user", "guard4b"
    t1_redis = T1FakeRedis()

    class FlakyRcv(JournalFakeRedis):
        def __init__(self):
            super().__init__()
            self.fail_next = False
        async def eval(self, script, numkeys, *ka):
            if self.fail_next and "RECEIVER_RELEASE_V1" in script:
                self.fail_next = False
                raise OSError("release blip")
            return await super().eval(script, numkeys, *ka)

    rcv_redis = FlakyRcv()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    cj, rj, _, stored, c0 = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1])
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    rcv_redis.fail_next = True
    first = await co._finish_acknowledged(scope, sid, dict(stored))
    assert first["status"] == "failed" and first["error"] == "release_failed"
    assert (await cj.load(scope, sid)).status == "trimmed"
    # Single entry stays in keep-tail (retained + marked, tokens cleared).
    assert await mem.store.get_entry(scope, sid, "e1") is not None
    assert (await mem.store.get_state(scope, sid))["unsummarized_tokens"] == 0
    assert "e1" in await mem.store.list_summarized_ids(scope, sid)
    # New gen2 batch claims same ID after release window; stale gen1 release
    # must not touch it, and retry of trimmed gen1 still succeeds idempotently.
    key = build_plan_cache_key(scope, sid, [{"entry_id": "e1", "role": "user", "content": "A2"}])
    # Release retry for gen1 (idempotent, owners still gen1).
    loaded = await cj.load(scope, sid)
    assert loaded.record is not None
    second = await co._finish_acknowledged(scope, sid, dict(loaded.record))
    assert second.get("status") == "ok"
    assert (await cj.load(scope, sid)).status == "missing"


# ------------------------------------------------- pre-journal gen validation


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_patch,err", [
    ({"receiver_generation": ""}, "bad_generation"),
    ({"receiver_generation": "ZZZ"}, "bad_generation"),
    ({"receiver_generation": "A" * 32}, "bad_generation"),
    ({"receiver_generation": "a" * 31}, "bad_generation"),
    ({}, "bad_generation"),
])
async def test_malformed_meaningful_gen_leaves_pending(ack_patch, err):
    scope, sid = "user", "guard5a"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    await mem.store.observe_entry(e1)
    ship = _ship([e1])
    cj = CallerJournal(t1_redis)
    pend = build_caller_pending(scope, sid, ship)
    assert (await cj.claim(scope, sid, pend))[0] == "claimed"
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    base = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0,
            "has_meaningful_content": True, "topics_stored": 1}
    base.update(ack_patch)
    out = await co._complete_validated_ack(scope, sid, pend, ["e1"], dict(base), shipped=1)
    assert out.get("status") == "failed" and out.get("error") == err
    assert (await cj.load(scope, sid)).status == "pending"
    assert await mem.store.get_entry(scope, sid, "e1") is not None
    assert rcv_redis._get_str(f"consol:own:{scope}:{sid}:e1") is None


@pytest.mark.asyncio
async def test_foreign_gen_rejected_before_journal():
    scope, sid = "user", "guard5b"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    await mem.store.observe_entry(e1)
    ship = _ship([e1])
    cj = CallerJournal(t1_redis)
    rj = ReceiverJournal(rcv_redis)
    pend = build_caller_pending(scope, sid, ship)
    assert (await cj.claim(scope, sid, pend))[0] == "claimed"
    c0 = await rj.claim_new(scope, sid, ship, "plan")
    assert c0.status == "claimed"
    foreign = "f" * 32 if c0.generation != "f" * 32 else "e" * 32
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    ack = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0,
           "has_meaningful_content": True, "topics_stored": 1,
           "receiver_generation": foreign}
    out = await co._complete_validated_ack(scope, sid, pend, ["e1"], dict(ack), shipped=1)
    assert out.get("status") == "failed"
    assert (await cj.load(scope, sid)).status == "pending"
    assert rcv_redis._get_str(f"consol:own:{scope}:{sid}:e1") == f"{batch_id_of_plan_key(build_plan_cache_key(scope, sid, ship))}:{c0.generation}"


@pytest.mark.asyncio
async def test_noise_with_gen_rejected_honest_noise_passes():
    scope, sid = "user", "guard5c"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "hi", 0)
    await mem.store.observe_entry(e1)
    ship = _ship([e1])
    cj = CallerJournal(t1_redis)
    pend = build_caller_pending(scope, sid, ship)
    assert (await cj.claim(scope, sid, pend))[0] == "claimed"
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    bad = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0,
           "has_meaningful_content": False, "topics_stored": 0,
           "receiver_generation": "a" * 32}
    out_bad = await co._complete_validated_ack(scope, sid, pend, ["e1"], dict(bad), shipped=1)
    assert out_bad.get("status") == "failed"
    assert out_bad.get("error") == "noise_with_generation"
    assert (await cj.load(scope, sid)).status == "pending"
    good = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0,
            "has_meaningful_content": False, "topics_stored": 0}
    out_good = await co._complete_validated_ack(scope, sid, pend, ["e1"], dict(good), shipped=1)
    assert out_good.get("status") == "ok"


# ------------------------------------------------- wrongtype / corrupt


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["index", "state", "summ", "caller", "entry", "counter", "caller_json"])
async def test_wrongtype_corrupt_no_partial(case):
    scope, sid = "user", f"guard6-{case}"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    e2 = _mkentry("e2", scope, sid, "B", 1)
    cj, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1, e2])
    co = ConsolidationCoordinator(
        mem, lambda: None, lambda: None,
        get_caller_redis=lambda: t1_redis,
        get_receiver_redis=lambda: rcv_redis)
    index_key = ActiveStore._index_key(scope, sid)  # noqa: SLF001
    state_key = ActiveStore._state_key(scope, sid)  # noqa: SLF001
    summ_key = ActiveStore._summarized_key(scope, sid)  # noqa: SLF001
    caller_key = caller_pending_key(scope, sid)
    entry_key = f"active:{scope}:{sid}:e1"
    before_state = await mem.store.get_state(scope, sid)
    if case == "index":
        t1_redis.strings[index_key] = "wrongtype"
    elif case == "state":
        t1_redis.strings[state_key] = "wrongtype"
    elif case == "summ":
        t1_redis.strings[summ_key] = "wrongtype"
    elif case == "caller":
        t1_redis.docs[caller_key] = "{}"
    elif case == "entry":
        t1_redis.strings[entry_key] = "wrongtype"
    elif case == "counter":
        t1_redis.hashes[state_key]["unsummarized_tokens"] = "not-an-int"
    elif case == "caller_json":
        t1_redis.strings[caller_key] = "{oops"
    out = await co._finish_acknowledged(scope, sid, dict(stored))
    assert out.get("status") == "failed"
    # No partial deletion/markers on guard failure (except caller_json case
    # where caller itself is corrupt; T1 must still be intact).
    if case != "caller_json":
        # Caller still acknowledged (Lua rejected before SET).
        loaded = await cj.load(scope, sid)
        # For caller wrongtype (docs vs strings), load sees missing/corrupt?
        # Either way, no trim must have happened.
        assert loaded.status in ("acknowledged", "missing", "error")
    assert await mem.store.get_entry(scope, sid, "e1") is not None or case == "entry"
    assert await mem.store.get_entry(scope, sid, "e2") is not None
    if case != "counter":
        # Counter case leaves the corrupt value untouched (no HINCRBY).
        cur = await mem.store.get_state(scope, sid) if case not in ("state",) else None
        if cur is not None:
            assert cur["unsummarized_tokens"] == before_state["unsummarized_tokens"]


# ------------------------------------------------- corrupt counter bounds


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    "-9223372036854775807",  # INT64_MIN+1: 1-token DEC then overflow
    "-1",
    "-0",
    "9223372036854775808",  # MAX+1
    "9999999999999999999",  # 19 digits > MAX
    "12345678901234567890",  # 20 digits
    "10000000000000000000",  # 20 digits
    "+1",
    "+0",
    "01",
    "00",
    "1.0",
    "1e3",
    " 1",
    "1 ",
    "",
    "abc",
    "0x10",
])
async def test_guarded_corrupt_counter_no_partial(bad):
    scope, sid = "user", "guard7"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    e1.tokens = 1
    e2 = _mkentry("e2", scope, sid, "B", 1)
    e2.tokens = 1
    cj, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1, e2])
    index_key = ActiveStore._index_key(scope, sid)  # noqa: SLF001
    state_key = ActiveStore._state_key(scope, sid)  # noqa: SLF001
    summ_key = ActiveStore._summarized_key(scope, sid)  # noqa: SLF001
    caller_key = caller_pending_key(scope, sid)
    t1_redis.hashes[state_key]["unsummarized_tokens"] = bad
    before_docs = dict(t1_redis.docs)
    before_index = dict(t1_redis.zsets.get(index_key, {}))
    before_summ = set(t1_redis.sets.get(summ_key, set()))
    before_hash = dict(t1_redis.hashes.get(state_key, {}))
    before_caller = t1_redis._get_str(caller_key)
    from twin.shared.memory.active.consolidation_trim import guarded_trim_batch
    out = await guarded_trim_batch(mem.store, scope, sid, dict(stored), 0)
    assert out.get("status") == "failed"
    assert out.get("reason") == "wrongtype"
    assert out.get("detail") == "counter"
    assert await mem.store.get_entry(scope, sid, "e1") is not None
    assert await mem.store.get_entry(scope, sid, "e2") is not None
    assert dict(t1_redis.docs) == before_docs
    assert dict(t1_redis.zsets.get(index_key, {})) == before_index
    assert set(t1_redis.sets.get(summ_key, set())) == before_summ
    assert dict(t1_redis.hashes.get(state_key, {})) == before_hash
    assert t1_redis._get_str(caller_key) == before_caller
    assert (await cj.load(scope, sid)).status == "acknowledged"


@pytest.mark.asyncio
@pytest.mark.parametrize(("good", "expect"), [
    ("0", 0),
    ("2", 0),
    ("9223372036854775807", 9223372036854775805),
])
async def test_guarded_counter_canonical_accepts(good, expect):
    scope, sid = "user", "guard7ok"
    t1_redis = T1FakeRedis()
    rcv_redis = JournalFakeRedis()
    mem, _ = _real_t1(t1_redis)
    e1 = _mkentry("e1", scope, sid, "A", 0)
    e1.tokens = 1
    e2 = _mkentry("e2", scope, sid, "B", 1)
    e2.tokens = 1
    cj, _, _, stored, _ = await _setup_acked(mem, t1_redis, rcv_redis, scope, sid, [e1, e2])
    state_key = ActiveStore._state_key(scope, sid)  # noqa: SLF001
    t1_redis.hashes[state_key]["unsummarized_tokens"] = good
    from twin.shared.memory.active.consolidation_trim import guarded_trim_batch
    out = await guarded_trim_batch(mem.store, scope, sid, dict(stored), 0)
    assert out.get("status") == "trimmed"
    assert await mem.store.get_entry(scope, sid, "e1") is None
    assert await mem.store.get_entry(scope, sid, "e2") is None
    assert int(t1_redis.hashes[state_key]["unsummarized_tokens"]) == expect
    assert (await cj.load(scope, sid)).status == "trimmed"
