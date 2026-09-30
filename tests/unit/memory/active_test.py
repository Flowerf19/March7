"""Unit tests for T1 active memory."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from unit.memory.journal_fake import JournalFakeRedis
from unit.memory.guarded_trim_fake import guarded_trim
from twin.shared.config.settings import Config
from twin.shared.memory.active import ActiveEntry, ActiveMemory, FastPathDetector
from twin.shared.memory.active.store import ActiveStore
from twin.shared.memory.vn_time import vn_day_str


class FakeRedis(JournalFakeRedis):
    """In-memory stand-in for redis.asyncio supporting the subset T1 needs.

    Supports: JSON.SET / JSON.GET, ZADD / ZRANGE / ZREM / ZSCORE, HSET /
    HGETALL / HINCRBY, SADD / SREM / SISMEMBER / SMEMBERS, DELETE / UNLINK,
    RPUSH / EXPIRE, GET / SET strings, and EVAL for the atomic T1 Lua
    scripts (observe/trim/clear/incr) plus the caller-journal scripts
    (inherited; the caller journal lives in the T1 DB).

    EVAL handlers execute synchronously on the internal dicts with no awaits
    that yield, so concurrent asyncio tasks linearize exactly like Redis
    single-threaded Lua. No unsafe non-atomic fallback exists in production.
    """

    def __init__(self) -> None:
        super().__init__()
        self.docs: dict[str, str] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.lists: dict[str, list[str]] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def execute_command(self, *args):
        cmd = args[0]
        if cmd == "JSON.SET":
            self.docs[args[1]] = args[3]
            return "OK"
        if cmd == "JSON.GET":
            return self.docs.get(args[1])
        raise RuntimeError(f"unsupported command: {cmd}")

    async def zadd(self, key: str, mapping: dict[str, float]):
        bucket = self.zsets.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    async def zrange(self, key: str, start: int, stop: int):
        bucket = self.zsets.get(key, {})
        ordered = sorted(bucket.items(), key=lambda kv: kv[1])
        if stop == -1:
            sliced = ordered[start:]
        else:
            sliced = ordered[start : stop + 1]
        return [k for k, _ in sliced]

    async def zrem(self, key: str, *members):
        bucket = self.zsets.get(key, {})
        removed = 0
        for m in members:
            if m in bucket:
                del bucket[m]
                removed += 1
        return removed

    async def zscore(self, key: str, member: str):
        return self.zsets.get(key, {}).get(member)

    async def zcard(self, key: str):
        return len(self.zsets.get(key, {}))

    async def sadd(self, key: str, *members):
        bucket = self.sets.setdefault(key, set())
        added = 0
        for m in members:
            if m not in bucket:
                bucket.add(m)
                added += 1
        return added

    async def srem(self, key: str, *members):
        bucket = self.sets.get(key, set())
        removed = 0
        for m in members:
            if m in bucket:
                bucket.discard(m)
                removed += 1
        return removed

    async def sismember(self, key: str, member: str):
        return 1 if member in self.sets.get(key, set()) else 0

    async def smembers(self, key: str):
        return set(self.sets.get(key, set()))

    async def hset(self, key: str, mapping: dict[str, str]):
        bucket = self.hashes.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    async def hincrby(self, key: str, field: str, amount: int = 1):
        bucket = self.hashes.setdefault(key, {})
        current = int(bucket.get(field, 0) or 0)
        new_value = current + amount
        bucket[field] = str(new_value)
        return new_value

    async def hgetall(self, key: str):
        return dict(self.hashes.get(key, {}))

    async def delete(self, *keys):
        n = await super().delete(*keys)
        for k in keys:
            if k in self.docs:
                del self.docs[k]
                n += 1
            if k in self.hashes:
                del self.hashes[k]
                n += 1
            if k in self.zsets:
                del self.zsets[k]
                n += 1
            if k in self.sets:
                del self.sets[k]
                n += 1
            if k in self.lists:
                del self.lists[k]
                n += 1
        return n

    async def unlink(self, *keys):
        return await self.delete(*keys)

    async def rpush(self, key: str, *values):
        bucket = self.lists.setdefault(key, [])
        bucket.extend(values)
        return len(bucket)

    async def expire(self, key: str, seconds: int):
        self.expire_calls.append((key, seconds))
        return 1

    async def eval(self, script: str, numkeys: int, *keys_and_args):
        # Atomic: pure sync dict ops, no awaits that yield.
        keys = [str(k) for k in keys_and_args[:numkeys]]
        args = [str(a) for a in keys_and_args[numkeys:]]
        if "T1_OBSERVE_V1" in script:
            entry_key, index_key, state_key = keys[0], keys[1], keys[2]
            entry_id, entry_json, ts_s, tok_s = args[0], args[1], args[2], args[3]
            self.docs[entry_key] = entry_json
            self.zsets.setdefault(index_key, {})[entry_id] = float(ts_s)
            h = self.hashes.setdefault(state_key, {})
            new_total = int(h.get("unsummarized_tokens", 0) or 0) + int(tok_s)
            h["unsummarized_tokens"] = str(new_total)
            cur_raw = h.get("last_entry_ts")
            ts_f = float(ts_s)
            if cur_raw is None:
                h["last_entry_ts"] = ts_s
            else:
                try:
                    cur_f = float(cur_raw)
                except ValueError:
                    cur_f = None
                if cur_f is None or ts_f > cur_f:
                    h["last_entry_ts"] = ts_s
            return new_total
        if "T1_INCR_TOKENS_V1" in script:
            state_key = keys[0]
            tok_s, ts_s = args[0], args[1]
            h = self.hashes.setdefault(state_key, {})
            new_total = int(h.get("unsummarized_tokens", 0) or 0) + int(tok_s)
            h["unsummarized_tokens"] = str(new_total)
            if ts_s != "":
                ts_f = float(ts_s)
                cur_raw = h.get("last_entry_ts")
                if cur_raw is None:
                    h["last_entry_ts"] = ts_s
                else:
                    try:
                        cur_f = float(cur_raw)
                    except ValueError:
                        cur_f = None
                    if cur_f is None or ts_f > cur_f:
                        h["last_entry_ts"] = ts_s
            return new_total
        if "T1_TRIM_GUARDED_V1" in script:
            return guarded_trim(self, keys, args)
        if "T1_TRIM_V1" in script:
            index_key, state_key, summ_key = keys[0], keys[1], keys[2]
            scope, scope_id = args[0], args[1]
            keep_recent, n = int(args[2]), int(args[3])
            summ_ids = args[4 : 4 + n]
            bucket = self.zsets.get(index_key, {})
            # Ensure the bucket reference exists for mutation.
            if index_key not in self.zsets:
                self.zsets[index_key] = bucket
            ordered = sorted(bucket.items(), key=lambda kv: kv[1])
            keep_set = set(dict(ordered[-keep_recent:]).keys()) if keep_recent > 0 else set()
            marked = self.sets.setdefault(summ_key, set())
            h = self.hashes.setdefault(state_key, {})
            deleted: list[str] = []
            subtracted = 0
            for eid in summ_ids:
                entry_key = f"active:{scope}:{scope_id}:{eid}"
                is_keep = eid in keep_set
                doc = self.docs.get(entry_key)
                has_index = eid in bucket
                if doc is None:
                    if has_index:
                        del bucket[eid]
                    marked.discard(eid)
                    if (not is_keep) and has_index:
                        deleted.append(eid)
                    continue
                try:
                    tok = int(json.loads(doc).get("tokens", 0) or 0)
                except (ValueError, AttributeError):
                    tok = 0
                if tok < 0:
                    tok = 0
                if is_keep:
                    if eid not in marked:
                        if has_index:
                            if tok > 0:
                                h["unsummarized_tokens"] = str(
                                    int(h.get("unsummarized_tokens", 0) or 0) - tok
                                )
                                subtracted += tok
                            marked.add(eid)
                        else:
                            del self.docs[entry_key]
                            marked.discard(eid)
                else:
                    if eid not in marked and tok > 0:
                        h["unsummarized_tokens"] = str(
                            int(h.get("unsummarized_tokens", 0) or 0) - tok
                        )
                        subtracted += tok
                    if entry_key in self.docs:
                        del self.docs[entry_key]
                    if eid in bucket:
                        del bucket[eid]
                    marked.discard(eid)
                    deleted.append(eid)
            try:
                cur_n = int(h.get("unsummarized_tokens", 0) or 0)
            except ValueError:
                cur_n = 0
            if cur_n < 0:
                h["unsummarized_tokens"] = "0"
                cur_n = 0
            return [subtracted, cur_n, *deleted]
        if "T1_CLEAR_V1" in script:
            index_key, state_key, summ_key = keys[0], keys[1], keys[2]
            prefix = f"active:{args[0]}:{args[1]}:"
            bucket = self.zsets.get(index_key, {})
            ids = [k for k, _ in sorted(bucket.items(), key=lambda kv: kv[1])]
            for eid in ids:
                self.docs.pop(prefix + eid, None)
            # SCAN orphans with the same prefix (pre-fix >10k leftovers).
            for k in [k for k in self.docs if k.startswith(prefix)]:
                del self.docs[k]
            self.zsets.pop(index_key, None)
            self.hashes.pop(state_key, None)
            self.sets.pop(summ_key, None)
            return len(ids)
        return await super().eval(script, numkeys, *keys_and_args)


def _make_memory(token_counter=None, trigger=None, redis=None) -> ActiveMemory:
    store = ActiveStore(redis or FakeRedis())
    return ActiveMemory(
        store=store,
        detector=FastPathDetector(),
        token_counter=token_counter,
        trigger_callback=trigger,
    )


# ---------------- store / service behaviour ----------------

@pytest.mark.asyncio
async def test_observe_appends_and_bumps_tokens():
    mem = _make_memory()
    await mem.observe("user", "u1", "user", "hello world")
    await mem.observe("user", "u1", "assistant", "hi back")
    state = await mem.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] > 0
    entries = await mem.get_context("user", "u1")
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_observe_concurrent_updates_both_counted():
    # Regression: two concurrent observes on the same scope must not lose an
    # update via a read-then-write race on unsummarized_tokens.
    mem = _make_memory(token_counter=lambda _: 10)
    await asyncio.gather(
        mem.observe("user", "u1", "user", "first"),
        mem.observe("user", "u1", "user", "second"),
    )
    state = await mem.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == 20


@pytest.mark.asyncio
async def test_get_context_returns_ascending_order():
    mem = _make_memory()
    a = await mem.observe("user", "u1", "user", "first")
    b = await mem.observe("user", "u1", "user", "second")
    c = await mem.observe("user", "u1", "user", "third")
    entries = await mem.get_context("user", "u1")
    assert [e.entry_id for e in entries] == [a.entry_id, b.entry_id, c.entry_id]


@pytest.mark.asyncio
async def test_trim_keeps_recent():
    mem = _make_memory()
    entries: list[ActiveEntry] = []
    for i in range(10):
        entries.append(await mem.observe("user", "u1", "user", f"msg-{i}"))

    summarized = [e.entry_id for e in entries[:7]]
    await mem.trim("user", "u1", summarized, keep_recent=5)

    remaining = await mem.get_context("user", "u1")
    remaining_ids = {e.entry_id for e in remaining}

    # Most recent 5 must survive even though some of them are also in
    # summarized list (entries 5,6 are both in summarized and in recent).
    last_five_ids = {e.entry_id for e in entries[-5:]}
    assert last_five_ids.issubset(remaining_ids)
    # Entries not in summarized list must also survive (entries 7,8,9 already
    # covered by the recent-five guard; entries 0-4 are summarized but only
    # 0-4 are not in recent-five, so they must be gone).
    deleted_expected = {e.entry_id for e in entries[:5]}
    assert deleted_expected.isdisjoint(remaining_ids)


@pytest.mark.asyncio
async def test_trim_clears_unsummarized_tokens_for_retained_entries():
    # Regression: retained-but-summarized entries (the keep_recent tail) must
    # NOT count toward unsummarized_tokens, or the scope stays hot forever and
    # the consolidation re-summarizes the same transcript on every poll.
    mem = _make_memory(token_counter=lambda _: 10)
    entries: list[ActiveEntry] = []
    for i in range(5):
        entries.append(await mem.observe("user", "u1", "user", f"msg-{i}"))

    # All 5 entries summarized; keep_recent=5 retains every one of them.
    summarized = [e.entry_id for e in entries]
    await mem.trim("user", "u1", summarized, keep_recent=5)

    state = await mem.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == 0


@pytest.mark.asyncio
async def test_trim_counts_entries_not_yet_summarized():
    # An entry that arrived after the consolidation snapshot (not in
    # summarized_entry_ids) and is still present MUST keep counting.
    mem = _make_memory(token_counter=lambda _: 10)
    entries: list[ActiveEntry] = []
    for i in range(4):
        entries.append(await mem.observe("user", "u1", "user", f"msg-{i}"))
    # A 5th message lands mid-consolidation, after the snapshot was taken.
    late = await mem.observe("user", "u1", "user", "late-arrival")

    summarized = [e.entry_id for e in entries]  # excludes `late`
    await mem.trim("user", "u1", summarized, keep_recent=5)

    state = await mem.store.get_state("user", "u1")
    # Only the late, un-summarized entry's tokens remain.
    assert state["unsummarized_tokens"] == late.tokens == 10


# ---------------- atomic observe/trim/clear (#22, #35) ----------------

@pytest.mark.asyncio
async def test_trim_preserves_observe_between_read_and_write(monkeypatch):
    # Finding #22: observe landing between trim's list (archive snapshot) and
    # the state write must not be overwritten by a stale absolute counter.
    mem = _make_memory(token_counter=lambda _: 10)
    first = await mem.observe("user", "u1", "user", "first")
    orig_trim = mem.store.trim_summarized

    async def injecting_trim(scope, scope_id, ids, *, keep_recent):
        await mem.observe(scope, scope_id, "user", "late-between")
        return await orig_trim(scope, scope_id, ids, keep_recent=keep_recent)

    monkeypatch.setattr(mem.store, "trim_summarized", injecting_trim)
    await mem.trim("user", "u1", [first.entry_id], keep_recent=5)

    state = await mem.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == 10
    entries = await mem.get_context("user", "u1")
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_observe_cross_client_concurrent():
    # Two ActiveMemory facades sharing one Redis server (different clients /
    # processes) must not lose increments or index entries.
    redis = FakeRedis()
    mem1 = _make_memory(token_counter=lambda _: 10, redis=redis)
    mem2 = _make_memory(token_counter=lambda _: 10, redis=redis)
    await asyncio.gather(
        *[mem1.observe("user", "u1", "user", f"a-{i}") for i in range(5)],
        *[mem2.observe("user", "u1", "user", f"b-{i}") for i in range(5)],
    )
    state = await mem1.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == 100
    entries = await mem1.get_context("user", "u1", limit=20)
    assert len(entries) == 10


@pytest.mark.asyncio
async def test_clear_scope_clears_beyond_10k_without_orphans():
    # Finding #35: clear must remove ALL old indexed entries (no 10k cap) and
    # leave no orphan JSON docs; new observes afterwards stay reachable.
    redis = FakeRedis()
    scope, scope_id = "user", "u1"
    index_key = ActiveStore._index_key(scope, scope_id)  # noqa: SLF001
    state_key = ActiveStore._state_key(scope, scope_id)  # noqa: SLF001
    summ_key = ActiveStore._summarized_key(scope, scope_id)  # noqa: SLF001
    prefix = f"active:{scope}:{scope_id}:"
    n = 10_500
    bucket: dict[str, float] = {}
    for i in range(n):
        eid = f"synth-{i:06d}"
        bucket[eid] = float(1000 + i)
        redis.docs[prefix + eid] = '{"tokens": 2}'
    redis.zsets[index_key] = bucket
    redis.hashes[state_key] = {
        "unsummarized_tokens": str(n * 2),
        "last_entry_ts": str(float(1000 + n)),
    }
    redis.sets[summ_key] = {"synth-000001", "synth-000002"}
    # An orphan doc not in the index (pre-fix leftover) must also be cleaned.
    redis.docs[prefix + "orphan-legacy"] = '{"tokens": 9}'

    store = ActiveStore(redis)
    await store.clear_scope(scope, scope_id)

    assert index_key not in redis.zsets
    assert state_key not in redis.hashes
    assert summ_key not in redis.sets
    assert not any(k.startswith(prefix) for k in redis.docs)
    state = await store.get_state(scope, scope_id)
    assert state == {"unsummarized_tokens": 0, "last_entry_ts": None, "recent_catalogs": []}

    mem = ActiveMemory(store=store, detector=FastPathDetector(), token_counter=lambda _: 7)
    fresh = await mem.observe(scope, scope_id, "user", "after-clear")
    assert (await store.get_state(scope, scope_id))["unsummarized_tokens"] == 7
    assert [e.entry_id for e in await mem.get_context(scope, scope_id)] == [fresh.entry_id]


@pytest.mark.asyncio
async def test_clear_observe_interleaving_no_orphans():
    # Concurrent clear + observes must never orphan docs/index and tokens must
    # equal the surviving unsummarized docs. Order is nondeterministic; assert
    # invariants that hold for every linearization.
    mem = _make_memory(token_counter=lambda _: 5)
    for i in range(3):
        await mem.observe("user", "u1", "user", f"old-{i}")
    redis: FakeRedis = mem.store.redis  # type: ignore[assignment]
    await asyncio.gather(
        mem.store.clear_scope("user", "u1"),
        *[mem.observe("user", "u1", "user", f"new-{i}") for i in range(5)],
    )
    index_key = ActiveStore._index_key("user", "u1")  # noqa: SLF001
    prefix = "active:user:u1:"
    members = set(redis.zsets.get(index_key, {}).keys())
    doc_ids = {k[len(prefix):] for k in redis.docs if k.startswith(prefix)}
    assert members == doc_ids
    total = 0
    for eid in members:
        total += int(json.loads(redis.docs[prefix + eid]).get("tokens", 0) or 0)
    state = await mem.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == total
    if members:
        assert state["last_entry_ts"] is not None
        hi = max(redis.zsets[index_key].values())
        assert abs(float(state["last_entry_ts"]) - hi) < 1e-6
    else:
        assert state["unsummarized_tokens"] == 0
        assert state["last_entry_ts"] is None


@pytest.mark.asyncio
async def test_no_peer_t1_reads_across_dbs():
    # Redis DB isolation: march7 (db0) and evernight (db1) use separate clients;
    # writes/clears on one must never appear on the other.
    mem_march7 = _make_memory(token_counter=lambda _: 10, redis=FakeRedis())
    mem_evernight = _make_memory(token_counter=lambda _: 10, redis=FakeRedis())
    await mem_march7.observe("user", "u1", "user", "march7-only")
    assert await mem_evernight.store.list_entries("user", "u1") == []
    assert (await mem_evernight.store.get_state("user", "u1"))["unsummarized_tokens"] == 0
    await mem_march7.store.clear_scope("user", "u1")
    assert await mem_march7.store.list_entries("user", "u1") == []
    await mem_evernight.observe("user", "u1", "user", "evernight-only")
    assert len(await mem_evernight.get_context("user", "u1")) == 1
    assert await mem_march7.store.list_entries("user", "u1") == []


@pytest.mark.asyncio
async def test_trim_does_not_regress_last_entry_ts():
    mem = _make_memory(token_counter=lambda _: 10)
    e1 = await mem.observe("user", "u1", "user", "first")
    await asyncio.sleep(0.01)
    await mem.observe("user", "u1", "user", "second")
    before = (await mem.store.get_state("user", "u1"))["last_entry_ts"]
    await mem.trim("user", "u1", [e1.entry_id], keep_recent=5)
    after = (await mem.store.get_state("user", "u1"))["last_entry_ts"]
    assert after == before
    # An out-of-order older observe must not regress the max either.
    from datetime import datetime, timedelta, timezone

    old = ActiveEntry(
        scope="user", scope_id="u1", role="user", content="old-clock", tokens=5,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    await mem.store.observe_entry(old)
    assert (await mem.store.get_state("user", "u1"))["last_entry_ts"] == before


@pytest.mark.asyncio
async def test_trim_idempotent_retry_does_not_double_subtract():
    mem = _make_memory(token_counter=lambda _: 10)
    entries = [await mem.observe("user", "u1", "user", f"m-{i}") for i in range(2)]
    ids = [e.entry_id for e in entries]
    await mem.trim("user", "u1", ids, keep_recent=5)
    assert (await mem.store.get_state("user", "u1"))["unsummarized_tokens"] == 0
    await mem.trim("user", "u1", ids, keep_recent=5)
    state = await mem.store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == 0
    assert len(await mem.get_context("user", "u1")) == 2


@pytest.mark.asyncio
async def test_increment_tokens_keeps_max_last_ts():
    store = ActiveStore(FakeRedis())
    assert await store.increment_tokens("user", "u1", 5, last_entry_ts=100.0) == 5
    assert await store.increment_tokens("user", "u1", 3, last_entry_ts=90.0) == 8
    state = await store.get_state("user", "u1")
    assert state["unsummarized_tokens"] == 8
    assert float(state["last_entry_ts"]) == 100.0


# ---------------- archive-on-trim (W3) ----------------

@pytest.mark.asyncio
async def test_trim_archives_deleted_entries(monkeypatch):
    monkeypatch.setattr(Config, "T1_ARCHIVE_ENABLED", True)
    monkeypatch.setattr(Config, "T1_ARCHIVE_TTL_DAYS", 90)
    mem = _make_memory()
    entries: list[ActiveEntry] = []
    for i in range(10):
        entries.append(await mem.observe("user", "u1", "user", f"msg-{i}"))

    summarized = [e.entry_id for e in entries[:7]]
    await mem.trim("user", "u1", summarized, keep_recent=5)

    redis: FakeRedis = mem.store.redis  # type: ignore[assignment]
    # Entries 0-4 were deleted (5,6 protected by keep_recent) → archived.
    day = vn_day_str(entries[0].created_at.timestamp())
    key = f"t1:archive:user:u1:{day}"
    assert key in redis.lists
    payloads = [json.loads(p) for p in redis.lists[key]]
    assert [p["entry_id"] for p in payloads] == [e.entry_id for e in entries[:5]]
    # Serialized entries must round-trip (verbatim transcript preserved).
    assert payloads[0]["content"] == "msg-0"
    assert payloads[0]["scope"] == "user"
    assert payloads[0]["created_at"]  # ISO datetime survived model_dump
    # TTL refreshed on the day key.
    assert (key, 90 * 86400) in redis.expire_calls
    # And the trim itself still happened.
    remaining_ids = {e.entry_id for e in await mem.get_context("user", "u1")}
    assert remaining_ids.isdisjoint({e.entry_id for e in entries[:5]})


@pytest.mark.asyncio
async def test_trim_archive_disabled_skips_archive_but_still_trims(monkeypatch):
    monkeypatch.setattr(Config, "T1_ARCHIVE_ENABLED", False)
    mem = _make_memory()
    entries: list[ActiveEntry] = []
    for i in range(10):
        entries.append(await mem.observe("user", "u1", "user", f"msg-{i}"))

    await mem.trim("user", "u1", [e.entry_id for e in entries[:7]], keep_recent=5)

    redis: FakeRedis = mem.store.redis  # type: ignore[assignment]
    assert redis.lists == {}
    remaining_ids = {e.entry_id for e in await mem.get_context("user", "u1")}
    assert remaining_ids.isdisjoint({e.entry_id for e in entries[:5]})


@pytest.mark.asyncio
async def test_trim_archive_failure_still_trims(monkeypatch, caplog):
    """Archive is best-effort: a failure must warn and NOT block the trim —
    blocking would leave the summarized transcript hot and re-consolidate it
    forever."""
    import logging

    monkeypatch.setattr(Config, "T1_ARCHIVE_ENABLED", True)
    mem = _make_memory()
    entries: list[ActiveEntry] = []
    for i in range(10):
        entries.append(await mem.observe("user", "u1", "user", f"msg-{i}"))

    async def exploding_archive(*args, **kwargs):
        raise RuntimeError("redis OOM")

    monkeypatch.setattr(mem.store, "archive_entries", exploding_archive)

    with caplog.at_level(logging.WARNING):
        await mem.trim("user", "u1", [e.entry_id for e in entries[:7]], keep_recent=5)

    assert any("archive-on-trim failed" in rec.message for rec in caplog.records)
    remaining_ids = {e.entry_id for e in await mem.get_context("user", "u1")}
    assert remaining_ids.isdisjoint({e.entry_id for e in entries[:5]})


@pytest.mark.asyncio
async def test_archive_entries_groups_by_vn_day():
    """Entries created on different VN days must land on separate day keys —
    a trim can carry messages from before midnight."""
    from datetime import datetime, timezone

    store = ActiveStore(FakeRedis())
    e1 = ActiveEntry(
        scope="user", scope_id="u1", role="user", content="tối qua",
        created_at=datetime(2026, 7, 2, 16, 0, tzinfo=timezone.utc),  # 23:00 VN 02/07
    )
    e2 = ActiveEntry(
        scope="user", scope_id="u1", role="user", content="sáng nay",
        created_at=datetime(2026, 7, 2, 18, 0, tzinfo=timezone.utc),  # 01:00 VN 03/07
    )

    await store.archive_entries("user", "u1", [e1, e2], ttl_seconds=86400)

    redis: FakeRedis = store.redis  # type: ignore[assignment]
    assert set(redis.lists) == {
        "t1:archive:user:u1:2026-07-02",
        "t1:archive:user:u1:2026-07-03",
    }
    assert ("t1:archive:user:u1:2026-07-02", 86400) in redis.expire_calls
    assert ("t1:archive:user:u1:2026-07-03", 86400) in redis.expire_calls


# ---------------- detector ----------------

def test_fast_path_detector_matches_identity():
    d = FastPathDetector()
    assert d.is_critical("tên tôi là Hoà") == "identity"


def test_fast_path_detector_matches_contact_email():
    d = FastPathDetector()
    assert d.is_critical("email tôi là a@b.com") == "contact"


def test_fast_path_detector_no_match():
    d = FastPathDetector()
    assert d.is_critical("ok") is None


# ---------------- threshold trigger ----------------

@pytest.mark.asyncio
async def test_threshold_trigger_fires():
    trigger = AsyncMock()
    mem = _make_memory(token_counter=lambda _: 2100, trigger=trigger)
    await mem.observe("user", "u1", "user", "anything")
    # Trigger now fires as a background task, not awaited inline.
    await asyncio.sleep(0)
    trigger.assert_awaited_once_with("user", "u1")


@pytest.mark.asyncio
async def test_threshold_not_fire_below():
    trigger = AsyncMock()
    mem = _make_memory(token_counter=lambda _: 1000, trigger=trigger)
    await mem.observe("user", "u1", "user", "anything")
    trigger.assert_not_awaited()


@pytest.mark.asyncio
async def test_threshold_trigger_does_not_block_observe():
    # observe() must return before the trigger callback resolves.
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_trigger(scope, scope_id):
        started.set()
        await release.wait()

    mem = _make_memory(token_counter=lambda _: 2100, trigger=slow_trigger)
    await mem.observe("user", "u1", "user", "anything")
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not release.is_set()  # sanity: trigger is still in flight
    release.set()
    # Drain the background task so it doesn't leak into other tests.
    await asyncio.gather(*mem._pending_tasks)


@pytest.mark.asyncio
async def test_threshold_trigger_skips_when_already_in_progress():
    async def slow_side_effect(*_args):
        await asyncio.sleep(0.05)

    trigger = AsyncMock(side_effect=slow_side_effect)
    mem = _make_memory(token_counter=lambda _: 2100, trigger=trigger)
    await mem.observe("user", "u1", "user", "first")
    # Second observe while the first trigger is still in flight.
    await mem.observe("user", "u1", "user", "second")
    await asyncio.gather(*mem._pending_tasks)
    trigger.assert_awaited_once_with("user", "u1")


@pytest.mark.asyncio
async def test_threshold_trigger_respects_cooldown_after_completion():
    trigger = AsyncMock()
    mem = _make_memory(token_counter=lambda _: 2100, trigger=trigger)
    await mem.observe("user", "u1", "user", "first")
    await asyncio.gather(*mem._pending_tasks)
    trigger.assert_awaited_once_with("user", "u1")

    # Tokens are still >= threshold (e.g. consolidation returned skipped),
    # but we're within the cooldown window, so no re-fire.
    await mem.observe("user", "u1", "user", "second")
    await asyncio.sleep(0)
    trigger.assert_awaited_once_with("user", "u1")


# ---------------- topic shift / push_catalog ----------------

def test_is_topic_shift_true():
    assert ActiveMemory.is_topic_shift(["work", "work", "work"], "identity", 0.85)


def test_is_topic_shift_low_conf():
    assert not ActiveMemory.is_topic_shift(["work", "work", "work"], "identity", 0.5)


def test_is_topic_shift_unstable_buffer():
    assert not ActiveMemory.is_topic_shift(["work", "identity", "work"], "habit", 0.9)


@pytest.mark.asyncio
async def test_push_catalog_returns_shift():
    mem = _make_memory()
    assert (await mem.push_catalog("user", "u1", "work", 0.9)) is False
    assert (await mem.push_catalog("user", "u1", "work", 0.9)) is False
    assert (await mem.push_catalog("user", "u1", "work", 0.9)) is False
    shift = await mem.push_catalog("user", "u1", "identity", 0.9)
    assert shift is True
    # Ensure the catalog actually got persisted.
    state = await mem.store.get_state("user", "u1")
    assert state["recent_catalogs"][-1] == "identity"
    # Verify the rolling window cap behaviour as a sanity check.
    assert len(state["recent_catalogs"]) <= 5
    # Make sure JSON serialised payload is sane (catches accidental bytes).
    raw_state = await mem.store.redis.hgetall(
        ActiveStore._state_key("user", "u1")  # noqa: SLF001
    )
    assert json.loads(raw_state["recent_catalogs"])[-1] == "identity"
