"""Durable canonical-plan + strict ACK regressions.

Fail-closed contract:
- A meaningful consolidation needs a witnessed durable plan (cache hit or
  atomic SET NX win) BEFORE any T2/T3 write. Cache I/O errors, corrupt
  payloads, lost races without a readable winner, and missing durable
  capability all fail with no acknowledgement: T1 preserved, writers
  never called. No own-plan fallback on transient failure.
- Plan keys bind the full ordered prompt/provenance fingerprint; any
  date/author/content/order change mints a fresh plan, while an
  ActiveEntry and its shipped dict share a key.
- The coordinator trims only on explicit durability proof.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from types import SimpleNamespace
from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.active import ActiveEntry
from twin.shared.memory.consolidation_coordinator import (
    ConsolidationCoordinator,
    entry_to_snapshot,
)
from twin.shared.memory.consolidation_journal import (
    ReceiverJournal,
    batch_id_of_plan_key,
)
from twin.shared.memory.profile.codec import profile_hash
from twin.shared.tools.modules.memory.consolidate_memory_tool import (
    ConsolidateMemoryTool,
)
from twin.shared.tools.modules.memory.consolidation_plan_cache import (
    PLAN_CACHE_TTL_SECONDS,
    CanonicalPlanCache,
    CanonicalPlanResolver,
    build_plan_cache_key,
    serialize_canonical_plan,
)
from twin.shared.tools.modules.memory.consolidation_schema import (
    ConsolidationPlan,
    ConsolidationPlanValidator,
    TopicPlan,
)


class FakePlanRedis(JournalFakeRedis):
    """Real-behavior async fake Redis strings + journal EVAL.

    Inherits bytes-on-GET, NX/EX SET, TTL clock (``now``), error injection
    (``get_error``/``set_error``/``eval_error``) and the journal Lua
    mirrors. Concurrent-winner scenarios seed the plan key directly — the
    atomic claim has no loser re-read to corrupt (see test notes).
    """


class FakeT2:
    """Idempotent diary double recording every store attempt."""

    def __init__(self, redis, fail_topics_once=None) -> None:
        self.redis = redis
        self.by_key: dict[str, str] = {}
        self.docs: dict[str, dict] = {}
        self.calls: list[str] = []
        self.fail_once = set(fail_topics_once or [])

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display,
        importance, period_start=None, period_end=None, source_entry_ids=None,
        idempotency_key=None,
    ):
        self.calls.append(topic)
        if topic in self.fail_once:
            self.fail_once.remove(topic)
            raise ValueError("injected T2 failure")
        if idempotency_key is not None and idempotency_key in self.by_key:
            return self.by_key[idempotency_key]
        sid = f"sum-{topic}-{len(self.docs)}"
        self.docs[sid] = {"topic": topic, "summary": summary}
        if idempotency_key is not None:
            self.by_key[idempotency_key] = sid
        return sid


class T2WithoutRedis(FakeT2):
    def __init__(self) -> None:
        self.by_key = {}
        self.docs = {}
        self.calls = []
        self.fail_once = set()
        # Deliberately no .redis: no durable plan capability.


class NoWriteProfile:
    async def read_raw(self, scope_id):
        return ""

    async def apply_consolidation_updates(self, *args, **kwargs):
        raise AssertionError("profile writer must not be called")


class FakeMemory:
    def __init__(self, profile) -> None:
        self.t1 = None
        self.profile = profile


class SeqLLM:
    model = "fake-model"

    def __init__(self, payloads) -> None:
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    async def generate_response(self, *args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else [])
        if messages:
            self.prompts.append(messages[0].get("content", ""))
        payload = self._payloads.pop(0)
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)


class FakeEmbed:
    async def get_embedding(self, text):
        return [0.1] * 8


def _topic(topic, summary, importance=4, display="Chủ đề"):
    return {
        "topic": topic, "topic_display": display,
        "summary": summary, "importance": importance,
    }


ENTRIES = [
    {"entry_id": "e1", "message_id": "m1", "role": "user",
     "content": "dự án X deadline 10/07, lỗi ImportError auth",
     "author_id": "u1", "author_name": "Hoà",
     "timestamp": "2026-07-01T03:00:00+00:00"},
    {"entry_id": "e2", "message_id": "m2", "role": "user",
     "content": "sáng nay chạy bộ 5km quanh hồ",
     "author_id": "u1", "author_name": "Hoà",
     "timestamp": "2026-07-01T03:05:00+00:00"},
]

MEANINGFUL = {"has_meaningful_content": True, "topics": [
    _topic("work", "User đang làm dự án X, deadline 10/07."),
    _topic("health", "User chạy bộ mỗi sáng quanh hồ."),
]}
NOISE = {"has_meaningful_content": False, "topics": []}


def _tool(redis, llm, t2=None):
    t2 = t2 if t2 is not None else FakeT2(redis)
    tool = ConsolidateMemoryTool(FakeMemory(NoWriteProfile()), llm, FakeEmbed(), t2)
    return tool, t2


def _winner_payload(topics, entry_ids=("e1", "e2")) -> str:
    plan = ConsolidationPlan(
        True,
        tuple(TopicPlan(t["topic"], t["topic_display"], t["summary"], t["importance"]) for t in topics),
        {}, {},
    )
    return serialize_canonical_plan(
        plan, scope="user", scope_id="u1",
        entry_ids=list(entry_ids), profile_hash=profile_hash(""),
    )


# ------------------------------------------------- fail-closed: cache errors


@pytest.mark.asyncio
async def test_cache_get_failure_fails_without_writes_or_llm():
    redis = FakePlanRedis()
    redis.get_error = OSError("redis down")
    llm = SeqLLM([MEANINGFUL])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_error"
    assert not res.get("entry_ids")
    assert llm.prompts == []  # failed before the plan was even fetched
    assert t2.calls == []


@pytest.mark.asyncio
async def test_journal_claim_failure_fails_without_writes():
    # Same intent as the old SET-failure test: the atomic claim is
    # unwitnessed, so the fetched plan must never reach T2/T3.
    redis = FakePlanRedis()
    redis.eval_error = OSError("redis down")
    llm = SeqLLM([MEANINGFUL])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_error"
    assert not res.get("entry_ids")
    assert len(llm.prompts) == 1  # plan fetched, claim unwitnessed
    assert t2.calls == []


@pytest.mark.asyncio
async def test_overlapping_owned_entries_rejected_before_llm():
    # A foreign batch owns e1 (partial attempt in flight): the grown batch
    # must fail closed before the full LLM call, with no writes.
    redis = FakePlanRedis()
    foreign = ReceiverJournal(redis)
    other = [dict(ENTRIES[0], content="foreign transcript"), dict(ENTRIES[1])]
    got = await foreign.claim_new("user", "u1", other, _winner_payload(
        [_topic("work", "Foreign.")], entry_ids=["e1", "e2"]))
    assert got.status == "claimed"
    llm = SeqLLM([MEANINGFUL])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "pending_overlap"
    assert not res.get("entry_ids")
    assert llm.prompts == []
    assert t2.calls == []


@pytest.mark.asyncio
async def test_tampered_batch_record_fails_before_llm():
    redis = FakePlanRedis()
    journal = ReceiverJournal(redis)
    got = await journal.claim_new(
        "user", "u1", ENTRIES, _winner_payload([_topic("work", "S.")]))
    assert got.status == "claimed"
    key = build_plan_cache_key("user", "u1", ENTRIES)
    batch_key = f"consol:batch:user:u1:{batch_id_of_plan_key(key)}"
    redis.strings[batch_key] = "not-json {"
    llm = SeqLLM([MEANINGFUL])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_error"
    assert not res.get("entry_ids")
    assert llm.prompts == []
    assert t2.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("injected, reason", [
    # Undecodable winner bytes: claim unwitnessed.
    (b"\xff\xfe not utf-8 \x00\x01", "plan_cache_error"),
    # Decodable but unparseable winner: corrupt.
    ('{"v": 999, "garbage": true}', "plan_cache_corrupt"),
])
async def test_concurrent_winner_corrupt_fails_without_writes(injected, reason):
    # A rival stored payload under our plan key (same batch, corrupt bytes
    # or wrong version): adoption must fail closed with no writes.
    redis = FakePlanRedis()
    key = build_plan_cache_key("user", "u1", ENTRIES)
    redis.strings[key] = injected
    tool, t2 = _tool(redis, SeqLLM([MEANINGFUL]))
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == reason
    assert not res.get("entry_ids")
    assert t2.calls == []


@pytest.mark.asyncio
async def test_corrupt_orphan_plan_fails_without_writes():
    # Ownership is probed before the cache is read, so an orphan plan (no
    # owners) is only discovered at adoption — after the LLM call, but
    # still with no acknowledgement and no writes.
    redis = FakePlanRedis()
    key = build_plan_cache_key("user", "u1", ENTRIES)
    redis.strings[key] = b"\xff\xfe not utf-8 \x00\x01"
    llm = SeqLLM([MEANINGFUL])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_error"
    assert not res.get("entry_ids")
    assert len(llm.prompts) == 1
    assert t2.calls == []


@pytest.mark.asyncio
async def test_nonstring_cached_entry_ids_rejected_without_writes():
    redis = FakePlanRedis()
    key = build_plan_cache_key("user", "u1", ENTRIES)
    tampered = json.loads(_winner_payload([_topic("work", "S.")]))
    tampered["entry_ids"] = ["e1", 2]  # untrusted cache side: no coercion
    redis.strings[key] = json.dumps(tampered)
    llm = SeqLLM([MEANINGFUL])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_corrupt"
    assert len(llm.prompts) == 1  # orphan discovered at adoption
    assert t2.calls == []


@pytest.mark.asyncio
async def test_nonstring_cached_profile_hash_rejected_without_writes():
    redis = FakePlanRedis()
    key = build_plan_cache_key("user", "u1", ENTRIES)
    tampered = json.loads(_winner_payload([_topic("work", "S.")]))
    tampered["profile_hash"] = 12345
    redis.strings[key] = json.dumps(tampered)
    tool, t2 = _tool(redis, SeqLLM([MEANINGFUL]))
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_corrupt"
    assert t2.calls == []


@pytest.mark.asyncio
async def test_missing_durable_capability_fails_meaningful_but_allows_noise():
    # Meaningful plan without Redis: no acknowledgement, no writes.
    llm = SeqLLM([MEANINGFUL])
    tool = ConsolidateMemoryTool(
        FakeMemory(NoWriteProfile()), llm, FakeEmbed(), T2WithoutRedis(),
    )
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_unavailable"
    assert not res.get("entry_ids")

    # Noise carries no writes, so it stays legitimate without the cache.
    llm2 = SeqLLM([NOISE])
    tool2 = ConsolidateMemoryTool(
        FakeMemory(NoWriteProfile()), llm2, FakeEmbed(), T2WithoutRedis(),
    )
    ok = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert ok["status"] == "ok"
    assert ok["entry_ids"] == ["e1", "e2"]
    assert ok["topics_stored"] == 0


@pytest.mark.asyncio
async def test_noise_plan_never_claims_cache():
    redis = FakePlanRedis()
    tool, _ = _tool(redis, SeqLLM([NOISE]))
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "ok"
    assert res["entry_ids"] == ["e1", "e2"]
    assert redis.set_calls == 0  # nothing to witness for a no-write plan
    assert redis.eval_calls == 0


# ------------------------------------------------------- winner adoption


@pytest.mark.asyncio
async def test_tool_adopts_concurrent_winner_topics():
    redis = FakePlanRedis()
    key = build_plan_cache_key("user", "u1", ENTRIES)
    redis.strings[key] = _winner_payload([_topic("work", "Winner summary.")])
    llm = SeqLLM([{"has_meaningful_content": True, "topics": [
        _topic("loser", "Loser summary, must be discarded."),
    ]}])
    tool, t2 = _tool(redis, llm)
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "ok"
    assert res["topics_stored"] == 1
    assert [d["summary"] for d in t2.docs.values()] == ["Winner summary."]


@pytest.mark.asyncio
async def test_concurrent_resolvers_adopt_same_durable_winner():
    redis = FakePlanRedis()
    cache = CanonicalPlanCache(redis)
    entries = [dict(e) for e in ENTRIES]

    async def fetch_a():
        plan, _ = ConsolidationPlanValidator.validate(
            {"has_meaningful_content": True,
             "topics": [_topic("work", "Summary A.")]})
        assert plan is not None
        return plan, None

    async def fetch_b():
        plan, _ = ConsolidationPlanValidator.validate(
            {"has_meaningful_content": True,
             "topics": [_topic("career", "Summary B.")]})
        assert plan is not None
        return plan, None

    async def no_patch(topics_json):
        return {}, {}, None

    r1 = CanonicalPlanResolver(cache, ReceiverJournal(redis))
    r2 = CanonicalPlanResolver(cache, ReceiverJournal(redis))
    res1, res2 = await asyncio.gather(
        r1.resolve(scope="user", scope_id="u1", entries=[dict(e) for e in entries],
                   profile_hash=None, fetch_full_plan=fetch_a,
                   fetch_profile_patch=no_patch),
        r2.resolve(scope="user", scope_id="u1", entries=[dict(e) for e in entries],
                   profile_hash=None, fetch_full_plan=fetch_b,
                   fetch_profile_patch=no_patch),
    )
    assert res1.error is None and res2.error is None
    assert res1.plan is not None and res2.plan is not None
    sum1 = [t.summary for t in res1.plan.topics]
    sum2 = [t.summary for t in res2.plan.topics]
    assert sum1 == sum2
    assert sum1 in (["Summary A."], ["Summary B."])
    # One canonical plan + one batch record + per-entry owners (same batch).
    plans = [k for k in redis.strings if k.startswith("consol:plan:")]
    assert len(plans) == 1
    assert len(redis.strings) == 4


# ------------------------------------------------- horizon + key binding


def test_plan_retention_covers_full_t2_durability_window():
    # The completion TTL covers the full T2 window (importance-5 max) and
    # starts only AFTER the last T2 write of the batch (release time), so a
    # late partial can never outlive its plan. Pending witnesses have NO
    # ttl at all: T1 unsummarized entries never expire either.
    assert PLAN_CACHE_TTL_SECONDS == 365 * 86400


@pytest.mark.asyncio
async def test_retry_after_thirty_minutes_reuses_canonical_plan():
    redis = FakePlanRedis()  # now = 0.0
    t2 = FakeT2(redis, fail_topics_once={"health"})
    tool1, _ = _tool(redis, SeqLLM([MEANINGFUL]), t2)
    first = json.loads(await tool1.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert first["status"] == "failed"
    assert len(t2.docs) == 1

    # Advance past the OLD 1800s TTL: the plan must still be pinned.
    redis.now = 1801.0
    llm2 = SeqLLM([{"has_meaningful_content": True, "topics": [
        _topic("career", "Drifted work summary."),
        _topic("fitness", "Drifted health summary."),
    ]}])
    tool2, _ = _tool(redis, llm2, t2)
    second = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]
    assert llm2.prompts == []
    assert len(t2.docs) == 2
    assert sorted(d["summary"] for d in t2.docs.values()) == sorted([
        "User đang làm dự án X, deadline 10/07.",
        "User chạy bộ mỗi sáng quanh hồ.",
    ])
    key = build_plan_cache_key("user", "u1", ENTRIES)
    assert key not in redis.ttls  # pending witness: no expiry


@pytest.mark.asyncio
async def test_pending_plan_persists_until_completion_ttl():
    # Pending plans never expire (a claim-time TTL would die before late T2
    # partials); the 365d TTL starts at release, after the last T2 write.
    redis = FakePlanRedis()
    cache = CanonicalPlanCache(redis)
    journal = ReceiverJournal(redis)
    key = build_plan_cache_key("user", "u1", ENTRIES)
    payload = _winner_payload([_topic("work", "S.")])
    claimed = await journal.claim_new("user", "u1", ENTRIES, payload)
    assert claimed.status == "claimed"
    assert key not in redis.ttls
    redis.now = 366 * 86400  # past the old fixed horizon: still pinned
    assert (await cache.load(
        key, scope="user", scope_id="u1", entry_ids=["e1", "e2"])).status == "hit"
    released = await journal.release(
        "user", "u1", ["e1", "e2"], batch_id_of_plan_key(key), key,
        claimed.generation)
    assert released.ok and released.released == 2
    assert redis.ttls.get(key) == PLAN_CACHE_TTL_SECONDS
    redis.now += PLAN_CACHE_TTL_SECONDS + 1
    assert (await cache.load(
        key, scope="user", scope_id="u1", entry_ids=["e1", "e2"])).status == "miss"


def test_plan_key_binds_dates_authors_order_and_provenance():
    base = [dict(e) for e in ENTRIES]
    k0 = build_plan_cache_key("user", "u1", base)
    assert build_plan_cache_key("user", "u1", [dict(e) for e in ENTRIES]) == k0

    def variant(idx, **overrides):
        changed = [dict(e) for e in base]
        changed[idx] = {**changed[idx], **overrides}
        return build_plan_cache_key("user", "u1", changed)

    assert variant(0, timestamp="2026-07-02T03:00:00+00:00") != k0
    assert variant(0, author_name="Lan") != k0
    assert variant(0, author_id="u2") != k0
    assert variant(0, message_id="m9") != k0
    assert variant(0, role="assistant") != k0
    assert variant(0, content="nội dung khác") != k0
    assert variant(0, entry_id="e9") != k0
    # Order matters: the prompt renders entries sequentially.
    assert build_plan_cache_key("user", "u1", [base[1], base[0]]) != k0
    # Scope binding stays strict.
    assert build_plan_cache_key("user", "u2", base) != k0
    assert build_plan_cache_key("channel", "u1", base) != k0
    # Delimiter-hostile content cannot collide with field boundaries.
    tricky = [dict(base[0], content="e1\x00user\x1fforged"), dict(base[1])]
    assert build_plan_cache_key("user", "u1", tricky) != k0


def test_plan_key_equivalent_for_objects_and_shipped_dicts():
    created = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
    created2 = datetime(2026, 7, 1, 3, 5, tzinfo=timezone.utc)
    e1 = ActiveEntry(
        entry_id="e1", scope="user", scope_id="u1", role="user",
        content="dự án X", author_id="u1", author_name="Hoà",
        message_id="m1", created_at=created,
    )
    e2 = ActiveEntry(
        entry_id="e2", scope="user", scope_id="u1", role="user",
        content="chạy bộ", author_id="u1", author_name="Hoà",
        message_id="m2", created_at=created2,
    )
    shipped = [entry_to_snapshot(e1), entry_to_snapshot(e2)]
    assert build_plan_cache_key("user", "u1", [e1, e2]) == \
        build_plan_cache_key("user", "u1", shipped)


# ------------------------------------------------------- strict ACK gating


class FakeT1Trim:
    def __init__(self, entries=None) -> None:
        self._entries = list(entries or [])
        self.get_context_calls: list[tuple] = []
        self.trim_calls: list[tuple] = []
        self.store = SimpleNamespace(redis=FakePlanRedis())

    async def get_context(self, scope, scope_id, limit=200):
        self.get_context_calls.append((scope, scope_id, limit))
        return list(self._entries)

    async def get_entries_by_ids(self, scope, scope_id, entry_ids):
        by_id = {e.entry_id: e for e in self._entries}
        return [by_id[i] for i in entry_ids if i in by_id]

    async def list_unsummarized_entries(self, scope, scope_id, limit=200):
        return list(self._entries[-limit:])

    async def trim(self, scope, scope_id, entry_ids, keep_recent=None):
        self.trim_calls.append((scope, scope_id, list(entry_ids)))

    async def trim_consolidated_batch(self, scope, scope_id, record, keep_recent=None):
        import json as _json
        from twin.shared.memory.consolidation_journal import caller_pending_key as _cpk
        key = _cpk(scope, scope_id)
        raw = self.store.redis._get_str(key)
        if raw is None:
            return {"status": "failed", "reason": "missing"}
        try:
            cur = _json.loads(raw)
        except Exception:
            return {"status": "failed", "reason": "corrupt"}
        exp_ids = list((record or {}).get("entry_ids") or [])
        if cur.get("entry_ids") != exp_ids or cur.get("caller_nonce") != (record or {}).get("caller_nonce"):
            return {"status": "failed", "reason": "mismatch"}
        if cur.get("stage") == "trimmed":
            return {"status": "already"}
        if cur.get("stage") != "acknowledged":
            return {"status": "failed", "reason": "not_acked"}
        self.trim_calls.append((scope, scope_id, list(exp_ids)))
        cur["stage"] = "trimmed"
        self.store.redis.strings[key] = _json.dumps(cur, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return {"status": "trimmed", "deleted_ids": list(exp_ids), "subtracted": 0, "unsummarized_tokens": 0}


class FakeClient:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def consolidate_scope(self, scope, scope_id, reason="auto", entries=None, **kw):
        self.calls.append({"scope": scope, "scope_id": scope_id, "entries": entries})
        return dict(self.result) if isinstance(self.result, dict) else self.result


def _ack_coords(t1, result):
    receiver = FakePlanRedis()
    return ConsolidationCoordinator(
        t1, lambda: FakeClient(result), lambda: None,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: receiver,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ack", [
    # The verified defect: bare ok + ids, zero durability proof.
    {"status": "ok", "entry_ids": ["e1"]},
    # Missing flag even with plausible counts.
    {"status": "ok", "entry_ids": ["e1"], "topics_stored": 2, "topics_failed": 0},
    # Bool is not int.
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": True, "topics_failed": 0},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": "2", "topics_failed": 0},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": 2, "topics_failed": "0"},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": 2, "topics_failed": True},
    # Contradictory counts.
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": 0, "topics_failed": 0},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": False,
     "topics_stored": 2, "topics_failed": 0},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": False,
     "topics_failed": 0},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_failed": 0},
    # Failed required writes veto trim.
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": 2, "topics_failed": 1},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": False,
     "topics_stored": 0, "topics_failed": 1},
    {"status": "ok", "entry_ids": ["e1"], "has_meaningful_content": True,
     "topics_stored": 2, "topics_failed": 0, "profile_conflict": True},
])
async def test_malformed_ack_never_trims(ack):
    t1 = FakeT1Trim()
    coord = _ack_coords(t1, ack)
    await coord.consolidate_scope("user", "u1", entries=[{"entry_id": "e1"}])
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_proper_meaningful_and_noise_acks_trim():
    meaningful = {"status": "ok", "entry_ids": ["e1", "e2"],
                  "has_meaningful_content": True, "topics_stored": 2,
                  "topics_failed": 0, "receiver_generation": "f" * 32}
    t1 = FakeT1Trim()
    await _ack_coords(t1, meaningful).consolidate_scope(
        "user", "u1", entries=[{"entry_id": "e1"}, {"entry_id": "e2"}])
    assert t1.trim_calls == [("user", "u1", ["e1", "e2"])]

    noise = {"status": "ok", "entry_ids": ["e1"],
             "has_meaningful_content": False, "topics_stored": 0,
             "topics_failed": 0}
    t1n = FakeT1Trim()
    await _ack_coords(t1n, noise).consolidate_scope(
        "user", "u1", entries=[{"entry_id": "e1"}])
    assert t1n.trim_calls == [("user", "u1", ["e1"])]


@pytest.mark.asyncio
async def test_foreign_ack_ids_never_trim():
    ack = {"status": "ok", "entry_ids": ["foreign"],
           "has_meaningful_content": True, "topics_stored": 1,
           "topics_failed": 0}
    t1 = FakeT1Trim()
    await _ack_coords(t1, ack).consolidate_scope(
        "user", "u1", entries=[{"entry_id": "e1"}])
    assert t1.trim_calls == []


def test_non_dict_result_never_trims():
    trim_ids, reason = ConsolidationCoordinator._ack_trim_ids("ok", {"e1"})
    assert trim_ids is None
    assert reason == "malformed_result"


# ------------------------------------------------------- local-branch parity


@pytest.mark.asyncio
async def test_local_branch_snapshots_own_t1_and_verifies_ack():
    e1 = ActiveEntry(entry_id="e1", scope="user", scope_id="u1",
                     role="user", content="hello")
    e2 = ActiveEntry(entry_id="e2", scope="user", scope_id="u1",
                     role="user", content="world")
    t1 = FakeT1Trim([e1, e2])
    received: dict = {}

    async def local(scope, scope_id, reason="auto", entries=None, **kw):
        received["entries"] = entries
        return {"status": "ok", "entry_ids": ["e1", "e2"],
                "has_meaningful_content": True, "topics_stored": 2,
                "topics_failed": 0, "receiver_generation": "f" * 32}

    receiver = FakePlanRedis()
    coord = ConsolidationCoordinator(
        t1, lambda: None, lambda: local,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: receiver,
    )
    res = await coord.consolidate_scope("user", "u1")
    assert res["status"] == "ok"
    # Pending-first selection ships the exact local snapshot (chat
    # get_context is NOT used on the consolidation path anymore).
    assert t1.get_context_calls == []
    assert received["entries"] is not None
    assert {e["entry_id"] for e in received["entries"]} == {"e1", "e2"}
    assert all("content" in e for e in received["entries"])
    assert t1.trim_calls == [("user", "u1", ["e1", "e2"])]


@pytest.mark.asyncio
async def test_local_branch_rejects_foreign_ack():
    e1 = ActiveEntry(entry_id="e1", scope="user", scope_id="u1",
                     role="user", content="hello")
    t1 = FakeT1Trim([e1])

    async def local(scope, scope_id, reason="auto", entries=None, **kw):
        return {"status": "ok", "entry_ids": ["foreign"],
                "has_meaningful_content": True, "topics_stored": 1,
                "topics_failed": 0}

    receiver = FakePlanRedis()
    coord = ConsolidationCoordinator(
        t1, lambda: None, lambda: local,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: receiver,
    )
    await coord.consolidate_scope("user", "u1")
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_local_branch_preserves_supplied_empty_shipment():
    t1 = FakeT1Trim()
    received: dict = {}

    async def local(scope, scope_id, reason="auto", entries=None, **kw):
        received["entries"] = entries
        return {"status": "skipped", "reason": "no_messages", "entry_ids": []}

    coord = ConsolidationCoordinator(t1, lambda: None, lambda: local)
    res = await coord.consolidate_scope("user", "u1", entries=[])
    assert res["status"] == "skipped"
    assert received["entries"] == []
    assert t1.get_context_calls == []
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_coordinator_reads_providers_per_call():
    holder: dict[str, Any] = {"client": None, "local": None}
    t1 = FakeT1Trim()
    receiver = FakePlanRedis()
    coord = ConsolidationCoordinator(
        t1, lambda: holder["client"], lambda: holder["local"],
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: receiver,
    )
    holder["client"] = FakeClient(
        {"status": "ok", "entry_ids": ["e1"],
         "has_meaningful_content": False, "topics_stored": 0,
         "topics_failed": 0})
    res = await coord.consolidate_scope("user", "u1", entries=[{"entry_id": "e1"}])
    assert res["status"] == "ok"
    assert t1.trim_calls == [("user", "u1", ["e1"])]
