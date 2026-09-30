"""Canonical-plan retry + coordinator ack-gating regressions.

- Partial two-topic failure, then retry with DIFFERENT LLM labels, must reuse
  the cached canonical plan (no duplicate of the successful summary).
- Cached T3 rewrite after an external append must recompute the profile
  portion only (never bless the stale rewrite with the new hash).
- Legacy T2 stores without idempotency_key must fail closed.
- Foreign/meaningless acknowledgements must never trim T1.
"""
from __future__ import annotations

import json

import pytest

from types import SimpleNamespace
from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.consolidation_journal import ReceiverJournal
from twin.shared.memory.manager import SharedMemoryManager
from twin.shared.memory.profile import MarkdownProfileStore
from twin.shared.tools.modules.memory.consolidate_memory_tool import (
    ConsolidateMemoryTool,
)
from twin.shared.tools.modules.memory.consolidation_plan_cache import (
    CanonicalPlanCache,
    build_plan_cache_key,
)


class FakeRedisKV(JournalFakeRedis):
    """Minimal async Redis string surface (bytes on GET, NX/EX on SET).

    Inherits the journal EVAL subset so every fake exercises the real
    receiver-claim capability (production has no non-EVAL fallback).
    """


class FakeT2Redis:
    """Idempotent diary double (one-shot failures) exposing .redis."""

    def __init__(self, redis, fail_topics_once=None) -> None:
        self.redis = redis
        self.by_key: dict[str, str] = {}
        self.docs: dict[str, dict] = {}
        self.fail_once = set(fail_topics_once or [])

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display,
        importance, period_start=None, period_end=None, source_entry_ids=None,
        idempotency_key=None,
    ):
        if topic in self.fail_once:
            self.fail_once.remove(topic)
            raise ValueError("injected T2 failure")
        if idempotency_key is not None and idempotency_key in self.by_key:
            sid = self.by_key[idempotency_key]
            assert sid in self.docs
            return sid
        sid = f"sum-{topic}-{len(self.docs)}"
        self.docs[sid] = {"topic": topic, "summary": summary}
        if idempotency_key is not None:
            self.by_key[idempotency_key] = sid
        return sid


class LegacyT2:
    """Unsafe store without idempotency_key support — must fail closed."""

    def __init__(self) -> None:
        # Durable plan capability only; store_summary still lacks the
        # idempotency_key kwarg so the T2 stage keeps failing closed.
        self.redis = FakeRedisKV()

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display,
        importance, period_start=None, period_end=None, source_entry_ids=None,
    ):
        return "sum-legacy"


class StrictT1:
    async def get_context(self, *args, **kwargs):
        raise AssertionError("shipped path must not read local T1")


class StrictProfile:
    """No profile writes expected — read empty, apply must stay uncalled."""

    async def read_raw(self, scope_id):
        return ""

    async def apply_consolidation_updates(self, *args, **kwargs):
        raise AssertionError("no profile writes expected")


class FakeMemory:
    def __init__(self, profile) -> None:
        self.t1 = StrictT1()
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
    {"entry_id": "e1", "role": "user", "content": "dự án X deadline 10/07, lỗi ImportError auth"},
    {"entry_id": "e2", "role": "user", "content": "sáng nay chạy bộ 5km quanh hồ"},
]


@pytest.mark.asyncio
async def test_retry_with_drifted_labels_reuses_canonical_plan():
    redis = FakeRedisKV()
    t2 = FakeT2Redis(redis, fail_topics_once={"health"})
    first_payload = {"has_meaningful_content": True, "topics": [
        _topic("work", "User đang làm dự án X, deadline 10/07, lỗi ImportError ở auth."),
        _topic("health", "User chạy bộ mỗi sáng quanh hồ."),
    ]}
    tool1 = ConsolidateMemoryTool(
        FakeMemory(StrictProfile()), SeqLLM([first_payload]), FakeEmbed(), t2,
    )
    first = json.loads(await tool1.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert first["status"] == "failed"
    assert not first.get("entry_ids")
    assert len(t2.docs) == 1

    # Retry from a SECOND tool instance with drifted LLM labels.
    llm2 = SeqLLM([{"has_meaningful_content": True, "topics": [
        _topic("career", "Tóm tắt khác hẳn về công việc."),
        _topic("fitness", "Tóm tắt khác hẳn về thể thao."),
    ]}])
    tool2 = ConsolidateMemoryTool(
        FakeMemory(StrictProfile()), llm2, FakeEmbed(), t2,
    )
    second = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))

    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]
    assert second["topics_stored"] == 2
    # Full LLM never consulted on retry; pinned summaries reused, no dup.
    assert llm2.prompts == []
    assert len(t2.docs) == 2
    summaries = sorted(d["summary"] for d in t2.docs.values())
    assert summaries == sorted([
        "User đang làm dự án X, deadline 10/07, lỗi ImportError ở auth.",
        "User chạy bộ mỗi sáng quanh hồ.",
    ])
    # Pending witness: plan + batch record + per-entry owners, and NONE of
    # them expire while the batch is unacknowledged (the completion TTL is
    # stamped only by the coordinator trim-release, after the last T2 write).
    key = build_plan_cache_key("user", "u1", ENTRIES)
    assert key in redis.strings
    assert key not in redis.ttls
    assert len(redis.strings) == 4


@pytest.mark.asyncio
async def test_cached_rewrite_after_external_append_recomputes_profile_only(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))
    await store.append_raw("u1", "interest", "Thích chơi game X")
    redis = FakeRedisKV()
    t2 = FakeT2Redis(redis)

    class RacingFullLLM(SeqLLM):
        async def generate_response(self, *args, **kwargs):
            await store.append_raw("u1", "interest", "Mê board game")
            return await super().generate_response(*args, **kwargs)

    full_payload = {"has_meaningful_content": True, "topics": [
        _topic("interest", "User chán game X, dạo này hay chơi board game.", 3),
    ], "profile_rewrites": {"interest": ["Hết thích game X"]}}
    tool1 = ConsolidateMemoryTool(
        FakeMemory(store), RacingFullLLM([full_payload]), FakeEmbed(), t2,
    )
    first = json.loads(await tool1.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert first["status"] == "failed"
    assert first["reason"] == "profile_conflict"
    assert not first.get("entry_ids")
    assert len(t2.docs) == 1

    patch_payload = {"profile_updates": {}, "profile_rewrites": {
        "interest": ["Hết thích game X", "Mê board game"],
    }}
    llm2 = SeqLLM([patch_payload])
    tool2 = ConsolidateMemoryTool(FakeMemory(store), llm2, FakeEmbed(), t2)
    second = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))

    assert second["status"] == "ok"
    assert second["rewritten_sections"] == ["interest"]
    # New fact preserved; stale rewrite never blessed with the new hash.
    assert await store.read_section("u1", "interest") == ["Hết thích game X", "Mê board game"]
    # T2 idempotent: same topic reused, no second doc.
    assert len(t2.docs) == 1
    # Only a profile recompute ran, against the fresh profile text.
    assert len(llm2.prompts) == 1
    assert "Memory Profiler" in llm2.prompts[0]
    assert "Mê board game" in llm2.prompts[0]
    # Cache refreshed with the recomputed profile portion.
    key = build_plan_cache_key("user", "u1", ENTRIES)
    cached = json.loads(redis.strings[key])
    assert cached["profile_rewrites"]["interest"] == ["Hết thích game X", "Mê board game"]
    assert cached["topics"][0]["topic"] == "interest"


@pytest.mark.asyncio
async def test_unchanged_profile_retry_replays_mutations_safely(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))

    class FlakyProfile:
        def __init__(self) -> None:
            self.fail_next = True

        async def read_raw(self, user_id):
            return await store.read_raw(user_id)

        async def apply_consolidation_updates(self, *args, **kwargs):
            if self.fail_next:
                self.fail_next = False
                raise OSError("injected disk fault")
            return await store.apply_consolidation_updates(*args, **kwargs)

    profile = FlakyProfile()
    redis = FakeRedisKV()
    t2 = FakeT2Redis(redis)
    payload = {"has_meaningful_content": True, "topics": [_topic("work", "User bận dự án X.")],
               "profile_updates": {"work": ["Đang làm dự án X"]}}
    tool1 = ConsolidateMemoryTool(FakeMemory(profile), SeqLLM([payload]), FakeEmbed(), t2)
    first = json.loads(await tool1.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert first["status"] == "failed"
    assert first["reason"] == "profile_write_failed"
    assert not first.get("entry_ids")

    llm2 = SeqLLM([payload])
    tool2 = ConsolidateMemoryTool(FakeMemory(profile), llm2, FakeEmbed(), t2)
    second = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert second["status"] == "ok"
    assert llm2.prompts == []
    assert await store.read_section("u1", "work") == ["Đang làm dự án X"]
    assert len(t2.docs) == 1


@pytest.mark.asyncio
async def test_grown_batch_overlapping_pending_is_rejected():
    # INTENTIONAL contract change (overlap-retry repair): the old suite
    # allowed a grown batch to bypass a failed batch's plan. That bypass is
    # exactly what duplicated OLD_A across drifted topics: the partial T2
    # write survives, so the grown snapshot must NOT mint a fresh plan. The
    # original batch must complete first; the grown input fails closed with
    # no new LLM call and no new writes — even when the first attempt
    # stored zero topics (a zero report cannot prove no writer committed).
    redis = FakeRedisKV()
    t2 = FakeT2Redis(redis, fail_topics_once={"work"})
    tool1 = ConsolidateMemoryTool(
        FakeMemory(StrictProfile()),
        SeqLLM([{"has_meaningful_content": True, "topics": [_topic("work", "Chuyện A.")]}]),
        FakeEmbed(), t2,
    )
    first = json.loads(await tool1.execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(ENTRIES[0])]))
    assert first["status"] == "failed"
    assert len(t2.docs) == 0

    llm2 = SeqLLM([{"has_meaningful_content": True, "topics": [
        _topic("work", "Chuyện A và B gộp lại."),
    ]}])
    tool2 = ConsolidateMemoryTool(FakeMemory(StrictProfile()), llm2, FakeEmbed(), t2)
    second = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert second["status"] == "failed"
    assert second["reason"] == "pending_overlap"
    assert not second.get("entry_ids")
    assert llm2.prompts == []
    assert len(t2.docs) == 0


@pytest.mark.asyncio
async def test_legacy_store_without_idempotency_fails_closed():
    tool = ConsolidateMemoryTool(
        FakeMemory(StrictProfile()),
        SeqLLM([{"has_meaningful_content": True, "topics": [_topic("work", "User bận dự án X.")]}]),
        FakeEmbed(), LegacyT2(),
    )
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=list(ENTRIES)))
    assert res["status"] == "failed"
    assert res["reason"] == "t2_store_failed"
    assert res["topics_failed"] == 1
    assert not res.get("entry_ids")


@pytest.mark.asyncio
async def test_claim_elects_single_winner_atomically():
    # Same intent as before (one atomic winner, loser adopts), now via the
    # journal claim that persists the plan WITH the entry ownership.
    redis = FakeRedisKV()
    journal = ReceiverJournal(redis)
    first = await journal.claim_new("user", "u1", ENTRIES, "plan-A")
    second = await journal.claim_new("user", "u1", ENTRIES, "plan-B")
    assert (first.status, first.payload) == ("claimed", "plan-A")
    assert (second.status, second.payload) == ("adopted", "plan-A")
    key = build_plan_cache_key("user", "u1", ENTRIES)
    assert redis.strings[key] == "plan-A"
    assert key not in redis.ttls


def test_plan_cache_key_binds_snapshot_identity():
    entries = list(ENTRIES)
    # Order matters: the prompt renders entries sequentially, so a reorder
    # must mint a fresh plan instead of reusing the cached one.
    reordered = [dict(ENTRIES[1]), dict(ENTRIES[0])]
    assert build_plan_cache_key("user", "u1", entries) != build_plan_cache_key("user", "u1", reordered)
    altered = [dict(ENTRIES[0], content="nội dung khác"), dict(ENTRIES[1])]
    assert build_plan_cache_key("user", "u1", entries) != build_plan_cache_key("user", "u1", altered)
    assert build_plan_cache_key("user", "u1", entries) != build_plan_cache_key("user", "u2", entries)
    assert build_plan_cache_key("user", "u1", entries) != build_plan_cache_key("channel", "u1", entries)


def test_plan_cache_load_rejects_snapshot_mismatch():
    import asyncio

    from twin.shared.tools.modules.memory.consolidation_plan_cache import (
        serialize_canonical_plan,
    )
    from twin.shared.tools.modules.memory.consolidation_schema import (
        ConsolidationPlan,
        TopicPlan,
    )

    async def go():
        redis = FakeRedisKV()
        cache = CanonicalPlanCache(redis)
        plan = ConsolidationPlan(
            True, (TopicPlan("work", "W", "S", 4),), {}, {},
        )
        key = build_plan_cache_key("user", "u1", ENTRIES)
        await redis.set(key, serialize_canonical_plan(
            plan, scope="user", scope_id="u1",
            entry_ids=["e1", "e2"], profile_hash="h",
        ))
        # Same key namespace but a grown snapshot must fail closed, not
        # silently miss and mint divergent topics.
        grown = list(ENTRIES) + [{"entry_id": "e3", "role": "user", "content": "mới"}]
        bad = await cache.load(key, scope="user", scope_id="u1",
                               entry_ids=["e1", "e2", "e3"])
        assert bad.status == "error"
        assert bad.plan is None
        good = await cache.load(key, scope="user", scope_id="u1",
                                entry_ids=["e1", "e2"])
        assert good.status == "hit"
        assert good.plan is not None

    asyncio.run(go())


# ------------------------------------------------------- coordinator ack gating


class FakeT1Trim:
    def __init__(self) -> None:
        self.trim_calls: list[tuple] = []
        self.store = SimpleNamespace(redis=FakeRedisKV())

    async def get_context(self, scope, scope_id, limit=200):
        return []

    async def get_entries_by_ids(self, scope, scope_id, entry_ids):
        return []

    async def list_unsummarized_entries(self, scope, scope_id, limit=200):
        return []

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

    async def consolidate_scope(self, scope, scope_id, reason="auto", entries=None, **kw):
        return dict(self.result)


def _manager(t1, result):
    return SharedMemoryManager(
        active=t1, profile_store=object(),  # type: ignore[arg-type]
        timeline_summary_store=SimpleNamespace(redis=FakeRedisKV()),
        consolidation_client=FakeClient(result),
    )


@pytest.mark.asyncio
async def test_manager_rejects_foreign_ack_ids():
    t1 = FakeT1Trim()
    mgr = _manager(t1, {"status": "ok", "entry_ids": ["foreign"]})
    res = await mgr.consolidate_scope("user", "u1", entries=[{"entry_id": "e1"}])
    assert res["status"] == "ok"
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_manager_rejects_meaningful_ack_with_zero_writes():
    t1 = FakeT1Trim()
    mgr = _manager(t1, {
        "status": "ok", "entry_ids": ["e1"],
        "has_meaningful_content": True, "topics_stored": 0, "topics_failed": 0,
    })
    await mgr.consolidate_scope("user", "u1", entries=[{"entry_id": "e1"}])
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_manager_trims_explicit_noise_ack():
    t1 = FakeT1Trim()
    mgr = _manager(t1, {
        "status": "ok", "entry_ids": ["e1"],
        "has_meaningful_content": False, "topics_stored": 0, "topics_failed": 0,
    })
    await mgr.consolidate_scope("user", "u1", entries=[{"entry_id": "e1"}])
    assert t1.trim_calls == [("user", "u1", ["e1"])]


@pytest.mark.asyncio
async def test_manager_failed_result_keeps_t1():
    t1 = FakeT1Trim()
    mgr = _manager(t1, {"status": "failed", "reason": "t2_store_failed"})
    res = await mgr.consolidate_scope("user", "u1", entries=[{"entry_id": "e1"}])
    assert res["status"] == "failed"
    assert t1.trim_calls == []
