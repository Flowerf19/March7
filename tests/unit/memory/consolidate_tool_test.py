"""Unit tests for ConsolidateMemoryTool shipped-entries path (Cách B, Bug 2/4)
and the P2 diary write path (prompt v2, provenance, profile_rewrites).

When entries are shipped over A2A, the tool must consolidate THOSE dicts and
return their entry_ids WITHOUT reading the local T1 store (which belongs to the
requesting agent, not Evernight).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from types import SimpleNamespace
from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.active import ActiveEntry
from twin.shared.tools.modules.memory.consolidate_memory_tool import ConsolidateMemoryTool


class FakeT1:
    def __init__(self) -> None:
        self.get_context_calls: list[dict] = []
        self._entries: list = []
        self.store = SimpleNamespace(redis=FakePlanRedis())

    async def get_context(self, scope, scope_id, *, limit: int = 50):
        self.get_context_calls.append({"scope": scope, "scope_id": scope_id, "limit": limit})
        return []

    async def get_entries_by_ids(self, scope, scope_id, entry_ids):
        by_id = {e.entry_id: e for e in self._entries}
        return [by_id[i] for i in entry_ids if i in by_id]

    async def list_unsummarized_entries(self, scope, scope_id, limit=200):
        return list(self._entries[-limit:])


class FakeProfileStore:
    def __init__(self) -> None:
        self.read_raw_calls: list[str] = []
        self.atomic_calls: list[tuple] = []

    async def read_raw(self, scope_id):
        self.read_raw_calls.append(scope_id)
        return ""

    async def apply_consolidation_updates(self, scope_id, rewrites, appends, expected_profile_hash):
        self.atomic_calls.append((scope_id, dict(rewrites or {}), dict(appends or {})))
        rewritten = [s for s, b in (rewrites or {}).items() if b]
        updated = [s for s, b in (appends or {}).items() if b and s not in rewritten]
        return {
            "ok": True, "conflict": False, "written": True,
            "updated_sections": updated, "rewritten_sections": rewritten,
        }


class FakeMemoryManager:
    def __init__(self) -> None:
        self.t1 = FakeT1()
        self.profile = FakeProfileStore()


class FakeLLM:
    model = "fake-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_response(self, *args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else [])
        if messages:
            self.prompts.append(messages[0].get("content", ""))
        return json.dumps({
            "has_meaningful_content": True,
            "topics": [
                {"topic": "work", "topic_display": "Công việc", "summary": "User đang bận dự án.", "importance": 4}
            ],
            "profile_updates": {},
        })


class FakeEmbeddingService:
    async def get_embedding(self, text):
        return [0.1] * 8


class FakePlanRedis(JournalFakeRedis):
    """Real-behavior async Redis string surface + journal EVAL subset."""


class FakeTimelineStore:
    def __init__(self) -> None:
        self.redis = FakePlanRedis()
        self.store_calls: list[dict] = []

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display, importance,
        period_start=None, period_end=None, source_entry_ids=None, idempotency_key=None,
    ):
        self.store_calls.append({
            "user_id": user_id,
            "summary": summary,
            "topic": topic,
            "importance": importance,
            "period_start": period_start,
            "period_end": period_end,
            "source_entry_ids": source_entry_ids,
            "idempotency_key": idempotency_key,
        })
        return f"sum-{topic}"


class FakeTimelineStoreAllFail:
    """store_summary always raises — models T2 fully down (dim mismatch or
    Redis error). Every topic store attempt fails, so nothing lands in T2."""

    def __init__(self) -> None:
        self.redis = FakePlanRedis()
        self.calls = 0

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display, importance,
        period_start=None, period_end=None, source_entry_ids=None, idempotency_key=None,
    ):
        self.calls += 1
        raise ValueError("embedding dim mismatch")


class FakeLLMWithProfileUpdates:
    """LLM stub that also returns non-empty profile_updates, to prove the
    channel-scope path genuinely skips step 6 rather than happening to have
    nothing to write."""

    model = "fake-model"

    async def generate_response(self, *args, **kwargs):
        return json.dumps({
            "has_meaningful_content": True,
            "topics": [
                {"topic": "work", "topic_display": "Công việc", "summary": "User đang bận dự án.", "importance": 4}
            ],
            "profile_updates": {"work": ["Đang làm dự án X"]},
        })


class FakeLLMWithRewrites:
    """LLM stub returning both profile_rewrites and profile_updates, with an
    overlapping section — proves rewrites run first and suppress the
    duplicate appends for the rewritten section only."""

    model = "fake-model"

    async def generate_response(self, *args, **kwargs):
        return json.dumps({
            "has_meaningful_content": True,
            "topics": [
                {"topic": "interest", "topic_display": "Sở thích", "summary": "User chán game X rồi.", "importance": 3}
            ],
            "profile_updates": {
                "interest": ["Hết thích game X"],
                "work": ["Đang làm dự án Y"],
            },
            "profile_rewrites": {
                "interest": ["Hết thích game X", "Vẫn mê board game"],
                "habit": [],  # empty rewrite must be ignored (would wipe the section)
            },
        })


@pytest.mark.asyncio
async def test_execute_with_shipped_entries_returns_ids_and_skips_local_t1():
    memory = FakeMemoryManager()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLM(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=FakeTimelineStore(),
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "dự án deadline tuần tới", "author_name": "Hoà"},
        {"entry_id": "e2", "role": "assistant", "content": "ghi nhận nhé"},
    ]

    result_str = await tool.execute(
        scope="channel",
        scope_id="chan1",
        reason="discussion",
        entries=entries,
    )
    result = json.loads(result_str)

    assert result["status"] == "ok"
    assert result["entry_ids"] == ["e1", "e2"]
    assert result["messages_summarized"] == 2
    # Shipped-entries path must NOT read the local T1 store.
    assert memory.t1.get_context_calls == []


@pytest.mark.asyncio
async def test_channel_scope_skips_profile_read_and_write():
    memory = FakeMemoryManager()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLMWithProfileUpdates(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=FakeTimelineStore(),
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "dự án deadline tuần tới", "author_name": "Hoà"},
    ]

    result_str = await tool.execute(
        scope="channel",
        scope_id="chan1",
        reason="discussion",
        entries=entries,
    )
    result = json.loads(result_str)

    assert result["status"] == "ok"
    # Channel scope must neither read nor write memories/<channel_id>.md, even
    # though the LLM returned non-empty profile_updates.
    assert memory.profile.read_raw_calls == []
    assert memory.profile.atomic_calls == []
    assert result["updated_sections"] == []


@pytest.mark.asyncio
async def test_all_t2_stores_fail_returns_failed_no_trim_shape():
    """When every meaningful T2 store attempt raises, the tool must report
    status=failed with no trim-triggering entry_ids (Finding 1 regression).

    The manager trims T1 on status=="ok" + entry_ids; returning "ok" here would
    delete the transcript while nothing landed in T2 (silent data loss). T1 must
    stay untouched — the tool must not present the trim-triggering shape.
    """
    memory = FakeMemoryManager()
    store = FakeTimelineStoreAllFail()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLM(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=store,
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "dự án deadline tuần tới", "author_name": "Hoà"},
    ]

    result_str = await tool.execute(
        scope="channel",
        scope_id="chan1",
        reason="discussion",
        entries=entries,
    )
    result = json.loads(result_str)

    # The store was attempted (meaningful topic) and it raised → nothing stored.
    assert store.calls == 1
    assert result["topics_stored"] == 0
    # Must NOT be the trim-triggering shape: manager trims only on status==ok,
    # and there must be no entry_ids to trim by.
    assert result["status"] == "failed"
    assert not result.get("entry_ids")
    # Shipped-entries path never reads local T1 — and certainly must not trim it.
    assert memory.t1.get_context_calls == []


# ---------------------------------------------------------------- P2.3: prompt v2


@pytest.mark.asyncio
async def test_summarizer_prompt_v2_contract():
    """Prompt must carry the current VN datetime (for relative→absolute date
    conversion) and the profile_rewrites key in its JSON contract."""
    memory = FakeMemoryManager()
    llm = FakeLLM()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=llm,
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=FakeTimelineStore(),
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "deadline tuần tới", "author_name": "Hoà"},
    ]
    await tool.execute(scope="channel", scope_id="chan1", reason="x", entries=entries)

    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    # Current VN date injected (dd/mm/yyyy somewhere after "Bây giờ là").
    assert "Bây giờ là" in prompt
    from twin.shared.memory.vn_time import vn_now
    assert vn_now().strftime("%d/%m/%Y") in prompt
    # Relative→absolute instruction + diary detail requirement + new contract key.
    assert "tuyệt đối" in prompt
    assert "3-5 câu" in prompt
    assert "profile_rewrites" in prompt
    # The old misleading promise ("mâu thuẫn thì ghi đè" on the append path)
    # must be gone — contradiction handling now goes through profile_rewrites.
    assert "mâu thuẫn thì ghi đè" not in prompt


# ---------------------------------------------------------------- P2.2/P2.5: provenance


@pytest.mark.asyncio
async def test_store_receives_period_span_and_provenance_from_shipped_entries():
    """period_start/period_end must span the source entries' OWN timestamps
    (shipped ISO strings), and source_entry_ids must carry the batch ids —
    not consolidate-time values."""
    memory = FakeMemoryManager()
    store = FakeTimelineStore()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLM(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=store,
    )

    t0 = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 7, 1, 5, 30, tzinfo=timezone.utc)
    entries = [
        {"entry_id": "e1", "role": "user", "content": "chuyện A", "timestamp": t0.isoformat()},
        {"entry_id": "e2", "role": "assistant", "content": "chuyện B", "timestamp": t1.isoformat()},
    ]

    await tool.execute(scope="channel", scope_id="chan1", reason="x", entries=entries)

    assert len(store.store_calls) == 1
    call = store.store_calls[0]
    assert call["period_start"] == pytest.approx(t0.timestamp())
    assert call["period_end"] == pytest.approx(t1.timestamp())
    assert call["source_entry_ids"] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_store_receives_period_from_local_active_entries():
    """Local (non-shipped) path: timestamps come from ActiveEntry.created_at."""
    memory = FakeMemoryManager()
    e1 = ActiveEntry(
        entry_id="a1", scope="user", scope_id="u1", role="user", content="chuyện A",
        created_at=datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc),
    )
    e2 = ActiveEntry(
        entry_id="a2", scope="user", scope_id="u1", role="user", content="chuyện B",
        created_at=datetime(2026, 7, 2, 2, 0, tzinfo=timezone.utc),
    )
    memory.t1._entries = [e1, e2]

    store = FakeTimelineStore()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLM(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=store,
    )

    await tool.execute(scope="user", scope_id="u1", reason="x")

    call = store.store_calls[0]
    assert call["period_start"] == pytest.approx(e1.created_at.timestamp())
    assert call["period_end"] == pytest.approx(e2.created_at.timestamp())
    assert call["source_entry_ids"] == ["a1", "a2"]


@pytest.mark.asyncio
async def test_entries_without_timestamps_fall_back_to_consolidate_time():
    """Malformed shipped entries (no timestamp) must not crash — the period
    falls back to consolidate-time (defensive path only)."""
    memory = FakeMemoryManager()
    store = FakeTimelineStore()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLM(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=store,
    )

    before = datetime.now(timezone.utc).timestamp()
    entries = [{"entry_id": "e1", "role": "user", "content": "chuyện A"}]
    result = json.loads(
        await tool.execute(scope="channel", scope_id="chan1", reason="x", entries=entries)
    )
    after = datetime.now(timezone.utc).timestamp()

    assert result["status"] == "ok"
    call = store.store_calls[0]
    assert before <= call["period_start"] <= after
    assert before <= call["period_end"] <= after


# ---------------------------------------------------------------- P2.5: profile_rewrites


@pytest.mark.asyncio
async def test_profile_rewrites_apply_atomically_and_suppress_duplicate_appends():
    memory = FakeMemoryManager()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLMWithRewrites(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=FakeTimelineStore(),
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "chán game X rồi", "author_name": "Hoà"},
    ]
    result = json.loads(
        await tool.execute(scope="user", scope_id="u1", reason="x", entries=entries)
    )

    assert result["status"] == "ok"
    # One atomic apply carrying rewrites + appends; the empty "habit"
    # rewrite never reaches the store (would wipe the section).
    assert memory.profile.atomic_calls == [
        ("u1",
         {"interest": ["Hết thích game X", "Vẫn mê board game"]},
         {"interest": ["Hết thích game X"], "work": ["Đang làm dự án Y"]}),
    ]
    assert result["rewritten_sections"] == ["interest"]
    # The rewritten section must NOT also be appended (would re-duplicate);
    # the untouched "work" section still goes through the normal append path.
    assert result["updated_sections"] == ["work"]


@pytest.mark.asyncio
async def test_channel_scope_skips_profile_rewrites_too():
    memory = FakeMemoryManager()
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLMWithRewrites(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=FakeTimelineStore(),
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "chán game X rồi", "author_name": "Hoà"},
    ]
    result = json.loads(
        await tool.execute(scope="channel", scope_id="chan1", reason="x", entries=entries)
    )

    assert result["status"] == "ok"
    assert memory.profile.atomic_calls == []
    assert result["rewritten_sections"] == []


@pytest.mark.asyncio
async def test_profile_apply_failure_fails_closed_without_ack_ids():
    """Fail-closed: a raising atomic apply must fail the whole snapshot
    with no acknowledgement IDs (no trim), not swallow per-section."""
    memory = FakeMemoryManager()

    async def failing_apply(scope_id, rewrites, appends, expected_profile_hash=None):
        raise ValueError("invalid section")

    memory.profile.apply_consolidation_updates = failing_apply
    tool = ConsolidateMemoryTool(
        memory_manager=memory,
        llm_service=FakeLLMWithRewrites(),
        embedding_service=FakeEmbeddingService(),
        timeline_summary_store=FakeTimelineStore(),
    )

    entries = [
        {"entry_id": "e1", "role": "user", "content": "chán game X rồi", "author_name": "Hoà"},
    ]
    result = json.loads(
        await tool.execute(scope="user", scope_id="u1", reason="x", entries=entries)
    )

    assert result["status"] == "failed"
    assert not result.get("entry_ids")
