"""Fail-closed consolidation + profile hardening regressions.

Covers findings #5,#6,#7,#23,#24,#26,#27,#28 focused scenarios:
incomplete JSON, malformed boolean, partial failure+idempotent retry,
profile I/O failure, stale rewrite conflict, explicit empty shipment,
newline validation, persona atomic write, VN timestamp anchoring.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from types import SimpleNamespace
from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.active import ActiveEntry
from twin.shared.memory.profile import MarkdownProfileStore
from twin.shared.memory.profile.codec import profile_hash
from twin.shared.tools.modules.memory.consolidate_memory_tool import (
    ConsolidateMemoryTool,
)
from twin.shared.tools.modules.memory.consolidation_ids import (
    build_idempotency_key,
)
from twin.shared.tools.modules.profile.update_personality_tool import (
    UpdatePersonalityTool,
)


class FakeT1:
    def __init__(self, entries=None):
        self._entries = list(entries or [])
        self.get_context_calls: list[dict] = []
        self.store = SimpleNamespace(redis=FakePlanRedis())

    async def get_context(self, scope, scope_id, *, limit=50):
        self.get_context_calls.append({"scope": scope, "scope_id": scope_id})
        return list(self._entries)

    async def get_entries_by_ids(self, scope, scope_id, entry_ids):
        by_id = {e.entry_id: e for e in self._entries}
        return [by_id[i] for i in entry_ids if i in by_id]

    async def list_unsummarized_entries(self, scope, scope_id, limit=200):
        return list(self._entries[-limit:])


class FakeProfile:
    def __init__(self):
        self.read_raw_calls: list[str] = []
        self.append_calls: list[tuple] = []
        self.replace_calls: list[tuple] = []

    async def read_raw(self, scope_id):
        self.read_raw_calls.append(scope_id)
        return ""

    async def append_raw(self, scope_id, section, bullet, source_memory_id=None):
        self.append_calls.append((scope_id, section, bullet))
        return True

    async def apply_consolidation_updates(self, scope_id, rewrites, appends, expected_profile_hash):
        self.append_calls.append((scope_id, "__atomic__", (dict(rewrites or {}), dict(appends or {}))))
        rewritten = [s for s, b in (rewrites or {}).items() if b]
        updated = [s for s, b in (appends or {}).items() if b and s not in rewritten]
        return {
            "ok": True, "conflict": False, "written": True,
            "updated_sections": updated, "rewritten_sections": rewritten,
        }

    async def replace_section(self, scope_id, section, bullets, expected_profile_hash=None):
        self.replace_calls.append((scope_id, section, list(bullets)))
        return {"ok": True, "conflict": False, "written": True}


class FakeMemory:
    def __init__(self, t1=None, profile=None):
        self.t1 = t1 or FakeT1()
        self.profile = profile or FakeProfile()


class FakeLLM:
    model = "fake-model"

    def __init__(self, payload):
        self._payload = payload
        self.prompts: list[str] = []

    async def generate_response(self, *args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else [])
        if messages:
            self.prompts.append(messages[0].get("content", ""))
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload, ensure_ascii=False)


class FakeEmbed:
    async def get_embedding(self, text):
        return [0.1] * 8


class FakePlanRedis(JournalFakeRedis):
    """Real-behavior async Redis string surface + journal EVAL subset."""


class FakeT2:
    def __init__(self):
        self.redis = FakePlanRedis()
        self.calls: list[dict] = []

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display,
        importance, period_start=None, period_end=None, source_entry_ids=None,
        idempotency_key=None,
    ):
        self.calls.append({"topic": topic, "summary": summary, "idempotency_key": idempotency_key})
        return f"sum-{topic}"


class FakeT2Idempotent:
    """Diary double with idempotency_key dedup + one-shot failure."""

    def __init__(self, fail_topics_once=None):
        self.redis = FakePlanRedis()
        self.by_key: dict[str, str] = {}
        self.docs: dict[str, dict] = {}
        self.fail_once = set(fail_topics_once or [])
        self.calls = 0

    async def store_summary(
        self, *, user_id, summary, embedding, topic, topic_display,
        importance, period_start=None, period_end=None, source_entry_ids=None,
        idempotency_key=None,
    ):
        self.calls += 1
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


def _valid_topic(topic="work", summary="User đang làm dự án X, deadline rõ ràng.", importance=4):
    return {
        "topic": topic, "topic_display": "Công việc",
        "summary": summary, "importance": importance,
    }


@pytest.mark.asyncio
async def test_incomplete_true_only_fails_without_ack():
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM({"has_meaningful_content": True}),
                                 FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "hello"}],
    ))
    assert res["status"] == "failed"
    assert not res.get("entry_ids")


@pytest.mark.asyncio
async def test_malformed_boolean_string_fails():
    payload = {"has_meaningful_content": "true", "topics": [_valid_topic()]}
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "hello"}],
    ))
    assert res["status"] == "failed"
    assert not res.get("entry_ids")


@pytest.mark.asyncio
async def test_malformed_boolean_int_fails():
    payload = {"has_meaningful_content": 1, "topics": [_valid_topic()]}
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "hello"}],
    ))
    assert res["status"] == "failed"
    assert not res.get("entry_ids")


@pytest.mark.asyncio
async def test_noise_false_with_topics_is_inconsistent():
    payload = {"has_meaningful_content": False, "topics": [_valid_topic()]}
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "hi"}],
    ))
    assert res["status"] == "failed"
    assert not res.get("entry_ids")


@pytest.mark.asyncio
async def test_noise_false_empty_succeeds_with_ack():
    payload = {"has_meaningful_content": False, "topics": [],
               "profile_updates": {}, "profile_rewrites": {}}
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "hi"}],
    ))
    assert res["status"] == "ok"
    assert res["entry_ids"] == ["e1"]
    assert res["topics_stored"] == 0


@pytest.mark.asyncio
async def test_invalid_topic_empty_summary_fails():
    payload = {"has_meaningful_content": True,
               "topics": [_valid_topic(summary="   ")]}
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "hello"}],
    ))
    assert res["status"] == "failed"
    assert not res.get("entry_ids")


@pytest.mark.asyncio
async def test_explicit_empty_shipment_skips_without_local_read():
    e = ActiveEntry(entry_id="local1", scope="user", scope_id="u1",
                    role="user", content="local only")
    t1 = FakeT1([e])
    mem = FakeMemory(t1=t1)
    tool = ConsolidateMemoryTool(mem, FakeLLM({"has_meaningful_content": False, "topics": []}),
                                 FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=[],
    ))
    assert res["status"] == "skipped"
    assert res["reason"] == "no_messages"
    assert res["entry_ids"] == []
    assert t1.get_context_calls == []
    assert mem.profile.read_raw_calls == []


@pytest.mark.asyncio
async def test_none_entries_reads_local_owned_scope():
    e = ActiveEntry(entry_id="local1", scope="user", scope_id="u1",
                    role="user", content="dự án X deadline rõ")
    t1 = FakeT1([e])
    mem = FakeMemory(t1=t1)
    payload = {"has_meaningful_content": True, "topics": [_valid_topic()]}
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(scope="user", scope_id="u1", reason="x", entries=None))
    assert res["status"] == "ok"
    assert res["entry_ids"] == ["local1"]
    # Pending-first local selection (chat get_context is not used here).
    assert t1.get_context_calls == []


@pytest.mark.asyncio
async def test_partial_failure_returns_failed_then_retry_is_idempotent():
    entries = [
        {"entry_id": "e1", "role": "user", "content": "chuyện A"},
        {"entry_id": "e2", "role": "user", "content": "chuyện B"},
    ]
    payload = {"has_meaningful_content": True,
               "topics": [_valid_topic("work"), _valid_topic("health", "User chạy bộ mỗi sáng.")]}
    t2 = FakeT2Idempotent(fail_topics_once={"health"})
    mem = FakeMemory()
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), t2)
    first = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x", entries=entries))
    assert first["status"] == "failed"
    assert not first.get("entry_ids")
    assert first["topics_failed"] == 1
    assert len(t2.docs) == 1

    # Retry same snapshot, same topics: first topic must reuse its key, no dup.
    tool2 = ConsolidateMemoryTool(FakeMemory(), FakeLLM(payload), FakeEmbed(), t2)
    second = json.loads(await tool2.execute(
        scope="user", scope_id="u1", reason="x", entries=entries))
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]
    assert second["topics_stored"] == 2
    assert len(t2.docs) == 2
    # Stable keys: same scope+IDs+topic => same key.
    k1 = build_idempotency_key("user", "u1", ["e1", "e2"], "work")
    k2 = build_idempotency_key("user", "u1", ["e2", "e1"], "work")
    k3 = build_idempotency_key("user", "u1", ["e1", "e2"], "health")
    assert k1 == k2
    assert k1 != k3


@pytest.mark.asyncio
async def test_profile_io_failure_fails_without_ack(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))
    await store.append_raw("u1", "basic", "Tên: A")

    class FailingProfile(MarkdownProfileStore):
        async def apply_consolidation_updates(self, *a, **k):
            raise OSError("disk full")

    failing = FailingProfile(base_path=str(tmp_path))
    mem = FakeMemory(profile=failing)
    payload = {"has_meaningful_content": True, "topics": [_valid_topic()],
               "profile_updates": {"work": ["Đang làm dự án X"]}}
    tool = ConsolidateMemoryTool(mem, FakeLLM(payload), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "dự án X"}],
    ))
    assert res["status"] == "failed"
    assert not res.get("entry_ids")
    # No partial profile write leaked the new fact.
    assert await store.read_section("u1", "work") == []


@pytest.mark.asyncio
async def test_stale_rewrite_conflict_keeps_new_facts(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))
    await store.append_raw("u1", "interest", "Thích chơi game")
    old_text = await store.read_raw("u1")
    old_hash = profile_hash(old_text)
    # External agent appends a new fact after our snapshot.
    await store.append_raw("u1", "interest", "Mê board game")
    # Stale rewrite from old snapshot must conflict, not overwrite.
    res = await store.apply_consolidation_updates(
        "u1", {"interest": ["Hết thích game"]}, {}, old_hash,
    )
    assert res["ok"] is False
    assert res["conflict"] is True
    assert await store.read_section("u1", "interest") == ["Thích chơi game", "Mê board game"]


@pytest.mark.asyncio
async def test_concurrent_append_during_llm_causes_tool_conflict(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))
    await store.append_raw("u1", "interest", "Thích chơi game")

    class RacingLLM:
        model = "fake-model"

        async def generate_response(self, *args, **kwargs):
            # External writer lands between profile-read and profile-write.
            await store.append_raw("u1", "interest", "Mê board game")
            return json.dumps({
                "has_meaningful_content": True,
                "topics": [_valid_topic("interest", "User chán game cũ.")],
                "profile_rewrites": {"interest": ["Hết thích game"]},
            })

    mem = FakeMemory(profile=store)
    tool = ConsolidateMemoryTool(mem, RacingLLM(), FakeEmbed(), FakeT2())
    res = json.loads(await tool.execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "chán game"}],
    ))
    assert res["status"] == "failed"
    assert res["reason"] == "profile_conflict"
    assert not res.get("entry_ids")
    assert await store.read_section("u1", "interest") == ["Thích chơi game", "Mê board game"]


@pytest.mark.asyncio
async def test_append_rejects_multiline_without_loss(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))
    with pytest.raises(ValueError):
        await store.append_raw("u1", "basic", "dòng một\ndòng hai")
    with pytest.raises(ValueError):
        await store.append_raw("u1", "basic", "has\rcarriage")
    with pytest.raises(ValueError):
        await store.append_raw("u1", "basic", "- prefixed")
    with pytest.raises(ValueError):
        await store.append_raw("u1", "basic", "bad\x00control")
    assert await store.read_section("u1", "basic") == []
    assert await store.append_raw("u1", "basic", "Tên: A") is True
    assert await store.read_section("u1", "basic") == ["Tên: A"]


@pytest.mark.asyncio
async def test_persona_write_fault_preserves_old_file(tmp_path, monkeypatch):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    target = persona_dir / "SOUL.md"
    target.write_text("old soul\n", encoding="utf-8")
    tool = UpdatePersonalityTool(base_memory_path=str(persona_dir))

    import twin.shared.tools.modules.profile.update_personality_tool as mod

    orig_replace = mod.os.replace

    def _fail_replace(src, dst):
        raise OSError("injected disk fault")

    monkeypatch.setattr(mod.os, "replace", _fail_replace)
    with pytest.raises(Exception):
        await tool.execute("new soul content", target_file="SOUL.md")
    assert target.read_text(encoding="utf-8") == "old soul\n"
    assert not (persona_dir / "SOUL.md.tmp").exists()
    monkeypatch.setattr(mod.os, "replace", orig_replace)
    ok = await tool.execute("new soul content", target_file="SOUL.md")
    assert "Đã cập nhật SOUL.md" in ok
    assert target.read_text(encoding="utf-8") == "new soul content\n"


@pytest.mark.asyncio
async def test_prompt_carries_vn_timestamps_and_entry_anchor():
    # Backlog entry from 01/07/2026 10:00 VN (03:00Z) saying "ngày mai".
    ts = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
    entries = [{
        "entry_id": "e1", "role": "user", "content": "ngày mai họp với anh Nam",
        "author_name": "Hoà", "timestamp": ts.isoformat(),
    }]
    llm = FakeLLM({"has_meaningful_content": False, "topics": []})
    tool = ConsolidateMemoryTool(FakeMemory(), llm, FakeEmbed(), FakeT2())
    await tool.execute(scope="user", scope_id="u1", reason="x", entries=entries)
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "10:00 01/07/2026" in prompt
    assert "thời điểm của từng tin nhắn" in prompt
    assert "KHÔNG dựa trên thời điểm hiện tại" in prompt


@pytest.mark.asyncio
async def test_prompt_includes_active_entry_vn_time():
    created = datetime(2026, 7, 2, 1, 30, tzinfo=timezone.utc)  # 08:30 VN
    e = ActiveEntry(entry_id="a1", scope="user", scope_id="u1",
                    role="user", content="ngày mai đi khám", created_at=created)
    t1 = FakeT1([e])
    llm = FakeLLM({"has_meaningful_content": False, "topics": []})
    tool = ConsolidateMemoryTool(FakeMemory(t1=t1), llm, FakeEmbed(), FakeT2())
    await tool.execute(scope="user", scope_id="u1", reason="x", entries=None)
    assert "08:30 02/07/2026" in llm.prompts[0]


@pytest.mark.asyncio
async def test_manager_does_not_trim_on_partial_ok(tmp_path):
    from twin.shared.memory.manager import SharedMemoryManager

    class T1:
        def __init__(self):
            self.trim_calls: list[tuple] = []

        async def get_context(self, scope, scope_id, limit=200):
            return []

        async def trim(self, scope, scope_id, entry_ids, keep_recent=None):
            self.trim_calls.append((scope, scope_id, list(entry_ids)))

    class Client:
        async def consolidate_scope(self, scope, scope_id, reason="auto", entries=None, **kw):
            return {"status": "ok", "entry_ids": ["e1"], "topics_failed": 1}

    t1 = T1()
    mgr = SharedMemoryManager(active=t1, profile_store=MarkdownProfileStore(base_path=str(tmp_path)),
                              consolidation_client=Client())
    res = await mgr.consolidate_scope("user", "u1", entries=[])
    assert res["status"] == "ok"
    assert t1.trim_calls == []


def test_idempotency_key_shape():
    k = build_idempotency_key("user", "u1", ["e1"], "work")
    assert k.startswith("consol:")
    assert k == build_idempotency_key("user", "u1", ["e1"], "work")
    assert k != build_idempotency_key("channel", "u1", ["e1"], "work")
    assert k != build_idempotency_key("user", "u1", ["e2"], "work")
