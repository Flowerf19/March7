"""Unit tests for SharedMemoryManager consolidation wiring (Cách B).

Covers the coordinated consolidation batch:
- Bug 1/2: consolidate_scope ships THIS agent's T1 entries over the wire and
  trims by the entry_ids the consolidator returns (never by count).
- Bug 1 guard: status=ok without entry_ids must NOT trim (data-loss guard).
- Bug 3: channel-scope recall searches T2 for both speaker and channel ids.

(Bug 4's no-re-observe guarantee lives in the tool path now — evernight's A2A
handler calls agent.consolidate_via_tool directly; covered in
consolidate_tool_test.py.)
"""
from __future__ import annotations

import logging

import pytest

from types import SimpleNamespace
from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.active import ActiveEntry
from twin.shared.memory.manager import SharedMemoryManager


class FakeT1:
    """ActiveMemory double recording get_context / trim / observe calls."""

    def __init__(self, entries: list[ActiveEntry] | None = None) -> None:
        self._entries = entries or []
        self.get_context_calls: list[dict] = []
        self.trim_calls: list[tuple] = []
        self.store = SimpleNamespace(redis=JournalFakeRedis())

    async def get_context(self, scope, scope_id, *, limit: int = 50):
        self.get_context_calls.append({"scope": scope, "scope_id": scope_id, "limit": limit})
        return list(self._entries)

    async def get_entries_by_ids(self, scope, scope_id, entry_ids):
        by_id = {e.entry_id: e for e in self._entries}
        return [by_id[i] for i in entry_ids if i in by_id]

    async def list_unsummarized_entries(self, scope, scope_id, limit=200):
        return list(self._entries[-limit:])

    async def trim(self, scope, scope_id, entry_ids, *, keep_recent=None):
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


class FakeConsolidationClient:
    """Records the consolidate_scope payload and returns a canned result."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def consolidate_scope(self, scope, scope_id, reason="auto", max_messages=200, entries=None):
        self.calls.append({
            "scope": scope,
            "scope_id": scope_id,
            "reason": reason,
            "max_messages": max_messages,
            "entries": entries,
        })
        return dict(self.result)


class FakeTimelineStore:
    """Records the user_ids searched so channel-vs-speaker recall is testable."""

    def __init__(self) -> None:
        self.searched_user_ids: list[str] = []

    async def search(self, user_id, query_embedding, limit=5, *, query_text=None, topic_filter=None):
        self.searched_user_ids.append(user_id)
        return [{"summary_id": f"sum-{user_id}", "summary": f"summary for {user_id}"}]


class FakeEmbeddingService:
    async def get_embedding(self, text):
        return [0.1] * 8


class FakeProfileStore:
    async def get_system_prompt_context(self, user_id):
        return ""


def _entry(entry_id: str, content: str) -> ActiveEntry:
    return ActiveEntry(
        entry_id=entry_id,
        scope="channel",
        scope_id="chan1",
        role="user",
        content=content,
    )


@pytest.mark.asyncio
async def test_consolidate_scope_ships_entries_and_trims_by_returned_ids():
    t1 = FakeT1([_entry("e1", "hello"), _entry("e2", "world"), _entry("e3", "extra")])
    # Complete batch receipt required: ACK must cover all pinned IDs exactly
    # (partial_ack rejected, no trim). Production tool always full-ACKs.
    client = FakeConsolidationClient({
        "status": "ok", "entry_ids": ["e1", "e2", "e3"],
        "has_meaningful_content": True, "topics_stored": 2, "topics_failed": 0,
        "receiver_generation": "f" * 32,
    })
    manager = SharedMemoryManager(
        active=t1,  # type: ignore[arg-type]
        profile_store=FakeProfileStore(),  # type: ignore[arg-type]
        timeline_summary_store=SimpleNamespace(redis=JournalFakeRedis()),
        consolidation_client=client,
    )

    result = await manager.consolidate_scope("channel", "chan1")

    assert result["status"] == "ok"
    # Entries were shipped over the wire with entry_id + content.
    shipped = client.calls[0]["entries"]
    assert shipped is not None and len(shipped) == 3
    assert {e["entry_id"] for e in shipped} == {"e1", "e2", "e3"}
    assert all("content" in e for e in shipped)
    # Trim used exactly the entry_ids the consolidator returned — full batch.
    assert t1.trim_calls == [("channel", "chan1", ["e1", "e2", "e3"])]


@pytest.mark.asyncio
async def test_consolidate_scope_ok_without_entry_ids_does_not_trim(caplog):
    t1 = FakeT1([_entry("e1", "hello")])
    client = FakeConsolidationClient({"status": "ok"})  # no entry_ids
    manager = SharedMemoryManager(
        active=t1,  # type: ignore[arg-type]
        profile_store=FakeProfileStore(),  # type: ignore[arg-type]
        consolidation_client=client,
    )

    with caplog.at_level(logging.WARNING):
        result = await manager.consolidate_scope("channel", "chan1")

    assert result["status"] == "ok"
    # Data-loss guard: never trim by count when entry_ids are absent.
    assert t1.trim_calls == []
    assert any("no entry_ids" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_get_context_channel_scope_does_not_touch_t2():
    """T2 recall is tool-only (2026-07-03): get_context must not search T2.

    The dual-scope (speaker+channel) recall this test used to assert now lives
    in the search_memory tool — the manager's job is only to expose the channel
    scope id in the header so the model can pass `channel_id` to the tool.
    """
    t1 = FakeT1([])
    timeline = FakeTimelineStore()
    manager = SharedMemoryManager(
        active=t1,  # type: ignore[arg-type]
        profile_store=FakeProfileStore(),  # type: ignore[arg-type]
        timeline_summary_store=timeline,
        embedding_service=FakeEmbeddingService(),
    )

    sys_prompt, _ = await manager.get_context(
        user_id="speaker1",
        current_query="nhớ gì không",
        channel_id="chan1",
    )

    assert timeline.searched_user_ids == []
    assert "Platform channel ID: chan1" in sys_prompt
