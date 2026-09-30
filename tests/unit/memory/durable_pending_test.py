"""Durable overlap-retry regressions (journal ownership + caller pending).

Parent reproduction: partial e1/e2 attempt (A stored, B failed, no ACK),
then e3 arrives and the grown e1/e2/e3 snapshot mints a fresh plan whose
drifted topics duplicate OLD_A across topics with overlapping provenance.
These tests pin the repair:

- Receiver journal (shared timeline Redis): per-entry owners + bounded
  batch record, Lua-atomic all-or-none claim. Disjoint batches proceed;
  overlapping different snapshots fail closed BEFORE any full LLM call or
  T2/T3 write; exact retries adopt the original canonical plan.
- Caller journal (own T1 Redis): the FIRST ordered batch is claimed before
  dispatch and exact-fetched by ID on retry (window-independent, >200-safe,
  summarized-tail excluded). Validated ACKs are journaled before trim;
  trim/release/clear failures all resume from the durable ACK record.
- Pending witnesses never expire and are never released on failure; the
  completion TTL starts only after the last T2 write (trim-release).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

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
    build_plan_cache_key,
    caller_pending_key,
)
from twin.shared.memory.profile import MarkdownProfileStore
from twin.shared.tools.modules.memory.consolidate_memory_tool import (
    ConsolidateMemoryTool,
)

OLD_A = "OLDFACT-A project X deadline 10/07 ImportError auth"
OLD_B = "OLDFACT-B morning run 5km lake"
NEW_C = "NEWFACT-C bought board game Catan"


def _topic(topic, summary, importance=4):
    return {"topic": topic, "topic_display": topic, "summary": summary,
            "importance": importance}


def _entry(entry_id, content, ts_min=0):
    return ActiveEntry(
        entry_id=entry_id, scope="user", scope_id="u1", role="user",
        content=content, author_id="u1", author_name="U",
        message_id=f"m-{entry_id}",
        created_at=datetime(2026, 7, 1, 3, ts_min, tzinfo=timezone.utc),
    )


class MemT1:
    """Minimal T1 facade double with the consolidation selection surface."""

    def __init__(self, entries=None) -> None:
        self._entries = list(entries or [])
        self.trim_calls: list[tuple] = []
        self.fail_next_trim: Exception | None = None
        self.store = SimpleNamespace(redis=JournalFakeRedis())

    async def get_entries_by_ids(self, scope, scope_id, entry_ids):
        by_id = {e.entry_id: e for e in self._entries}
        return [by_id[i] for i in entry_ids if i in by_id]

    async def list_unsummarized_entries(self, scope, scope_id, limit=200):
        return list(self._entries[-limit:])

    async def trim(self, scope, scope_id, entry_ids, keep_recent=None):
        self.trim_calls.append((scope, scope_id, list(entry_ids)))
        if self.fail_next_trim is not None:
            err, self.fail_next_trim = self.fail_next_trim, None
            raise err
        gone = set(entry_ids or [])
        self._entries = [e for e in self._entries if e.entry_id not in gone]

    async def trim_consolidated_batch(self, scope, scope_id, record, keep_recent=None):
        import json as _json
        from twin.shared.memory.consolidation_journal import caller_pending_key as _cpk
        if self.fail_next_trim is not None:
            err, self.fail_next_trim = self.fail_next_trim, None
            self.trim_calls.append((scope, scope_id, list((record or {}).get("entry_ids") or [])))
            raise err
        key = _cpk(scope, scope_id)
        raw = self.store.redis._get_str(key)
        if raw is None:
            return {"status": "failed", "reason": "missing"}
        try:
            cur = _json.loads(raw)
        except Exception:
            return {"status": "failed", "reason": "corrupt"}
        exp_ids = list((record or {}).get("entry_ids") or [])
        if cur.get("entry_ids") != exp_ids or cur.get("caller_nonce") != (record or {}).get("caller_nonce") or cur.get("plan_key") != (record or {}).get("plan_key"):
            return {"status": "failed", "reason": "mismatch"}
        if cur.get("stage") == "trimmed":
            return {"status": "already"}
        if cur.get("stage") != "acknowledged":
            return {"status": "failed", "reason": "not_acked"}
        self.trim_calls.append((scope, scope_id, list(exp_ids)))
        gone = set(exp_ids)
        self._entries = [e for e in self._entries if e.entry_id not in gone]
        cur["stage"] = "trimmed"
        self.store.redis.strings[key] = _json.dumps(cur, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return {"status": "trimmed", "deleted_ids": list(exp_ids), "subtracted": 0, "unsummarized_tokens": 0}


class FakeT2:
    """Idempotent diary double (keyed dedup + one-shot topic failures)."""

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
        self.docs[sid] = {"topic": topic, "summary": summary,
                          "source_entry_ids": list(source_entry_ids or [])}
        if idempotency_key is not None:
            self.by_key[idempotency_key] = sid
        return sid


class StrictProfile:
    async def read_raw(self, scope_id):
        return ""

    async def apply_consolidation_updates(self, *args, **kwargs):
        raise AssertionError("no profile writes expected")


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


class ToolClient:
    """Coordinator client delegating to a real ConsolidateMemoryTool."""

    def __init__(self, tool) -> None:
        self.tool = tool
        self.calls: list[dict] = []

    async def consolidate_scope(self, scope, scope_id, reason="auto", entries=None, **kw):
        self.calls.append({"scope": scope, "scope_id": scope_id, "entries": entries})
        return json.loads(await self.tool.execute(
            scope=scope, scope_id=scope_id, reason=reason, entries=entries))


def _tool(llm, t2, profile=None):
    mem = SimpleNamespace(t1=None, profile=profile or StrictProfile())
    return ConsolidateMemoryTool(mem, llm, FakeEmbed(), t2)


def _coords(t1, client, receiver_redis):
    return ConsolidationCoordinator(
        t1, lambda: client, lambda: None,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: receiver_redis,
    )


P1 = {"has_meaningful_content": True, "topics": [
    _topic("project_alpha", OLD_A), _topic("health", OLD_B)]}
P3 = {"has_meaningful_content": True, "topics": [_topic("leisure", NEW_C)]}


# ------------------------------------------------- flagship: restart pins

@pytest.mark.asyncio
async def test_auto_restart_pins_original_batch_then_processes_new():
    """e1/e2 partial -> restart -> e3 arrives: e1/e2 complete first with no
    drift, then e3 is processed alone with no OLD_A duplication."""
    t1 = MemT1([_entry("e1", "project X deadline 10/07 ImportError auth", 0),
                _entry("e2", "morning run 5km lake", 5)])
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"health"})
    llm = SeqLLM([P1, P3])
    tool = _tool(llm, t2)

    first = await _coords(t1, ToolClient(tool), receiver).consolidate_scope("user", "u1")
    assert first["status"] == "failed"  # B failed; A durable; no ACK
    assert t1.trim_calls == []
    assert len(t2.docs) == 1

    # "Restart": brand-new coordinator over the same durable state, e3 lands.
    t1._entries.append(_entry("e3", "bought board game Catan", 10))
    client2 = ToolClient(tool)
    second = await _coords(t1, client2, receiver).consolidate_scope("user", "u1")
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]  # pinned original, NOT grown
    assert [e["entry_id"] for e in client2.calls[0]["entries"]] == ["e1", "e2"]
    assert len(llm.prompts) == 1  # exact retry adopted the witness, no re-LLM
    assert t1.trim_calls == [("user", "u1", ["e1", "e2"])]
    assert len(t2.docs) == 2  # A idempotent (same batch+topic), B stored

    third = await _coords(t1, ToolClient(tool), receiver).consolidate_scope("user", "u1")
    assert third["status"] == "ok"
    assert third["entry_ids"] == ["e3"]
    summaries = sorted(d["summary"] for d in t2.docs.values())
    assert summaries == sorted([OLD_A, OLD_B, NEW_C])
    assert sum(OLD_A in s for s in summaries) == 1
    assert not any(OLD_A in s and NEW_C in s for s in summaries)


@pytest.mark.asyncio
async def test_direct_grown_shipment_rejected_with_zero_new_llm_or_writes():
    """Parent proof inverted: grown e1/e2/e3 over partial e1/e2 is rejected."""
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"health"})
    e1 = {"entry_id": "e1", "role": "user", "content": "project X"}
    e2 = {"entry_id": "e2", "role": "user", "content": "run 5km"}
    e3 = {"entry_id": "e3", "role": "user", "content": "catan"}
    first = json.loads(await _tool(SeqLLM([P1]), t2).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(e1), dict(e2)]))
    assert first["status"] == "failed"
    assert len(t2.docs) == 1

    llm2 = SeqLLM([{"has_meaningful_content": True, "topics": [
        _topic("work_stuff", f"{OLD_A} ; ALSO {NEW_C}"), _topic("health", OLD_B)]}])
    grown = json.loads(await _tool(llm2, t2).execute(
        scope="user", scope_id="u1", reason="x",
        entries=[dict(e1), dict(e2), dict(e3)]))
    assert grown["status"] == "failed"
    assert grown["reason"] == "pending_overlap"
    assert not grown.get("entry_ids")
    assert llm2.prompts == []
    assert len(t2.docs) == 1  # still only the original A


# ------------------------------------------------------- concurrency shape

@pytest.mark.asyncio
async def test_same_concurrent_batch_single_canonical_winner():
    import asyncio

    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)
    entries = [{"entry_id": "e1", "role": "user", "content": "A"},
               {"entry_id": "e2", "role": "user", "content": "B"}]
    pa = {"has_meaningful_content": True, "topics": [_topic("work", "Summary A.")]}
    pb = {"has_meaningful_content": True, "topics": [_topic("career", "Summary B.")]}
    ta = _tool(SeqLLM([pa]), t2)
    tb = _tool(SeqLLM([pb]), t2)
    ra, rb = await asyncio.gather(
        ta.execute(scope="user", scope_id="u1", reason="x",
                   entries=[dict(e) for e in entries]),
        tb.execute(scope="user", scope_id="u1", reason="x",
                   entries=[dict(e) for e in entries]),
    )
    assert json.loads(ra)["status"] == "ok"
    assert json.loads(rb)["status"] == "ok"
    assert len(t2.docs) == 1  # one winner, adopted by the loser
    assert [d["summary"] for d in t2.docs.values()] in (["Summary A."], ["Summary B."])


@pytest.mark.asyncio
async def test_disjoint_concurrent_batches_both_proceed():
    import asyncio

    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)
    a = [{"entry_id": "e1", "role": "user", "content": "A"}]
    b = [{"entry_id": "e2", "role": "user", "content": "B"}]
    pa = {"has_meaningful_content": True, "topics": [_topic("wa", "Sum A.")]}
    pb = {"has_meaningful_content": True, "topics": [_topic("wb", "Sum B.")]}
    ra, rb = await asyncio.gather(
        _tool(SeqLLM([pa]), t2).execute(
            scope="user", scope_id="u1", reason="x", entries=[dict(a[0])]),
        _tool(SeqLLM([pb]), t2).execute(
            scope="user", scope_id="u2", reason="x", entries=[dict(b[0])]),
    )
    assert json.loads(ra)["status"] == "ok"
    assert json.loads(rb)["status"] == "ok"
    assert len(t2.docs) == 2


@pytest.mark.asyncio
async def test_zero_store_failure_still_owned_auto_pins_original():
    """A total T2 failure (zero stored) keeps ownership: auto pins the
    original batch even after new input arrives (stronger than the old
    zero-commit bypass, which could not prove no writer committed)."""
    t1 = MemT1([_entry("e1", "A", 0)])
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"work"})
    llm = SeqLLM([{"has_meaningful_content": True, "topics": [_topic("work", "Sum A.")]},
                  {"has_meaningful_content": True, "topics": [_topic("work", "Sum B.")]}])
    tool = _tool(llm, t2)
    first = await _coords(t1, ToolClient(tool), receiver).consolidate_scope("user", "u1")
    assert first["status"] == "failed"
    assert len(t2.docs) == 0

    t1._entries.append(_entry("e2", "B", 5))
    client2 = ToolClient(tool)
    second = await _coords(t1, client2, receiver).consolidate_scope("user", "u1")
    assert [e["entry_id"] for e in client2.calls[0]["entries"]] == ["e1"]
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1"]


# ------------------------------------------------- provenance sensitivity

ENTRIES = [
    {"entry_id": "e1", "message_id": "m1", "role": "user", "content": "A",
     "author_id": "u1", "author_name": "U", "timestamp": "2026-07-01T03:00:00+00:00"},
    {"entry_id": "e2", "message_id": "m2", "role": "user", "content": "B",
     "author_id": "u1", "author_name": "U", "timestamp": "2026-07-01T03:05:00+00:00"},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutate", [
    lambda es: [dict(es[0], content="CHANGED"), dict(es[1])],
    lambda es: [dict(es[0], role="assistant"), dict(es[1])],
    lambda es: [dict(es[1]), dict(es[0])],
    lambda es: [dict(es[0], timestamp="2026-07-02T03:00:00+00:00"), dict(es[1])],
    lambda es: [dict(es[0], author_name="Stranger"), dict(es[1])],
])
async def test_altered_batch_overlapping_owned_rejected(mutate):
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"health"})
    first = json.loads(await _tool(SeqLLM([P1]), t2).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(e) for e in ENTRIES]))
    assert first["status"] == "failed"  # owns e1/e2 with original fingerprints

    llm2 = SeqLLM([P1])
    retry = json.loads(await _tool(llm2, t2).execute(
        scope="user", scope_id="u1", reason="x", entries=mutate(ENTRIES)))
    assert retry["status"] == "failed"
    assert retry["reason"] == "pending_overlap"
    assert llm2.prompts == []
    assert len(t2.docs) == 1


@pytest.mark.asyncio
async def test_explicit_grown_shipment_rejected_before_dispatch():
    t1 = MemT1([_entry("e1", "A", 0), _entry("e2", "B", 5)])
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"health"})
    tool = _tool(SeqLLM([P1]), t2)
    coords = _coords(t1, ToolClient(tool), receiver)
    assert (await coords.consolidate_scope("user", "u1"))["status"] == "failed"

    client2 = ToolClient(tool)
    coords2 = _coords(t1, client2, receiver)
    grown = [{"entry_id": "e1", "role": "user", "content": "A"},
             {"entry_id": "e2", "role": "user", "content": "B"},
             {"entry_id": "e3", "role": "user", "content": "C"}]
    res = await coords2.consolidate_scope("user", "u1", entries=grown)
    assert res["status"] == "failed"
    assert res["error"] == "pending_mismatch"
    assert client2.calls == []  # rejected before dispatch, input never replaced
    assert t1.trim_calls == []


# ------------------------------------------------- window + tail behavior

def _real_t1():
    redis = T1FakeRedis()
    mem = ActiveMemory(store=ActiveStore(redis), detector=FastPathDetector(),
                       token_counter=lambda _: 1)
    return mem, redis


@pytest.mark.asyncio
async def test_pending_exact_fetch_beyond_200_window():
    mem, _ = _real_t1()
    o1 = await mem.observe("user", "u1", "user", "orig one")
    o2 = await mem.observe("user", "u1", "user", "orig two")
    orig_ids = [o1.entry_id, o2.entry_id]
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"health"})
    llm = SeqLLM([P1, P3])
    tool = _tool(llm, t2)
    coords = ConsolidationCoordinator(
        mem, lambda: ToolClient(tool), lambda: None,
        get_caller_redis=lambda: mem.store.redis,
        get_receiver_redis=lambda: receiver,
    )
    assert (await coords.consolidate_scope("user", "u1"))["status"] == "failed"

    for i in range(250):  # push the pending batch out of any recent window
        await mem.observe("user", "u1", "user", f"filler {i}")
    client2 = ToolClient(tool)
    coords2 = ConsolidationCoordinator(
        mem, lambda: client2, lambda: None,
        get_caller_redis=lambda: mem.store.redis,
        get_receiver_redis=lambda: receiver,
    )
    second = await coords2.consolidate_scope("user", "u1")
    assert second["status"] == "ok"
    assert second["entry_ids"] == orig_ids  # exact ID fetch, not the window
    assert [e["entry_id"] for e in client2.calls[0]["entries"]] == orig_ids

    client3 = ToolClient(tool)
    coords3 = ConsolidationCoordinator(
        mem, lambda: client3, lambda: None,
        get_caller_redis=lambda: mem.store.redis,
        get_receiver_redis=lambda: receiver,
    )
    third = await coords3.consolidate_scope("user", "u1")
    assert third["status"] == "ok"
    shipped3 = [e["entry_id"] for e in client3.calls[0]["entries"]]
    assert len(shipped3) == 200  # bounded fresh batch from the 250 new
    assert not (set(shipped3) & set(orig_ids))


@pytest.mark.asyncio
async def test_summarized_tail_never_reconsolidated():
    mem, _ = _real_t1()
    for i in range(7):
        await mem.observe("user", "u1", "user", f"msg {i}")
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)
    llm = SeqLLM([{"has_meaningful_content": True, "topics": [_topic("g", "Seven.")]},
                  {"has_meaningful_content": True, "topics": [_topic("g", "Eight. customer new")]}])

    def _c(client):
        return ConsolidationCoordinator(
            mem, lambda: client, lambda: None,
            get_caller_redis=lambda: mem.store.redis,
            get_receiver_redis=lambda: receiver,
        )

    tool = _tool(llm, t2)
    first = await _c(ToolClient(tool)).consolidate_scope("user", "u1")
    assert first["status"] == "ok"
    assert len(first["entry_ids"]) == 7
    # Real trim keeps the 5 most recent (marked summarized), deletes oldest 2.
    assert len(await mem.get_context("user", "u1", limit=50)) == 5

    idle_client = ToolClient(tool)
    idle = await _c(idle_client).consolidate_scope("user", "u1")
    assert idle["status"] == "skipped"  # retained tail excluded, no dispatch
    assert idle_client.calls == []

    await mem.observe("user", "u1", "user", "brand new eight")
    fresh_client = ToolClient(tool)
    fresh = await _c(fresh_client).consolidate_scope("user", "u1")
    assert fresh["status"] == "ok"
    assert len(fresh_client.calls[0]["entries"]) == 1
    assert fresh_client.calls[0]["entries"][0]["content"] == "brand new eight"


# ------------------------------------------------- ack/trim/release paths

class _FlakyClient(ToolClient):
    """Drops the first successful response (lost ACK), then behaves."""

    def __init__(self, tool) -> None:
        super().__init__(tool)
        self.dropped = False

    async def consolidate_scope(self, scope, scope_id, reason="auto", entries=None, **kw):
        self.calls.append({"scope": scope, "scope_id": scope_id, "entries": entries})
        result = json.loads(await self.tool.execute(
            scope=scope, scope_id=scope_id, reason=reason, entries=entries))
        if not self.dropped:
            self.dropped = True
            return {"status": "failed", "scope": scope, "scope_id": scope_id,
                    "error": "transport timeout after commit"}
        return result


@pytest.mark.asyncio
async def test_lost_ack_retries_exact_batch_idempotently():
    t1 = MemT1([_entry("e1", "A", 0), _entry("e2", "B", 5)])
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)
    llm = SeqLLM([P1])
    tool = _tool(llm, t2)
    client = _FlakyClient(tool)
    coords = _coords(t1, client, receiver)
    assert (await coords.consolidate_scope("user", "u1"))["status"] == "failed"
    assert len(t2.docs) == 2  # receiver actually committed everything
    assert t1.trim_calls == []

    llm2 = SeqLLM([P1])  # fresh LLM: exact retry must not need it
    tool.llm_service = llm2
    second = await coords.consolidate_scope("user", "u1")
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]
    assert llm2.prompts == []
    assert len(t2.docs) == 2  # idempotent re-store, no duplicates
    assert t1.trim_calls == [("user", "u1", ["e1", "e2"])]


@pytest.mark.asyncio
async def test_trim_failure_resumes_from_durable_ack_without_redispatch():
    t1 = MemT1([_entry("e1", "A", 0), _entry("e2", "B", 5)])
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)
    tool = _tool(SeqLLM([P1]), t2)
    client = ToolClient(tool)
    coords = _coords(t1, client, receiver)
    t1.fail_next_trim = OSError("redis blip")

    first = await coords.consolidate_scope("user", "u1")
    assert first["status"] == "failed"
    assert first["error"] == "trim_failed"
    assert first["ack_durable"] is True
    assert len(t1._entries) == 2  # T1 preserved for retry
    loaded = await CallerJournal(t1.store.redis).load("user", "u1")
    assert loaded.status == "acknowledged"

    second = await coords.consolidate_scope("user", "u1")
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]
    assert len(client.calls) == 1  # no redispatch, no LLM rerun
    assert len(t1.trim_calls) == 2
    assert len(t2.docs) == 2
    assert (await CallerJournal(t1.store.redis).load("user", "u1")).status == "missing"


class _FlakyEvalRedis(JournalFakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_eval: Exception | None = None

    async def eval(self, script, numkeys, *keys_and_args):
        if self.fail_next_eval is not None:
            err, self.fail_next_eval = self.fail_next_eval, None
            raise err
        return await super().eval(script, numkeys, *keys_and_args)


@pytest.mark.asyncio
async def test_release_failure_after_trim_resumes_without_entries_or_llm():
    t1 = MemT1([_entry("e1", "A", 0), _entry("e2", "B", 5)])
    receiver = _FlakyEvalRedis()
    t2 = FakeT2(receiver)
    llm = SeqLLM([P1])
    tool = _tool(llm, t2)
    client = ToolClient(tool)
    orig_execute = tool.execute

    async def _execute_and_arm(*args, **kwargs):
        out = await orig_execute(*args, **kwargs)
        receiver.fail_next_eval = OSError("redis blip")  # fail the release only
        return out

    tool.execute = _execute_and_arm  # type: ignore[method-assign]
    coords = _coords(t1, client, receiver)
    first = await coords.consolidate_scope("user", "u1")
    assert first["status"] == "failed"
    assert first["error"] == "release_failed"
    assert first["ack_durable"] is True
    assert t1._entries == []  # trim already deleted the raw entries

    second = await coords.consolidate_scope("user", "u1")
    assert second["status"] == "ok"
    assert second["entry_ids"] == ["e1", "e2"]
    assert len(client.calls) == 1  # resumed from ACK: no redispatch/LLM
    assert len(llm.prompts) == 1
    assert len(t2.docs) == 2
    assert (await CallerJournal(t1.store.redis).load("user", "u1")).status == "missing"
    key = build_plan_cache_key("user", "u1", client.calls[0]["entries"])
    # Owners released; completed plan carries the fresh completion TTL.
    assert receiver.strings.get("consol:own:user:u1:e1") is None
    assert receiver.ttls.get(key) == 365 * 86400


# ------------------------------------------------- journal failure closure

@pytest.mark.asyncio
async def test_corrupt_caller_record_fails_without_dispatch_or_trim():
    t1 = MemT1([_entry("e1", "A", 0)])
    t1.store.redis.strings[caller_pending_key("user", "u1")] = "{oops"
    receiver = JournalFakeRedis()
    client = ToolClient(_tool(SeqLLM([P1]), FakeT2(receiver)))
    res = await _coords(t1, client, receiver).consolidate_scope("user", "u1")
    assert res["status"] == "failed"
    assert res["error"] == "journal_error"
    assert client.calls == []
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_missing_caller_journal_fails_closed():
    t1 = MemT1([_entry("e1", "A", 0)])
    coords = ConsolidationCoordinator(t1, lambda: ToolClient(None), lambda: None)
    res = await coords.consolidate_scope("user", "u1")
    assert res["status"] == "failed"
    assert res["error"] == "journal_error"
    assert t1.trim_calls == []


@pytest.mark.asyncio
async def test_receiver_owner_without_batch_fails_before_llm():
    receiver = JournalFakeRedis()
    receiver.strings["consol:own:user:u1:e1"] = "deadbeef"
    t2 = FakeT2(receiver)
    llm = SeqLLM([P1])
    res = json.loads(await _tool(llm, t2).execute(
        scope="user", scope_id="u1", reason="x",
        entries=[{"entry_id": "e1", "role": "user", "content": "A"}]))
    assert res["status"] == "failed"
    assert res["reason"] == "plan_cache_error"
    assert llm.prompts == []
    assert t2.calls == []


@pytest.mark.asyncio
async def test_exact_retry_with_lost_plan_fails_without_fresh_mint():
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver, fail_topics_once={"health"})
    entries = [{"entry_id": "e1", "role": "user", "content": "A"},
               {"entry_id": "e2", "role": "user", "content": "B"}]
    first = json.loads(await _tool(SeqLLM([P1]), t2).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(e) for e in entries]))
    assert first["status"] == "failed"
    key = build_plan_cache_key("user", "u1", entries)
    del receiver.strings[key]  # out-of-band plan loss; owners/batch survive

    llm2 = SeqLLM([P1])
    retry = json.loads(await _tool(llm2, t2).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(e) for e in entries]))
    assert retry["status"] == "failed"
    assert retry["reason"] == "plan_cache_error"
    assert llm2.prompts == []  # never a fresh mint for a known batch
    assert len(t2.docs) == 1


# ------------------------------------------------- profile interplay

@pytest.mark.asyncio
async def test_profile_write_failure_keeps_ownership_retry_succeeds(tmp_path):
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
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)
    payload = {"has_meaningful_content": True,
               "topics": [_topic("work", "User bận dự án X.")],
               "profile_updates": {"work": ["Đang làm dự án X"]}}
    entries = [{"entry_id": "e1", "role": "user", "content": "dự án X deadline"}]
    first = json.loads(await _tool(SeqLLM([payload]), t2, profile).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(entries[0])]))
    assert first["status"] == "failed"
    assert first["reason"] == "profile_write_failed"
    assert len(t2.docs) == 1  # T2 partial durable; ownership retained

    grown_llm = SeqLLM([payload])
    grown = json.loads(await _tool(grown_llm, t2, profile).execute(
        scope="user", scope_id="u1", reason="x",
        entries=[dict(entries[0]), {"entry_id": "e2", "role": "user", "content": "new"}]))
    assert grown["reason"] == "pending_overlap"
    assert grown_llm.prompts == []

    llm2 = SeqLLM([payload])
    second = json.loads(await _tool(llm2, t2, profile).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(entries[0])]))
    assert second["status"] == "ok"
    assert llm2.prompts == []  # adopted witness, no re-LLM
    assert len(t2.docs) == 1  # idempotent: same batch+topic
    assert await store.read_section("u1", "work") == ["Đang làm dự án X"]


@pytest.mark.asyncio
async def test_profile_drift_recomputes_only_profile_under_journal(tmp_path):
    store = MarkdownProfileStore(base_path=str(tmp_path))
    await store.append_raw("u1", "interest", "Thích chơi game X")
    receiver = JournalFakeRedis()
    t2 = FakeT2(receiver)

    class RacingFullLLM(SeqLLM):
        async def generate_response(self, *args, **kwargs):
            await store.append_raw("u1", "interest", "Mê board game")
            return await super().generate_response(*args, **kwargs)

    entries = [{"entry_id": "e1", "role": "user", "content": "chán game X"}]
    full = {"has_meaningful_content": True, "topics": [
        _topic("interest", "User chán game X, mê board game.", 3)],
        "profile_rewrites": {"interest": ["Hết thích game X"]}}
    first = json.loads(await _tool(RacingFullLLM([full]), t2, store).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(entries[0])]))
    assert first["status"] == "failed"
    assert first["reason"] == "profile_conflict"

    patch = {"profile_updates": {}, "profile_rewrites": {
        "interest": ["Hết thích game X", "Mê board game"]}}
    llm2 = SeqLLM([patch])
    second = json.loads(await _tool(llm2, t2, store).execute(
        scope="user", scope_id="u1", reason="x", entries=[dict(entries[0])]))
    assert second["status"] == "ok"
    assert second["rewritten_sections"] == ["interest"]
    assert await store.read_section("u1", "interest") == ["Hết thích game X", "Mê board game"]
    assert len(t2.docs) == 1  # topic pinned, no divergent re-mint
    assert len(llm2.prompts) == 1 and "Memory Profiler" in llm2.prompts[0]


# ------------------------------------------------- retention horizon

@pytest.mark.asyncio
async def test_pending_witness_survives_100_366_days_then_completion_ttl():
    receiver = JournalFakeRedis()
    journal = ReceiverJournal(receiver)
    entries = [{"entry_id": "e1", "role": "user", "content": "A"}]
    key = build_plan_cache_key("user", "u1", entries)
    got = await journal.claim_new("user", "u1", entries, "plan-payload")
    assert got.status == "claimed"

    # Late T2 partials persist 365d from EACH write: a claim-time TTL would
    # die first, so pending witnesses carry no TTL and are never reminted.
    for day in (100, 366, 465):
        receiver.now = day * 86400
        probe = await journal.probe("user", "u1", entries)
        assert probe.status == "exact"
        assert probe.plan_key == key
        assert key not in receiver.ttls

    # Completion (after trim) stamps one fresh 365d TTL starting then.
    receiver.now = 100 * 86400
    released = await journal.release(
        "user", "u1", ["e1"], batch_id_of_plan_key(key), key, got.generation)
    assert released.ok
    assert receiver.ttls.get(key) == 365 * 86400
    receiver.now = 465 * 86400 + 1  # 100 + 365: aligned with the T2 window
    assert receiver._get_str(key) is None
