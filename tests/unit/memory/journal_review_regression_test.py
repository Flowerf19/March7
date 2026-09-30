"""Journal review regressions: privacy (issue 1) + later ACK/generation cases.

Owned by journal repair agent. Each test pins a confirmed review finding
against the actual classes + JournalFakeRedis (no network, no real data).
Synthetic storage only; own UUID-free fixed scope_ids with isolated fakes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from unit.memory.journal_fake import JournalFakeRedis
from twin.shared.memory.active import ActiveEntry
from twin.shared.memory.consolidation_coordinator import ConsolidationCoordinator
from twin.shared.memory.consolidation_journal import (
    CallerJournal,
    ReceiverJournal,
    build_caller_pending,
    build_plan_cache_key,
    batch_id_of_plan_key,
    caller_pending_key,
    entry_hashes_of,
    fingerprint_hash,
    incoming_matches_pending,
    parse_caller_record,
    receiver_batch_key,
)

NEEDLE = "SYNTHETIC-SENSITIVE-NEEDLE-9f3k7q2m"


def _entries():
    return [
        {"entry_id": "e1", "message_id": "m1", "role": "user",
         "content": f"hello {NEEDLE} world", "author_id": "u1",
         "author_name": "Alice", "timestamp": "2026-07-01T03:00:00+00:00"},
        {"entry_id": "e2", "message_id": "m2", "role": "assistant",
         "content": "reply ok", "author_id": "bot",
         "author_name": "Bot", "timestamp": "2026-07-01T03:05:00+00:00"},
    ]


@pytest.mark.asyncio
async def test_privacy_no_raw_in_any_journal_payload():
    entries = _entries()
    caller_redis = JournalFakeRedis()
    receiver_redis = JournalFakeRedis()
    cj = CallerJournal(caller_redis)
    rj = ReceiverJournal(receiver_redis)

    pending = build_caller_pending("user", "priv1", entries)
    # Persisted fingerprints are 64-lower-hex hashes, not dicts/raw.
    assert all(isinstance(h, str) and len(h) == 64 for h in pending["fingerprints"])
    assert NEEDLE not in json.dumps(pending)

    assert (await cj.claim("user", "priv1", pending))[0] == "claimed"
    got = await rj.claim_new("user", "priv1", entries, "plan-payload")
    assert got.status == "claimed"

    for store in (caller_redis.strings, receiver_redis.strings):
        for key, val in store.items():
            text = val.decode("utf-8") if isinstance(val, bytes) else str(val)
            assert NEEDLE not in text, f"needle leaked in {key}"
            # No whole-transcript copy: raw content must not appear verbatim.
            assert "hello " not in text or "plan-payload" in text or "consol:plan" in key

    # After caller T1 reset (discard), receiver still holds only hashes.
    assert await cj.discard("user", "priv1") is True
    assert caller_pending_key("user", "priv1") not in caller_redis.strings
    batch_key = receiver_batch_key(
        "user", "priv1", batch_id_of_plan_key(build_plan_cache_key("user", "priv1", entries)))
    assert batch_key in receiver_redis.strings
    assert NEEDLE not in str(receiver_redis.strings[batch_key])
    # Hash match still works after discard (receiver independent).
    probe = await rj.probe("user", "priv1", entries)
    assert probe.status == "exact"


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [
    ("content", "MUTATED body"),
    ("role", "assistant"),
    ("timestamp", "2026-07-02T03:00:00+00:00"),
    ("author_name", "Mallory"),
    ("author_id", "u9"),
    ("message_id", "m9"),
])
async def test_privacy_mutated_field_rejects_hash_match(field, value):
    entries = _entries()
    pending = build_caller_pending("user", "priv2", entries)
    assert incoming_matches_pending(entries, pending) is True
    assert entry_hashes_of(entries) == pending["fingerprints"]

    mutated = [dict(entries[0], **{field: value}), dict(entries[1])]
    assert incoming_matches_pending(mutated, pending) is False

    receiver_redis = JournalFakeRedis()
    rj = ReceiverJournal(receiver_redis)
    assert (await rj.claim_new("user", "priv2", entries, "p")).status == "claimed"
    assert (await rj.probe("user", "priv2", entries)).status == "exact"
    assert (await rj.probe("user", "priv2", mutated)).status == "conflict"


@pytest.mark.asyncio
async def test_privacy_order_mutation_rejects():
    entries = _entries()
    pending = build_caller_pending("user", "priv3", entries)
    swapped = [entries[1], entries[0]]
    assert incoming_matches_pending(swapped, pending) is False
    receiver_redis = JournalFakeRedis()
    rj = ReceiverJournal(receiver_redis)
    assert (await rj.claim_new("user", "priv3", entries, "p")).status == "claimed"
    assert (await rj.probe("user", "priv3", swapped)).status == "conflict"


def test_privacy_strict_parse_rejects_v1_and_malformed():
    good = build_caller_pending("user", "priv4", _entries())
    assert parse_caller_record(json.dumps(good)) is not None

    # V1 raw dict fingerprints rejected (no legacy accept, no migration).
    v1 = dict(good, v=1, fingerprints=[{"entry_id": "e1", "content": "raw"}])
    assert parse_caller_record(json.dumps(v1)) is None

    # Non-hex / dict / count-mismatch fingerprints rejected.
    for bad_fp in ([{"x": 1}, {"y": 2}], ["not-hex", "also-bad"],
                   ["a" * 64], ["A" * 64, "B" * 64], ["a" * 63 + "g", "b" * 64]):
        bad = dict(good, fingerprints=list(bad_fp))
        assert parse_caller_record(json.dumps(bad)) is None

    # Empty / whitespace / duplicate / unbounded IDs rejected.
    for bad_ids in ([], [""], ["  "], ["e1", "e1"], ["e1", ""],
                    [f"e{i}" for i in range(201)]):
        bad = dict(good, entry_ids=list(bad_ids),
                   fingerprints=["a" * 64 for _ in bad_ids])
        assert parse_caller_record(json.dumps(bad)) is None


def test_privacy_hash_stable_objects_and_dicts():
    created = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
    obj = ActiveEntry(
        entry_id="e1", scope="user", scope_id="u1", role="user",
        content="hello", author_id="u1", author_name="Alice",
        message_id="m1", created_at=created,
    )
    shipped = {"entry_id": "e1", "message_id": "m1", "role": "user",
               "content": "hello", "author_id": "u1",
               "author_name": "Alice", "timestamp": created.isoformat()}
    assert fingerprint_hash(obj) == fingerprint_hash(shipped)
    assert len(fingerprint_hash(obj)) == 64


# ------------------------------------------------- issue 2: adopted TTL


@pytest.mark.asyncio
async def test_adopt_persists_completion_ttl_fake():
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    entries = [{"entry_id": "e1", "role": "user", "content": "A"}]
    key = build_plan_cache_key("user", "ttl1", entries)
    c1 = await rj.claim_new("user", "ttl1", entries, "plan-gen1")
    assert c1.status == "claimed"
    rel = await rj.release("user", "ttl1", ["e1"], batch_id_of_plan_key(key), key, c1.generation)
    assert rel.ok and rcv.ttls.get(key) == 365 * 86400
    rcv.now = 100 * 86400
    got = await rj.claim_new("user", "ttl1", entries, "plan-gen2")
    assert got.status == "adopted" and got.payload == "plan-gen1"
    assert key not in rcv.ttls and key not in rcv.expiries
    rcv.now = 366 * 86400
    assert (await rj.probe("user", "ttl1", entries)).status == "exact"
    assert rcv._get_str(key) == "plan-gen1"


@pytest.mark.asyncio
async def test_adopt_persists_legacy_ttl_plan():
    rcv = JournalFakeRedis()
    entries = [{"entry_id": "e1", "role": "user", "content": "A"}]
    key = build_plan_cache_key("user", "ttl2", entries)
    rcv.strings[key] = "legacy-plan"
    rcv.ttls[key] = 999
    rcv.expiries[key] = 999.0
    rj = ReceiverJournal(rcv)
    got = await rj.claim_new("user", "ttl2", entries, "mine")
    assert got.status == "adopted" and got.payload == "legacy-plan"
    assert key not in rcv.ttls and key not in rcv.expiries


# ------------------------------------------------- issue 3: full ACK


class _MemT1:
    def __init__(self, ids_or_entries):
        from types import SimpleNamespace
        if ids_or_entries and isinstance(ids_or_entries[0], dict):
            self._by_id = {e["entry_id"]: dict(e) for e in ids_or_entries}
            self.ids = [e["entry_id"] for e in ids_or_entries]
        else:
            self._by_id = {}
            self.ids = list(ids_or_entries)
        self.trims = []
        self.store = SimpleNamespace(redis=JournalFakeRedis())

    async def get_entries_by_ids(self, s, sid, eids):
        out = []
        for i in eids:
            if i in self._by_id:
                out.append(dict(self._by_id[i]))
            elif i in self.ids:
                out.append({"entry_id": i})
        return out

    async def list_unsummarized_entries(self, s, sid, limit=200):
        return []

    async def trim(self, s, sid, eids, keep_recent=None):
        self.trims.append(list(eids))
        self.ids = [i for i in self.ids if i not in set(eids)]

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
        if cur.get("entry_ids") != exp_ids or cur.get("caller_nonce") != (record or {}).get("caller_nonce") or cur.get("plan_key") != (record or {}).get("plan_key"):
            return {"status": "failed", "reason": "mismatch"}
        if cur.get("stage") == "trimmed":
            return {"status": "already"}
        if cur.get("stage") != "acknowledged":
            return {"status": "failed", "reason": "not_acked"}
        self.trims.append(list(exp_ids))
        gone = set(exp_ids)
        self.ids = [i for i in self.ids if i not in gone]
        for i in list(gone):
            self._by_id.pop(i, None)
        cur["stage"] = "trimmed"
        self.store.redis.strings[key] = _json.dumps(cur, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return {"status": "trimmed", "deleted_ids": list(exp_ids), "subtracted": 0, "unsummarized_tokens": 0}


def test_ack_trim_ids_rejects_partial_and_duplicates():
    full = {"status": "ok", "entry_ids": ["e1", "e2"],
            "topics_failed": 0, "has_meaningful_content": True,
            "topics_stored": 1}
    trim, reason = ConsolidationCoordinator._ack_trim_ids(full, {"e1", "e2"})
    assert trim == ["e1", "e2"] and reason is None
    for bad, why in [
        (["e1"], "subset"),
        (["e1", "e2", "e2"], "dup"),
        (["e1", "e1"], "dup-subset"),
        ([""], "empty"),
        (["  "], "ws"),
    ]:
        res = dict(full, entry_ids=list(bad))
        t, r = ConsolidationCoordinator._ack_trim_ids(res, {"e1", "e2"})
        assert t is None and r in ("partial_ack", "no_entry_ids"), why
    foreign = dict(full, entry_ids=["e1", "foreign"])
    t, r = ConsolidationCoordinator._ack_trim_ids(foreign, {"e1", "e2"})
    assert t is None and r == "foreign_entry_ids"


@pytest.mark.asyncio
async def test_partial_ack_keeps_owners_and_blocks_grown():
    t1 = _MemT1(["e1", "e2"])
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    assert (await rj.claim_new("user", "ack3", e12, "plan")).status == "claimed"
    pend = build_caller_pending("user", "ack3", e12)
    co = ConsolidationCoordinator(
        t1, lambda: None, lambda: None,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: rcv)
    cj = CallerJournal(t1.store.redis)
    assert (await cj.claim("user", "ack3", pend))[0] == "claimed"
    res = {"status": "ok", "entry_ids": ["e1"], "topics_failed": 0,
           "has_meaningful_content": True, "topics_stored": 1}
    out = await co._complete_validated_ack(
        "user", "ack3", pend, ["e1"], dict(res), shipped=2)
    assert out.get("status") == "failed"
    assert out.get("error") == "partial_ack"
    assert t1.trims == []
    assert rcv._get_str("consol:own:user:ack3:e1") is not None
    assert rcv._get_str("consol:own:user:ack3:e2") is not None
    assert (await cj.load("user", "ack3")).status == "pending"
    grown = [{"entry_id": "e2", "role": "user", "content": "B"},
             {"entry_id": "e3", "role": "user", "content": "C"}]
    assert (await rj.claim_new("user", "ack3", grown, "g")).status == "conflict"


# ------------------------------------------------- issue 4: ACK winner


@pytest.mark.asyncio
async def test_already_returns_stored_winner_not_empty():
    redis = JournalFakeRedis()
    cj = CallerJournal(redis)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    pend = build_caller_pending("user", "win4", e12)
    assert (await cj.claim("user", "win4", pend))[0] == "claimed"
    ack1 = {"status": "ok", "entry_ids": ["e1", "e2"],
            "topics_failed": 0, "has_meaningful_content": True,
            "topics_stored": 1, "winner": 1,
            "receiver_generation": "c" * 32}
    st1, rec1 = await cj.mark_acknowledged(
        "user", "win4", pend, ["e1", "e2"], dict(ack1))
    assert st1 == "acked" and isinstance(rec1, dict)
    assert rec1["ack"]["winner"] == 1
    ack2 = dict(ack1, winner=2)
    st2, rec2 = await cj.mark_acknowledged(
        "user", "win4", pend, ["e1", "e2"], ack2)
    assert st2 == "already" and isinstance(rec2, dict)
    assert rec2["ack"]["winner"] == 1
    assert rec2["trim_ids"] == ["e1", "e2"]


@pytest.mark.asyncio
async def test_coordinator_finalizes_stored_winner():
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    t1 = _MemT1(e12)
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    c0 = await rj.claim_new("user", "win4b", e12, "plan")
    assert c0.status == "claimed"
    pend = build_caller_pending("user", "win4b", e12)
    cj = CallerJournal(t1.store.redis)
    assert (await cj.claim("user", "win4b", pend))[0] == "claimed"
    co = ConsolidationCoordinator(
        t1, lambda: None, lambda: None,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: rcv)
    ack1 = {"status": "ok", "entry_ids": ["e1", "e2"],
            "topics_failed": 0, "has_meaningful_content": True,
            "topics_stored": 1, "winner": 1,
            "receiver_generation": c0.generation}
    out1 = await co._complete_validated_ack(
        "user", "win4b", pend, ["e1", "e2"], dict(ack1), shipped=2)
    assert out1.get("status") == "ok" and out1.get("winner") == 1
    # Second finisher with different payload must not resurrect; record cleared.
    assert (await cj.load("user", "win4b")).status == "missing"
    assert t1.trims == [["e1", "e2"]]


@pytest.mark.asyncio
async def test_ack_cas_binds_fingerprints_and_plan():
    redis = JournalFakeRedis()
    cj = CallerJournal(redis)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    pend = build_caller_pending("user", "win4c", e12)
    assert (await cj.claim("user", "win4c", pend))[0] == "claimed"
    ack = {"status": "ok", "entry_ids": ["e1", "e2"],
           "topics_failed": 0, "has_meaningful_content": True,
           "topics_stored": 1, "receiver_generation": "d" * 32}
    # Same IDs but drifted fingerprints/plan must mismatch, not ack.
    drifted = dict(pend, fingerprints=["a" * 64, "b" * 64])
    st, _ = await cj.mark_acknowledged(
        "user", "win4c", drifted, ["e1", "e2"], dict(ack))
    assert st == "mismatch"
    assert (await cj.load("user", "win4c")).status == "pending"


# ------------------------------------------------- issue 5: stored ACK


def _good_ack(ids, gen="e" * 32):
    return {"status": "ok", "entry_ids": list(ids),
            "topics_failed": 0, "has_meaningful_content": True,
            "topics_stored": 1, "receiver_generation": gen}


def test_parse_rejects_corrupt_acknowledged():
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    base = build_caller_pending("user", "ack5", e12)
    good_ack = _good_ack(["e1", "e2"])
    good_rec = dict(base, stage="acknowledged", trim_ids=["e1", "e2"],
                    ack=dict(good_ack), receiver_generation="e" * 32)
    assert parse_caller_record(json.dumps(good_rec)) is not None
    cases = []
    cases.append(dict(good_rec, trim_ids=["", "foreign"]))
    cases.append(dict(good_rec, trim_ids=[]))
    cases.append(dict(good_rec, trim_ids=["e1"]))
    cases.append(dict(good_rec, trim_ids=["e1", "e2", "e2"]))
    cases.append(dict(good_rec, trim_ids=["e1", "  "]))
    cases.append(dict(good_rec, trim_ids=["e1", "foreign"]))
    cases.append(dict(good_rec, ack=dict(good_ack, status="failed")))
    cases.append(dict(good_rec, ack=dict(good_ack, topics_stored=True)))
    cases.append(dict(good_rec, ack=dict(good_ack, topics_stored="1")))
    cases.append(dict(good_rec, ack=dict(good_ack, topics_failed="0")))
    cases.append(dict(good_rec, ack=dict(good_ack, entry_ids=["e1"])))
    cases.append(dict(good_rec, ack=dict(good_ack, entry_ids=["e1", "foreign"])))
    cases.append(dict(good_rec, ack=dict(good_ack, entry_ids=["e1", "e2", "e2"])))
    cases.append(dict(good_rec, ack="not-a-dict"))
    cases.append(dict(good_rec, plan_key="tampered"))
    for bad in cases:
        assert parse_caller_record(json.dumps(bad)) is None


@pytest.mark.asyncio
async def test_finish_revalidates_before_destructive():
    t1 = _MemT1(["e1", "e2"])
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    assert (await rj.claim_new("user", "ack5b", e12, "plan")).status == "claimed"
    pend = build_caller_pending("user", "ack5b", e12)
    co = ConsolidationCoordinator(
        t1, lambda: None, lambda: None,
        get_caller_redis=lambda: t1.store.redis,
        get_receiver_redis=lambda: rcv)
    bad_ack = dict(_good_ack(["e1"]), entry_ids=["e1"])
    bad_rec = dict(pend, stage="acknowledged",
                   trim_ids=["e1"], ack=bad_ack)
    out = await co._finish_acknowledged("user", "ack5b", bad_rec)
    assert out.get("status") == "failed"
    assert out.get("error") == "journal_error"
    assert t1.trims == []
    assert rcv._get_str("consol:own:user:ack5b:e1") is not None
    assert rcv._get_str("consol:own:user:ack5b:e2") is not None


# ------------------------------------------------- issue 6: generation ABA


@pytest.mark.asyncio
async def test_generation_claim_racers_converge_winner():
    import asyncio
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    ra, rb = await asyncio.gather(
        rj.claim_new("user", "gen6", [dict(e) for e in e12], "plan-A"),
        rj.claim_new("user", "gen6", [dict(e) for e in e12], "plan-B"),
    )
    assert sorted([ra.status, rb.status]) == ["adopted", "claimed"]
    assert ra.payload == rb.payload and ra.payload in ("plan-A", "plan-B")
    assert ra.generation and ra.generation == rb.generation
    assert len(ra.generation) == 32
    o1 = rcv._get_str("consol:own:user:gen6:e1")
    o2 = rcv._get_str("consol:own:user:gen6:e2")
    assert o1 and o2 and o1 == o2 and o1.endswith(":" + ra.generation)
    probe = await rj.probe("user", "gen6", e12)
    assert probe.status == "exact" and probe.generation == ra.generation


@pytest.mark.asyncio
async def test_generation_stale_release_keeps_new():
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    key = build_plan_cache_key("user", "gen6b", e12)
    bid = batch_id_of_plan_key(key)
    c1 = await rj.claim_new("user", "gen6b", e12, "plan-g1")
    assert c1.status == "claimed"
    rel1 = await rj.release("user", "gen6b", ["e1", "e2"], bid, key, c1.generation)
    assert rel1.ok and rel1.released == 2
    rcv.strings.pop(key, None); rcv.ttls.pop(key, None); rcv.expiries.pop(key, None)
    c2 = await rj.claim_new("user", "gen6b", e12, "plan-g2")
    assert c2.status == "claimed" and c2.payload == "plan-g2"
    assert c2.generation != c1.generation
    stale = await rj.release("user", "gen6b", ["e1", "e2"], bid, key, c1.generation)
    assert not stale.ok
    assert rcv._get_str("consol:own:user:gen6b:e1") == f"{bid}:{c2.generation}"
    assert rcv._get_str("consol:own:user:gen6b:e2") == f"{bid}:{c2.generation}"
    assert key not in rcv.ttls
    grown = [{"entry_id": "e2", "role": "user", "content": "B"},
             {"entry_id": "e3", "role": "user", "content": "C"}]
    assert (await rj.claim_new("user", "gen6b", grown, "g")).status == "conflict"


@pytest.mark.asyncio
async def test_generation_release_foreign_no_partial_no_ttl():
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    a = [{"entry_id": "e1", "role": "user", "content": "A"}]
    b = [{"entry_id": "e1", "role": "user", "content": "A-CHANGED"}]
    ca = await rj.claim_new("user", "gen6c", a, "plan-A")
    assert ca.status == "claimed"
    ka = build_plan_cache_key("user", "gen6c", a)
    rcv.strings["consol:own:user:gen6c:e1"] = "foreign-batch:" + "f" * 32
    rel = await rj.release("user", "gen6c", ["e1"], batch_id_of_plan_key(ka), ka, ca.generation)
    assert not rel.ok
    assert rcv._get_str("consol:own:user:gen6c:e1") == "foreign-batch:" + "f" * 32
    assert ka not in rcv.ttls
    assert f"consol:batch:user:gen6c:{batch_id_of_plan_key(ka)}" in rcv.strings


@pytest.mark.asyncio
async def test_generation_caller_nonce_blocks_stale_clear():
    redis = JournalFakeRedis()
    cj = CallerJournal(redis)
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    p1 = build_caller_pending("user", "gen6d", e12)
    assert (await cj.claim("user", "gen6d", p1))[0] == "claimed"
    ack = _good_ack(["e1", "e2"], gen="a" * 32)
    st, rec = await cj.mark_acknowledged("user", "gen6d", p1, ["e1", "e2"], dict(ack))
    assert st == "acked"
    assert (await cj.clear("user", "gen6d", ["e1", "e2"], p1["caller_nonce"]))[0] == "cleared"
    p2 = build_caller_pending("user", "gen6d", e12)
    assert p2["caller_nonce"] != p1["caller_nonce"]
    assert (await cj.claim("user", "gen6d", p2))[0] == "claimed"
    st2, _ = await cj.mark_acknowledged("user", "gen6d", p2, ["e1", "e2"], dict(ack))
    assert st2 == "acked"
    stale, _ = await cj.clear("user", "gen6d", ["e1", "e2"], p1["caller_nonce"])
    assert stale == "mismatch"
    assert (await cj.load("user", "gen6d")).status == "acknowledged"


@pytest.mark.asyncio
async def test_generation_stale_finish_preserves_new():
    e12 = [{"entry_id": "e1", "role": "user", "content": "A"},
           {"entry_id": "e2", "role": "user", "content": "B"}]
    t1 = _MemT1(e12)
    rcv = JournalFakeRedis()
    rj = ReceiverJournal(rcv)
    c1 = await rj.claim_new("user", "gen6e", e12, "plan-g1")
    assert c1.status == "claimed"
    pend1 = build_caller_pending("user", "gen6e", e12)
    cj = CallerJournal(t1.store.redis)
    assert (await cj.claim("user", "gen6e", pend1))[0] == "claimed"
    co = ConsolidationCoordinator(t1, lambda: None, lambda: None,
        get_caller_redis=lambda: t1.store.redis, get_receiver_redis=lambda: rcv)
    ack1 = _good_ack(["e1", "e2"], gen=c1.generation)
    out1 = await co._complete_validated_ack("user", "gen6e", pend1, ["e1", "e2"], dict(ack1), shipped=2)
    assert out1.get("status") == "ok"
    old_rec = dict(pend1, stage="acknowledged", trim_ids=["e1", "e2"],
                   ack=dict(ack1), receiver_generation=c1.generation)
    # Simulate TTL expiry + same-ID restore/reship as gen2
    key = build_plan_cache_key("user", "gen6e", e12)
    rcv.strings.pop(key, None); rcv.ttls.pop(key, None); rcv.expiries.pop(key, None)
    t1.ids = ["e1", "e2"]
    t1._by_id = {e["entry_id"]: dict(e) for e in e12}
    c2 = await rj.claim_new("user", "gen6e", e12, "plan-g2")
    assert c2.status == "claimed" and c2.generation != c1.generation
    pend2 = build_caller_pending("user", "gen6e", e12)
    assert (await cj.claim("user", "gen6e", pend2))[0] == "claimed"
    stale_out = await co._finish_acknowledged("user", "gen6e", old_rec)
    assert stale_out.get("status") == "failed"
    assert t1.ids == ["e1", "e2"] and t1.trims == [["e1", "e2"]]
    assert rcv._get_str("consol:own:user:gen6e:e1").endswith(c2.generation)
    assert (await cj.load("user", "gen6e")).record["caller_nonce"] == pend2["caller_nonce"]


@pytest.mark.asyncio
async def test_minor_undecodable_bytes_fail_closed():
    class BadRedis(JournalFakeRedis):
        async def get(self, key):
            if key == "consol:own:user:bad1:e1":
                return b"\xff\xfe-bad"
            if key == "consol:caller:user:bad2":
                return b"\xff\xfe-bad"
            return await super().get(key)
    rj = ReceiverJournal(BadRedis())
    assert (await rj.probe("user", "bad1", [{"entry_id": "e1"}])).status == "error"
    cj = CallerJournal(BadRedis())
    assert (await cj.load("user", "bad2")).status == "error"



# ------------------------------------------------- selection: >1000 tail


@pytest.mark.asyncio
async def test_selection_finds_older_behind_1000_summarized():
    from unit.memory.active_test import FakeRedis as T1FakeRedis
    from twin.shared.memory.active.store import ActiveStore
    import json as _json
    redis = T1FakeRedis()
    store = ActiveStore(redis)
    scope, sid = "user", "sel7"
    # 1000 recent summarized + 1 older unsummarized behind.
    recent = [f"e{i:04d}" for i in range(1, 1001)]
    await redis.zadd(f"active_index:{scope}:{sid}", {"e_old": 1.0, **{eid: float(1 + idx) for idx, eid in enumerate(recent, start=1)}})
    await redis.sadd(f"active_summarized:{scope}:{sid}", *recent)
    e_old = ActiveEntry(entry_id="e_old", scope=scope, scope_id=sid, role="user", content="old unsummarized")
    redis.docs[f"active:{scope}:{sid}:e_old"] = _json.dumps(e_old.model_dump(mode="json"), ensure_ascii=False)
    found = await store.list_unsummarized_entries(scope, sid, limit=200)
    assert [e.entry_id for e in found] == ["e_old"]
    assert len(found) <= 200
