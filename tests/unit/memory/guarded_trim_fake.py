"""Guarded-trim EVAL simulator shared by T1 fakes.

Extracted from FakeRedis._guarded_trim to keep the test double under the
300-line cap. Synchronous dict ops linearize like single-threaded Redis Lua.
"""
from __future__ import annotations

import json


def guarded_trim(state, keys: list[str], args: list[str]):
    # Mirror T1_TRIM_GUARDED_V1: validate caller + snapshots, then trim core.
    # No awaits: linearizes like Redis Lua. No mutation on any failure.
    try:
        index_key, state_key, summ_key, caller_key = keys[0], keys[1], keys[2], keys[3]
        scope, scope_id = args[0], args[1]
        keep_recent, n = int(args[2]), int(args[3])
    except Exception:
        return [0, "bad_args", ""]
    if n <= 0 or n > 200:
        return [0, "bad_args", ""]
    if keep_recent < 0:
        keep_recent = 0
    if len(args) < 10 + 2 * n:
        return [0, "bad_args", ""]
    ids = args[4:4 + n]
    if any(not i for i in ids):
        return [0, "bad_args", ""]
    fp_json, exp_plan, exp_nonce = args[4 + n], args[5 + n], args[6 + n]
    exp_gen, trim_json, ack_json = args[7 + n], args[8 + n], args[9 + n]
    snaps = args[10 + n:10 + 2 * n]
    # Wrongtype preflight (keys must live in expected namespaces only).
    if caller_key in state.docs or caller_key in state.zsets or caller_key in state.hashes or caller_key in state.sets or caller_key in state.lists:
        return [0, "wrongtype", "caller"]
    if index_key in state.strings or index_key in state.docs or index_key in state.hashes or index_key in state.sets or index_key in state.lists:
        return [0, "wrongtype", "index"]
    if state_key in state.strings or state_key in state.docs or state_key in state.zsets or state_key in state.sets or state_key in state.lists:
        return [0, "wrongtype", "state"]
    if summ_key in state.strings or summ_key in state.docs or summ_key in state.zsets or summ_key in state.hashes or summ_key in state.lists:
        return [0, "wrongtype", "summarized"]
    if index_key in state.zsets and not isinstance(state.zsets[index_key], dict):
        return [0, "wrongtype", "index"]
    if state_key in state.hashes and not isinstance(state.hashes[state_key], dict):
        return [0, "wrongtype", "state"]
    if summ_key in state.sets and not isinstance(state.sets[summ_key], set):
        return [0, "wrongtype", "summarized"]
    cur_raw = state._get_str(caller_key)
    if cur_raw is None:
        return [0, "missing", "caller"]
    try:
        rec = json.loads(cur_raw)
        exp_fp = json.loads(fp_json)
        exp_trim = json.loads(trim_json)
        exp_ack = json.loads(ack_json)
    except Exception:
        return [0, "corrupt", "caller"]
    if not isinstance(rec, dict) or not isinstance(exp_fp, list) or not isinstance(exp_trim, list) or not isinstance(exp_ack, dict):
        return [0, "corrupt", "caller_shape"]
    cur_ids, cur_fp, cur_trim, cur_ack = rec.get("entry_ids"), rec.get("fingerprints"), rec.get("trim_ids"), rec.get("ack")
    if not isinstance(cur_ids, list) or not isinstance(cur_fp, list) or not isinstance(cur_trim, list) or not isinstance(cur_ack, dict):
        return [0, "corrupt", "caller_shape"]
    if not isinstance(rec.get("plan_key"), str) or not isinstance(rec.get("caller_nonce"), str):
        return [0, "corrupt", "caller_shape"]
    if cur_ids != ids:
        return [0, "mismatch", "ids"]
    if cur_fp != exp_fp:
        return [0, "mismatch", "fp"]
    if rec.get("plan_key") != exp_plan:
        return [0, "mismatch", "plan"]
    if rec.get("caller_nonce") != exp_nonce:
        return [0, "mismatch", "nonce"]
    cur_gen = rec.get("receiver_generation", "")
    if cur_gen is None:
        cur_gen = ""
    if not isinstance(cur_gen, str):
        return [0, "corrupt", "gen"]
    if cur_gen != exp_gen:
        return [0, "mismatch", "gen"]
    if len(cur_trim) != len(exp_trim) or set(cur_trim) != set(exp_trim):
        return [0, "mismatch", "trim"]
    if cur_ack != exp_ack:
        return [0, "mismatch", "ack"]
    if rec.get("stage") == "trimmed":
        return [1, "already"]
    if rec.get("stage") != "acknowledged":
        return [0, "not_acked", ""]
    try:
        nxt = json.dumps(dict(rec, stage="trimmed"), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:
        return [0, "corrupt", "encode"]
    if state_key in state.hashes:
        cur_tok = state.hashes[state_key].get("unsummarized_tokens")
        if cur_tok is not None:
            s = str(cur_tok)
            # Canonical 0<=counter<=MAXINT64 (no sign/leading zeros, len<=19,
            # lexicographic max; mirrors Lua, no float compare).
            ok_c = False
            if s == "0":
                ok_c = True
            elif 1 <= len(s) <= 19 and s[0] in "123456789" and all(c in "0123456789" for c in s):
                if len(s) < 19 or s <= "9223372036854775807":
                    ok_c = True
            if not ok_c:
                return [0, "wrongtype", "counter"]
    for eid, snap in zip(ids, snaps):
        try:
            exp_doc = json.loads(snap)
        except Exception:
            return [0, "bad_args", "snap_json"]
        if not isinstance(exp_doc, dict):
            return [0, "bad_args", "snap_json"]
        if "tokens" in exp_doc and exp_doc["tokens"] is not None:
            tv = exp_doc["tokens"]
            if isinstance(tv, bool):
                return [0, "corrupt", "tokens"]
            if isinstance(tv, int):
                pass
            elif isinstance(tv, float):
                if not tv.is_integer():
                    return [0, "corrupt", "tokens"]
            elif isinstance(tv, str):
                if not tv.lstrip("-").isdigit():
                    return [0, "corrupt", "tokens"]
            else:
                return [0, "corrupt", "tokens"]
        entry_key = f"active:{scope}:{scope_id}:{eid}"
        if entry_key in state.strings or entry_key in state.zsets or entry_key in state.hashes or entry_key in state.sets or entry_key in state.lists:
            return [0, "wrongtype", eid]
        cur_doc_raw = state.docs.get(entry_key)
        if cur_doc_raw is None:
            return [0, "t1_missing", eid]
        try:
            cur_doc = json.loads(cur_doc_raw)
        except Exception:
            return [0, "corrupt", eid]
        if cur_doc != exp_doc:
            return [0, "t1_changed", eid]
        if eid not in state.zsets.get(index_key, {}):
            return [0, "t1_missing", eid]
    # Validated: run shared trim core (same semantics as ordinary).
    bucket = state.zsets.get(index_key, {})
    if index_key not in state.zsets:
        state.zsets[index_key] = bucket
    ordered = sorted(bucket.items(), key=lambda kv: kv[1])
    keep_set = set(dict(ordered[-keep_recent:]).keys()) if keep_recent > 0 else set()
    marked = state.sets.setdefault(summ_key, set())
    h = state.hashes.setdefault(state_key, {})
    deleted: list[str] = []
    subtracted = 0
    for eid in ids:
        entry_key = f"active:{scope}:{scope_id}:{eid}"
        is_keep = eid in keep_set
        doc = state.docs.get(entry_key)
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
        except (ValueError, AttributeError, TypeError):
            tok = 0
        if tok < 0:
            tok = 0
        if is_keep:
            if eid not in marked:
                if has_index:
                    if tok > 0:
                        h["unsummarized_tokens"] = str(int(h.get("unsummarized_tokens", 0) or 0) - tok)
                        subtracted += tok
                    marked.add(eid)
                else:
                    del state.docs[entry_key]
                    marked.discard(eid)
        else:
            if eid not in marked and tok > 0:
                h["unsummarized_tokens"] = str(int(h.get("unsummarized_tokens", 0) or 0) - tok)
                subtracted += tok
            if entry_key in state.docs:
                del state.docs[entry_key]
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
    state.strings[caller_key] = nxt
    return [1, "trimmed", subtracted, cur_n, *deleted]
