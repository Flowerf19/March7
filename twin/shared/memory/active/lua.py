"""Lua scripts for atomic T1 active-memory operations.

All observe/trim/clear linearization happens inside Redis via EVAL, so
concurrent clients/processes cannot interleave between index/entry/state
updates. Single-node Redis Stack is assumed (entry keys are constructed
inside Lua from scope/scope_id, matching the existing non-cluster DEL
multi-key usage).
"""
from __future__ import annotations

# Atomically: JSON.SET doc + ZADD index + HINCRBY tokens + max(last_entry_ts).
# KEYS[1]=entry_key KEYS[2]=index_key KEYS[3]=state_key
# ARGV[1]=entry_id ARGV[2]=entry_json ARGV[3]=ts ARGV[4]=tokens
T1_OBSERVE_V1 = """-- T1_OBSERVE_V1
redis.call('JSON.SET', KEYS[1], '$', ARGV[2])
redis.call('ZADD', KEYS[2], tonumber(ARGV[3]), ARGV[1])
local new_total = redis.call('HINCRBY', KEYS[3], 'unsummarized_tokens', tonumber(ARGV[4]))
local cur = redis.call('HGET', KEYS[3], 'last_entry_ts')
local ts_n = tonumber(ARGV[3])
if cur == false then
  redis.call('HSET', KEYS[3], 'last_entry_ts', ARGV[3])
else
  local cur_n = tonumber(cur)
  if cur_n == nil or ts_n > cur_n then
    redis.call('HSET', KEYS[3], 'last_entry_ts', ARGV[3])
  end
end
return new_total
"""

# Atomically: HINCRBY tokens + max(last_entry_ts). Used by increment_tokens.
# KEYS[1]=state_key ARGV[1]=tokens ARGV[2]=last_entry_ts or ''
T1_INCR_TOKENS_V1 = """-- T1_INCR_TOKENS_V1
local new_total = redis.call('HINCRBY', KEYS[1], 'unsummarized_tokens', tonumber(ARGV[1]))
if ARGV[2] ~= '' then
  local cur = redis.call('HGET', KEYS[1], 'last_entry_ts')
  local ts_n = tonumber(ARGV[2])
  if cur == false then
    redis.call('HSET', KEYS[1], 'last_entry_ts', ARGV[2])
  else
    local cur_n = tonumber(cur)
    if cur_n == nil or (ts_n ~= nil and ts_n > cur_n) then
      redis.call('HSET', KEYS[1], 'last_entry_ts', ARGV[2])
    end
  end
end
return new_total
"""

# Shared T1 trim core: keep-tail + token decrement + index/marker updates.
# Wrapped by ordinary T1_TRIM_V1 and guarded T1_TRIM_GUARDED_V1 so the
# ~100-line body exists once. Ordinary reply contract unchanged.
_T1_TRIM_CORE_LUA = """
local function t1_trim_core(index_key, state_key, summ_key, scope, scope_id, keep_recent, ids)
  local keep_set = {}
  if keep_recent > 0 then
    local keep_ids = redis.call('ZRANGE', index_key, -keep_recent, -1)
    for i=1,#keep_ids do
      keep_set[keep_ids[i]] = true
    end
  end
  local deleted = {}
  local subtracted = 0
  for i=1,#ids do
    local eid = ids[i]
    local entry_key = 'active:' .. scope .. ':' .. scope_id .. ':' .. eid
    local is_keep = keep_set[eid] == true
    local ok, doc = pcall(function() return redis.call('JSON.GET', entry_key) end)
    local doc_exists = (ok and doc ~= false and doc ~= nil)
    local in_index = redis.call('ZSCORE', index_key, eid)
    local has_index = (in_index ~= false)
    if not doc_exists then
      if has_index then
        redis.call('ZREM', index_key, eid)
      end
      redis.call('SREM', summ_key, eid)
      if (not is_keep) and has_index then
        deleted[#deleted+1] = eid
      end
    else
      local tok = 0
      local pok, obj = pcall(cjson.decode, doc)
      if pok and type(obj) == 'table' and obj['tokens'] ~= nil then
        tok = tonumber(obj['tokens']) or 0
        if tok < 0 then tok = 0 end
      end
      if is_keep then
        if redis.call('SISMEMBER', summ_key, eid) == 0 then
          if has_index then
            if tok > 0 then
              redis.call('HINCRBY', state_key, 'unsummarized_tokens', -tok)
              subtracted = subtracted + tok
            end
            redis.call('SADD', summ_key, eid)
          else
            redis.call('DEL', entry_key)
            redis.call('SREM', summ_key, eid)
          end
        end
      else
        if redis.call('SISMEMBER', summ_key, eid) == 0 then
          if tok > 0 then
            redis.call('HINCRBY', state_key, 'unsummarized_tokens', -tok)
            subtracted = subtracted + tok
          end
        end
        redis.call('DEL', entry_key)
        if has_index then
          redis.call('ZREM', index_key, eid)
        end
        redis.call('SREM', summ_key, eid)
        deleted[#deleted+1] = eid
      end
    end
  end
  local cur = redis.call('HGET', state_key, 'unsummarized_tokens')
  if cur ~= false and cur ~= nil then
    local cur_n = tonumber(cur)
    if cur_n ~= nil and cur_n < 0 then
      redis.call('HSET', state_key, 'unsummarized_tokens', '0')
    end
  end
  local new_total = 0
  local nt = redis.call('HGET', state_key, 'unsummarized_tokens')
  if nt ~= false and nt ~= nil then
    new_total = tonumber(nt) or 0
  end
  return subtracted, new_total, deleted
end
"""

_DEEP_EQUAL_LUA = """
local function deep_equal(a, b)
  if type(a) ~= type(b) then return false end
  if type(a) ~= 'table' then return a == b end
  for k,v in pairs(a) do
    if b[k] == nil then return false end
    if not deep_equal(v, b[k]) then return false end
  end
  for k,v in pairs(b) do
    if a[k] == nil then return false end
  end
  return true
end
local function rtype(k)
  local ok, t = pcall(function() return redis.call('TYPE', k) end)
  if not ok then return nil end
  if type(t) == 'table' then
    if t['ok'] ~= nil then return t['ok'] end
    return nil
  end
  return t
end
"""

# Atomically trim exact summarized IDs: keep most-recent N (by index score),
# delete the rest, subtract tokens once (via JSON.GET at linearization time),
# and mark retained-but-summarized IDs in a SET for idempotent retries.
# Never touches last_entry_ts (no regress). Clamps counter >= 0.
# KEYS[1]=index_key KEYS[2]=state_key KEYS[3]=summarized_key
# ARGV[1]=scope ARGV[2]=scope_id ARGV[3]=keep_recent ARGV[4]=n ARGV[5..]=ids
T1_TRIM_V1 = (
    "-- T1_TRIM_V1\n"
    + _T1_TRIM_CORE_LUA
    + """\nlocal index_key = KEYS[1]
local state_key = KEYS[2]
local summ_key = KEYS[3]
local scope = ARGV[1]
local scope_id = ARGV[2]
local keep_recent = tonumber(ARGV[3]) or 0
local n = tonumber(ARGV[4]) or 0
local ids = {}
for i=1,n do ids[#ids+1] = ARGV[4+i] end
local subtracted, new_total, deleted = t1_trim_core(index_key, state_key, summ_key, scope, scope_id, keep_recent, ids)
local ret = {subtracted, new_total}
for i=1,#deleted do ret[#ret+1] = deleted[i] end
return ret
"""
)

# Guarded consolidation trim: single OwnT1 EVAL verifies CURRENT caller
# acknowledged identity (ordered IDs + SHA64 + plan + nonce + gen + trim + ACK),
# canonical counter 0<=unsummarized_tokens<=9223372036854775807 (no sign/
# leading zeros, len<=19, lexicographic max, no float compare), and exact T1
# snapshots BEFORE any mutation, then runs the shared trim core and sets caller
# stage trimmed atomically. Same-identity already-trimmed returns idempotent
# success WITHOUT touching T1. Any mismatch/missing/corrupt/wrongtype fails
# with NO mutation (preflight before first write; Lua has no rollback on
# runtime errors, so all fallible checks precede DEL/HINCRBY/SADD/SET.
# Residual: server crash/OOM mid-script may leave partial writes; single-node
# Redis assumed, no cross-DB txn).
# KEYS[1]=index KEYS[2]=state KEYS[3]=summ KEYS[4]=caller
# ARGV[1]=scope [2]=sid [3]=keep [4]=n [5..4+n]=ids
# [5+n]=fp_json [6+n]=plan [7+n]=nonce [8+n]=gen [9+n]=trim_json
# [10+n]=ack_json [11+n..10+2n]=snapshot_jsons (ordered as ids, transient)
T1_TRIM_GUARDED_V1 = (
    "-- T1_TRIM_GUARDED_V1\n"
    + _T1_TRIM_CORE_LUA
    + _DEEP_EQUAL_LUA
    + """\nlocal index_key = KEYS[1]
local state_key = KEYS[2]
local summ_key = KEYS[3]
local caller_key = KEYS[4]
local scope = ARGV[1]
local scope_id = ARGV[2]
local keep_recent = tonumber(ARGV[3])
local n = tonumber(ARGV[4])
if keep_recent == nil or n == nil or n <= 0 or n > 200 then
  return {0, 'bad_args', ''}
end
if keep_recent < 0 then keep_recent = 0 end
local ids = {}
for i=1,n do
  local v = ARGV[4+i]
  if v == nil or v == false or v == '' then
    return {0, 'bad_args', ''}
  end
  ids[#ids+1] = v
end
local fp_json = ARGV[5+n]
local exp_plan = ARGV[6+n]
local exp_nonce = ARGV[7+n]
local exp_gen = ARGV[8+n]
local trim_json = ARGV[9+n]
local ack_json = ARGV[10+n]
if fp_json == nil or exp_plan == nil or exp_nonce == nil or exp_gen == nil or trim_json == nil or ack_json == nil then
  return {0, 'bad_args', ''}
end
local ct = rtype(caller_key)
if ct ~= 'string' and ct ~= 'none' then
  return {0, 'wrongtype', 'caller'}
end
local it = rtype(index_key)
if it ~= 'zset' and it ~= 'none' then
  return {0, 'wrongtype', 'index'}
end
local st = rtype(state_key)
if st ~= 'hash' and st ~= 'none' then
  return {0, 'wrongtype', 'state'}
end
local sut = rtype(summ_key)
if sut ~= 'set' and sut ~= 'none' then
  return {0, 'wrongtype', 'summarized'}
end
local cur_raw = redis.call('GET', caller_key)
if cur_raw == false or cur_raw == nil then
  return {0, 'missing', 'caller'}
end
local ok, rec = pcall(cjson.decode, cur_raw)
if not ok or type(rec) ~= 'table' then
  return {0, 'corrupt', 'caller'}
end
local ok1, exp_fp = pcall(cjson.decode, fp_json)
local ok2, exp_trim = pcall(cjson.decode, trim_json)
local ok3, exp_ack = pcall(cjson.decode, ack_json)
if not ok1 or type(exp_fp) ~= 'table' then return {0, 'bad_args', 'fp'} end
if not ok2 or type(exp_trim) ~= 'table' then return {0, 'bad_args', 'trim'} end
if not ok3 or type(exp_ack) ~= 'table' then return {0, 'bad_args', 'ack'} end
local cur_ids = rec['entry_ids']
local cur_fp = rec['fingerprints']
local cur_trim = rec['trim_ids']
local cur_ack = rec['ack']
local cur_stage = rec['stage']
if type(cur_ids) ~= 'table' or type(cur_fp) ~= 'table' or type(cur_trim) ~= 'table' or type(cur_ack) ~= 'table' then
  return {0, 'corrupt', 'caller_shape'}
end
if type(rec['plan_key']) ~= 'string' or type(rec['caller_nonce']) ~= 'string' then
  return {0, 'corrupt', 'caller_shape'}
end
if #cur_ids ~= n then return {0, 'mismatch', 'ids_len'} end
for i=1,n do if cur_ids[i] ~= ids[i] then return {0, 'mismatch', 'ids'} end end
if #cur_fp ~= n then return {0, 'mismatch', 'fp_len'} end
for i=1,n do if cur_fp[i] ~= exp_fp[i] then return {0, 'mismatch', 'fp'} end end
if rec['plan_key'] ~= exp_plan then return {0, 'mismatch', 'plan'} end
if rec['caller_nonce'] ~= exp_nonce then return {0, 'mismatch', 'nonce'} end
local cur_gen = rec['receiver_generation']
if cur_gen == nil then cur_gen = '' end
if type(cur_gen) ~= 'string' then return {0, 'corrupt', 'gen'} end
if cur_gen ~= exp_gen then return {0, 'mismatch', 'gen'} end
if #cur_trim ~= #exp_trim then return {0, 'mismatch', 'trim_len'} end
local trim_set = {}
for i=1,#cur_trim do trim_set[cur_trim[i]] = true end
for i=1,#exp_trim do if trim_set[exp_trim[i]] ~= true then return {0, 'mismatch', 'trim'} end end
if not deep_equal(cur_ack, exp_ack) then return {0, 'mismatch', 'ack'} end
if cur_stage == 'trimmed' then
  return {1, 'already'}
end
if cur_stage ~= 'acknowledged' then
  return {0, 'not_acked', ''}
end
rec['stage'] = 'trimmed'
local ok_enc, nxt = pcall(cjson.encode, rec)
if not ok_enc or type(nxt) ~= 'string' then
  return {0, 'corrupt', 'encode'}
end
if st == 'hash' then
  local cur_tok = redis.call('HGET', state_key, 'unsummarized_tokens')
  if cur_tok ~= false and cur_tok ~= nil then
    local s = tostring(cur_tok)
    local ok_c = false
    if s == '0' then
      ok_c = true
    elseif string.match(s, '^[1-9][0-9]*$') ~= nil then
      if #s < 19 then
        ok_c = true
      elseif #s == 19 and s <= '9223372036854775807' then
        ok_c = true
      end
    end
    if not ok_c then
      return {0, 'wrongtype', 'counter'}
    end
  end
end
for i=1,n do
  local eid = ids[i]
  local snap_json = ARGV[10+n+i]
  if snap_json == nil or snap_json == false then
    return {0, 'bad_args', 'snap'}
  end
  local ok_s, exp_doc = pcall(cjson.decode, snap_json)
  if not ok_s or type(exp_doc) ~= 'table' then
    return {0, 'bad_args', 'snap_json'}
  end
  if exp_doc['tokens'] ~= nil then
    local num = tonumber(exp_doc['tokens'])
    if num == nil or math.floor(num) ~= num then
      return {0, 'corrupt', 'tokens'}
    end
  end
  local entry_key = 'active:' .. scope .. ':' .. scope_id .. ':' .. eid
  local ok_g, cur_doc_raw = pcall(function() return redis.call('JSON.GET', entry_key) end)
  if not ok_g then
    return {0, 'wrongtype', eid}
  end
  if cur_doc_raw == false or cur_doc_raw == nil then
    return {0, 't1_missing', eid}
  end
  local ok_c, cur_doc = pcall(cjson.decode, cur_doc_raw)
  if not ok_c or type(cur_doc) ~= 'table' then
    return {0, 'corrupt', eid}
  end
  if not deep_equal(cur_doc, exp_doc) then
    return {0, 't1_changed', eid}
  end
  local sc = redis.call('ZSCORE', index_key, eid)
  if sc == false or sc == nil then
    return {0, 't1_missing', eid}
  end
end
local subtracted, new_total, deleted = t1_trim_core(index_key, state_key, summ_key, scope, scope_id, keep_recent, ids)
redis.call('SET', caller_key, nxt)
local ret = {1, 'trimmed', subtracted, new_total}
for i=1,#deleted do ret[#ret+1] = deleted[i] end
return ret
"""
)

# Atomically clear ALL indexed entries (no 10k cap) + orphan docs via SCAN,
# then DEL index/state/markers. Concurrent observes linearize before (cleared)
# or after (survive); never orphaned.
# KEYS[1]=index_key KEYS[2]=state_key KEYS[3]=summarized_key
# ARGV[1]=scope ARGV[2]=scope_id
T1_CLEAR_V1 = """-- T1_CLEAR_V1
local index_key = KEYS[1]
local state_key = KEYS[2]
local summ_key = KEYS[3]
local prefix = 'active:' .. ARGV[1] .. ':' .. ARGV[2] .. ':'
local ids = redis.call('ZRANGE', index_key, 0, -1)
for i=1,#ids do
  redis.call('UNLINK', prefix .. ids[i])
end
local cursor = '0'
repeat
  local res = redis.call('SCAN', cursor, 'MATCH', prefix .. '*', 'COUNT', 1000)
  cursor = res[1]
  local keys = res[2]
  for i=1,#keys do
    redis.call('UNLINK', keys[i])
  end
until cursor == '0'
redis.call('DEL', index_key)
redis.call('DEL', state_key)
redis.call('DEL', summ_key)
return #ids
"""
