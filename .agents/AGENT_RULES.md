# AGENT_RULES

Rules for coding agents working in this repository.

## Safety

- Do not reveal secrets. Never print full `.env` files, Discord tokens, API
  keys, or URLs containing credentials.
- **System Gateway is the host boundary.** Use `host_system` for new host
  interactions. Do not add direct host shell calls from containers.
- **System Gateway invariants (Plan A, staged, no deployment has occurred):**
  - Evernight owner DM/UI is the trusted issuer; March7 may request but cannot
    sign approvals. Only the configured owner (`EVERNIGHT_OWNER_USER_ID`)
    approves; non-owner requesters may trigger a request, never approve it.
    Missing owner or missing approval key fails closed (deny, no fallback).
  - Request-HMAC key `SYSTEM_GATEWAY_SHARED_SECRET` is DISTINCT from the
    private approval key FILE. The approval value lives only in a host-private
    FILE outside the repo, mounted read-only into Evernight alone
    (`/run/secrets/system_gateway_approval`; March7 never mounts/reads it).
    Equal credentials are rejected. Secrets stay out of shared `.env`/repo.
  - Shell requires an owner-signed structured grant bound to the canonical
    execution action (`canonical_approval_action`) + actor + expiry + durable
    single use (SQLite ledger + nonce). Bare approval IDs or unconsumed tokens
    are not enough; March7 passes the grant through, never mints.
  - Native generic shell is the only execution path; no structured actions.
    Raw shell is denied by default (`SYSTEM_GATEWAY_RAW_SHELL=false`); enabling
    it requires explicit config and still requires an owner-signed grant.
  - Never bypass the gateway by having Evernight or March7 execute host
    commands through Redis, Docker socket, or any other side channel.
  - Do not log the shared secret, approval key, or grant tokens.
- Keep the March7/Evernight A2A boundary intact. Evernight must not read or
  clear March7 T1 state by direct Redis key access.
- Keep the gateway/platform boundary intact. Discord, Zalo, and future chat
  surfaces are adapters only; core gateway handlers, agents, shared memory, and
  shared tools must not require `discord.Message`, Discord views, Discord
  logger names, or Discord-only prompt labels.

## Working Style

- **Use CodeGraph first** for structural questions (definitions, signatures, callers, callees, impact, and feature context).
- **NEVER re-read code files** with native read/grep for exploration once `codegraph` tools (`codegraph_explore`, `codegraph_node`) have already returned the symbol/source. Native read/search is only for confirming specifics CodeGraph didn't cover, or for inspecting config/docs/manifests.
- **Plan before coding.** For any non-trivial feature/bug, produce a plan (via the `implementation-planner` skill) and get user approval before editing code. The plan is part of the conversation — do not write it into an external app-data directory.
- Make small, task-scoped changes. Avoid broad refactors, speculative abstractions, unrelated formatting, and new tooling unless requested.
- If a change affects runtime flow, env vars, Docker, memory schema, or public
  behavior, update relevant docs in this folder and project READMEs.
- If touching `gateway/`, first read the current boundaries in
  [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md); the earlier
  `plans/gateway-platform-abstraction.md` is absent. Avoid adding dependencies from
  `twin/shared/*`, `twin/march7/*`, or gateway core files back into
  `gateway.adapters.discord`.

## Skills

Curated skills live globally at `~/.claude/skills/` (clone of [Flowerf19/agents-skills](https://github.com/Flowerf19/agents-skills)) — applies to all agents in every project, not just this repo.

- **Claude Code** auto-discovers them at `~/.claude/skills/`. Invoke with `/<skill-name>` or via the Skill tool.
- **Other agents** (Antigravity, Gemini, Cursor, generic LLMs): point them at `~/.claude/skills/<name>/SKILL.md`.

Available: `implementation-planner` (plan before code), `thoughtful-coder` (surgical changes), `debug-investigator` (root cause before any fix), `code-reviewer` (independent review before merge), `architecture-docs` (refresh `.agents/`), `create-readme` (root README from evidence).

Update upstream: `cd ~/.claude/skills && git pull`.

## Python Conventions

- Prefer clear async I/O with explicit timeouts for HTTP, Discord, Redis, and
  LLM calls.
- Handle cancellation and clean shutdown for background tasks.
- Use type hints on public functions and methods. Prefer `str | None` over
  `Optional[str]` for new Python 3.10+ code.
- Log enough context to debug, but never log secrets.
- Add retry/backoff only for concrete transient failure modes.

## Verification And PR Hygiene

- Choose focused tests from [TESTING_GUIDE.md](TESTING_GUIDE.md).
- For external I/O changes, verify timeout/error paths where practical.
- PR summaries should include context, change list, and exact verification.
- Commit messages should be short and specific, preferably
  `type(scope): message`.

## Verified Gotchas

- **`discord.Client.start()` does not retry login failures.** Its internal backoff
  only covers the websocket loop, so anything that fails before a socket exists
  (TLS during `static_login`, e.g. `certificate is not yet valid` when the
  container boots before NTP steps the clock) raises out of the task and the bot
  stays silently disconnected while the process keeps serving A2A. Both adapters
  run `start()` under `gateway.adapters.discord.connect.supervise_bot`. Do not
  call `bot.close()` to reset between attempts — it sets `loop = MISSING` and the
  next `start()` raises; `supervise_bot` closes only `bot.http` and resets
  `bot.http.connector`.
- **`/.well-known/agent.json` is not a health signal.** It answers 200 whenever
  the A2A server is up, including when Discord is disconnected, so it cannot
  detect the failure above. Use `/health` (503 when not connected) for
  containers/watchers and keep the agent card for liveness of the A2A API.

- **Gateway abstraction is partially refactored.** Production boot now uses
  `gateway.core.GatewayChatHandler` and `gateway.core.AgentRouter`; Discord
  policy/send behavior belongs in `gateway.adapters.discord`. Approval and
  some tool docs still contain Discord-specific assumptions, so do not copy
  those into new platforms.
- **`manage_user_profile` tool supports two modes:** single-section mode (when `section` and `bullets` are provided) and whole-file/all mode (when `section` is omitted and `sections` map is passed). Omitted sections in whole-file mode are deleted, and a shrink check prevents dropping >50% of profile bullets unless `allow_shrink=True` is explicitly passed. Both modes require `expected_profile_hash`.
- **Agent loop (Think/Act) and prerequisite routing:** Chat/tool orchestration is now `Think(Decide) -> Think(Refine) -> Act -> ... -> Think(Decide)` in `twin/shared/agent/agent_loop.py`. `Think` is the only LLM-facing stage; `Act` has no LLM access and loads no persona. `Think(Decide)` can answer the user directly or select the next tool. `Think(Refine)` remains mandatory before tool execution and may route/switch to a prerequisite tool (e.g. `get_profile`) when the selected tool is missing required arguments such as `expected_profile_hash`. Refine cancellation/error and loop-limit exits ask a final `Think(Decide)` pass with native tools disabled; do not reintroduce a Resolve stage.
- **No Zalo adapter implementation exists yet.** `gateway/adapters/factory.py`
  raises `NotImplementedError` for `zalo`. Treat Zalo as planned/held until the
  Zalo webhook/token settings and adapter contract are defined.
- **Run tests through the conda env interpreter.** Use
  `conda run -n discord_bot python -m pytest ...`, not
  `conda run -n discord_bot pytest ...`; the latter may resolve to the wrong
  pytest executable/interpreter.
- T3 storage is Markdown via `MarkdownProfileStore`, not YAML.
- `README.md` is the project README filename currently used at repo root.
- Root lint/format/type-check config is not currently established; do not add
  or run repo-wide formatters as part of unrelated work.
- Memory source lives under `twin/shared/memory/`. Do not recreate
  `twin/march7/memories/`, `twin/evernight/memories/`, or
  `twin/shared/memories/`.
- T2 timeline keeps memories user-centric for user scope, and also stores
  channel scope summaries with `user_id=channel_id`. Channel scope is
  consolidated via A2A with shipped entries; do not invent channel sentinel ids
  as `user_id` in `TimelineSummaryStore`.
- The legacy local Python consolidation paths (including `DiscussionConsolidator`, `consolidate_t2_memory`, `Consolidator`, `CleanupScheduler`, `TimelineStore`, `TimelineSearch`, and `T2Memory`) were removed. Current flow uses **A2A Consolidation**: `InactivityTrigger` on Evernight detects idle scopes → March7 sends task via `ConsolidationClient` (A2A) with shipped T1 entries → Evernight runs `ConsolidateMemoryTool` using those shipped entries (it does not re-read March7's T1) and writes to `TimelineSummaryStore` / `MarkdownProfileStore`.
- Each agent builds the shared stack (`ActiveMemory`, `MarkdownProfileStore`, `TimelineSummaryStore`) in its container. T2 RediSearch index must use Redis DB 0 (`TIMELINE_REDIS_DB=0`).
- **`twin/shared/tools/` consolidated 2026-05-26.** Core types live in `twin.shared.tools.registry`; individual tool classes live under `twin/shared.tools.modules.<domain>.<tool>` (domains: `execution`, `memory`, `profile`, `web`). The paths `twin.shared.tools.base_tool` / `tool_registry` / `tool_discovery` / `implementations.system.*` no longer exist — do not recreate them.
- **LLM/embedding endpoints chạy trên host phải dùng `host.docker.internal`, không phải `localhost`.** Container march7/evernight có `extra_hosts: host.docker.internal:host-gateway` trong compose; `localhost` trong `.env` sẽ trỏ vào chính container và fail với `Cannot connect to host localhost:<port>`. Áp dụng cho `OPENAI_API_URL`, `LM_STUDIO_API_URL`, `TOOL_LLM_ENDPOINT`. (Embeddings là Harrier ONNX local, không qua API.) Service nội-mạng Docker (redis, codebox, evernight) thì dùng service name.
- **Docker entry for March7 is `python -m gateway`** (`gateway/__main__.py`),
  not `python -m twin.march7`. When wiring new background tasks (triggers,
  workers, schedulers) for March7, add them to `gateway/__main__.py` so they
  run inside the container. `twin/march7/__main__.py` is a thinner local-dev
  entry — keep it in sync but treat the gateway entry as authoritative.
- Local Redis/T2 data may be disposable during development because T3 Markdown
  is the durable profile/core memory. Confirm before deleting production data.
- New host interactions must use the `host_system` tool through the native
  `system-gateway` service. Do not recreate privileged container-side host
  bridges.
- **System Gateway runs on the host, not in a container.** Containers reach it
  via `SYSTEM_GATEWAY_URL` (e.g. `http://host.docker.internal:8380`). Request
  signing uses `SYSTEM_GATEWAY_SHARED_SECRET`; the separate approval key is a
  host-private FILE mounted read-only into Evernight only. Source binds
  (`twin/`, `gateway/`, `models/`) are read-only; only the agent's OWN persona
  dir is a writable data-only overlay. Owner-absent boot yields `None` and
  denies approval/DM without crashing.
- **T2 recall is tool-only.** `SharedMemoryManager.get_context` returns only
  T1 + T3 context and no longer embeds or searches T2. The model obtains
  timeline context by calling `search_memory` (`user_id`, optional `channel_id`
  dual-scope, optional `query`/`days_back`, `limit`). No peer reads another
  agent's T1 by direct Redis access; consolidation ships entries over A2A and
  trims only on a fully successful acknowledgement with exact `entry_ids` and
  zero failed required writes — no trim on incomplete writes. T1 observe/trim
  are Redis-atomic (Lua) with bounded trigger dispatch; diary vectors validate
  dim/finite and fail closed (BM25-only invalid vectors are dropped, never
  bypass); diary merge is CAS with recompute on conflict; profile writes are
  CAS on `expected_profile_hash` with recompute-on-drift. Pending caller/receiver
  journals and plans have no expiry; never erase receiver witnesses on caller
  reset or a reported zero-store failure. Guarded trim validates nonce/snapshot
  and journals `trimmed` in one OwnT1 EVAL. Only matching-generation release
  follows; audited recovery requirements are in `ARCHITECTURE.md` §3.
- **T2 index dimension is 640 (Harrier q4 ONNX only).** T2 uses local Harrier q4
  with `VECTOR HNSW FLOAT32 COSINE DIM=640` (`EMBEDDING_VECTOR_SIZE`).
  Dimension change uses `scripts/migrate_t2_harrier.py` dry-run + verified
  backup + resumable re-embed from source HASH `summary` (dry-run by default
  without `--apply`/`--rollback`; parser has no `--dry-run` flag;
  `--apply` requires `--backup PATH`; `--recreate-index` only after validation
  with `DROPINDEX` without `DD`; `--rollback --backup PATH`). Only the
  `embedding` field is rewritten; never `DEL`/`UNLINK`/`FLUSHALL`/`DD`, never
  pad/truncate legacy 1024 vectors. Backup before explicit migration; production
  migration/redeploy is a manual owner action; no production data deletion.
  Adding `day`, `period_start`, and `period_end` is done via `FT.ALTER` without
  reindex.
- **New T2/T1 env vars control current behavior:** `HARRIER_MODEL_DIR`,
  `EMBEDDING_VECTOR_SIZE` (`640`), `EMBEDDING_QUERY_PREFIX`,
  `EMBEDDING_PASSAGE_PREFIX`, `T2_MIN_COSINE` (default `0.60`),
  `T2_MERGE_MIN_COSINE` (default `0.75`), `T2_MERGE_MAX_CHARS`
  (default `1500`), `T1_ARCHIVE_ENABLED` (default `true`),
  `T1_ARCHIVE_TTL_DAYS` (default `90`). Legacy Qwen-era `0.0`/`0.35` retrieve
  and `0.60` merge cause proven Harrier false positives; below-default explicit
  overrides warn at import but are respected.
- **`search_memory` is the only supported T2 retrieval tool.** It accepts
  `user_id`, optional `channel_id` (dual-scope), optional `query`, optional
  `days_back`, and `limit`. It does not accept `mode`, `topic`, `hours`, or
  `days`. BM25-only fused docs are still cosine-gated by `T2_MIN_COSINE`.
- **`TimelineSummaryStore` is a package (`twin/shared/memory/diary/`).** Import
  `TimelineSummaryStore` from `twin.shared.memory.diary`; import `_rrf_fuse`
  from `twin.shared.memory.diary.store` if needed for tests.
- **Embedding service must be wired into `TimelineSummaryStore`.** If
  `embedding_service` is `None`, same-day diary merge is disabled and the store
  is append-only. Verify probe/production wiring passes the service, not just
  `embedding_dim`.
