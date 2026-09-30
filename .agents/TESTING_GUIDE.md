# TESTING_GUIDE

March7 dùng `pytest`. Test layout phản ánh ranh giới runtime: memory stack, gateway,
transport, và external services. Dùng file này để chọn test focused; dùng CodeGraph
để tìm symbol cụ thể bên trong mỗi test.

## Test Layout

- `tests/unit/` — logic cô lập, không cần service ngoài:
  - `tests/unit/memory/` — memory stack: `active_test.py`,
    `manager_consolidation_test.py`, `consolidate_tool_test.py`,
    `consolidation_failclosed_test.py`, `consolidation_retry_test.py`,
    `profile_test.py`, `profile_security_test.py`, `test_store_schema.py`,
    `test_store_idempotency.py`, `test_merge_cas.py`, `test_diary_recency.py`,
    `tools_test.py`, `test_rrf.py`, `test_embedding_prefix.py`.
  - `tests/unit/llm/embedding/` — Harrier + T2 ops: `test_harrier_embedding_service.py`,
    `test_pull_harrier_model.py`, `test_migrate_t2_harrier.py` (fake Redis/embedder),
    `test_t2_calibration.py` (defaults `0.60`/`0.75` + recorded zones; live model test
    skip khi thiếu `.installed.json` marker).
  - `tests/unit/` (gốc) — `inactivity_trigger`, `a2a_client`, `a2a_task_bounds`,
    `owner_approval`, `source_ro_boot_isolation`, `system_gateway_*`,
    `http_transport`, `tool_bootstrap`, `march7_handle_chat_scope`,
    `discord_send_response`.
- `tests/gateway/` — gateway core/adapter + models (`test_gateway`,
  `test_core_handler`, `test_models`).
- `tests/services/external/` — external I/O clients such as `codebox_client`;
  networked web search now goes through the Tavily MCP-backed tool wrapper.
- `tests/services/tools/` — tool wrappers (`code_interpreter_tool`,
  `tavily_search_tool`, `host_system_tool`).
- `tests/integration/test_diary_merge_redis.py`, `test_t1_active_redis.py`,
  `test_consolidation_journal_redis.py` — opt-in Redis Stack thật: diary merge,
  T1 observe/trim/clear Lua, journal claim/adopt/release/CAS. Không chạy mặc định
  khi thiếu service; chỉ dùng disposable Redis, không production.
- `tests/e2e/REPORT.md` — báo cáo e2e lịch sử, không phải suite runnable; không
  chạy Discord/host-exec/external network test từ đây.
- `tests/manual/` — hiện chỉ còn scaffolding (`__init__.py`).

## Common Commands

Use the conda env interpreter when that env is installed:

```bash
conda run -n discord_bot python -m pytest tests/unit -q
conda run -n discord_bot python -m pytest tests/unit/memory -q
conda run -n discord_bot python -m pytest tests/gateway -q
conda run -n discord_bot python -m pytest tests/services -q
conda run -n discord_bot python -m pytest tests -q
```

Host hiện tại KHÔNG có env `discord_bot` (chỉ `base`); các lệnh trên chỉ valid
khi env đã cài. Phân biệt: installed-container verification (deps đầy đủ trong
image) vs host deps (thiếu `discord.py`/Redis/model sẽ fail collection hoặc skip).
Skipped tests (ví dụ thiếu Harrier marker) không phải proof. Tránh
`conda run -n discord_bot pytest ...`; nó có thể resolve sai pytest executable.
Nếu không dùng conda, `python -m pytest` vẫn ưu tiên hơn bare `pytest`.

Session-specific: disposable test runner ở `/tmp/march7-fixes-*/run-tests.sh`
(docker read-only, network none) chỉ là artifact của session tích hợp này,
không phải repo supported command; không document nó như workflow chính thức.

`pytest.ini` discovers both legacy `*_test.py` files and gateway-style
`test_*.py` files.

Full suite với workaround phoenix/strawberry import conflict và bỏ qua các test
cần `discord.py`/gateway CLI:

```bash
conda run -n discord_bot python -m pytest tests -q \
  --ignore=tests/unit/discord_send_response_test.py \
  --ignore=tests/unit/evernight_discord_adapter_test.py \
  --ignore=tests/services/system_gateway_cli_test.py \
  -p no:phoenix
```

## Service Dependencies

- `tests/unit/*` chạy không cần service ngoài (mock Redis/LLM), trừ live Harrier
  subset trong `test_t2_calibration.py` (skip khi thiếu model marker).
- `tests/services/external/*_integration_test.py` gọi network thật (ví dụ
  codebox) — bỏ qua nếu không có endpoint.
- `tests/integration/test_diary_merge_redis.py` cần Redis Stack thật; ưu tiên
  provision qua Docker (xem [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)).
- Cần `discord.py` cài đặt để collect `tests/unit/discord_send_response_test.py`
  và các test import `gateway.adapters.discord`.
- Không chạy actual Discord/host-exec/external network test để verify docs.

## Selection Guide

- T1 active memory / archive / trim → `tests/unit/memory/active_test.py` +
  `manager_consolidation_test.py`
- Cross-DB consolidation / A2A shipped entries →
  `tests/unit/memory/consolidate_tool_test.py` + `manager_consolidation_test.py`
- `InactivityTrigger` → `tests/unit/inactivity_trigger_test.py`
- T2 timeline/vector → `tests/unit/memory/test_store_schema.py`,
  `tools_test.py`, `test_rrf.py`, `test_embedding_prefix.py`,
  `test_store_idempotency.py`, `test_merge_cas.py`, `test_diary_recency.py`
- T2 migration/calibration → `tests/unit/llm/embedding/test_migrate_t2_harrier.py`,
  `test_t2_calibration.py`, `test_pull_harrier_model.py`
- Owner approval / source-RO boot → `tests/unit/owner_approval_test.py`,
  `tests/unit/source_ro_boot_isolation_test.py`
- A2A bounds/completion → `tests/unit/a2a_task_bounds_test.py`,
  `a2a_failclosed_stream_test.py`, `a2a_stream_completion_test.py`.
  `COMPLETED` metadata không đủ: client phải nhận marker/count hợp lệ và EOF
  sạch. March7/Evernight phải upgrade cùng nhau; không fallback server cũ.
- T3 profile → `tests/unit/memory/profile_test.py` + `tests/unit/manage_profile_tool_test.py`
- March7 chat scope / A2A → `tests/unit/march7_handle_chat_scope_test.py`, `a2a_client_test.py`
- Gateway / Discord → `tests/gateway -q`,
  `tests/unit/discord_send_response_test.py`
- External services (codebox) → `tests/services/external/*`
- Tavily web search MCP wrapper → `tests/services/tools/tavily_search_tool_test.py`
- Tool wrappers → `tests/services/tools/*`

## Last Verified

- Staged isolated-container run (source-only read-only, no external network):
  `python -m pytest tests services/system_gateway/tests --ignore=tests/services/external/codebox_client_integration_test.py`
  → **1121 passed, 30 skipped, 3 warnings**. Disposable pinned Redis chạy riêng
  journal/T1/diary → **24 passed**, container riêng đã remove; production không
  bị dùng. Host vẫn thiếu env `discord_bot`.
- Muse đã approve SSE/recall và targeted journal closeout/source/docs scopes;
  coordinated application còn pending. Sol quota errors, không có verdict.
  Không coi subset approval là blanket verdict cho 37 findings, không coi
  skips/simulated macOS/Windows là deployment proof.
- 2026-07-03: `conda run -n discord_bot python -m pytest tests -q --ignore=tests/unit/discord_send_response_test.py --ignore=tests/unit/evernight_discord_adapter_test.py --ignore=tests/services/system_gateway_cli_test.py -p no:phoenix`
  → 479 passed, 6 skipped.
- 2026-07-03: Unit memory subset → 399 passed (T2 diary/retrieval + T1
  archive/trim + cross-DB consolidation).
- 2026-07-03: Live probes: T2 recall end-to-end PASS (KNN_RESULT cos 0.499),
  cross-DB consolidation "Cách B" E2E PASS, FT.ALTER thêm `day`/`period_start`/
  `period_end` vào index v2 thành công không cần reindex.
- 2026-06-12: `conda run -n discord_bot python -m pytest tests/unit/manage_profile_tool_test.py -q`
  → passed.

## Historical snapshots (pre-T2-diary; kept for archaeology)

- 2026-06-07: `conda run -n discord_bot python -m pytest tests/unit/approval_gate_test.py tests/gateway tests/unit/evernight_discord_adapter_test.py tests/unit/march7_handle_chat_scope_test.py tests/unit/evernight_agent_test.py tests/unit/discord_send_response_test.py tests/unit/tool_bootstrap_test.py tests/unit/memory/manager_test.py -q`
  → 55 passed.
- 2026-06-07: Docker rebuild/restart via `docker compose -f docker/docker-compose.yml up -d --build`;
  `march7`, `evernight`, the then-current host executor, `codebox`, and `redis` healthy. Evernight
  A2A chat smoke and Chrome snapshot of `http://localhost:8001/.well-known/agent.json`
  passed; Chrome console only showed favicon 404.
- 2026-06-07: `conda run -n discord_bot python -m pytest tests/gateway/test_gateway.py tests/gateway/test_models.py tests/gateway/test_core_handler.py tests/unit/march7_handle_chat_scope_test.py tests/unit/evernight_agent_test.py tests/unit/discord_send_response_test.py tests/unit/memory/manager_test.py -q`
  → 44 passed.
- 2026-06-07: `conda run -n discord_bot python -m pytest tests/gateway -q`
  → 22 passed.
- 2026-05-28: `pytest` → 221 passed, 13 skipped; `docker compose ps` healthy cho
  march7/evernight/redis/codebox plus the then-current host executor.
- Lưu ý (2026-06-02): chạy lại đầy đủ cần `discord.py` + Redis trong môi trường;
  thiếu deps sẽ fail ở collection (`ModuleNotFoundError: discord`).
