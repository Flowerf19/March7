# Nợ kỹ thuật (DEBT)

Ghi nhanh cuối ngày 2026-09-29. Mỗi mục: mức độ, bằng chứng, hướng fix. Không xóa mục đã xong — chuyển `status: done` + ghi ngày.

---

## DEBT-001 — Lỗ hổng approval: ai cũng bấm được nút duyệt lệnh host [CRITICAL, security]

- `status`: open
- `date`: 2026-09-29

**Bằng chứng live:**
- Log Evernight 15:36: `APPROVAL REQUESTED: host_system -> host shell: docker logs march7...` → `✅ APPROVED` — người bấm là user `1378359549379084432`, KHÔNG phải owner (`726302130318868500`). Ngay sau đó user này nhắn: "ủa, em nhấn đc approve".
- `gateway/adapters/discord/views/approve_view.py`: không có `interaction_check` → ai thấy nút cũng bấm được. Repo đã biết, ghi nợ ở TASK-026 (`ARCHITECTURE.md` §7.4) nhưng chưa làm.
- Evernight không bao giờ gửi DM cho owner: `ApprovalGate` của Evernight build trần (`use_evernight_dm_approval` default `False`, chỉ `twin/march7/container.py:70` bật `True`). Log cả 2 bên: 0 lần thử gửi DM approval → Evernight ra thẳng nút channel.

**Hướng fix (user chưa chốt, mai làm):**
1. Gấp: `interaction_check` — chỉ owner (`EVERNIGHT_OWNER_USER_ID`) được bấm; người khác bấm → ephemeral "không có quyền". Áp dụng cả `ApproveView` + `DMApproveView`.
2. Đúng thiết kế: Evernight tự DM owner trước (qua discord client của nó / route `POST /dm`), fallback channel sau.
3. Policy còn mở: có chặn requester non-owner ngay từ đầu không (fail fast "chỉ owner mới được"), hay giữ "ai request cũng được, owner duyệt"?

**Staged Plan A (chưa deploy, giữ open):** `ApproveView.interaction_check` + re-verify trong callback đã staged (`gateway/adapters/discord/views/approve_view.py`; `DMApproveView` kế thừa) — chỉ `interaction.user.id` khớp configured owner mới tính, non-owner nhận ephemeral deny. Evernight là owner-trusted issuer duy nhất (`twin/evernight/server/approval_issuer.py` + `twin/shared/system_gateway/auth.py` `canonical_approval_action`/`mint_approval_token`/`verify_approval_token`); March7 chỉ request + consume grant, không sign. Grant bound canonical execution + actor + expiry + durable single use; missing owner/key fails closed. Policy staged: non-owner requesters được request, chỉ configured owner approve. Evernight cũng bật owner DM approval; mọi `gateway_admin update` cần owner consent mới qua `authorize_host`, không mint tại shared tool. Chưa deploy/verify live; Muse subset review không phải overall acceptance. Sol quota errors và không có verdict; giữ open.

---

## DEBT-002 — Làm lại A2A theo hướng health-check/ops

- `status`: open
- `date`: 2026-09-29

**Hiện trạng (không giống mục đích):**
- A2A hiện tại là JSON-RPC tự chế (`twin/shared/a2a/`): card + `tasks/send|get|cancel` + skills chat/consolidate/snapshot. Dùng cho chat delegate + consolidate + snapshot — không có gì cho health/ops.
- Health đi đường vòng: `SelfHealMonitor` poll `/.well-known/agent.json` của March7 để check sống chết (`twin/evernight/self_heal/monitor.py:161`); `/health` chỉ trả `{status, connected}` câm.
- Hệ quả tối nay: owner hỏi "Bé Bảy ra sao" → Evernight không có đường lấy số thật nên chém gió ("buộc phải đi qua gateway"). Đã chữa cháy tạm bằng 2 tools LLM 2 chiều (`march7_snapshot`, `request_consolidation` + guides) — nhưng đó là tool chat, không phải hạ tầng ops.

**Hướng đã bàn (chưa chốt, mai thiết kế):**
1. Liveness giữ `GET /health` thuần (Docker healthcheck/monitor cần endpoint ngu, nhanh).
2. Rich status/doctor qua A2A skills: uptime, model, discord link, T1 entries, T2 docs, redis/embedding OK không.
3. `SelfHealMonitor` chuyển từ poll agent card sang `/health` (+ skill `status` khi cần chi tiết).

**Staged (giữ open):** `SelfHealMonitor._check_health` đã poll `/health` (không dùng agent card; `twin/evernight/self_heal/monitor.py`); A2A task lifecycle đã bounded in-memory (`twin/shared/a2a/tasks.py`: active/completed caps, buffer/subscriber bounds, TTL/purge) với tests ở `tests/unit/a2a_task_bounds_test.py`. Rich status/doctor skills chưa staged. Muse đã approve scope SSE/recall: completion marker/count và EOF bắt buộc, không dựa riêng vào final `COMPLETED`. March7/Evernight phải nâng cấp cùng nhau; client mới từ chối server cũ thiếu marker. Chưa deploy; journal rereview và overall acceptance còn pending.

**Ghi chú:** 2 thư mục xác `twin/march7/a2a/`, `twin/evernight/a2a/` (rỗng, chỉ còn `__pycache__`) đã xóa 2026-09-29. Code A2A thật nằm ở `twin/shared/a2a/` + `*/server/a2a_server.py`.

---

## DEBT-003 — Staged remediation acceptance và follow-ups

- `status`: open
- Muse đã chứng minh guarded trim giữ nguyên dữ liệu mới khi reset/restore,
  same-ID save và mất EVAL response. Muse đã approve targeted closeout
  class-cap/counter/source/docs; không phải blanket verdict cho mọi finding.
  Không coi test xanh là overall acceptance, không áp dụng net diff hoặc deploy
  trước coordinated acceptance.
- Pending receiver ownership/plan là recovery witness, không được tự xóa/TTL để
  né lỗi. Reset caller/T1 không chứng minh T2 partial writes chưa tồn tại. Recovery
  cần owner reconciliation và backup/quiesce trước cleanup; contract được ghi
  tại `ARCHITECTURE.md` §3, chưa chạy recovery/migration trên production.
- Các Minor chưa đóng: concurrent model pulls dùng chung `.part`; migration
  index-recreation retry ergonomics; shell/cwd validation trước claim; existing
  ledger POSIX modes; request-nonce capacity; issuer symlink/mount trust;
  denied-version auditing; replay-after-failed-adapter coverage; grant/native actor
  consistency; self-update actor policy; persona fault-fixture kwargs; single-event
  SSE parser byte cap. Không tự suy ra đã fix từ subset approval.
- Source docstring migration vẫn có example `--dry-run` sai; parser không hỗ trợ
  flag này. Dùng dry-run mặc định như runbook, không copy example cũ.
- Live Discord delivery, actual macOS/Windows install, NTFS ACL và firewall chưa
  verify; POSIX `chmod 0600` không phải bằng chứng Windows ACL an toàn.
- Main-preservation exception: bốn docs edits sai checkout đã revert đúng từng
  edit; original HEAD/status và staged object IDs/modes giữ nguyên. Original index
  byte checksum chưa khôi phục. Phải công bố và xử lý exception tại apply guard,
  không đổi baseline hoặc ghi đè dirty work để che divergence.
