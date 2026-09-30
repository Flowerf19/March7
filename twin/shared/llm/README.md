# LLM Providers & Embeddings

Cấu hình env vars cho chat + embedding của `march7`/`evernight`. Code lookup: hỏi codegraph (`codegraph_search GeminiService`, `codegraph_files twin/shared/llm`...).

## Chat providers

| `LLM_PROVIDER` | Protocol | Env vars |
|---|---|---|
| `gemini` | Gemini native | `GEMINI_API_KEY`, `LLM_MODEL` (vd `gemini-2.5-flash`), optional `GEMINI_API_URL` |
| `openai_compat` (alias: `openai`) | OpenAI-compatible `POST /chat/completions` | `OPENAI_API_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` |

`OPENAI_API_URL` chấp nhận mọi endpoint OpenAI-compat: OpenAI gốc, OpenRouter, LM Studio (`http://host.docker.internal:1234/v1`), Qwen DashScope compatible-mode, ...

## Embeddings (local Harrier q4 ONNX only, no API)

Backend duy nhất: `HarrierEmbeddingService` chạy `harrier-oss-v1-270m` q4 bằng ONNX Runtime CPU trong process. Pull model 1 lần (host, stdlib-only, 5 files SHA-256 verified, marker `.installed.json` publish atomic sau cùng):

```bash
python scripts/pull_harrier_model.py [--model-dir models/harrier-q4]
```

Model vào `models/harrier-q4/`, bind-mount read-only vào container (`/app/models:ro`).

| Env | Ý nghĩa |
|---|---|
| `HARRIER_MODEL_DIR` | Thư mục model (default `models/harrier-q4`) |
| `EMBEDDING_VECTOR_SIZE` | Phải là `640` (dim Harrier; code default `640` trong `twin/shared/config/settings.py`) |
| `T2_MIN_COSINE` | Retrieval gate, default `0.60` (Harrier-calibrated 2026-09-30; legacy Qwen-era `0.0`/`0.35` gây false recall, below-default warn ở import) |
| `T2_MERGE_MIN_COSINE` | Same-day merge gate, default `0.75` (legacy `0.60` false-merge cross-topic với Harrier) |

Đổi dim KHÔNG dùng `FT.DROPINDEX`/restart thô. Migration thật là dry-run + backup + resumable re-embed từ `summary` trong source HASH (`scripts/migrate_t2_harrier.py`, tests ở `tests/unit/llm/embedding/test_migrate_t2_harrier.py`):

```bash
python3 scripts/migrate_t2_harrier.py --dry-run --redis-url redis://localhost:6379
python3 scripts/migrate_t2_harrier.py --apply --backup /safe/t2.bak.jsonl
python3 scripts/migrate_t2_harrier.py --apply --backup /safe/t2.bak.jsonl --recreate-index
python3 scripts/migrate_t2_harrier.py --rollback --backup /safe/t2.bak.jsonl
```

Contract: dry-run mặc định, zero writes; `--apply` bắt buộc `--backup PATH` (backup full HASH + SHA-256, verify trước mọi write, checkpoint resumable); chỉ rewrite field `embedding`, giữ key/ID/summary/topic/provenance/TTL byte-for-byte và validate lại với backup. KHÔNG bao giờ `DEL`/`UNLINK`/`FLUSHALL`/`DROPINDEX ... DD`; recreate index (`DROPINDEX` không `DD` + `FT.CREATE`) chỉ với `--recreate-index`, chỉ sau khi mọi record validate và không còn legacy-dim vector. KHÔNG pad/truncate vector 1024 cũ. Thêm field additive (`day`/`period_start`/`period_end`) dùng `FT.ALTER`, không full reindex. Production migration/redeploy là owner action thủ công, không xóa dữ liệu thật; backup trước khi migrate explicit.

## Ví dụ cấu hình

### LM Studio chat + Harrier local embeddings

```env
LLM_PROVIDER=openai_compat
OPENAI_API_URL=http://host.docker.internal:1234/v1
OPENAI_API_KEY=dummy-key
OPENAI_MODEL=<model-name-in-lm-studio>

HARRIER_MODEL_DIR=models/harrier-q4
EMBEDDING_VECTOR_SIZE=640
```

### Gemini chat + Harrier local embeddings

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
LLM_MODEL=gemini-2.5-flash

HARRIER_MODEL_DIR=models/harrier-q4
EMBEDDING_VECTOR_SIZE=640
```

> [!NOTE]
> Docker Compose: từ container → container dùng service name (vd `http://redis:6379`); container → host dùng `host.docker.internal` nếu môi trường hỗ trợ.

Calibration (`scripts/calibrate_t2.py`, tests ở `tests/unit/llm/embedding/test_t2_calibration.py`): `python3 scripts/calibrate_t2.py --offline` (no Redis, embed labeled probes bằng local read-only Harrier, assert separation cho cả 2 gates) hoặc live read-only (`--redis-url ...`, `SCAN` + `HGETALL` only, gate tắt in-process để đo). Offline là labeled probes cố định, không thay production trace; vector/model errors và tracing là best-effort, invalid vector fail-closed (không bypass gate).

## Thêm provider mới

Chat: kế thừa `BaseLLMService` + thêm 1 nhánh vào factory chat. Embeddings chỉ có Harrier local (không thêm provider API). Codegraph lookup: `codegraph_node BaseLLMService` / `codegraph_search create_embedding_service`.
