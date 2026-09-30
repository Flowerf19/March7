# Docker Setup — March7

> **Dành cho agent (Copilot/LLM)**: Trước khi làm việc với Docker, đọc [.agents/README.md](../.agents/README.md) để nắm kiến trúc, boundary, và workflow.

## Mục tiêu

Docker Compose là runtime chính cho kiến trúc Twin-Soul của dự án:

- `march7`: main chat runtime + gateway + A2A `:8000`
- `evernight`: independent Discord agent for DM/tag/`!9` chat, notifications/approvals, consolidation/self-heal + A2A `:8001`
- shared infra: Redis, Codebox

## Structure

```
docker/
├── docker-compose.yml           # Master compose (network + includes)
├── shared/
│   ├── Dockerfile.base
│   ├── docker-compose.base.yml
│   ├── docker-compose.redis.yml      # redis_data là NAMED volume (không phải bind ./volumes)
│   └── docker-compose.codebox.yml
├── march7/
│   ├── Dockerfile
│   └── docker-compose.yml       # source :ro + own persona overlay writable
├── evernight/
│   ├── Dockerfile
│   └── docker-compose.yml       # source :ro + own persona overlay + approval-key mount (Evernight only)
└── README.md                    # This file
```

Agent mounts (xem 2 compose files): `twin/` + `gateway/` read-only (`:ro,z`); overlay writable chỉ cho OWN persona dir (`twin/<agent>/personas`, data-only `.md`, peer path vẫn read-only qua parent mount); `models/` read-only (`/app/models:ro`, host pull bằng `scripts/pull_harrier_model.py`); `memories/` + `data/` writable.

## Architecture

### Twin-Soul Agent Services

| Service | Container | Image | Port | Purpose |
|---------|-----------|-------|------|---------|
| `redis` | `march7-redis` | `redis/redis-stack-server` | 6379 | Shared T1 storage + coordination markers |
| `codebox` | `march7-codebox` | `shroominic/codebox` | 8069 | Python sandbox |
| `base` | — | `march7-base` | — | Shared Python runtime |
| `march7` | `march7` | `march7-agent` | 8000 | Gateway + March7 Discord bot + March7 A2A |
| `evernight` | `evernight` | `evernight-agent` | 8001 | Evernight Discord bot for DM/tag/`!9` chat, notifications/approvals, consolidation + self-healing |

Xem [ARCHITECTURE.md](ARCHITECTURE.md) để biết boundary private/shared chi tiết.

> [!WARNING]
> Evernight không đọc trực tiếp T1 keys của March7; truy cập memory qua A2A boundary (`get_snapshot`, `clear_session`).

### Current Build

- **shared/Dockerfile.base**: runtime Python + dependencies chung.
- **march7/Dockerfile**: March7-owned image, entrypoint `python -m gateway`.
- **evernight/Dockerfile**: Evernight-owned image, entrypoint `python -m twin.evernight`.

## Quick Start

### Clean start (first time or after changes)

```bash
cd docker
docker compose down
DOCKER_BUILDKIT=1 docker compose build
docker compose up -d
docker compose logs -f
```

Health endpoints sau khi chạy:

- `http://localhost:8000/health`, `http://localhost:8001/health` — 503 khi Discord bot
  chưa connected (process vẫn sống nên vẫn phải phân biệt được hai trạng thái này)
- Agent card A2A: `http://localhost:8000/.well-known/agent.json`,
  `http://localhost:8001/.well-known/agent.json`

### From project root

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs -f
```

### Individual service operations

```bash
# Chỉ start infrastructure
docker compose -f docker/shared/docker-compose.redis.yml \
    -f docker/shared/docker-compose.codebox.yml up -d

# Start riêng agents
docker compose -f docker/docker-compose.yml up -d march7 evernight

# Rebuild một service
DOCKER_BUILDKIT=1 docker compose -f docker/docker-compose.yml build march7
```

## Data Persistence

| Mount | Purpose | Storage |
|-----------|---------|---------|
| named volume `redis_data` (`/data`) | T1 + T2 + coordination | Redis AOF/RDB (Redis Stack `7.2.0-v18`) |
| bind `twin/`, `gateway/` `:ro` | runtime source (read-only) | host repo |
| bind own `twin/<agent>/personas` writable | persona `.md` data-only | host repo |
| bind `memories/`, `data/` writable | T3 Markdown + app data | host repo |
| bind `models/` `:ro` | Harrier q4 ONNX (host-pulled) | host repo |

## Commands Reference

```bash
# Start all
docker compose up -d

# Stop all
docker compose down

# Stop + remove volumes (⚠️ xóa toàn bộ dữ liệu)
docker compose down -v

# Rebuild tất cả
DOCKER_BUILDKIT=1 docker compose build

# Logs
docker compose logs -f march7     # March7 agent
docker compose logs -f evernight  # Evernight agent
docker compose logs -f            # All

# Restart
docker compose restart march7

# Exec
docker exec -it march7 bash
docker exec -it evernight bash

# Health check
docker compose ps
```

## Hot-Reload

Code changes trong `twin/`, `gateway/`, hoặc `memories/` được watch tự động bởi `watchmedo` — không cần rebuild container.

Chỉ rebuild image khi:
- Thay đổi `requirements.txt` (dependencies mới)
- Sửa Dockerfile

## Environment Variables

Bot services dùng chung `env_file: ../../.env` (common config), KHÔNG bind `.env` file vào container. Peer credential isolation bằng explicit blank override (compose precedence `environment:` > `env_file`): march7 set `DISCORD_EVERNIGHT_TOKEN=` trống, evernight set `DISCORD_MARCH7_TOKEN=` trống; march7 còn blank `SYSTEM_GATEWAY_APPROVAL_SECRET=` và `SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=` để stray `.env` entry không reintroduce được. Key variables (không in giá trị thật, không commit secret vào repo):

```env
# Discord (mỗi agent chỉ thấy token của mình; peer token bị blank trong compose)
DISCORD_MARCH7_TOKEN=<march7-bot-token>
DISCORD_MARCH7_CLIENT_ID=<march7-client-id>
DISCORD_EVERNIGHT_TOKEN=<evernight-bot-token>
EVERNIGHT_OWNER_USER_ID=<configured-owner-platform-id>

# Gateway
GATEWAY_ENABLED_PLATFORMS=discord
DISCORD_GATEWAY_ENABLED=true

# LLM chat
LLM_PROVIDER=openai
OPENAI_API_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=dummy-key
OPENAI_MODEL=<model-name>

# Embeddings (local Harrier q4 ONNX only, dim 640)
HARRIER_MODEL_DIR=models/harrier-q4
EMBEDDING_VECTOR_SIZE=640
T2_MIN_COSINE=0.60
T2_MERGE_MIN_COSINE=0.75

# Infrastructure (internal Docker network)
REDIS_URL=redis://redis:6379
TIMELINE_REDIS_DB=0
CODEBOX_API_URL=http://codebox:8069

# Native System Gateway (Plan A trust)
SYSTEM_GATEWAY_URL=http://host.docker.internal:8380
SYSTEM_GATEWAY_SHARED_SECRET=<request-hmac-key>
SYSTEM_GATEWAY_RAW_SHELL=false
```

Plan A secrets (confirm đầy đủ, không rút gọn): request-HMAC key `SYSTEM_GATEWAY_SHARED_SECRET` KHÁC private approval key; approval key value KHÔNG BAO GIỜ nằm trong env/shared `.env`/repo, chỉ nằm ở host-private FILE ngoài repo và mount read-only vào Evernight duy nhất (`target: /run/secrets/system_gateway_approval`, `read_only: true`, `bind.create_host_path: false`, `selinux: z`; `docker/evernight/docker-compose.yml`). Host source override (nonsecret path): `${SYSTEM_GATEWAY_APPROVAL_SECRET_HOST_PATH:-${HOME}/.config/system-gateway/approval_secret}` (user default `~/.config/system-gateway/approval_secret`; root native CLI/service align `/etc/system-gateway/approval_secret` qua cùng resolver + explicit override). Trong container Evernight: `SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=/run/secrets/system_gateway_approval`. March7 KHÔNG mount key, KHÔNG đọc key, chỉ request + consume grant, không bao giờ sign approval. Missing owner (`EVERNIGHT_OWNER_USER_ID` trống) hoặc missing key fails closed (deny, không fallback sang request secret). Raw shell denied by default (`SYSTEM_GATEWAY_RAW_SHELL=false`), native generic shell là path duy nhất, mọi shell cần owner-signed grant. Chưa có deployment nào xảy ra; sau khi đổi `.env`, owner recreate thủ công (`docker compose -f docker/docker-compose.yml up -d --force-recreate march7 evernight`), không cần rebuild image trừ khi đổi deps/Dockerfile.

## Redis Stack Notes

T1 active memory can use each agent's own Redis DB (`MARCH7_REDIS_DB`,
`EVERNIGHT_REDIS_DB`). T2 timeline memory uses RediSearch indexes and must run
on Redis DB 0, so compose sets `TIMELINE_REDIS_DB=0` for both agents.

> [!TIP]
> Chi tiết provider và mapping endpoint xem `../twin/shared/llm/README.md`.
