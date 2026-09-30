# Shared runtime configuration loaded from environment.
#
# Keep only values that genuinely vary per deployment. Defaults that rarely
# need tuning live as plain constants below the Config class.
import logging
import os


# === Constants (rarely change; don't expose to .env) ===

# Discord message pacing — keeps the bot from looking instant/spammy.
PART_BREAK_DELAY = float(os.getenv("PART_BREAK_DELAY", "0.6"))

# Tavily MCP defaults.
TAVILY_TIMEOUT_DEFAULT = 30

# Codebox runtime.
CODEBOX_TIMEOUT_DEFAULT = 60
CODEBOX_MAX_OUTPUT_CHARS_DEFAULT = 2000
CODEBOX_SESSION_TTL_DEFAULT = 1800  # 30 min

# System Gateway.
SYSTEM_GATEWAY_TIMEOUT_DEFAULT = 30

# Search / T2 retrieval — only SEMANTIC is currently consumed by the
# orchestrator; the time/topic knobs were never wired.
SEARCH_TOP_K_SEMANTIC_DEFAULT = 5
SEARCH_MIN_RELEVANCE_DEFAULT = 0.3

# Harrier-calibrated T2 cosine gates (2026-09-30, real harrier-oss-v1-270m q4
# ONNX, deployed query/passage prefixes; probes in scripts/calibrate_t2.py
# --offline). Owner may still override via env; below-default explicit values
# log a warning at import (see _warn_if_legacy_t2_gate).
T2_MIN_COSINE_DEFAULT = 0.60
T2_MERGE_MIN_COSINE_DEFAULT = 0.75


class Config:
    # === Discord / Gateway ===
    DISCORD_BOT_TOKEN = os.getenv("DISCORD_MARCH7_TOKEN")
    DISCORD_BOT_CLIENT_ID = os.getenv("DISCORD_MARCH7_CLIENT_ID")
    SYNC_COMMANDS = os.getenv("SYNC_COMMANDS", "0")

    # === LLM provider ===
    # Supported: "gemini" or "openai" (OpenAI-compatible — also covers
    # OpenRouter, LM Studio, Qwen compatible-mode, etc. via OPENAI_API_URL).
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

    # Generation parameters (tuned per deployment).
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
    LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))
    LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.4"))
    LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.1"))

    # Reasoning effort for OpenAI-compat endpoint.
    # Ollama openai.go maps this to native `think` param:
    #   "none"      -> think=false
    #   "low"|"medium"|"high"|"max" -> think="<level>"
    # Empty / unset = don't send the field, let provider/model decide.
    _LLM_REASONING_EFFORT_RAW = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    _LLM_REASONING_EFFORT_VALID = {"", "none", "low", "medium", "high", "max"}
    if _LLM_REASONING_EFFORT_RAW not in _LLM_REASONING_EFFORT_VALID:
        raise ValueError(
            f"LLM_REASONING_EFFORT must be one of "
            f"{{'', 'none', 'low', 'medium', 'high', 'max'}}, "
            f"got: {_LLM_REASONING_EFFORT_RAW!r}"
        )
    LLM_REASONING_EFFORT = _LLM_REASONING_EFFORT_RAW or None

    # Consolidation is a JSON-extraction (Summarizer) task, not open reasoning.
    # Running it at the global reasoning_effort ("high") makes minimax "think"
    # for minutes on a large T1 prompt and blow past LLM_REQUEST_TIMEOUT, so it
    # gets its own lower effort + tighter token cap. Empty = don't send field.
    _LLM_CONSOLIDATION_EFFORT_RAW = os.getenv("LLM_CONSOLIDATION_REASONING_EFFORT", "low").strip().lower()
    if _LLM_CONSOLIDATION_EFFORT_RAW not in _LLM_REASONING_EFFORT_VALID:
        raise ValueError(
            f"LLM_CONSOLIDATION_REASONING_EFFORT must be one of "
            f"{{'', 'none', 'low', 'medium', 'high', 'max'}}, "
            f"got: {_LLM_CONSOLIDATION_EFFORT_RAW!r}"
        )
    LLM_CONSOLIDATION_REASONING_EFFORT = _LLM_CONSOLIDATION_EFFORT_RAW or None
    LLM_CONSOLIDATION_MAX_TOKENS = int(os.getenv("LLM_CONSOLIDATION_MAX_TOKENS", "4000"))

    # OpenAI tool_choice enforcement for Decide stage. Allowed: "" (off, don't send) /
    # "auto" / "required" / "none". Empty = current behavior (let LLM decide). Set
    # "required" to force a tool call when user intent clearly needs a tool — but note
    # Ollama OpenAI-compat proxy may silently ignore this field; test runtime before
    # relying on it.
    _LLM_TOOL_CHOICE_RAW = os.getenv("LLM_TOOL_CHOICE", "").strip().lower()
    LLM_TOOL_CHOICE = _LLM_TOOL_CHOICE_RAW or None

    LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    LLM_CONNECT_TIMEOUT = int(os.getenv("LLM_CONNECT_TIMEOUT", "10"))

    # OpenAI-compatible endpoint.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Gemini.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models")

    # === Discord message pacing ===
    PART_BREAK_DELAY = PART_BREAK_DELAY

    # === Logging ===
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # === Redis / T1 ===
    REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    TIMELINE_REDIS_DB = int(os.getenv("TIMELINE_REDIS_DB", "0"))
    # T1 storage phases:
    #   redis_stack  — Redis Stack JSON/Search (default)
    #   legacy       — Redis HASH fallback
    T1_STORAGE_PHASE = os.getenv("T1_STORAGE_PHASE", "redis_stack").lower()
    T1_CONTEXT_MAX_TOKENS = int(os.getenv("T1_CONTEXT_MAX_TOKENS", "1800"))
    T1_CONTEXT_MAX_MESSAGES = int(os.getenv("T1_CONTEXT_MAX_MESSAGES", "32"))

    # === Tavily (web search tool) ===
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", None)
    TAVILY_TIMEOUT = int(os.getenv("TAVILY_TIMEOUT", str(TAVILY_TIMEOUT_DEFAULT)))
    TAVILY_MCP_URL = os.getenv("TAVILY_MCP_URL", "https://mcp.tavily.com/mcp")

    # === Codebox (sandboxed Python) ===
    CODEBOX_API_URL = os.getenv("CODEBOX_API_URL", "http://localhost:8069")
    CODEBOX_TIMEOUT = CODEBOX_TIMEOUT_DEFAULT
    CODEBOX_MAX_OUTPUT_CHARS = CODEBOX_MAX_OUTPUT_CHARS_DEFAULT
    CODEBOX_SESSION_TTL = CODEBOX_SESSION_TTL_DEFAULT

    # === System Gateway (native host boundary) ===
    SYSTEM_GATEWAY_URL = os.getenv("SYSTEM_GATEWAY_URL", "http://host.docker.internal:8380")
    SYSTEM_GATEWAY_TIMEOUT = int(
        os.getenv("SYSTEM_GATEWAY_TIMEOUT", str(SYSTEM_GATEWAY_TIMEOUT_DEFAULT))
    )
    SYSTEM_GATEWAY_SHARED_SECRET = os.getenv("SYSTEM_GATEWAY_SHARED_SECRET") or None

    # === Evernight A2A endpoint (March7 calls Evernight) ===
    EVERNIGHT_A2A_URL = os.getenv("EVERNIGHT_A2A_URL", "http://evernight:8001")

    # === March7 A2A endpoint (Evernight calls March7) ===
    MARCH7_URL = os.getenv("MARCH7_URL", "http://march7:8000")

    # === A2A peer authentication (HMAC shared secret) ===
    # Deployed via shared .env (both agents must hold the same value).
    A2A_SHARED_SECRET = os.getenv("A2A_SHARED_SECRET") or None

    # === Owner approval authority (Evernight-issued grants) ===
    # Explicit owner platform user id. Empty/None means unknown -> deny.
    # No hardcoded fallback: authority must be configured, never assumed.
    EVERNIGHT_OWNER_USER_ID = os.getenv("EVERNIGHT_OWNER_USER_ID") or None
    # Host-private approval-key FILE path. The key VALUE is never in env or
    # the shared repo .env; the file is mounted read-only into Evernight
    # only (March7 must never read it).
    SYSTEM_GATEWAY_APPROVAL_SECRET_FILE = os.getenv(
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", "/run/secrets/system_gateway_approval"
    )

    # === Search Memory ===
    SEARCH_TOP_K_SEMANTIC = SEARCH_TOP_K_SEMANTIC_DEFAULT
    SEARCH_MIN_RELEVANCE = SEARCH_MIN_RELEVANCE_DEFAULT

    # === Embedding model (local Harrier q4 ONNX only, no API) ===
    HARRIER_MODEL_DIR = os.getenv("HARRIER_MODEL_DIR", "models/harrier-q4")
    EMBEDDING_VECTOR_SIZE = int(os.getenv("EMBEDDING_VECTOR_SIZE", "640"))
    EMBEDDING_TRACE_LOG_ENABLED = (
        os.getenv("EMBEDDING_TRACE_LOG_ENABLED", "false").lower() == "true"
    )
    EMBEDDING_TRACE_LOG_PATH = os.getenv(
        "EMBEDDING_TRACE_LOG_PATH", "logs/embedding_trace.jsonl"
    )
    # Retrieval prefixes. Qwen3-Embedding is instruction-aware and ASYMMETRIC:
    # the QUERY side wants an instruct wrapper ("Instruct: ...\nQuery: <text>")
    # while the PASSAGE side wants raw text (empty prefix) — so changing only
    # the query prefix needs NO reindex. (e5-style models instead use
    # "query: "/"passage: ".) Provider-agnostic so the embedding model can be
    # switched via .env without touching call sites. Env values are stored on
    # one line with a literal backslash-n; both dotenv and docker env_file may
    # deliver it as two raw chars, so unescape it into a real newline here.
    EMBEDDING_QUERY_PREFIX = os.getenv(
        "EMBEDDING_QUERY_PREFIX",
        "Instruct: Given a user message, retrieve relevant memory summaries "
        "about the user and past conversation\nQuery: ",
    ).replace("\\n", "\n")
    EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "").replace("\\n", "\n")
    # T2 semantic-recall relevance gate: drop KNN hits whose cosine similarity
    # is below this before injecting into the prompt.
    # Harrier default 0.60 (2026-09-30, real q4 ONNX, 19 labeled queries +
    # 19 negatives: precision 1.00, recall 0.79; noise peaks 0.588, clear
    # positives >= 0.626, midpoint 0.607). Vague sub-gate queries return no
    # memory rather than wrong memory. Legacy Qwen-era values (0.0/0.35/0.42/
    # 0.45) admit cross-topic noise with Harrier (pet<->hiking 0.64,
    # weather-query<->hiking 0.52) — explicit below-default values warn at import.
    T2_MIN_COSINE = float(os.getenv("T2_MIN_COSINE", str(T2_MIN_COSINE_DEFAULT)))
    # T2 diary model (write path). A new summary is merged into an existing
    # same-user same-VN-day doc when their cosine similarity reaches this
    # floor. Harrier default 0.75 (2026-09-30: same-event follow-ups
    # 0.82-0.93 all merge; cross-topic max 0.71 blocked; missed merges safely
    # append, while a false merge glues unrelated topics together). Legacy
    # Qwen-era 0.60 false-merges 26/60 cross-topic pairs with Harrier.
    T2_MERGE_MIN_COSINE = float(os.getenv("T2_MERGE_MIN_COSINE", str(T2_MERGE_MIN_COSINE_DEFAULT)))
    # Char cap for a merged diary doc: beyond this, append a new doc instead
    # of growing a mega-doc whose embedding averages into mush.
    T2_MERGE_MAX_CHARS = int(os.getenv("T2_MERGE_MAX_CHARS", "1500"))
    # Archive raw T1 entries to a cold per-day Redis list on trim instead of
    # hard-deleting (W3): t1:archive:{scope}:{scope_id}:{day}. Best-effort —
    # an archive failure never blocks the trim.
    T1_ARCHIVE_ENABLED = os.getenv("T1_ARCHIVE_ENABLED", "true").lower() == "true"
    T1_ARCHIVE_TTL_DAYS = int(os.getenv("T1_ARCHIVE_TTL_DAYS", "90"))


def _warn_if_legacy_t2_gate(env_name: str, default: float, legacy: str) -> None:
    """Warn when an explicit T2 gate override sits below the Harrier default.

    Overrides stay owner-configurable and are respected; the warning exists
    because pre-Harrier values silently cause false recalls/merges. Unset or
    unparsable values stay quiet (unparsable ones fail loudly at float()).
    """
    raw = os.getenv(env_name)
    if raw is None:
        return
    try:
        value = float(raw)
    except ValueError:
        return
    if value < default:
        logging.getLogger(__name__).warning(
            "%s=%s is below the Harrier-calibrated default %.2f (%s); "
            "pre-Harrier values cause false recalls/merges — override respected.",
            env_name, raw, default, legacy,
        )


_warn_if_legacy_t2_gate("T2_MIN_COSINE", T2_MIN_COSINE_DEFAULT, "legacy Qwen-era 0.0/0.35/0.42/0.45")
_warn_if_legacy_t2_gate("T2_MERGE_MIN_COSINE", T2_MERGE_MIN_COSINE_DEFAULT, "legacy Qwen-era 0.60")
