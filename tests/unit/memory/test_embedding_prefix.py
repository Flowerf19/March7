"""Unit tests for the qwen3 asymmetric embedding prefixes + tool-only T2 recall.

qwen3-embedding is instruction-aware and ASYMMETRIC: the query side gets an
instruct wrapper ("Instruct: ...\nQuery: "), the passage side stays raw. This
module also guards the tool-only recall decision (2026-07-03): get_context
must never embed or search T2 — recall happens only via the search_memory tool.
"""
from __future__ import annotations

import importlib.util

from unittest.mock import AsyncMock, MagicMock

import pytest

from twin.shared.config.settings import Config
from twin.shared.memory.manager import SharedMemoryManager
from twin.shared.tools.modules.memory.search_memory_tool import SearchMemoryTool


class CapturingEmbeddingService:
    """Records all texts passed to get_embedding."""

    def __init__(self, dim: int = 384):
        self.texts: list[str] = []
        self.dim = dim

    async def get_embedding(self, text: str) -> list[float]:
        self.texts.append(text)
        return [0.1] * self.dim


class RecordingStore:
    def __init__(self):
        self.search_calls: list[dict] = []
        self.recent_calls: list[dict] = []

    async def search(self, user_id, query_embedding, limit, *, query_text=None, topic_filter=None):
        self.search_calls.append({"user_id": user_id})
        return []

    async def get_recent(self, user_id, limit):
        self.recent_calls.append({"user_id": user_id})
        return []


def _manager(store, svc):
    active_mock = MagicMock()
    active_mock.get_context = AsyncMock(return_value=[])
    profile_mock = MagicMock()
    profile_mock.get_system_prompt_context = AsyncMock(return_value="")
    return SharedMemoryManager(
        active=active_mock,
        profile_store=profile_mock,
        timeline_summary_store=store,
        embedding_service=svc,
    )


# --------------------------------------------------------------- prefix value


def test_default_query_prefix_is_qwen3_instruct_format():
    assert Config.EMBEDDING_QUERY_PREFIX.startswith("Instruct: ")
    # The real newline and the trailing space must survive loading.
    assert Config.EMBEDDING_QUERY_PREFIX.endswith("\nQuery: ")
    assert "\\n" not in Config.EMBEDDING_QUERY_PREFIX


def test_default_passage_prefix_is_raw():
    # qwen3 passage side takes raw text; T2 docs are indexed WITHOUT a prefix,
    # so a non-empty default here would silently poison new embeddings.
    assert Config.EMBEDDING_PASSAGE_PREFIX == ""


def test_prefix_unescapes_literal_backslash_n(monkeypatch):
    """.env stores the prefix on one line with a literal backslash-n."""
    import twin.shared.config.settings as settings_module

    original_config = settings_module.Config
    original_prefix = original_config.EMBEDDING_QUERY_PREFIX
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "Instruct: test\\nQuery: ")
    # Isolated evaluation: never importlib.reload() the shared settings module
    # (reload replaces Config globally and breaks later owner-approval tests).
    # Same spec_from_file_location pattern as bootstrap script tests.
    spec = importlib.util.spec_from_file_location(
        "twin_shared_config_settings_isolated_prefix",
        settings_module.__file__,
    )
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert isolated.Config.EMBEDDING_QUERY_PREFIX == "Instruct: test\nQuery: "
    # Parent Config identity/values must survive this test.
    assert settings_module.Config is original_config
    assert settings_module.Config.EMBEDDING_QUERY_PREFIX == original_prefix
    assert Config is original_config


# --------------------------------------------------------- query composition


@pytest.mark.asyncio
async def test_search_memory_tool_applies_query_prefix():
    svc = CapturingEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=RecordingStore(), embedding_service=svc)

    await tool.execute(user_id="12345", query="phim hoạt hình")

    assert len(svc.texts) == 1
    assert svc.texts[0] == f"{Config.EMBEDDING_QUERY_PREFIX}phim hoạt hình"
    assert svc.texts[0].endswith("\nQuery: phim hoạt hình")


# ------------------------------------------------------- tool-only T2 recall


@pytest.mark.asyncio
async def test_get_context_does_not_inject_t2():
    """T2 recall is tool-only: get_context must not embed nor search T2."""
    svc = CapturingEmbeddingService()
    store = RecordingStore()
    manager = _manager(store, svc)

    sys_prompt, messages = await manager.get_context("12345", "nhạc jazz")

    assert svc.texts == []
    assert store.search_calls == []
    assert "Ngữ cảnh nhớ liên quan" not in sys_prompt
    assert messages == []


@pytest.mark.asyncio
async def test_get_context_channel_header_exposes_channel_id():
    """The model needs the channel scope id to pass channel_id to search_memory."""
    manager = _manager(RecordingStore(), CapturingEmbeddingService())

    sys_prompt, _ = await manager.get_context("12345", "hi", channel_id="777")

    assert "Platform channel ID: 777" in sys_prompt


@pytest.mark.asyncio
async def test_get_context_user_scope_has_no_channel_header():
    manager = _manager(RecordingStore(), CapturingEmbeddingService())

    sys_prompt, _ = await manager.get_context("12345", "hi")

    assert "Platform channel ID" not in sys_prompt


def test_preflight_helpers_removed():
    """Preflight was removed twice already — keep it removed (see get_context)."""
    import twin.shared.memory.manager as manager_module

    assert not hasattr(manager_module, "format_preflight_for_prompt")
    assert not hasattr(SharedMemoryManager, "_preflight_context")
