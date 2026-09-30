"""Trace-fault isolation for T2 SEARCH observability (memory_trace only).

Observability must never change user-visible recall: dim mismatch or a
throwing emitter is traced best-effort per hit (cs=None / skip) while the
recall path keeps matched text, sources, ranks, and later hits.
Strict dim gates in ranking/inference are untouched (cosine still raises).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from twin.shared.llm.embedding.embedding_trace_logger import cosine_similarity
from twin.shared.tools.modules.memory.memory_trace import trace_semantic_results
from twin.shared.tools.modules.memory.search_memory_tool import SearchMemoryTool


class FakeTraceService:
    """Enabled collector: records SEARCH events without I/O."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.trace_logger = SimpleNamespace(enabled=enabled)
        self.events: list[dict] = []

    async def get_embedding(self, text: str) -> list[float]:
        return [0.1] * 640

    def _trace_embedding_event(self, **kwargs) -> None:  # noqa: ANN003
        self.events.append(kwargs)


class ThrowingTraceService(FakeTraceService):
    def _trace_embedding_event(self, **kwargs) -> None:  # noqa: ANN003
        raise OSError("trace backend down")


class FakeStore:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs

    async def search(self, user_id, query_embedding, limit, **kwargs):  # noqa: ANN001, ANN002
        return list(self.docs)

    async def get_recent(self, user_id, limit, **kwargs):  # noqa: ANN001, ANN002
        return list(self.docs)


def test_mismatched_dim_traced_as_none_and_next_hit_ok() -> None:
    """640 query + legacy 1024 hit must not abort; next 640 hit still traced."""
    # Strict gate preserved: raw cosine still rejects mixed dims.
    with pytest.raises(ValueError):
        cosine_similarity([0.1] * 640, [0.1] * 1024)

    svc = FakeTraceService(enabled=True)
    trace_semantic_results(
        svc,
        input_text="query: phim",
        current_query="phim",
        query_embedding=[0.1] * 640,
        summaries=[
            {"summary": "legacy hit", "embedding": [0.1] * 1024, "score": 0.2},
            {"summary": "valid hit", "embedding": [0.1] * 640, "score": 0.3},
        ],
        scope_ids=["12345"],
        sources=["12345", "12345"],
    )
    assert len(svc.events) == 2
    assert svc.events[0]["cosine_similarity"] is None
    assert svc.events[0]["matched_text"] == "legacy hit"
    assert svc.events[0]["rrf_rank"] == 1
    assert svc.events[0]["action"] == "KNN_RESULT"
    assert svc.events[1]["cosine_similarity"] == pytest.approx(1.0)
    assert svc.events[1]["matched_text"] == "valid hit"
    assert svc.events[1]["rrf_rank"] == 2
    assert svc.events[1]["action"] == "KNN_CANDIDATE"
    assert svc.events[0]["extra"]["source_scope_id"] == "12345"


def test_non_numeric_embedding_traced_as_none() -> None:
    """Malformed vectors (TypeError in cosine) also degrade to cs=None."""
    svc = FakeTraceService(enabled=True)
    trace_semantic_results(
        svc,
        input_text="q",
        current_query="q",
        query_embedding=[0.1, 0.2],
        summaries=[
            {"summary": "bad vec", "embedding": ["a", "b"]},
            {"summary": "good vec", "embedding": [0.1, 0.2]},
        ],
    )
    assert len(svc.events) == 2
    assert svc.events[0]["cosine_similarity"] is None
    assert svc.events[1]["cosine_similarity"] == pytest.approx(1.0)


def test_disabled_trace_skips_dim_calculation() -> None:
    """Disabled logger: no events and mismatched dims never reach cosine."""
    svc = FakeTraceService(enabled=False)
    trace_semantic_results(
        svc,
        input_text="q",
        current_query="q",
        query_embedding=[0.1] * 640,
        summaries=[{"summary": "legacy", "embedding": [0.1] * 1024}],
    )
    assert svc.events == []

    svc.trace_logger = None
    trace_semantic_results(
        svc,
        input_text="q",
        current_query="q",
        query_embedding=[0.1] * 640,
        summaries=[{"summary": "legacy", "embedding": [0.1] * 1024}],
    )
    assert svc.events == []


def test_throwing_emitter_does_not_abort_trace_loop() -> None:
    svc = ThrowingTraceService(enabled=True)
    trace_semantic_results(
        svc,
        input_text="q",
        current_query="q",
        query_embedding=[0.1] * 640,
        summaries=[
            {"summary": "one", "embedding": [0.1] * 640},
            {"summary": "two", "embedding": [0.1] * 640},
        ],
    )  # must not raise


@pytest.mark.asyncio
async def test_search_recall_unaffected_by_trace_dim_mismatch() -> None:
    """End-to-end via tool: mixed dims + enabled trace keeps normal recall."""
    store = FakeStore([
        {"summary_id": "legacy", "summary": "Chuyện cũ legacy.",
         "embedding": [0.1] * 1024, "score": 0.2, "created_at": 1718360000.0},
        {"summary_id": "valid", "summary": "Chuyện mới khớp.",
         "embedding": [0.1] * 640, "score": 0.3, "created_at": 1718361000.0},
    ])
    svc = FakeTraceService(enabled=True)
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=svc)

    result = await tool.execute(user_id="12345", query="chuyện")

    assert "Tìm thấy 2 ký ức" in result
    assert "Chuyện cũ legacy" in result and "Chuyện mới khớp" in result
    assert "relevance=0.80" in result and "relevance=0.70" in result
    assert len(svc.events) == 2
    assert svc.events[0]["cosine_similarity"] is None


@pytest.mark.asyncio
async def test_search_recall_unaffected_by_throwing_emitter() -> None:
    store = FakeStore([
        {"summary_id": "s1", "summary": "Ký ức bình thường.",
         "embedding": [0.1] * 640, "score": 0.1, "created_at": 1718360000.0},
    ])
    svc = ThrowingTraceService(enabled=True)
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=svc)

    result = await tool.execute(user_id="12345", query="ký ức")

    assert "Tìm thấy 1 ký ức" in result
    assert "Ký ức bình thường" in result
    assert "Lỗi khi tìm kiếm" not in result
