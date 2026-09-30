"""Unit tests for memory/profile system tools."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from twin.shared.config.settings import Config
from twin.shared.memory.vn_time import VN_TZ, vn_now
from twin.shared.tools.modules.memory.search_memory_tool import SearchMemoryTool
from twin.shared.tools.modules.profile.get_profile_tool import GetProfileTool
from twin.shared.tools.modules.profile.update_profile_tool import UpdateUserProfileTool


class FakeTimelineSearch:
    def __init__(self, results_by_user=None, recent_by_user=None):
        self.calls: list[dict] = []
        # Optional per-scope fixtures for dual-scope (speaker+channel) tests;
        # None keeps the legacy single-result behavior.
        self.results_by_user = results_by_user
        self.recent_by_user = recent_by_user

    async def search(
        self,
        user_id,
        query_embedding,
        limit,
        *,
        query_text=None,
        topic_filter=None,
        since_ts=None,
        until_ts=None,
    ):
        self.calls.append({
            "method": "search",
            "user_id": user_id,
            "query_embedding": query_embedding,
            "limit": limit,
            "query_text": query_text,
            "topic_filter": topic_filter,
            "since_ts": since_ts,
            "until_ts": until_ts,
        })
        if self.results_by_user is not None:
            return list(self.results_by_user.get(user_id, []))
        return [
            {
                "summary_id": "sum-1",
                "user_id": user_id,
                "summary": "User thích phim tâm lý.",
                "topic": "interest",
                "importance": 3,
                "created_at": 1718360000.0,
            }
        ]

    async def get_recent(self, user_id, limit, *, since_ts=None, until_ts=None):
        self.calls.append({
            "method": "get_recent",
            "user_id": user_id,
            "limit": limit,
            "since_ts": since_ts,
            "until_ts": until_ts,
        })
        if self.recent_by_user is not None:
            return list(self.recent_by_user.get(user_id, []))
        return [
            {
                "summary_id": "sum-2",
                "user_id": user_id,
                "summary": "User nói chào Bé Bảy.",
                "importance": 4,
                "created_at": 1718370000.0,
            }
        ]


class FakeEmbeddingService:
    def __init__(self):
        self.calls: list[str] = []

    async def get_embedding(self, text):
        self.calls.append(text)
        return [0.1] * 384


class FakeProfileStore:
    def __init__(self):
        self.append_calls: list[dict] = []

    async def append_raw(self, user_id, section, content, source_memory_id=None):
        self.append_calls.append(
            {
                "user_id": user_id,
                "section": section,
                "content": content,
                "source_memory_id": source_memory_id,
            }
        )
        return content != "duplicate"

    async def read_raw(self, user_id):
        return f"## Thông tin cơ bản\n- Danh xưng: {user_id}\n"

    async def read_section(self, user_id, section):
        if section == "interest":
            return ["Phim tâm lý", "Cờ vua"]
        return []


# ------------------------------------------------------- v3 behavior matrix
# query + days_back -> hybrid, time-filtered | query only -> hybrid, all-time
# days_back only -> timeline in range | neither -> recent


@pytest.mark.asyncio
async def test_search_memory_tool_query_only_resolves_hybrid_all_time():
    """query with no days_back: v3 always hybrid (KNN+BM25), no time filter."""
    store = FakeTimelineSearch()
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", query="phim")
    assert "Tìm thấy 1 ký ức" in result
    assert "User thích phim tâm lý" in result
    assert "summary_id=sum-1" in result
    last_call = store.calls[-1]
    assert last_call["method"] == "search"
    # qwen3 instruct query prefix applied
    assert embeddings.calls[-1] == f"{Config.EMBEDDING_QUERY_PREFIX}phim"
    # v3: query always triggers hybrid (query_text set) — no more semantic-only mode
    assert last_call["query_text"] == "phim"
    assert last_call["since_ts"] is None


@pytest.mark.asyncio
async def test_search_memory_tool_query_with_days_back_applies_since_ts():
    store = FakeTimelineSearch()
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", query="phim", days_back=7)
    assert "Tìm thấy 1 ký ức" in result
    last_call = store.calls[-1]
    assert last_call["method"] == "search"
    assert last_call["query_text"] == "phim"
    assert last_call["since_ts"] is not None
    assert embeddings.calls[-1] == f"{Config.EMBEDDING_QUERY_PREFIX}phim"


@pytest.mark.asyncio
async def test_search_memory_tool_days_back_only_uses_timeline():
    """days_back alone (no query): timeline path, sorted new->old, no embedding call."""
    store = FakeTimelineSearch()
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", days_back=7)
    assert "Tìm thấy 1 ký ức" in result
    last_call = store.calls[-1]
    assert last_call["method"] == "get_recent"
    assert last_call["since_ts"] is not None
    assert embeddings.calls == []


@pytest.mark.asyncio
async def test_search_memory_tool_neither_param_uses_recent():
    store = FakeTimelineSearch()
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345")
    assert "Tìm thấy 1 ký ức" in result
    assert "User nói chào Bé Bảy" in result
    assert "summary_id=sum-2" in result
    last_call = store.calls[-1]
    assert last_call["method"] == "get_recent"
    assert last_call["since_ts"] is None


def test_legacy_params_and_dead_code_removed():
    """v3 (fix B1) drops mode/topic/hours/days — regression guard."""
    tool = SearchMemoryTool()
    props = tool.parameters_schema["properties"]
    for removed in ("mode", "topic", "hours", "days"):
        assert removed not in props
    assert not hasattr(SearchMemoryTool, "_VALID_TOOL_MODES")
    assert not hasattr(SearchMemoryTool, "_resolve_hours")
    assert not hasattr(SearchMemoryTool, "_timeline_mode")


@pytest.mark.asyncio
async def test_search_memory_tool_dual_scope_channel():
    """A channel turn searches BOTH speaker and channel scopes, deduped."""
    shared_doc = {
        "summary_id": "sum-user",
        "user_id": "12345",
        "summary": "User kể về dự án X.",
        "score": 0.4,
        "created_at": 1718360000.0,
    }
    store = FakeTimelineSearch(results_by_user={
        "12345": [shared_doc],
        "99999": [
            {
                "summary_id": "sum-chan",
                "user_id": "99999",
                "summary": "Cả kênh bàn về game Y.",
                "score": 0.5,
                "created_at": 1718361000.0,
            },
            shared_doc,  # duplicate across scopes — must appear once
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(
        user_id="12345", channel_id="99999", query="dự án",
    )

    searched = [c["user_id"] for c in store.calls if c["method"] == "search"]
    assert searched == ["12345", "99999"]
    assert "Tìm thấy 2 ký ức" in result
    assert "sum-user" in result and "sum-chan" in result


@pytest.mark.asyncio
async def test_search_memory_tool_channel_id_equal_user_searches_once():
    store = FakeTimelineSearch()
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    await tool.execute(user_id="12345", channel_id="12345", query="phim")
    assert len([c for c in store.calls if c["method"] == "search"]) == 1


@pytest.mark.asyncio
async def test_search_memory_tool_invalid_channel_id_ignored():
    """A bad channel_id must not fail the recall — it just drops that scope."""
    store = FakeTimelineSearch()
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(
        user_id="12345", channel_id="général", query="phim",
    )
    assert "Tìm thấy 1 ký ức" in result
    assert [c["user_id"] for c in store.calls if c["method"] == "search"] == ["12345"]


@pytest.mark.asyncio
async def test_search_memory_tool_no_results_message():
    """Nothing past the cosine gate → explicit message, never empty output."""
    store = FakeTimelineSearch(results_by_user={})
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", query="phim")
    assert result == "Không tìm thấy ký ức phù hợp."


@pytest.mark.asyncio
async def test_search_memory_tool_shows_relevance_score():
    """KNN distance is surfaced to the LLM as relevance = 1 - score."""
    store = FakeTimelineSearch(results_by_user={
        "12345": [
            {
                "summary_id": "sum-1",
                "user_id": "12345",
                "summary": "User thích phim tâm lý.",
                "score": 0.4,
                "created_at": 1718360000.0,
            }
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", query="phim")
    assert "relevance=0.60" in result


@pytest.mark.asyncio
async def test_search_memory_tool_bm25_only_shows_match_label():
    """A BM25-only doc (WITHSCORES → `_score`, no KNN `score`) renders
    `match=bm25`, not a cosine `relevance=` — the B3-fix survivor the LLM
    must be able to treat with caution. Guards the `_score` field name
    (parse_results writes `_score`, not `_bm25_score`)."""
    store = FakeTimelineSearch(results_by_user={
        "12345": [
            {
                "summary_id": "lex-hit",
                "user_id": "12345",
                "summary": "Bàn về AMD Ryzen.",
                "_score": 1.5,
                "created_at": 1718360000.0,
            }
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", query="phim")
    assert "match=bm25" in result
    assert "relevance=" not in result


@pytest.mark.asyncio
async def test_search_memory_tool_recent_dual_scope_sorted():
    """Recent (no query/days_back) merges both scopes newest-first."""
    store = FakeTimelineSearch(recent_by_user={
        "12345": [
            {"summary_id": "old", "summary": "Chuyện cũ.", "created_at": 1718300000.0},
        ],
        "99999": [
            {"summary_id": "new", "summary": "Chuyện mới ở kênh.", "created_at": 1718400000.0},
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", channel_id="99999")
    assert result.index("summary_id=new") < result.index("summary_id=old")


# ------------------------------------------------------------- P3.3 widen-on-empty


@pytest.mark.asyncio
async def test_search_memory_tool_widens_days_back_then_labels_result():
    """First round (days_back) empty -> widen x3 -> label the wider result."""

    class WideningStore:
        def __init__(self):
            self.since_ts_seen: list[float | None] = []

        async def get_recent(self, user_id, limit, *, since_ts=None, until_ts=None):
            self.since_ts_seen.append(since_ts)
            if len(self.since_ts_seen) < 2:
                return []
            return [{
                "summary_id": "wide-hit",
                "summary": "Tin cũ hơn.",
                "created_at": 1718300000.0,
            }]

    store = WideningStore()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=FakeEmbeddingService())

    result = await tool.execute(user_id="12345", days_back=2)

    assert len(store.since_ts_seen) == 2
    assert "wide-hit" in result
    assert "không thấy trong 2 ngày" in result
    assert "kết quả từ 6 ngày" in result


@pytest.mark.asyncio
async def test_search_memory_tool_widens_to_all_time_when_still_empty():
    """All 3 rounds empty -> plain no-result message, no crash."""

    class EmptyStore:
        def __init__(self):
            self.calls = 0

        async def get_recent(self, user_id, limit, *, since_ts=None, until_ts=None):
            self.calls += 1
            return []

    store = EmptyStore()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=FakeEmbeddingService())

    result = await tool.execute(user_id="12345", days_back=2)

    assert store.calls == 3
    assert result == "Không tìm thấy ký ức phù hợp."


@pytest.mark.asyncio
async def test_search_memory_tool_query_with_days_back_widens_on_empty():
    """Widen ladder also applies to the query+days_back hybrid branch."""

    class WideningSemanticStore:
        def __init__(self):
            self.since_ts_seen: list[float | None] = []

        async def search(self, user_id, query_embedding, limit, *, query_text=None,
                          topic_filter=None, since_ts=None, until_ts=None):
            self.since_ts_seen.append(since_ts)
            if len(self.since_ts_seen) < 3:
                return []
            return [{"summary_id": "wide-hit", "summary": "Tin cũ.", "created_at": 1718300000.0}]

    store = WideningSemanticStore()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=FakeEmbeddingService())

    result = await tool.execute(user_id="12345", query="phim", days_back=2)

    assert len(store.since_ts_seen) == 3
    assert "wide-hit" in result
    assert "không thấy trong 2 ngày" in result
    assert "kết quả từ toàn bộ" in result


# --------------------------------------------------------------------- helpers


def test_since_ts_from_days_back_uses_vn_calendar():
    ts = SearchMemoryTool._since_ts_from_days_back(1)
    expected = (vn_now() - timedelta(days=1)).timestamp()
    assert abs(ts - expected) < 2


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        (0, None),
        (-3, None),
        ("abc", None),
        (5, 5),
        ("7", 7),
    ],
)
def test_positive_int_or_none(value, expected):
    assert SearchMemoryTool._positive_int_or_none(value) == expected


def test_days_ago_display_uses_period_end_over_created_at(monkeypatch):
    import twin.shared.tools.modules.memory.search_memory_tool as tool_module

    fixed_now = datetime(2026, 7, 3, 12, 0, tzinfo=VN_TZ)
    monkeypatch.setattr(tool_module, "vn_now", lambda: fixed_now)

    four_days_ago = datetime(2026, 6, 29, 15, 30, tzinfo=VN_TZ)
    memory = {
        "period_end": four_days_ago.timestamp(),
        "created_at": fixed_now.timestamp(),  # must be ignored — period_end wins
    }
    display = SearchMemoryTool._days_ago_display(memory)
    assert display == "4 ngày trước — 29/06"


def test_days_ago_display_falls_back_to_created_at(monkeypatch):
    import twin.shared.tools.modules.memory.search_memory_tool as tool_module

    fixed_now = datetime(2026, 7, 3, 12, 0, tzinfo=VN_TZ)
    monkeypatch.setattr(tool_module, "vn_now", lambda: fixed_now)

    two_days_ago = datetime(2026, 7, 1, 9, 0, tzinfo=VN_TZ)
    memory = {"created_at": two_days_ago.timestamp()}
    display = SearchMemoryTool._days_ago_display(memory)
    assert display == "2 ngày trước — 01/07"


def test_days_ago_display_today_when_same_calendar_day(monkeypatch):
    import twin.shared.tools.modules.memory.search_memory_tool as tool_module

    fixed_now = datetime(2026, 7, 3, 23, 0, tzinfo=VN_TZ)
    monkeypatch.setattr(tool_module, "vn_now", lambda: fixed_now)

    earlier_today = datetime(2026, 7, 3, 8, 0, tzinfo=VN_TZ)
    memory = {"period_end": earlier_today.timestamp()}
    display = SearchMemoryTool._days_ago_display(memory)
    assert display == "hôm nay — 03/07"


def test_days_ago_display_none_when_no_timestamp():
    assert SearchMemoryTool._days_ago_display({}) is None


@pytest.mark.asyncio
async def test_update_profile_tool_appends_valid_section():
    store = FakeProfileStore()
    tool = UpdateUserProfileTool(profile_store=store)

    result = await tool.execute(
        user_id="123",
        section="interest",
        content="Thích phim tâm lý",
        source_memory_id="mem-1",
    )

    assert result.startswith("Đã cập nhật")
    assert store.append_calls == [
        {
            "user_id": "123",
            "section": "interest",
            "content": "Thích phim tâm lý",
            "source_memory_id": "mem-1",
        }
    ]


@pytest.mark.asyncio
async def test_update_profile_tool_rejects_invalid_section():
    store = FakeProfileStore()
    tool = UpdateUserProfileTool(profile_store=store)

    result = await tool.execute(user_id="123", section="bad", content="x")

    assert result.startswith("Lỗi: section")
    assert store.append_calls == []


@pytest.mark.asyncio
async def test_get_profile_tool_reads_raw_and_section():
    store = FakeProfileStore()
    tool = GetProfileTool(profile_store=store)

    raw = await tool.execute(user_id="123")
    section = await tool.execute(user_id="123", section="interest")

    assert "## Thông tin cơ bản" in raw
    assert "- Phim tâm lý" in section
    assert "- Cờ vua" in section


# ------------------------------------------------- #32 global top-K ranking


@pytest.mark.asyncio
async def test_search_memory_tool_global_ranking_channel_outranks_user():
    """Dual-scope semantic recall is globally ranked by relevance (#32):
    a stronger channel hit (score 0.1 -> sim 0.9) outranks a weaker user
    hit (score 0.5 -> sim 0.5) instead of user-first truncation."""
    store = FakeTimelineSearch(results_by_user={
        "12345": [
            {
                "summary_id": "weak-user",
                "user_id": "12345",
                "summary": "Chuyện user mờ nhạt.",
                "score": 0.5,
                "created_at": 1718360000.0,
            },
        ],
        "99999": [
            {
                "summary_id": "strong-chan",
                "user_id": "99999",
                "summary": "Chuyện kênh rất khớp.",
                "score": 0.1,
                "created_at": 1718361000.0,
            },
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", channel_id="99999", query="khớp")

    assert "Tìm thấy 2 ký ức" in result
    assert result.index("summary_id=strong-chan") < result.index("summary_id=weak-user")


@pytest.mark.asyncio
async def test_search_memory_tool_global_ranking_respects_limit():
    """limit=1 across two scopes returns the single globally-best hit."""
    store = FakeTimelineSearch(results_by_user={
        "12345": [
            {"summary_id": "weak-user", "summary": "Mờ.", "score": 0.5,
             "created_at": 1718360000.0},
        ],
        "99999": [
            {"summary_id": "strong-chan", "summary": "Khớp.", "score": 0.1,
             "created_at": 1718361000.0},
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(
        user_id="12345", channel_id="99999", query="khớp", limit=1,
    )

    assert "Tìm thấy 1 ký ức" in result
    assert "strong-chan" in result
    assert "weak-user" not in result


@pytest.mark.asyncio
async def test_search_memory_tool_recent_sorts_updated_old_first():
    """Recent branch uses content time (#33): an old-created summary with a
    fresh period_end (merged today) outranks a newer-created legacy doc."""
    store = FakeTimelineSearch(recent_by_user={
        "12345": [
            {"summary_id": "legacy", "summary": "Tạo sau, không period.",
             "created_at": 1718400000.0},
        ],
        "99999": [
            {"summary_id": "updated", "summary": "Tạo trước, vừa merge.",
             "created_at": 1718300000.0, "period_end": 1718500000.0},
        ],
    })
    embeddings = FakeEmbeddingService()
    tool = SearchMemoryTool(timeline_summary_store=store, embedding_service=embeddings)

    result = await tool.execute(user_id="12345", channel_id="99999")

    assert result.index("summary_id=updated") < result.index("summary_id=legacy")
