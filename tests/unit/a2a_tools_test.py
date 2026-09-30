"""Unit tests for the two A2A cross-agent tools (mocked transports)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twin.shared.tools.approval_context import (
    ApprovalRequestContext,
    clear_current_approval_context,
    set_current_approval_context,
)
from twin.shared.tools.modules.a2a.march7_snapshot_tool import March7SnapshotTool
from twin.shared.tools.modules.a2a.request_consolidation_tool import (
    RequestConsolidationTool,
)


@pytest.fixture
def approval_context():
    set_current_approval_context(
        ApprovalRequestContext(
            platform="test",
            user_id="123",
            conversation_id="456",
            channel_id="456",
        )
    )
    yield
    clear_current_approval_context()


@pytest.fixture(autouse=True)
def _clean_approval_context():
    clear_current_approval_context()
    yield
    clear_current_approval_context()


def _snapshot(n: int = 3) -> list[dict]:
    return [
        {
            "entry_id": f"e{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"tin {i}",
            "author_name": "Hòa",
        }
        for i in range(n)
    ]


class TestMarch7SnapshotTool:
    def test_name(self) -> None:
        assert March7SnapshotTool().name == "march7_snapshot"

    async def test_invalid_user_id(self, approval_context) -> None:
        out = await March7SnapshotTool().execute(user_id="abc")
        assert "không hợp lệ" in out

    async def test_missing_context_fails(self) -> None:
        out = await March7SnapshotTool().execute(user_id="123")
        assert "thiếu ngữ cảnh xác thực" in out

    async def test_foreign_user_denied(self, approval_context) -> None:
        out = await March7SnapshotTool().execute(user_id="333")
        assert "từ chối" in out

    async def test_empty_snapshot(self, approval_context) -> None:
        tool = March7SnapshotTool(march7_url="http://march7:8000")
        with patch(
            "twin.shared.tools.modules.a2a.march7_snapshot_tool.A2AClient"
        ) as mock_client_cls:
            client = mock_client_cls.return_value
            client.send_data_task = AsyncMock(return_value={"snapshot": []})
            client.close = AsyncMock()
            out = await tool.execute(user_id="123")
        assert "trống" in out
        client.send_data_task.assert_called_once_with(
            skill="get_snapshot", session_id="123"
        )
        _, kwargs = mock_client_cls.call_args
        assert kwargs["actor"] == "evernight"

    async def test_formats_entries(self, approval_context) -> None:
        tool = March7SnapshotTool(march7_url="http://march7:8000")
        with patch(
            "twin.shared.tools.modules.a2a.march7_snapshot_tool.A2AClient"
        ) as mock_client_cls:
            client = mock_client_cls.return_value
            client.send_data_task = AsyncMock(
                return_value={"snapshot": _snapshot(3)}
            )
            client.close = AsyncMock()
            out = await tool.execute(user_id="123", limit=2)
        assert "2/3" in out
        assert "tin 1" in out and "tin 2" in out
        assert "tin 0" not in out

    async def test_a2a_failure(self, approval_context) -> None:
        tool = March7SnapshotTool(march7_url="http://march7:8000")
        with patch(
            "twin.shared.tools.modules.a2a.march7_snapshot_tool.A2AClient"
        ) as mock_client_cls:
            client = mock_client_cls.return_value
            client.send_data_task = AsyncMock(side_effect=ConnectionError("down"))
            client.close = AsyncMock()
            out = await tool.execute(user_id="123")
        assert "không kết nối được March7" in out


class TestRequestConsolidationTool:
    def test_name(self) -> None:
        tool = RequestConsolidationTool(memory_manager=MagicMock())
        assert tool.name == "request_consolidation"

    async def test_invalid_user_id(self, approval_context) -> None:
        tool = RequestConsolidationTool(memory_manager=MagicMock())
        out = await tool.execute(user_id="abc")
        assert "không hợp lệ" in out

    async def test_no_manager(self, approval_context) -> None:
        out = await RequestConsolidationTool(memory_manager=None).execute(
            user_id="123"
        )
        assert "chưa sẵn sàng" in out

    async def test_missing_context_fails(self) -> None:
        manager = MagicMock()
        manager.consolidate_scope = AsyncMock(return_value={"status": "ok"})
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="123"
        )
        assert "thiếu ngữ cảnh xác thực" in out
        manager.consolidate_scope.assert_not_called()

    async def test_foreign_user_denied(self, approval_context) -> None:
        manager = MagicMock()
        manager.consolidate_scope = AsyncMock(return_value={"status": "ok"})
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="333"
        )
        assert "từ chối" in out
        manager.consolidate_scope.assert_not_called()

    async def test_foreign_channel_denied(self, approval_context) -> None:
        manager = MagicMock()
        manager.consolidate_scope = AsyncMock(return_value={"status": "ok"})
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="123", channel_id="444"
        )
        assert "từ chối" in out
        manager.consolidate_scope.assert_not_called()

    async def test_ok_user_scope(self, approval_context) -> None:
        manager = MagicMock()
        manager.consolidate_scope = AsyncMock(
            return_value={"status": "ok", "entry_ids": ["a", "b"]}
        )
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="123"
        )
        assert "Đã nén T1 của user 123: 2 tin" in out
        manager.consolidate_scope.assert_called_once_with("user", "123")

    async def test_ok_channel_scope(self, approval_context) -> None:
        manager = MagicMock()
        manager.consolidate_scope = AsyncMock(
            return_value={"status": "ok", "entry_ids": ["a"]}
        )
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="123", channel_id="456"
        )
        assert "kênh 456" in out
        manager.consolidate_scope.assert_called_once_with("channel", "456")

    async def test_skipped_and_failed(self, approval_context) -> None:
        manager = MagicMock()
        manager.consolidate_scope = AsyncMock(return_value={"status": "skipped"})
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="123"
        )
        assert "Bỏ qua" in out

        manager.consolidate_scope = AsyncMock(return_value={"status": "failed"})
        out = await RequestConsolidationTool(memory_manager=manager).execute(
            user_id="123"
        )
        assert "thất bại" in out
