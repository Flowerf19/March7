"""RequestConsolidationTool - March7 triggers Evernight consolidation on demand."""
from __future__ import annotations

import logging
from typing import Any, Optional

from twin.shared.tools.approval_context import get_current_approval_context
from twin.shared.tools.registry.base import BaseTool

logger = logging.getLogger(__name__)


class RequestConsolidationTool(BaseTool):
    """Manually trigger T1 consolidation via Evernight (A2A).

    March7-only. Auto-consolidation (2000 từ / idle 15 phút) already covers
    the normal path — this is only for explicit user requests ("lưu lại đi",
    "nén memory đi"). Reuses SharedMemoryManager.consolidate_scope, so trim
    and archive behave exactly like the automatic flow.
    """

    def __init__(self, memory_manager: Optional[Any] = None):
        self.memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "request_consolidation"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID (số) cần nén T1.",
                },
                "channel_id": {
                    "type": "string",
                    "description": (
                        "ID kênh chung — chỉ truyền khi muốn nén T1 của kênh "
                        "(lấy 'Platform channel ID' từ system prompt)."
                    ),
                },
            },
            "required": ["user_id"],
        }

    async def execute(
        self, user_id: str, channel_id: Optional[str] = None
    ) -> str:
        user_id = str(user_id or "").strip()
        channel_id = str(channel_id or "").strip() or None
        if not user_id.isdigit():
            return f"Lỗi: user_id '{user_id}' không hợp lệ. user_id phải là số ID Discord."
        if self.memory_manager is None:
            return "Lỗi: memory manager chưa sẵn sàng."
        context = get_current_approval_context()
        if context is None or not context.user_id:
            return "Lỗi: thiếu ngữ cảnh xác thực — không thể xác định phạm vi cần nén."
        if user_id != str(context.user_id):
            return (
                f"Lỗi: từ chối nén T1 của user {user_id} — chỉ được nén đúng "
                f"user đang trò chuyện ({context.user_id})."
            )

        if channel_id:
            if not channel_id.isdigit():
                return f"Lỗi: channel_id '{channel_id}' không hợp lệ."
            ctx_channel = context.channel_id or context.conversation_id
            if not ctx_channel or channel_id != str(ctx_channel):
                return (
                    f"Lỗi: từ chối nén T1 của kênh {channel_id} — chỉ được nén đúng "
                    f"kênh đang trò chuyện ({ctx_channel or 'không rõ'})."
                )
            scope, scope_id = "channel", channel_id
        else:
            scope, scope_id = "user", user_id

        try:
            result = await self.memory_manager.consolidate_scope(scope, scope_id)
        except Exception as exc:
            logger.warning(
                "request_consolidation failed scope=%s/%s: %s", scope, scope_id, exc
            )
            return "Lỗi: consolidate thất bại (Evernight không phản hồi?)."

        status = result.get("status")
        if status == "ok":
            count = len(result.get("entry_ids") or [])
            where = f"kênh {scope_id}" if scope == "channel" else f"user {scope_id}"
            return f"Đã nén T1 của {where}: {count} tin → T2/T3, T1 đã trim gọn."
        if status == "skipped":
            return "Bỏ qua: không có gì mới để nén."
        return f"Consolidate thất bại (status={status or 'unknown'})."
