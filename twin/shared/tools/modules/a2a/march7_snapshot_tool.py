"""March7SnapshotTool - Evernight reads March7's T1 via A2A get_snapshot."""
from __future__ import annotations

import logging
from typing import Any, Optional

from twin.shared.a2a.client import A2AClient
from twin.shared.config.settings import Config
from twin.shared.tools.approval_context import get_current_approval_context
from twin.shared.tools.registry.base import BaseTool

logger = logging.getLogger(__name__)

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50


class March7SnapshotTool(BaseTool):
    """Fetch March7's short-term (T1) context for a user over A2A.

    Evernight-only. Evernight must never read March7's Redis keys directly;
    this tool is the sanctioned bridge (A2A skill `get_snapshot`).
    """

    def __init__(
        self,
        march7_url: Optional[str] = None,
        *,
        actor: str = "evernight",
        secret: Optional[str] = None,
    ):
        self._march7_url = (march7_url or Config.MARCH7_URL).rstrip("/")
        self._actor = actor
        self._secret = secret

    @property
    def name(self) -> str:
        return "march7_snapshot"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID (số) cần xem T1 của March7.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Số tin gần nhất (mặc định 20, tối đa 50).",
                },
            },
            "required": ["user_id"],
        }

    async def execute(self, user_id: str, limit: int = _DEFAULT_LIMIT) -> str:
        user_id = str(user_id or "").strip()
        if not user_id.isdigit():
            return f"Lỗi: user_id '{user_id}' không hợp lệ. user_id phải là số ID Discord."
        context = get_current_approval_context()
        if context is None or not context.user_id:
            return "Lỗi: thiếu ngữ cảnh xác thực — không thể kiểm tra phạm vi snapshot."
        if user_id != str(context.user_id):
            return (
                f"Lỗi: từ chối đọc T1 của user {user_id} — chỉ được đọc đúng "
                f"user đang trò chuyện ({context.user_id})."
            )
        limit = max(1, min(int(limit or _DEFAULT_LIMIT), _MAX_LIMIT))

        client = A2AClient(
            base_url=self._march7_url,
            timeout=30.0,
            actor=self._actor,
            secret=self._secret,
        )
        try:
            result = await client.send_data_task(
                skill="get_snapshot",
                session_id=user_id,
            )
        except Exception as exc:
            logger.warning("march7_snapshot A2A failed user=%s: %s", user_id, exc)
            return "Lỗi: không kết nối được March7 qua A2A."
        finally:
            await client.close()

        snapshot = result.get("snapshot") or []
        if not snapshot:
            return f"March7 không còn T1 nào cho user {user_id} (trống hoặc đã consolidate)."
        return self._format(snapshot[-limit:], user_id, len(snapshot))

    @staticmethod
    def _format(entries: list[dict], user_id: str, total: int) -> str:
        lines = [f"T1 của March7 cho user {user_id} ({len(entries)}/{total} tin gần nhất):"]
        for e in entries:
            role = e.get("role") or "user"
            author = e.get("author_name") or e.get("author_id") or role
            content = str(e.get("content") or "")
            if len(content) > 300:
                content = content[:300] + "…"
            lines.append(f"- [{role}/{author}] {content}")
        return "\n".join(lines)
