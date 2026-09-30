"""A2A HTTP client for March7 gateway calls to Evernight."""

from __future__ import annotations

import logging

from twin.shared.a2a.client import A2AClient
from twin.shared.observability import a2a_parent_headers
from twin.shared.observability.langsmith import traceable

logger = logging.getLogger(__name__)


class EvernightClient:
    """A2A HTTP client that communicates with the Evernight agent."""

    def __init__(
        self,
        base_url: str = "http://evernight:8001",
        timeout: float = 120.0,
        *,
        actor: str = "march7",
        secret: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = A2AClient(
            base_url=self.base_url, timeout=timeout, actor=actor, secret=secret
        )

    @traceable(name="a2a.evernight_chat", run_type="chain", tags=["a2a", "evernight", "chat"])
    async def send_chat(self, user_id: str, content: str) -> str:
        """Send a chat message to Evernight and return the response text."""
        try:
            return await self._client.send_text_task(
                skill="chat",
                session_id=user_id,
                text=content,
                trace_parent=a2a_parent_headers(),
            )
        except Exception:
            logger.exception("Failed to send chat to Evernight at %s", self.base_url)
            return "Xin lỗi, không thể kết nối đến Evernight."

    @traceable(name="a2a.evernight_consolidation", run_type="chain", tags=["a2a", "evernight", "consolidation"])
    async def request_consolidation(self, payload: dict) -> dict:
        """Ask Evernight to consolidate a SUMMARY_REQUESTED payload."""
        session_id = payload.get("scope_id") or "unknown"
        try:
            return await self._client.send_data_task(
                skill="consolidate_discussion",
                session_id=session_id,
                params={
                    "payload": {
                        **payload,
                        "_langsmith_parent": a2a_parent_headers(),
                    },
                },
            )
        except Exception as exc:
            logger.exception(
                "Failed to request consolidation from Evernight at %s",
                self.base_url,
            )
            return {
                "status": "failed",
                "scope": payload.get("scope"),
                "scope_id": payload.get("scope_id"),
                "reason": f"a2a_error: {exc}",
                "retry_after_seconds": 600,
            }

    async def health_check(self) -> bool:
        """Check Evernight health via /.well-known/agent.json."""
        try:
            return await self._client.get_agent_card() is not None
        except Exception:
            logger.warning("Evernight health check failed at %s", self.base_url)
            return False

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._client.close()
