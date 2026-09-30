"""Consolidation client - sends consolidation requests to Evernight via A2A."""
from __future__ import annotations

import logging
from typing import Any

from twin.shared.a2a.client import A2AClient
from twin.shared.observability import a2a_parent_headers
from twin.shared.observability.langsmith import traceable

logger = logging.getLogger(__name__)


class ConsolidationClient:
    """
    Client for sending consolidation requests to Evernight.
    
    This replaces the old local consolidation pipeline with A2A calls.
    """

    def __init__(
        self,
        evernight_url: str,
        timeout: float = 300.0,
        *,
        actor: str = "march7",
        secret: str | None = None,
    ):
        self.a2a_client = A2AClient(
            base_url=evernight_url, timeout=timeout, actor=actor, secret=secret
        )
        logger.info("ConsolidationClient initialized for %s", evernight_url)

    @traceable(name="a2a.consolidate_scope", run_type="chain", tags=["a2a", "consolidation"])
    async def consolidate_scope(
        self,
        scope: str,
        scope_id: str,
        reason: str = "auto",
        max_messages: int = 200,
        entries: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Send consolidation request to Evernight.

        ``entries`` carries THIS agent's own T1 snapshot so Evernight can
        consolidate the shipped messages instead of reading its (empty for our
        scopes) local T1.

        Returns dict with status, timeline_summary, profile_updates, etc.
        """
        logger.info(
            "ConsolidationClient: requesting consolidation scope=%s scope_id=%s reason=%s entries=%d",
            scope, scope_id, reason, len(entries or []),
        )

        try:
            result = await self.a2a_client.send_data_task(
                skill="consolidate_discussion",
                session_id=scope_id,
                params={
                    "payload": {
                        "scope": scope,
                        "scope_id": scope_id,
                        "reason": reason,
                        "max_messages": max_messages,
                        "entries": entries or [],
                        "_langsmith_parent": a2a_parent_headers(),
                    }
                },
            )
            
            logger.info(
                "ConsolidationClient: consolidation completed scope=%s/%s status=%s",
                scope, scope_id, result.get("status"),
            )
            return result
            
        except Exception as exc:
            logger.error(
                "ConsolidationClient: consolidation failed scope=%s/%s: %s",
                scope, scope_id, exc, exc_info=True,
            )
            return {
                "status": "failed",
                "scope": scope,
                "scope_id": scope_id,
                "error": str(exc),
            }

    async def close(self) -> None:
        """Close the A2A client session."""
        await self.a2a_client.close()
