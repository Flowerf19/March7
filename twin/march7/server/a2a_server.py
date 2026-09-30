"""March7 A2A Server."""
import asyncio
import logging
from typing import AsyncIterator

from twin.shared.a2a.auth import skill_peers_for_agent
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.types import A2AMessage, Part, AgentCard
from twin.march7.agent import March7Agent

logger = logging.getLogger(__name__)


class March7A2AHandler:
    def __init__(self, agent: March7Agent):
        self.agent = agent

    async def handle_chat_task(self, params: dict) -> AsyncIterator[A2AMessage]:
        session_id = params.get("sessionId", "unknown")
        msg = params.get("message", {})
        parts = msg.get("parts", [])
        content = ""
        for p in parts:
            if p.get("type") == "text":
                content += p.get("text", "")

        user_id = session_id
        logger.info(f"March7 handling chat for user {user_id}")

        yield A2AMessage(
            role="agent",
            parts=[Part(type="text", text="...")],
        )

        try:
            response = await self.agent.handle_chat(user_id=user_id, content=content)
            yield A2AMessage(
                role="agent",
                parts=[Part(type="text", text=response)],
            )
        except Exception as e:
            logger.exception("Chat handler error")
            yield A2AMessage(
                role="agent",
                parts=[Part(type="text", text=f"Error: {e}")],
            )

    async def handle_get_snapshot(self, params: dict) -> AsyncIterator[A2AMessage]:
        session_id = params.get("sessionId", "unknown")
        snapshot = await self.agent.handle_get_snapshot(user_id=session_id)
        yield A2AMessage(
            role="agent",
            parts=[Part(type="data", data={"snapshot": snapshot})],
        )

    async def handle_clear_session(self, params: dict) -> AsyncIterator[A2AMessage]:
        session_id = params.get("sessionId", "unknown")
        success = await self.agent.handle_clear_session(user_id=session_id)
        yield A2AMessage(
            role="agent",
            parts=[Part(type="data", data={"success": success})],
        )


def validate_march7_session(skill: str, session_id: object) -> str | None:
    """Validate the session scope for a March7 skill.

    Returns an error message when invalid, else None. Memory skills only ever
    address one Discord user id; anything else-shaped is rejected instead of
    being trusted as a lookup key.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return f"invalid sessionId for skill '{skill}': session is required"
    if skill in {"get_snapshot", "clear_session"} and not session_id.isdigit():
        return f"invalid sessionId for skill '{skill}': user id must be numeric"
    return None


def start_server(
    agent: March7Agent,
    host="0.0.0.0",
    port=8000,
    health_probe=None,
    *,
    shared_secret: str | None = None,
    skill_peers: dict[str, frozenset] | None = None,
) -> A2AServer:
    handler = March7A2AHandler(agent)
    server = A2AServer(
        agent_card=agent.get_agent_card(),
        skill_handlers={
            "chat": handler.handle_chat_task,
            "get_snapshot": handler.handle_get_snapshot,
            "clear_session": handler.handle_clear_session,
        },
        host=host,
        port=port,
        health_probe=health_probe,
        agent_name="march7",
        shared_secret=shared_secret,
        skill_peers=skill_peers if skill_peers is not None else skill_peers_for_agent("march7"),
        session_validator=validate_march7_session,
    )
    return server
