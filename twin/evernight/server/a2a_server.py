"""Evernight A2A Server.

Neutral orchestrator: A2A skill handlers plus owner DM/approval routes.
Concrete Discord I/O lives in the gateway adapter backend; approval grants
come from the owner-trusted issuer after an actual owner approval. This
module imports no platform UI code.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncIterator

from aiohttp import web

from twin.evernight.server.approval_issuer import (
    ApprovalDMBackend,
    decide_host_approval,
)
from twin.shared.a2a.auth import EVERNIGHT_DM_ALLOWED_PEERS
from twin.shared.a2a.server import A2AServer
from twin.shared.a2a.types import A2AMessage, Part
from twin.shared.config.settings import Config
from twin.shared.tools.approval_context import (
    ApprovalRequestContext,
    clear_current_approval_context,
    set_current_approval_context,
)

if TYPE_CHECKING:
    from twin.evernight.agent import EvernightAgent

logger = logging.getLogger(__name__)

_ALLOWED_SCOPES = frozenset({"user", "channel"})


def validate_evernight_session(skill: str, session_id: object) -> str | None:
    """Validate the session scope for an Evernight skill (shared server contract).

    Returns an error message when invalid, else None. All Evernight skills
    address a Discord snowflake; anything else-shaped is rejected instead of
    being trusted as a lookup key. No unsigned/dev bypass.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        return f"invalid sessionId for skill '{skill}': session is required"
    if not session_id.isdigit():
        return f"invalid sessionId for skill '{skill}': user id must be numeric"
    return None


def _normalize_scope_id(value: object) -> str | None:
    """Normalize a numeric scope id to str; None when malformed."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.strip().isdigit():
        return value.strip()
    return None


def _validate_entries(value: object) -> tuple[list[dict] | None, str | None]:
    """Validate shipped entries; preserves [] (only missing key means local read)."""
    if not isinstance(value, list):
        return None, "entries must be a list"
    for item in value:
        if not isinstance(item, dict):
            return None, "entries must be a list of objects"
    return value, None


def _validate_max_messages(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 1 <= value <= 5000:
        return value
    return None


def _fail_data(scope: object, scope_id: object, reason: str) -> A2AMessage:
    return A2AMessage(
        role="agent",
        parts=[Part(type="data", data={
            "status": "failed",
            "scope": scope,
            "scope_id": scope_id,
            "reason": reason,
        })],
    )


class EvernightA2AHandler:
    def __init__(
        self,
        agent: EvernightAgent,
        *,
        dm_backend: ApprovalDMBackend | None = None,
        owner_user_id: str | int | None = None,
    ):
        self.agent = agent
        self.dm_backend = dm_backend
        explicit = str(owner_user_id or "").strip()
        self.owner_user_id = (
            explicit or str(getattr(Config, "EVERNIGHT_OWNER_USER_ID", "") or "")
        ).strip()

    async def handle_chat_task(self, params: dict) -> AsyncIterator[A2AMessage]:
        session_id = params.get("sessionId", "")
        if validate_evernight_session("chat", session_id):
            yield A2AMessage(
                role="agent",
                parts=[Part(type="text", text="Error: invalid session")],
            )
            return
        msg = params.get("message", {})
        if not isinstance(msg, dict) or not isinstance(msg.get("parts", []), list):
            yield A2AMessage(
                role="agent",
                parts=[Part(type="text", text="Error: invalid message")],
            )
            return
        content = ""
        for p in msg.get("parts", []):
            if not isinstance(p, dict):
                yield A2AMessage(
                    role="agent",
                    parts=[Part(type="text", text="Error: invalid message")],
                )
                return
            if p.get("type") == "text":
                text = p.get("text", "")
                if not isinstance(text, str):
                    yield A2AMessage(
                        role="agent",
                        parts=[Part(type="text", text="Error: invalid message")],
                    )
                    return
                content += text

        user_id = str(session_id)
        logger.info(f"Evernight handling chat for user {user_id}")

        # Neutral validated context: only the middleware-verified peer route
        # and the validated session id are eligible. No Discord objects here;
        # agent/shared code stays platform-neutral. Cleaned on completion.
        set_current_approval_context(
            ApprovalRequestContext(platform="a2a", user_id=user_id)
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
        finally:
            clear_current_approval_context()

    async def handle_consolidate_task(self, params: dict) -> AsyncIterator[A2AMessage]:
        session_id = params.get("sessionId", "")
        if validate_evernight_session("consolidate", session_id):
            yield _fail_data("user", session_id, "invalid session")
            return
        reason = params.get("reason", "manual")
        max_messages = params.get("max_messages", 200)
        if not isinstance(reason, str) or not reason.strip():
            yield _fail_data("user", session_id, "invalid reason")
            return
        validated_max = _validate_max_messages(max_messages)
        if validated_max is None:
            yield _fail_data("user", session_id, "invalid max_messages")
            return

        scope_id = str(session_id)
        logger.info(f"Evernight handling consolidation via tool for user {scope_id}, reason={reason}")
        try:
            result = await self.agent.consolidate_via_tool(
                scope="user",
                scope_id=scope_id,
                reason=reason,
                max_messages=validated_max,
            )
            yield A2AMessage(
                role="agent",
                parts=[Part(type="data", data=result)],
            )
        except Exception as e:
            logger.exception("Consolidation handler error")
            yield A2AMessage(
                role="agent",
                parts=[Part(type="data", data={"status": "failed", "error": str(e)})],
            )

    async def handle_consolidate_discussion_task(self, params: dict) -> AsyncIterator[A2AMessage]:
        """A2A consolidation via tool - replaces old pipeline."""
        session_id = params.get("sessionId", "")
        if validate_evernight_session("consolidate_discussion", session_id):
            yield _fail_data(None, session_id, "invalid session")
            return
        payload = params.get("payload")
        if not isinstance(payload, dict):
            yield _fail_data(None, session_id, "invalid payload")
            return
        scope = payload.get("scope", "user")
        if scope not in _ALLOWED_SCOPES:
            yield _fail_data(scope, payload.get("scope_id"), "invalid scope")
            return
        scope_id = _normalize_scope_id(payload.get("scope_id"))
        if scope_id is None:
            yield _fail_data(scope, payload.get("scope_id"), "invalid scope_id")
            return
        if str(session_id) != scope_id:
            yield _fail_data(scope, scope_id, "mismatched scope/session")
            return
        reason = payload.get("reason", "discussion")
        if not isinstance(reason, str) or not reason.strip():
            yield _fail_data(scope, scope_id, "invalid reason")
            return
        validated_max = _validate_max_messages(payload.get("max_messages", 200))
        if validated_max is None:
            yield _fail_data(scope, scope_id, "invalid max_messages")
            return
        # Preserve shipped [] (empty snapshot stays []); only a missing key
        # means "read local T1". Malformed entries are rejected, not coerced.
        if "entries" in payload:
            entries, entries_err = _validate_entries(payload["entries"])
            if entries_err is not None:
                yield _fail_data(scope, scope_id, entries_err)
                return
        else:
            entries = None

        logger.info(
            "Evernight handling consolidate_discussion via tool scope=%s scope_id=%s entries=%s",
            scope,
            scope_id,
            len(entries) if entries is not None else "local",
        )

        try:
            result = await self.agent.consolidate_via_tool(
                scope=scope,
                scope_id=scope_id,
                reason=reason,
                max_messages=validated_max,
                entries=entries,
            )
            # Add scope info to result for compatibility
            result["scope"] = scope
            result["scope_id"] = scope_id
            yield A2AMessage(
                role="agent",
                parts=[Part(type="data", data=result)],
            )
        except Exception as e:
            logger.exception("Consolidation discussion handler error")
            yield A2AMessage(
                role="agent",
                parts=[Part(type="data", data={
                    "status": "failed",
                    "scope": scope,
                    "scope_id": scope_id,
                    "error": str(e),
                })],
            )

    def _peer(self, request: web.Request) -> str | None:
        """Authenticated A2A peer from the inherited auth middleware."""
        return request.get("a2a_peer")

    def _owner_or_deny(self, user_id: object) -> web.Response | None:
        if not self.owner_user_id:
            logger.warning("DM denied: owner is not configured")
            return web.json_response(
                {"success": False, "error": "owner_unknown"}, status=503
            )
        if str(user_id) != self.owner_user_id:
            logger.warning("DM rejected for non-owner user_id=%s", user_id)
            return web.json_response(
                {"success": False, "error": "Access denied: owner only"}, status=403
            )
        return None

    async def handle_dm(self, request: web.Request) -> web.Response:
        """Handle DM requests from authenticated A2A peers.

        Expects JSON body with `type: "message" | "approval"`. Approval
        requests carry exact execution fields and return a structured
        decision: {"approved": bool, "grant": str | None, "reason": str}.
        """
        peer = self._peer(request)
        if not peer:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        if body.get("type", "message") == "approval":
            return await self._handle_approval_request(body, peer)
        return await self._handle_message_request(body)

    async def handle_approval_dm(self, request: web.Request) -> web.Response:
        """Legacy approval endpoint (use /dm with type="approval" instead)."""
        peer = self._peer(request)
        if not peer:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not body.get("user_id") or not body.get("command"):
            return web.json_response(
                {"error": "Missing user_id or command"}, status=400
            )
        return await self._handle_approval_request(body, peer)

    async def _handle_message_request(self, body: dict) -> web.Response:
        user_id = body.get("user_id")
        content = body.get("content", "")
        if not user_id:
            return web.json_response({"success": False, "error": "Missing user_id"}, status=400)
        denied = self._owner_or_deny(user_id)
        if denied is not None:
            return denied
        if self.dm_backend is None:
            return web.json_response(
                {"success": False, "error": "Discord bot not available"}, status=503
            )
        try:
            await self.dm_backend.send_owner_dm(
                owner_user_id=self.owner_user_id, content=content
            )
            return web.json_response({"success": True})
        except Exception as exc:
            logger.exception("Error sending DM to owner")
            return web.json_response({"success": False, "error": str(exc)}, status=500)

    async def _handle_approval_request(self, body: dict, peer: str) -> web.Response:
        action = str(body.get("action") or "shell").strip()
        if action in ("self.update", "self_update"):
            if not body.get("from_version"):
                return web.json_response(
                    {"approved": False, "grant": None,
                     "reason": "missing from_version for approval"},
                    status=400,
                )
        elif not body.get("command"):
            return web.json_response(
                {"approved": False, "grant": None,
                 "reason": "missing command for approval"},
                status=400,
            )
        denied = self._owner_or_deny(body.get("user_id"))
        if denied is not None:
            # Normalize recipient denials to the structured decision shape.
            status = denied.status
            if status == 403:
                return web.json_response(
                    {"approved": False, "grant": None, "reason": "owner_only"},
                    status=403,
                )
            return web.json_response(
                {"approved": False, "grant": None, "reason": "owner_unknown"}
            )
        if self.dm_backend is None:
            return web.json_response(
                {"success": False, "error": "Discord bot not available"}, status=503
            )
        decision = await decide_host_approval(
            backend=self.dm_backend,
            owner_user_id=self.owner_user_id,
            peer=peer,
            body=body,
        )
        return web.json_response(decision.to_dict())


def _default_dm_backend(discord_bot: object) -> ApprovalDMBackend | None:
    if discord_bot is None:
        return None
    from gateway.adapters.discord.dm_delivery import DiscordDMDelivery

    return DiscordDMDelivery(discord_bot)


def start_server(
    agent: EvernightAgent,
    host="0.0.0.0",
    port=8001,
    discord_bot=None,
    owner_user_id: str | int | None = None,
    *,
    dm_backend: ApprovalDMBackend | None = None,
    shared_secret: str | None = None,
) -> A2AServer:
    resolved_owner = str(owner_user_id or "").strip() or (
        str(getattr(Config, "EVERNIGHT_OWNER_USER_ID", "") or "").strip()
    )
    handler = EvernightA2AHandler(
        agent,
        dm_backend=dm_backend if dm_backend is not None else _default_dm_backend(discord_bot),
        owner_user_id=resolved_owner,
    )
    server = A2AServer(
        agent_card=agent.get_agent_card(),
        skill_handlers={
            "chat": handler.handle_chat_task,
            "consolidate": handler.handle_consolidate_task,
            "consolidate_discussion": handler.handle_consolidate_discussion_task,
        },
        host=host,
        port=port,
        # Health reflects the Discord link (approval DMs depend on it); with no
        # Discord bot configured Evernight is A2A-only, so report healthy.
        health_probe=(lambda: discord_bot.is_ready()) if discord_bot is not None else None,
        agent_name="evernight",
        shared_secret=shared_secret,
        session_validator=validate_evernight_session,
        dm_allowed_peers=EVERNIGHT_DM_ALLOWED_PEERS,
    )
    # Patch: add the DM endpoints. The inherited app-level auth middleware
    # covers these routes too (authenticated peers only; public health/card).
    original_build_app = server.build_app

    def patched_build_app():
        app = original_build_app()
        app.router.add_post("/dm", handler.handle_dm)
        app.router.add_post("/approval_dm", handler.handle_approval_dm)
        return app

    server.build_app = patched_build_app
    return server
