"""DMClient — authenticated HTTP client from March7 to Evernight DMs.

Sends owner DMs and structured host-approval requests via the Evernight bot.
Every request is signed with A2A headers over the exact serialized body bytes;
the approval recipient is always the configured owner. March7 consumes the
returned grant and never mints approvals (it holds no approval key).
"""
from __future__ import annotations

import json as jsonlib
import logging
from typing import Any, Optional

import aiohttp

from twin.shared.a2a.auth import A2AAuthError, make_a2a_headers
from twin.shared.config.settings import Config
from twin.shared.tools.approval_context import (
    ApprovalDecision,
    ApprovalRequestContext,
)

logger = logging.getLogger(__name__)


class DMUnavailableError(Exception):
    """Raised when Evernight bot is not available for DM."""
    pass


# Backward compatibility alias
ApprovalDMUnavailableError = DMUnavailableError


def _resolve_secret(explicit: str | None) -> str | None:
    return explicit or getattr(Config, "A2A_SHARED_SECRET", None)


def _resolve_owner_id(explicit: str | int | None) -> int | None:
    raw = explicit if explicit not in (None, "") else getattr(
        Config, "EVERNIGHT_OWNER_USER_ID", None
    )
    if raw in (None, ""):
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _signed_post_parts(
    *, actor: str, path: str, payload: dict[str, Any], secret: str | None
) -> tuple[bytes, dict[str, str]]:
    """Serialize once and sign the exact bytes the server will verify."""
    body = jsonlib.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        **make_a2a_headers(actor, "POST", path, body, secret=secret),
    }
    return body, headers


def _context_from_native_message(message: object) -> ApprovalRequestContext:
    """Build neutral context from a legacy native message object."""
    channel = getattr(message, "channel", None)
    guild = getattr(message, "guild", None)
    author = getattr(message, "author", None)
    channel_name = str(channel) if channel is not None else None
    if channel is not None and getattr(channel, "name", None):
        channel_name = f"#{channel.name}"
    if guild is not None and channel_name:
        channel_name = f"{getattr(guild, 'name', guild)}/{channel_name}"

    channel_id = str(getattr(channel, "id", "")) if channel is not None else None
    return ApprovalRequestContext(
        platform="discord",
        user_id=str(getattr(author, "id", "")),
        conversation_id=channel_id,
        channel_id=channel_id,
        message_id=str(getattr(message, "id", "")),
        channel_name=channel_name,
        space_id=str(getattr(guild, "id", "")) if guild is not None else None,
        user_name=getattr(author, "display_name", None),
        native_message=message,
    )


def _coerce_int_if_numeric(value: str | int | None) -> str | int | None:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


class DMClient:
    """Authenticated HTTP client that sends DMs via the Evernight bot."""

    def __init__(
        self,
        evernight_url: str = "http://evernight:8001",
        timeout: float = 120.0,
        *,
        actor: str = "march7",
        secret: str | None = None,
        owner_user_id: str | int | None = None,
    ):
        self.base_url = evernight_url.rstrip("/")
        self._timeout = timeout
        self.actor = actor
        self._secret = secret
        self._owner_user_id = owner_user_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        return self._session

    def _post_signed(
        self, session: aiohttp.ClientSession, path: str, payload: dict[str, Any]
    ):
        url = f"{self.base_url}{path}"
        try:
            body, headers = _signed_post_parts(
                actor=self.actor,
                path=path,
                payload=payload,
                secret=_resolve_secret(self._secret),
            )
        except A2AAuthError as exc:
            raise DMUnavailableError(f"Cannot sign DM request: {exc}") from exc
        return session.post(url, data=body, headers=headers)

    def _owner_or_raise(self, caller_user_id: Any = None) -> int:
        owner = _resolve_owner_id(self._owner_user_id)
        if owner is None:
            raise DMUnavailableError("Owner user id is not configured")
        if caller_user_id not in (None, ""):
            try:
                asserted = int(str(caller_user_id).strip())
            except (TypeError, ValueError):
                asserted = None
            if asserted is not None and asserted != owner:
                logger.warning(
                    "DM recipient forced to configured owner (caller asked %s)",
                    caller_user_id,
                )
        return owner

    async def send_message(self, user_id: int, content: str) -> bool:
        """Send a general DM via Evernight bot (owner-only recipient)."""
        session = await self._get_session()
        target = self._owner_or_raise(user_id)
        payload = {"user_id": target, "content": content, "type": "message"}
        logger.info("Sending DM: user=%s content=%.60s", target, content)
        try:
            async with self._post_signed(session, "/dm", payload) as resp:
                if resp.status == 200:
                    logger.info("DM sent successfully")
                    return True
                if resp.status == 503:
                    raise DMUnavailableError(
                        f"Evernight bot at {self.base_url} is not available for DM"
                    )
                error_text = await resp.text()
                logger.error("DM request failed: status=%s body=%s",
                             resp.status, error_text[:500])
                raise DMUnavailableError(f"DM request failed: HTTP {resp.status}")
        except DMUnavailableError:
            raise
        except aiohttp.ClientConnectionError:
            raise DMUnavailableError(
                f"Cannot connect to Evernight at {self.base_url}"
            )
        except Exception:
            logger.exception("Unexpected error in DM request")
            raise DMUnavailableError("Unexpected error in DM request")

    async def request_authorization(
        self,
        *,
        action: str = "shell",
        command: str | None = None,
        shell: str | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
        actor: str | None = None,
        from_version: str | None = None,
        to_version: str | None = None,
        context: ApprovalRequestContext | None = None,
        channel_id: str | int | None = None,
        message_id: str | int | None = None,
        channel_name: str | None = None,
    ) -> ApprovalDecision:
        """Request a structured owner-approval decision from Evernight.

        Returns the issuer decision (approved/grant/reason). Raises
        DMUnavailableError only for transport/auth failures so callers can
        distinguish "ask elsewhere" from an explicit owner denial.
        """
        owner = _resolve_owner_id(self._owner_user_id)
        if owner is None:
            return ApprovalDecision.denied("owner_unknown")
        if context is not None:
            channel_id = channel_id or context.channel_id or context.conversation_id
            message_id = message_id or context.message_id
            channel_name = channel_name or context.channel_name
        payload: dict[str, Any] = {
            "user_id": owner,
            "type": "approval",
            "action": action,
            "actor": actor or self.actor,
            "command": command,
            "shell": shell,
            "cwd": cwd,
            "timeout": timeout,
            "from_version": from_version,
            "to_version": to_version,
            "channel_id": _coerce_int_if_numeric(channel_id),
            "message_id": _coerce_int_if_numeric(message_id),
            "channel_name": channel_name or "unknown",
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        session = await self._get_session()
        logger.info("Requesting approval DM: user=%s action=%s", owner, action)
        try:
            async with self._post_signed(session, "/dm", payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    decision = ApprovalDecision.from_dict(data)
                    # Never log the grant token itself.
                    logger.info(
                        "Approval DM result: %s reason=%.80s",
                        "approved" if decision.approved else "rejected",
                        decision.reason,
                    )
                    if decision.approved and not decision.grant:
                        return ApprovalDecision.denied("issuer returned no grant")
                    return decision
                if resp.status == 503:
                    raise DMUnavailableError(
                        f"Evernight bot at {self.base_url} is not available"
                    )
                error_text = await resp.text()
                logger.error("Approval DM failed: status=%s body=%s",
                             resp.status, error_text[:500])
                raise DMUnavailableError(
                    f"Approval DM request failed: HTTP {resp.status}"
                )
        except DMUnavailableError:
            raise
        except aiohttp.ClientConnectionError:
            raise DMUnavailableError(
                f"Cannot connect to Evernight at {self.base_url}"
            )
        except Exception:
            logger.exception("Unexpected error in approval DM request")
            raise DMUnavailableError("Unexpected error in approval DM request")

    async def request_approval(
        self,
        command: str,
        context: ApprovalRequestContext | None = None,
        user_id: int | None = None,
        *,
        original_message: object | None = None,
        channel_id: str | int | None = None,
        message_id: str | int | None = None,
        channel_name: str | None = None,
    ) -> bool:
        """Legacy bool approval request (recipient forced to owner)."""
        if context is None and original_message is not None:
            context = _context_from_native_message(original_message)
        if user_id is not None:
            self._owner_or_raise(user_id)
        decision = await self.request_authorization(
            action="shell",
            command=command,
            context=context,
            channel_id=channel_id,
            message_id=message_id,
            channel_name=channel_name,
        )
        return decision.approved

    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Backward compatibility alias
ApprovalDMClient = DMClient
