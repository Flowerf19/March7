"""Owner-trusted approval issuer (Evernight side, plan A).

March7 holds only the gateway request-signing key and consumes grants; it
must never import this module. Only the owner-trusted issuer (Evernight, plus
the owner CLI on the host) holds the separate approval key, read from a
host-private FILE that is mounted read-only into Evernight alone. A missing
key fails closed -- there is never a fallback to the request secret.

Grants bind the exact execution fields through native
``canonical_approval_action`` plus the execution actor, so a grant for one
command/timeout/actor cannot authorize anything else.
"""
from __future__ import annotations

import json as jsonlib
import logging
from typing import Any, Protocol, runtime_checkable

from twin.shared.config.settings import Config
from twin.shared.system_gateway.auth import (
    canonical_approval_action,
    mint_approval_token,
)
from twin.shared.tools.approval_context import (
    ApprovalDecision,
    HostApprovalRequest,
    build_owner_approval_message,
    render_execution_display,
)

logger = logging.getLogger(__name__)

DEFAULT_APPROVAL_SECRET_PATH = "/run/secrets/system_gateway_approval"
SHELL_ACTION = "shell"
SELF_UPDATE_ACTION = "self.update"
SELF_UPDATE_ALIASES = frozenset({SELF_UPDATE_ACTION, "self_update"})
# Peers that may request grants over A2A. self.update may be REQUESTED by
# owner tooling or the evernight agent itself; the request role never confers
# approver authority -- a grant still needs genuine configured-owner UI
# approval. March7 stays untrusted for self.update (shell requests only).
SHELL_GRANT_PEERS = frozenset({"march7", "evernight", "owner"})
SELF_UPDATE_GRANT_PEERS = frozenset({"owner", "evernight"})
# Local (non-A2A) Evernight flows may only mint for the evernight actor.
LOCAL_GRANT_ACTOR = "evernight"


class ApprovalUnavailableError(RuntimeError):
    """Raised when the approval key or request cannot produce a grant."""


def approval_secret_path(explicit: str | None = None) -> str:
    """Resolve the host-private approval-key file path (never a value)."""
    if explicit:
        return explicit
    configured = getattr(Config, "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", None)
    return configured or DEFAULT_APPROVAL_SECRET_PATH


def read_approval_secret(path: str | None = None) -> str | None:
    """Read the owner approval key from its host-private file only."""
    try:
        with open(approval_secret_path(path), "r", encoding="utf-8") as handle:
            return handle.read().strip() or None
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None


def _require_secret(explicit: str | None = None) -> str:
    secret = explicit if explicit else read_approval_secret()
    if not secret:
        raise ApprovalUnavailableError("owner approval key is not available")
    return secret


def issue_shell_grant(
    *,
    actor: str,
    command: str,
    shell: str | None = None,
    cwd: str | None = None,
    timeout: int | None = None,
    approval_secret: str | None = None,
) -> str:
    """Mint an actor/execution-bound shell grant (post-approval only)."""
    canonical = canonical_approval_action(
        SHELL_ACTION,
        {"command": command, "shell": shell, "cwd": cwd, "timeout": timeout},
    )
    return mint_approval_token(
        secret=_require_secret(approval_secret), action=canonical, actor=actor
    )


def issue_self_update_grant(
    *,
    actor: str,
    from_version: str,
    to_version: str | None = None,
    approval_secret: str | None = None,
) -> str:
    """Mint an actor/version-bound self.update grant (post-approval only)."""
    canonical = canonical_approval_action(
        SELF_UPDATE_ACTION,
        {"from_version": from_version, "to_version": to_version},
    )
    return mint_approval_token(
        secret=_require_secret(approval_secret), action=canonical, actor=actor
    )


def issue_grant(
    *,
    action: str,
    actor: str,
    payload: dict[str, Any],
    approval_secret: str | None = None,
) -> str:
    """Mint a grant for a normalized action through the single helper path."""
    normalized = str(action or "").strip()
    if normalized == SHELL_ACTION:
        return issue_shell_grant(
            actor=actor,
            command=str(payload.get("command") or ""),
            shell=payload.get("shell"),
            cwd=payload.get("cwd"),
            timeout=payload.get("timeout"),
            approval_secret=approval_secret,
        )
    if normalized in SELF_UPDATE_ALIASES:
        return issue_self_update_grant(
            actor=actor,
            from_version=str(payload.get("from_version") or ""),
            to_version=payload.get("to_version"),
            approval_secret=approval_secret,
        )
    raise ValueError(f"unsupported approval action: {action!r}")


@runtime_checkable
class ApprovalDMBackend(Protocol):
    """Neutral Discord delivery capability, implemented by the gateway adapter."""

    async def request_owner_approval(
        self,
        *,
        owner_user_id: str,
        message_text: str,
        label: str,
        timeout: float,
    ) -> bool:
        """Show the exact prompt to the owner and wait for a decision."""

    async def send_owner_dm(self, *, owner_user_id: str, content: str) -> None:
        """Send a plain DM to the owner."""

    async def notify_channel(
        self, *, channel_id: Any, approved: bool, label: str
    ) -> None:
        """Report the decision back; must never include the grant token."""


def _grant_actor_for_peer(peer: str, body: dict[str, Any]) -> str:
    """Bind the grant actor to the authenticated peer, not caller assertion."""
    if peer == "owner":
        requested = str(body.get("actor") or "").strip()
        if requested in SHELL_GRANT_PEERS:
            return requested
    return peer


def _request_from_body(
    *, action: str, actor: str, body: dict[str, Any]
) -> HostApprovalRequest:
    """Build the display request from normalized effective execution values.

    Uses native ``canonical_approval_action`` as the single source of truth
    so the owner sees exactly what will execute/bind. Raises ValueError when
    invalid so callers deny pre-DM without bothering the owner.
    """
    if action in SELF_UPDATE_ALIASES:
        payload: dict[str, Any] = {
            "from_version": body.get("from_version"),
            "to_version": body.get("to_version"),
        }
        normalized = jsonlib.loads(
            canonical_approval_action(SELF_UPDATE_ACTION, payload)
        )
        return HostApprovalRequest(
            action=SELF_UPDATE_ACTION,
            actor=actor,
            from_version=normalized.get("from_version"),
            to_version=normalized.get("to_version"),
            channel_name=body.get("channel_name"),
        )
    payload = {
        "command": body.get("command"),
        "shell": body.get("shell"),
        "cwd": body.get("cwd"),
        "timeout": body.get("timeout"),
    }
    normalized = jsonlib.loads(canonical_approval_action(SHELL_ACTION, payload))
    return HostApprovalRequest(
        action=SHELL_ACTION,
        actor=actor,
        command=normalized.get("command"),
        shell=normalized.get("shell"),
        cwd=normalized.get("cwd"),
        timeout=normalized.get("timeout"),
        channel_name=body.get("channel_name"),
    )


async def decide_host_approval(
    *,
    backend: ApprovalDMBackend | None,
    owner_user_id: str | None,
    peer: str,
    body: dict[str, Any],
    dm_timeout: float = 60.0,
) -> ApprovalDecision:
    """Run one owner-approval round for an authenticated A2A request.

    Fails closed at every step: unknown owner, disallowed action/peer,
    non-displayable payload, missing approval key, unavailable DM path, or a
    reject/timeout all deny without minting. A grant is minted only after an
    actual owner approval, bound to the authenticated peer.
    """
    owner = str(owner_user_id or "").strip()
    if not owner:
        return ApprovalDecision.denied("owner_unknown")
    action = str(body.get("action") or SHELL_ACTION).strip()
    if action == SHELL_ACTION:
        allowed = SHELL_GRANT_PEERS
    elif action in SELF_UPDATE_ALIASES:
        allowed = SELF_UPDATE_GRANT_PEERS
        action = SELF_UPDATE_ACTION
    else:
        return ApprovalDecision.denied(f"unsupported action: {action}")
    if peer not in allowed:
        return ApprovalDecision.denied(f"peer '{peer}' may not request {action}")
    if backend is None:
        return ApprovalDecision.denied("approval_dm_unavailable")

    grant_actor = _grant_actor_for_peer(peer, body)
    try:
        request = _request_from_body(action=action, actor=grant_actor, body=body)
    except ValueError as exc:
        return ApprovalDecision.denied(str(exc))
    prompt, reason = build_owner_approval_message(
        render_execution_display(request),
        channel_name=request.channel_name,
        timeout_seconds=int(dm_timeout),
    )
    if prompt is None:
        return ApprovalDecision.denied(reason)
    if not read_approval_secret():
        # Fail before bothering the owner: no key means no grant is possible.
        return ApprovalDecision.denied("owner approval key is not available")

    label = request.command if action == SHELL_ACTION else action
    try:
        approved = await backend.request_owner_approval(
            owner_user_id=owner,
            message_text=prompt,
            label=label,
            timeout=dm_timeout,
        )
    except Exception as exc:
        logger.exception("Owner approval DM failed")
        return ApprovalDecision.denied(f"approval_dm_error: {exc}")

    channel_id = body.get("channel_id")
    if channel_id is not None:
        try:
            await backend.notify_channel(
                channel_id=channel_id, approved=approved, label=label
            )
        except Exception:
            logger.exception("Approval channel notification failed")

    if not approved:
        return ApprovalDecision.denied("rejected_or_timeout")
    try:
        grant = issue_grant(
            action=action, actor=grant_actor, payload=dict(request.execution_payload())
        )
    except (ApprovalUnavailableError, ValueError) as exc:
        return ApprovalDecision.denied(str(exc))
    return ApprovalDecision.approved_with_grant(grant)


async def approve_local_shell(
    *,
    backend: ApprovalDMBackend | None,
    owner_user_id: str | int | None,
    command: str,
    shell: str | None = None,
    cwd: str | None = None,
    timeout: int | None = None,
    channel_name: str | None = None,
    dm_timeout: float = 60.0,
) -> ApprovalDecision:
    """Request real owner approval for an Evernight-local shell (self-heal).

    Same authority as the A2A path: exact full display, owner-bound DM, grant
    minted only after approval. Local flows always bind the evernight actor.
    """
    owner = str(owner_user_id or "").strip()
    if not owner:
        return ApprovalDecision.denied("owner_unknown")
    if backend is None:
        return ApprovalDecision.denied("approval_dm_unavailable")
    try:
        request = _request_from_body(
            action=SHELL_ACTION,
            actor=LOCAL_GRANT_ACTOR,
            body={
                "command": command,
                "shell": shell,
                "cwd": cwd,
                "timeout": timeout,
                "channel_name": channel_name or "self-heal",
            },
        )
    except ValueError as exc:
        return ApprovalDecision.denied(str(exc))
    prompt, reason = build_owner_approval_message(
        render_execution_display(request),
        channel_name=request.channel_name,
        timeout_seconds=int(dm_timeout),
    )
    if prompt is None:
        return ApprovalDecision.denied(reason)
    if not read_approval_secret():
        return ApprovalDecision.denied("owner approval key is not available")
    try:
        approved = await backend.request_owner_approval(
            owner_user_id=owner,
            message_text=prompt,
            label=request.command or SHELL_ACTION,
            timeout=dm_timeout,
        )
    except Exception as exc:
        logger.exception("Local owner approval DM failed")
        return ApprovalDecision.denied(f"approval_dm_error: {exc}")
    if not approved:
        return ApprovalDecision.denied("rejected_or_timeout")
    try:
        grant = issue_shell_grant(
            actor=LOCAL_GRANT_ACTOR,
            command=request.command or "",
            shell=request.shell,
            cwd=request.cwd,
            timeout=request.timeout,
        )
    except (ApprovalUnavailableError, ValueError) as exc:
        return ApprovalDecision.denied(str(exc))
    return ApprovalDecision.approved_with_grant(grant)
