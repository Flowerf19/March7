"""Platform-neutral approval context for dangerous tool execution.

Gateway adapters set this context before routing a message into the agent.
Shared tools read only neutral metadata and an optional approval backend.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ApprovalRequestContext:
    """Neutral metadata for requesting user approval.

    `native_message` is intentionally opaque. Only the platform adapter/backend
    that created it should inspect it.
    """

    platform: str
    user_id: str
    conversation_id: str | None = None
    channel_id: str | None = None
    message_id: str | None = None
    channel_name: str | None = None
    space_id: str | None = None
    user_name: str | None = None
    native_message: object | None = None
    approval_backend: "ApprovalBackend | None" = None


@runtime_checkable
class ApprovalBackend(Protocol):
    """Platform adapter capability for rendering approval UI."""

    async def request_channel_approval(
        self,
        context: ApprovalRequestContext,
        command: str,
    ) -> bool:
        """Ask for approval in the source conversation/channel."""


# Discord message-content bound. Approval prompts must fit entirely; anything
# that does not fit is denied (owner CLI remains available), never truncated.
DISCORD_MESSAGE_LIMIT = 2000

SHELL_APPROVAL_ACTION = "shell"
SELF_UPDATE_APPROVAL_ACTION = "self.update"


@dataclass(frozen=True)
class HostApprovalRequest:
    """Neutral, exact execution fields for a host-approval decision.

    The owner sees every bound field in full. `actor` is the requested
    execution actor; the Evernight issuer rebinds it to the authenticated
    A2A peer instead of trusting the caller assertion.
    """

    action: str
    actor: str = ""
    command: str | None = None
    shell: str | None = None
    cwd: str | None = None
    timeout: int | None = None
    from_version: str | None = None
    to_version: str | None = None
    channel_name: str | None = None

    def execution_payload(self) -> dict[str, Any]:
        """Payload bound by native `canonical_approval_action`."""
        if self.action == SELF_UPDATE_APPROVAL_ACTION:
            return {
                "from_version": self.from_version or "",
                "to_version": self.to_version,
            }
        return {
            "command": self.command or "",
            "shell": self.shell,
            "cwd": self.cwd,
            "timeout": self.timeout,
        }


@dataclass(frozen=True)
class ApprovalDecision:
    """Structured owner-approval outcome from the Evernight issuer."""

    approved: bool
    grant: str | None = None
    reason: str = ""

    @classmethod
    def denied(cls, reason: str) -> "ApprovalDecision":
        return cls(approved=False, grant=None, reason=reason)

    @classmethod
    def approved_with_grant(cls, grant: str) -> "ApprovalDecision":
        return cls(approved=True, grant=grant, reason="approved")

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "grant": self.grant,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalDecision":
        return cls(
            approved=bool(data.get("approved", False)),
            grant=data.get("grant"),
            reason=str(data.get("reason") or ""),
        )


def render_execution_display(request: HostApprovalRequest) -> str:
    """Render the EXACT bound execution fields, in full, no truncation."""
    lines = [f"action: {request.action}", f"actor: {request.actor or 'unknown'}"]
    if request.action == SELF_UPDATE_APPROVAL_ACTION:
        lines.append(f"from_version: {request.from_version or ''}")
        lines.append(f"to_version: {request.to_version or 'latest'}")
        return "\n".join(lines)
    lines.append(f"command: {request.command or ''}")
    lines.append(f"shell: {request.shell or 'default'}")
    lines.append(f"cwd: {request.cwd or 'default'}")
    lines.append(f"timeout: {request.timeout if request.timeout is not None else 'default'}s")
    return "\n".join(lines)


def _display_rejection(display: str) -> str | None:
    """Return a denial reason when raw text cannot be shown unambiguously."""
    if "```" in display:
        return "payload contains code fences and cannot be displayed unambiguously"
    for char in display:
        if char in ("\n", "\t"):
            continue
        if ord(char) < 0x20 or ord(char) == 0x7F:
            return "payload contains control characters"
    return None


def build_owner_approval_message(
    display: str,
    *,
    channel_name: str | None = None,
    timeout_seconds: int = 60,
) -> tuple[str | None, str]:
    """Build the full owner-facing approval prompt, or deny with a reason.

    Returns (message, "") when the exact payload fits Discord bounds, else
    (None, reason). Never truncates: oversized input must go through the
    owner CLI instead.
    """
    rejection = _display_rejection(display)
    if rejection is not None:
        return None, rejection
    prompt = (
        "\U0001f510 **Bot mu\u1ed1n ch\u1ea1y l\u1ec7nh tr\u00ean host:**\n"
        f"```\n{display}\n```\n"
        f"\U0001f4cd K\u00eanh g\u1ed1c: {channel_name or 'unknown'}\n"
        f"Cho ph\u00e9p? (Timeout: {timeout_seconds} gi\u00e2y)"
    )
    rejection = _display_rejection(channel_name or "")
    if rejection is not None:
        return None, f"channel label {rejection}"
    if len(prompt) > DISCORD_MESSAGE_LIMIT:
        return (
            None,
            "payload too large for Discord approval; use the owner CLI instead",
        )
    return prompt, ""


_current_approval_context: ContextVar[ApprovalRequestContext | None] = ContextVar(
    "current_approval_context", default=None
)


def set_current_approval_context(context: ApprovalRequestContext) -> None:
    """Set the active approval context for the current task."""
    _current_approval_context.set(context)


def get_current_approval_context() -> ApprovalRequestContext | None:
    """Return the active approval context, if one exists."""
    return _current_approval_context.get(None)


def clear_current_approval_context() -> None:
    """Clear the active approval context."""
    _current_approval_context.set(None)


def set_current_message(message: object) -> None:
    """Compatibility shim for old Discord call sites.

    New code should set a full ApprovalRequestContext from the platform adapter.
    """
    channel = getattr(message, "channel", None)
    guild = getattr(message, "guild", None)
    author = getattr(message, "author", None)
    channel_name = str(channel) if channel is not None else None
    if channel is not None and getattr(channel, "name", None):
        channel_name = f"#{channel.name}"
    if guild is not None and channel_name:
        channel_name = f"{getattr(guild, 'name', guild)}/{channel_name}"

    context = ApprovalRequestContext(
        platform="discord",
        user_id=str(getattr(author, "id", "")),
        conversation_id=str(getattr(channel, "id", "")) if channel is not None else None,
        channel_id=str(getattr(channel, "id", "")) if channel is not None else None,
        message_id=str(getattr(message, "id", "")),
        channel_name=channel_name,
        space_id=str(getattr(guild, "id", "")) if guild is not None else None,
        user_name=getattr(author, "display_name", None),
        native_message=message,
        approval_backend=None,
    )
    set_current_approval_context(context)


def get_current_message() -> object | None:
    """Compatibility shim returning the opaque native message."""
    context = get_current_approval_context()
    if context is None:
        return None
    return context.native_message


def clear_current_message() -> None:
    """Compatibility shim for old Discord call sites."""
    clear_current_approval_context()
