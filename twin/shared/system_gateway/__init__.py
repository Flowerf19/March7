"""Shared client/protocol types for the native System Gateway."""
from twin.shared.system_gateway.audit import (
    AuditEvent,
    AuditOutcome,
    audit_event,
)
from twin.shared.system_gateway.auth import (
    ACTOR_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    ApprovalTokenResult,
    NonceStore,
    SignedRequest,
    approval_replay_protection,
    canonical_approval_action,
    extract_auth_headers,
    headers_from_signed,
    is_timestamp_within_skew,
    mint_approval_token,
    sign_request,
    verify_approval_token,
    verify_signature,
)
from twin.shared.system_gateway.client import HostGatewayClient
from twin.shared.system_gateway.errors import (
    HostGatewayError,
    HostGatewayUnavailableError,
)
from twin.shared.system_gateway.policy import (
    PolicyContext,
    PolicyDecision,
    PolicyReason,
    PolicyVerdict,
    evaluate_action_policy,
    evaluate_shell_policy,
)
from twin.shared.system_gateway.types import (
    GatewayActionResponse,
    GatewayCapabilities,
    GatewayHealth,
    GatewayShellRequest,
)

__all__ = [
    "ACTOR_HEADER",
    "ApprovalTokenResult",
    "AuditEvent",
    "AuditOutcome",
    "GatewayActionResponse",
    "GatewayCapabilities",
    "GatewayHealth",
    "GatewayShellRequest",
    "HostGatewayClient",
    "HostGatewayError",
    "HostGatewayUnavailableError",
    "NONCE_HEADER",
    "NonceStore",
    "PolicyContext",
    "PolicyDecision",
    "PolicyReason",
    "PolicyVerdict",
    "SIGNATURE_HEADER",
    "SignedRequest",
    "TIMESTAMP_HEADER",
    "approval_replay_protection",
    "audit_event",
    "canonical_approval_action",
    "evaluate_action_policy",
    "evaluate_shell_policy",
    "extract_auth_headers",
    "headers_from_signed",
    "is_timestamp_within_skew",
    "mint_approval_token",
    "sign_request",
    "verify_approval_token",
    "verify_signature",
]