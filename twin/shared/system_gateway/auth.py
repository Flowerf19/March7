"""Authentication helpers for System Gateway.

The shared request secret is supplied via configuration (env var or secret
file) on both client and server. Each request carries a timestamp and nonce,
and the server validates that the HMAC signature matches.

Replay protection: the server keeps an in-memory set of recently seen nonces
and rejects any request whose timestamp is too old or whose nonce has already
been seen.

Approval tokens (mint_approval_token / verify_approval_token) bind an approval
to a specific canonical action, actor, and short TTL, and are single-use.
Trust model (plan A): the approval HMAC key is DIFFERENT from the request
signing key. Only the owner-trusted approval issuer (Evernight, plus the
owner CLI on the host) holds the approval key; March7 holds only the request
key and consumes grants, never mints. The gateway verifies approvals with the
approval key and requests with the shared request key; a missing approval key
fails closed. The ``action`` passed to mint/verify MUST be the output of
``canonical_approval_action`` so command/shell/cwd/timeout (or update
versions) are bound, not just the action name.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json as jsonlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


SIGNATURE_HEADER = "X-System-Gateway-Signature"
TIMESTAMP_HEADER = "X-System-Gateway-Timestamp"
NONCE_HEADER = "X-System-Gateway-Nonce"
ACTOR_HEADER = "X-System-Gateway-Actor"

DEFAULT_MAX_CLOCK_SKEW_SECONDS = 300
NONCE_TTL_SECONDS = DEFAULT_MAX_CLOCK_SKEW_SECONDS * 2


@dataclass(frozen=True)
class SignedRequest:
    """Material a client must attach to a request."""

    timestamp: str
    nonce: str
    actor: str
    signature: str


def derive_key(secret: str | None) -> bytes:
    """Derive the HMAC key bytes from the configured secret."""

    if not secret:
        return b""
    return hashlib.sha256(secret.encode("utf-8")).digest()


def canonical_message(
    *,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    actor: str,
    body: bytes,
) -> bytes:
    """Build the canonical message that gets HMAC-signed.

    Format (LF-separated):
        METHOD
        PATH
        TIMESTAMP
        NONCE
        ACTOR
        <body bytes>

    Body bytes are hashed separately so the canonical message stays small even
    for large payloads, while still binding the signature to the body.
    """

    body_digest = hashlib.sha256(body).hexdigest()
    lines = [method.upper(), path, timestamp, nonce, actor, body_digest]
    return "\n".join(lines).encode("utf-8")


def sign_request(
    *,
    secret: str | None,
    method: str,
    path: str,
    actor: str,
    body: bytes,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> SignedRequest:
    """Build a SignedRequest for a client call.

    Generates a fresh nonce and timestamp if not supplied.
    """

    ts = timestamp if timestamp is not None else str(int(time.time()))
    n = nonce if nonce is not None else secrets.token_hex(16)
    msg = canonical_message(
        method=method,
        path=path,
        timestamp=ts,
        nonce=n,
        actor=actor,
        body=body,
    )
    sig = hmac.new(derive_key(secret), msg, hashlib.sha256).hexdigest()
    return SignedRequest(timestamp=ts, nonce=n, actor=actor, signature=sig)


def verify_signature(
    secret: str | None,
    *,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    actor: str,
    body: bytes,
    signature: str,
) -> bool:
    """Constant-time signature verification."""

    if not secret:
        return False
    expected_msg = canonical_message(
        method=method,
        path=path,
        timestamp=timestamp,
        nonce=nonce,
        actor=actor,
        body=body,
    )
    expected_sig = hmac.new(
        derive_key(secret), expected_msg, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_sig, signature or "")


@dataclass
class NonceStore:
    """In-memory nonce store with TTL-based eviction.

    This is intentionally simple: it tracks recently seen nonces within a TTL
    window. A persistent store would be a later hardening step; Phase B only
    needs single-instance replay protection.
    """

    ttl_seconds: float = NONCE_TTL_SECONDS
    _seen: dict[str, float] = field(default_factory=dict)

    def is_fresh(self, nonce: str, now: float | None = None) -> bool:
        """Return True iff nonce is unseen and within TTL window.

        Marks the nonce as seen as a side effect so the next call returns False.
        """

        self._evict(now if now is not None else time.time())
        if nonce in self._seen:
            return False
        self._seen[nonce] = now if now is not None else time.time()
        return True

    def _evict(self, now: float) -> None:
        expired = [
            nonce
            for nonce, seen_at in self._seen.items()
            if now - seen_at > self.ttl_seconds
        ]
        for nonce in expired:
            self._seen.pop(nonce, None)

    def __len__(self) -> int:
        return len(self._seen)


def headers_from_signed(signed: SignedRequest) -> dict[str, str]:
    """Map a SignedRequest to the headers a client should send."""

    return {
        SIGNATURE_HEADER: signed.signature,
        TIMESTAMP_HEADER: signed.timestamp,
        NONCE_HEADER: signed.nonce,
        ACTOR_HEADER: signed.actor,
    }


def extract_auth_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Lower-cased lookup of the four auth headers."""

    lowered = {key.lower(): value for key, value in headers.items()}
    return {
        "signature": lowered.get(SIGNATURE_HEADER.lower(), ""),
        "timestamp": lowered.get(TIMESTAMP_HEADER.lower(), ""),
        "nonce": lowered.get(NONCE_HEADER.lower(), ""),
        "actor": lowered.get(ACTOR_HEADER.lower(), ""),
    }


def is_timestamp_within_skew(
    timestamp: str,
    *,
    now: float | None = None,
    max_skew_seconds: float = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
) -> bool:
    """Return True iff timestamp parses and is within +/-max_skew_seconds of now."""

    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = now if now is not None else time.time()
    return abs(current - ts) <= max_skew_seconds


def approval_replay_protection(
    consumed: Iterable[str],
    approval_id: str | None,
) -> bool:
    """Return True iff approval_id is present and not in the consumed set."""

    if not approval_id:
        return False
    return approval_id not in frozenset(consumed)


APPROVAL_TOKEN_VERSION = 1
DEFAULT_APPROVAL_TTL_SECONDS = 120


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _action_digest(action: str) -> str:
    return hashlib.sha256(action.encode("utf-8")).hexdigest()


def _approval_canonical(
    *,
    version: int,
    issued_at: int,
    expiry: int,
    nonce: str,
    actor: str,
    action_digest: str,
) -> bytes:
    """Canonical message HMAC-signed by an approval token."""

    lines = [
        str(version),
        str(issued_at),
        str(expiry),
        nonce,
        actor,
        action_digest,
    ]
    return "\n".join(lines).encode("utf-8")


@dataclass(frozen=True)
class ApprovalTokenResult:
    """Outcome of verifying an action-bound approval token."""

    valid: bool
    reason: str
    nonce: str | None = None


SHELL_TIMEOUT_DEFAULT = 30
SHELL_TIMEOUT_MIN = 1
SHELL_TIMEOUT_MAX = 300

SELF_UPDATE_ACTION = "self.update"
SELF_UPDATE_ALIASES = frozenset({"self.update", "self_update"})


def _normalize_shell_timeout(value: Any) -> int:
    """Normalize an effective shell timeout (server execution semantics)."""

    try:
        if value is None:
            return SHELL_TIMEOUT_DEFAULT
        seconds = int(value)
    except (TypeError, ValueError):
        return SHELL_TIMEOUT_DEFAULT
    return max(SHELL_TIMEOUT_MIN, min(seconds, SHELL_TIMEOUT_MAX))


def _normalize_optional_str(value: Any) -> str | None:
    """Normalize shell/cwd: missing or empty string becomes None, else exact."""

    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    return text


def canonical_approval_action(action: str, payload: Mapping[str, Any]) -> str:
    """Build the canonical action string bound by an approval token.

    Both the owner-trusted issuer (Evernight / owner CLI) and the gateway
    verifier MUST call this exact function with the effective execution
    fields. The returned string is passed as ``action=`` to
    ``mint_approval_token`` / ``verify_approval_token``.

    Protocol-normalization decisions (issuer and verifier must match):

    - ``action`` is stripped; ``"shell"`` and ``"self.update"``
      (alias ``"self_update"`` accepted, normalized to ``"self.update"``)
      are supported. Anything else raises ``ValueError``; removed legacy
      structured actions are never revived here.
    - ``approval_id`` and unknown keys are always excluded; only effective
      execution fields are bound.
    - shell: ``command`` is stripped and required (empty raises
      ``ValueError``); ``shell``/``cwd`` use ``_normalize_optional_str``
      (missing/``""`` -> ``None``, otherwise exact, no stripping);
      ``timeout`` uses server execution semantics (default 30, clamp
      1..300, invalid -> 30). ``max_output_chars`` is output-shaping only
      and intentionally NOT bound.
    - self.update: ``from_version`` is stripped (may be empty; the server
      still enforces version match after token verification);
      ``to_version`` missing/empty/whitespace -> ``None``, else stripped.
    - Encoding is JSON with ``sort_keys=True`` and compact separators, so
      field order and whitespace cannot cause a mismatch.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("approval payload must be a mapping")
    norm_action = str(action or "").strip()
    if norm_action == "shell":
        command = str(payload.get("command") or "").strip()
        if not command:
            raise ValueError("shell approval requires a non-empty command")
        canonical = {
            "action": "shell",
            "command": command,
            "cwd": _normalize_optional_str(payload.get("cwd")),
            "shell": _normalize_optional_str(payload.get("shell")),
            "timeout": _normalize_shell_timeout(payload.get("timeout")),
        }
        return jsonlib.dumps(canonical, sort_keys=True, separators=(",", ":"))
    if norm_action in SELF_UPDATE_ALIASES:
        from_version = str(payload.get("from_version") or "").strip()
        raw_target = payload.get("to_version")
        if raw_target is None:
            to_version = None
        else:
            stripped = str(raw_target).strip()
            to_version = stripped or None
        canonical_update = {
            "action": SELF_UPDATE_ACTION,
            "from_version": from_version,
            "to_version": to_version,
        }
        return jsonlib.dumps(
            canonical_update, sort_keys=True, separators=(",", ":")
        )
    raise ValueError(f"unsupported approval action: {action!r}")


def mint_approval_token(
    *,
    secret: str | None,
    action: str,
    actor: str,
    ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    now: float | None = None,
    nonce: str | None = None,
) -> str:
    """Mint a compact, action-bound, single-use approval token.

    The token binds issued_at, expiry, a fresh nonce, the actor, and
    sha256(action) under an HMAC-SHA256 signature derived from the approval
    secret (which MUST differ from the request-signing shared secret).
    ``action`` MUST be ``canonical_approval_action(...)`` output for
    shell/self.update so execution fields are bound. Format:
    ``<urlsafe-b64(payload-json)>.<hex-signature>``.
    """

    issued_at = int(now if now is not None else time.time())
    expiry = issued_at + int(ttl_seconds)
    token_nonce = nonce if nonce is not None else secrets.token_hex(16)
    action_digest = _action_digest(action)
    payload = {
        "v": APPROVAL_TOKEN_VERSION,
        "iat": issued_at,
        "exp": expiry,
        "nonce": token_nonce,
        "actor": actor,
        "ah": action_digest,
    }
    payload_bytes = jsonlib.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    message = _approval_canonical(
        version=APPROVAL_TOKEN_VERSION,
        issued_at=issued_at,
        expiry=expiry,
        nonce=token_nonce,
        actor=actor,
        action_digest=action_digest,
    )
    signature = hmac.new(derive_key(secret), message, hashlib.sha256).hexdigest()
    return f"{_b64encode(payload_bytes)}.{signature}"


def verify_approval_token(
    *,
    secret: str | None,
    token: str | None,
    action: str,
    actor: str,
    now: float | None = None,
) -> ApprovalTokenResult:
    """Verify an action-bound approval token.

    Returns ``ApprovalTokenResult`` with ``valid``, a ``reason`` string, and the
    token ``nonce`` (used as the single-use replay key) when the format parses.
    Rejects bad format, signature mismatch, expiry, action mismatch, and actor
    mismatch.
    """

    if not secret:
        return ApprovalTokenResult(False, "approval_secret_unset")
    if not token or "." not in token:
        return ApprovalTokenResult(False, "approval_malformed")

    payload_part, signature_part = token.rsplit(".", 1)
    try:
        payload = jsonlib.loads(_b64decode(payload_part).decode("utf-8"))
    except Exception:
        return ApprovalTokenResult(False, "approval_malformed")
    if not isinstance(payload, dict):
        return ApprovalTokenResult(False, "approval_malformed")

    try:
        version = int(payload["v"])
        issued_at = int(payload["iat"])
        expiry = int(payload["exp"])
        token_nonce = str(payload["nonce"])
        token_actor = str(payload["actor"])
        token_action_digest = str(payload["ah"])
    except (KeyError, TypeError, ValueError):
        return ApprovalTokenResult(False, "approval_malformed")

    message = _approval_canonical(
        version=version,
        issued_at=issued_at,
        expiry=expiry,
        nonce=token_nonce,
        actor=token_actor,
        action_digest=token_action_digest,
    )
    expected_sig = hmac.new(derive_key(secret), message, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, signature_part):
        return ApprovalTokenResult(False, "approval_signature_mismatch", nonce=token_nonce)

    current = now if now is not None else time.time()
    if current >= expiry:
        return ApprovalTokenResult(False, "approval_expired", nonce=token_nonce)
    if not hmac.compare_digest(token_action_digest, _action_digest(action)):
        return ApprovalTokenResult(False, "approval_action_mismatch", nonce=token_nonce)
    if not hmac.compare_digest(token_actor, actor):
        return ApprovalTokenResult(False, "approval_actor_mismatch", nonce=token_nonce)

    return ApprovalTokenResult(True, "approved", nonce=token_nonce)
