"""HMAC authentication for A2A agent-to-agent HTTP.

Every protected request carries four ``X-A2A-*`` headers produced by
:func:`make_a2a_headers`. The signature binds actor + method + path +
timestamp + nonce + body hash, so a signed request cannot be retargeted,
replayed, or tampered with.

Fail-closed rules:

- Missing/empty shared secret denies every protected route.
- Unknown actors, expired timestamps, and reused nonces are rejected.
- Only ``GET /.well-known/agent.json`` and ``GET /health`` are public.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from collections import OrderedDict
from typing import Callable, Mapping

from aiohttp import web

from twin.shared.config.settings import Config

logger = logging.getLogger(__name__)

A2A_ACTOR_MARCH7 = "march7"
A2A_ACTOR_EVERNIGHT = "evernight"
A2A_ACTOR_OWNER = "owner"
A2A_ACTORS = frozenset({A2A_ACTOR_MARCH7, A2A_ACTOR_EVERNIGHT, A2A_ACTOR_OWNER})

HEADER_ACTOR = "X-A2A-Actor"
HEADER_TIMESTAMP = "X-A2A-Timestamp"
HEADER_NONCE = "X-A2A-Nonce"
HEADER_SIGNATURE = "X-A2A-Signature"

TIMESTAMP_SKEW_SECONDS = 300
NONCE_TTL_SECONDS = 2 * TIMESTAMP_SKEW_SECONDS
NONCE_STORE_MAX_ENTRIES = 10_000
NONCE_PURGE_INTERVAL_SECONDS = 60

PUBLIC_ROUTES = frozenset({"/.well-known/agent.json", "/health"})
DM_ROUTE_PATHS = frozenset({"/dm", "/approval_dm"})

# Deliberate owner-tooling policy: local owner tooling may call either agent,
# but only with a valid signature like any other peer.
DEFAULT_DM_ALLOWED_PEERS = frozenset({A2A_ACTOR_MARCH7, A2A_ACTOR_OWNER})
# Evernight self-DM: the evernight agent may REQUEST (never auto-approve) via
# its own /dm; the owner UI still approves. Other agents keep the default.
EVERNIGHT_DM_ALLOWED_PEERS = frozenset(
    {A2A_ACTOR_MARCH7, A2A_ACTOR_EVERNIGHT, A2A_ACTOR_OWNER}
)

MARCH7_SKILL_PEERS: dict[str, frozenset[str]] = {
    "chat": frozenset({A2A_ACTOR_OWNER}),
    "get_snapshot": frozenset({A2A_ACTOR_EVERNIGHT, A2A_ACTOR_OWNER}),
    "clear_session": frozenset({A2A_ACTOR_EVERNIGHT, A2A_ACTOR_OWNER}),
}
EVERNIGHT_SKILL_PEERS: dict[str, frozenset[str]] = {
    "chat": frozenset({A2A_ACTOR_MARCH7, A2A_ACTOR_OWNER}),
    "consolidate": frozenset({A2A_ACTOR_MARCH7, A2A_ACTOR_OWNER}),
    "consolidate_discussion": frozenset({A2A_ACTOR_MARCH7, A2A_ACTOR_OWNER}),
}


class A2AAuthError(ValueError):
    """Raised when an A2A request cannot be authenticated."""


def make_a2a_headers(
    actor: str,
    method: str,
    path: str,
    body: bytes,
    secret: str | None = None,
    *,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Build signed ``X-A2A-*`` headers for one request.

    Args:
        actor: One of ``march7`` / ``evernight`` / ``owner``.
        method: HTTP method (``POST`` for RPC/DM, ``GET`` for streams).
        path: Exact request path (no query string).
        body: Exact body bytes that will be sent (``b""`` for GET).
        secret: Shared secret; falls back to ``Config.A2A_SHARED_SECRET``.
        timestamp/nonce: Overrides for deterministic tests.

    Raises:
        A2AAuthError: On unknown actor or missing secret.
    """
    if actor not in A2A_ACTORS:
        raise A2AAuthError(f"unknown A2A actor: {actor!r}")
    resolved = secret or Config.A2A_SHARED_SECRET
    if not resolved:
        raise A2AAuthError("missing A2A shared secret")
    body_bytes = bytes(body)
    ts = int(timestamp) if timestamp is not None else int(time.time())
    nonce_value = nonce or secrets.token_hex(16)
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    canonical = "\n".join([actor, method.upper(), path, str(ts), nonce_value, body_hash])
    signature = hmac.new(
        resolved.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return {
        HEADER_ACTOR: actor,
        HEADER_TIMESTAMP: str(ts),
        HEADER_NONCE: nonce_value,
        HEADER_SIGNATURE: signature,
    }


class NonceStore:
    """Bounded replay cache for A2A request nonces."""

    def __init__(
        self,
        *,
        ttl_seconds: float = NONCE_TTL_SECONDS,
        max_entries: int = NONCE_STORE_MAX_ENTRIES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._last_purge = 0.0

    def __len__(self) -> int:
        return len(self._seen)

    def check_and_add(self, nonce: str, now: float | None = None) -> bool:
        """Record *nonce*; return False when it is a live replay."""
        current = self._clock() if now is None else now
        if (
            current - self._last_purge >= NONCE_PURGE_INTERVAL_SECONDS
            or len(self._seen) >= self._max_entries
        ):
            self.purge(current)
        existing = self._seen.get(nonce)
        if existing is not None and existing > current:
            return False
        while len(self._seen) >= self._max_entries:
            self._seen.popitem(last=False)
        self._seen[nonce] = current + self._ttl
        return True

    def purge(self, now: float | None = None) -> int:
        """Drop expired nonces; return the number removed."""
        current = self._clock() if now is None else now
        self._last_purge = current
        expired = [key for key, exp in self._seen.items() if exp <= current]
        for key in expired:
            del self._seen[key]
        return len(expired)


def verify_a2a_request(
    *,
    headers: Mapping[str, str],
    method: str,
    path: str,
    body: bytes,
    secret: str | None,
    nonce_store: NonceStore,
    now: float | None = None,
) -> str:
    """Verify signed headers; return the authenticated actor.

    Raises:
        A2AAuthError: On any authentication failure.
    """
    if not secret:
        raise A2AAuthError("missing A2A shared secret")
    actor = (headers.get(HEADER_ACTOR) or "").strip()
    ts_raw = (headers.get(HEADER_TIMESTAMP) or "").strip()
    nonce = (headers.get(HEADER_NONCE) or "").strip()
    provided = (headers.get(HEADER_SIGNATURE) or "").strip()
    if not actor or not ts_raw or not nonce or not provided:
        raise A2AAuthError("missing auth headers")
    if actor not in A2A_ACTORS:
        raise A2AAuthError(f"unknown A2A actor: {actor!r}")
    try:
        ts = int(ts_raw)
    except ValueError:
        raise A2AAuthError("invalid timestamp") from None
    current = time.time() if now is None else now
    if abs(current - ts) > TIMESTAMP_SKEW_SECONDS:
        raise A2AAuthError("timestamp outside window")
    body_hash = hashlib.sha256(bytes(body)).hexdigest()
    canonical = "\n".join([actor, method.upper(), path, str(ts), nonce, body_hash])
    expected = hmac.new(
        secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    try:
        matched = hmac.compare_digest(expected, provided)
    except TypeError:
        raise A2AAuthError("invalid signature") from None
    if not matched:
        raise A2AAuthError("invalid signature")
    if not nonce_store.check_and_add(nonce, current):
        raise A2AAuthError("replayed nonce")
    return actor


def resolve_agent_name(agent_card: object, explicit: str | None = None) -> str:
    """Resolve the served agent name, defaulting from the agent card."""
    if explicit:
        return explicit
    name = (getattr(agent_card, "name", "") or "").strip().lower()
    if name in (A2A_ACTOR_MARCH7, A2A_ACTOR_EVERNIGHT):
        return name
    return "unknown"


def skill_peers_for_agent(agent_name: str) -> dict[str, frozenset[str]]:
    """Return the default skill -> allowed-peers policy for an agent."""
    if agent_name == A2A_ACTOR_MARCH7:
        return dict(MARCH7_SKILL_PEERS)
    if agent_name == A2A_ACTOR_EVERNIGHT:
        return dict(EVERNIGHT_SKILL_PEERS)
    return {}


def a2a_auth_middleware(
    *,
    secret_resolver: Callable[[], str | None],
    dm_allowed_peers: frozenset[str] = DEFAULT_DM_ALLOWED_PEERS,
    nonce_store: NonceStore,
):
    """Build app-level auth middleware covering RPC, streams, and DM routes."""

    @web.middleware
    async def middleware(request: web.Request, handler):
        if request.method == "GET" and request.path in PUBLIC_ROUTES:
            return await handler(request)
        body = await request.read()
        try:
            peer = verify_a2a_request(
                headers=request.headers,
                method=request.method,
                path=request.path,
                body=body,
                secret=secret_resolver(),
                nonce_store=nonce_store,
            )
        except A2AAuthError as exc:
            logger.warning(
                "A2A auth denied: %s %s (%s)", request.method, request.path, exc
            )
            return web.json_response({"error": f"unauthorized: {exc}"}, status=401)
        if request.path in DM_ROUTE_PATHS and peer not in dm_allowed_peers:
            logger.warning("A2A DM denied for peer %s on %s", peer, request.path)
            return web.json_response({"error": "forbidden"}, status=403)
        request["a2a_peer"] = peer
        return await handler(request)

    return middleware
