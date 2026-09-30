"""Owner-approval regression tests (plan A: Evernight is the issuer).

Covers: owner-bound approval views, exact full-display enforcement, A2A-signed
DM client with forced owner recipient, structured grant flow (March7 consumes,
never mints), peer-bound issuance, self.update authority, and deployment
wiring (approval key in Evernight-only mount, never in March7/.env).
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from twin.shared.a2a.auth import NonceStore, verify_a2a_request
from twin.shared.config.settings import Config
from twin.shared.system_gateway.auth import (
    canonical_approval_action,
    verify_approval_token,
)
from twin.shared.tools.approval_context import (
    ApprovalDecision,
    ApprovalRequestContext,
    HostApprovalRequest,
    build_owner_approval_message,
    clear_current_approval_context,
    render_execution_display,
    set_current_approval_context,
)
from twin.shared.tools.approval_gate import ApprovalGate
from twin.shared.tools.dm_client import DMClient, DMUnavailableError

APPROVAL_KEY = "owner-approval-key-1"
REQUEST_KEY = "request-signing-key-2"
A2A_SECRET = "a2a-test-secret-3"
OWNER = "726302130318868500"


@pytest.fixture(autouse=True)
def _clear_context():
    clear_current_approval_context()
    yield
    clear_current_approval_context()


@pytest.fixture()
def approval_key_file(monkeypatch, tmp_path):
    path = tmp_path / "approval_secret"
    path.write_text(APPROVAL_KEY, encoding="utf-8")
    monkeypatch.setattr(Config, "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(path))
    return path


# ---------------------------------------------------------------------------
# A. Owner-bound approval views (real discord lib)
# ---------------------------------------------------------------------------


def _interaction(user_id: Any):
    sent: list[dict] = []
    edited: list[dict] = []

    class _Response:
        async def send_message(self, content=None, **kwargs):
            sent.append({"content": content, **kwargs})

        async def edit_message(self, **kwargs):
            edited.append(kwargs)

    return SimpleNamespace(
        user=SimpleNamespace(id=user_id), response=_Response(),
        _sent=sent, _edited=edited,
    )


@pytest.mark.asyncio
async def test_owner_click_approves():
    from gateway.adapters.discord.views.approve_view import ApproveView

    view = ApproveView(command="echo hi", owner_user_id=OWNER)
    interaction = _interaction(int(OWNER))

    await view.approve_button.callback(interaction)

    assert view._approved is True
    assert view._result.is_set()
    assert await view.wait_for_decision() is True


@pytest.mark.asyncio
async def test_owner_click_rejects():
    from gateway.adapters.discord.views.approve_view import ApproveView

    view = ApproveView(command="echo hi", owner_user_id=OWNER)
    interaction = _interaction(int(OWNER))

    await view.reject_button.callback(interaction)

    assert view._approved is False
    assert view._result.is_set()
    assert await view.wait_for_decision() is False


@pytest.mark.asyncio
async def test_nonowner_click_is_rejected_including_direct_callback():
    from gateway.adapters.discord.views.approve_view import ApproveView

    view = ApproveView(command="echo hi", owner_user_id=OWNER)
    stranger = _interaction(999)

    # interaction_check path.
    assert await view.interaction_check(stranger) is False
    assert view._result.is_set() is False

    # Direct callback bypass path (interaction_check skipped).
    await view.approve_button.callback(stranger)
    assert view._approved is False
    assert view._result.is_set() is False
    assert stranger._sent  # ephemeral denial notice


@pytest.mark.asyncio
async def test_unknown_owner_denies_everyone():
    from gateway.adapters.discord.views.approve_view import ApproveView

    for unknown in (None, "", "   "):
        view = ApproveView(command="echo hi", owner_user_id=unknown)
        owner_click = _interaction(int(OWNER))
        assert await view.interaction_check(owner_click) is False
        await view.approve_button.callback(owner_click)
        assert view._approved is False
        assert view._result.is_set() is False


@pytest.mark.asyncio
async def test_dm_view_passes_owner_through():
    from gateway.adapters.discord.views.dm_approve_view import DMApproveView

    view = DMApproveView(command="echo hi", owner_user_id=OWNER)
    assert view._owner_user_id == OWNER
    assert await view.interaction_check(_interaction(int(OWNER))) is True
    assert await view.interaction_check(_interaction(999)) is False


# ---------------------------------------------------------------------------
# B. Exact full-display enforcement (finding #11)
# ---------------------------------------------------------------------------


def test_render_shows_exact_full_execution_fields():
    request = HostApprovalRequest(
        action="shell",
        actor="march7",
        command="rm -rf /tmp/x --and --more --flags",
        shell="bash",
        cwd="/srv/app",
        timeout=45,
    )
    display = render_execution_display(request)
    assert "rm -rf /tmp/x --and --more --flags" in display
    assert "bash" in display
    assert "/srv/app" in display
    assert "45s" in display
    assert "march7" in display


def test_oversized_payload_is_rejected_not_truncated():
    prompt, reason = build_owner_approval_message(
        "x" * 5000, channel_name="#ops", timeout_seconds=60
    )
    assert prompt is None
    assert "CLI" in reason


def test_deceptive_fences_and_control_chars_are_rejected():
    prompt, _ = build_owner_approval_message("echo hi ```\nrm -rf /")
    assert prompt is None
    prompt, _ = build_owner_approval_message("echo hi\x00hidden")
    assert prompt is None
    prompt, reason = build_owner_approval_message("echo hi\nsecond line")
    assert prompt is not None and reason == ""


# ---------------------------------------------------------------------------
# C. Authenticated DM client
# ---------------------------------------------------------------------------


class _FakeDMResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {"approved": True, "grant": "g", "reason": "ok"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return dict(self._payload)

    async def text(self):
        return json.dumps(self._payload)


class _FakeDMSession:
    def __init__(self, response=None):
        self.posts: list[dict] = []
        self._response = response or _FakeDMResponse()

    def post(self, url, *, data, headers):
        self.posts.append({"url": url, "data": data, "headers": headers})
        return self._response


def _dm_client(session, **kwargs) -> DMClient:
    import asyncio

    kwargs.setdefault("actor", "march7")
    kwargs.setdefault("secret", A2A_SECRET)
    kwargs.setdefault("owner_user_id", OWNER)
    client = DMClient(evernight_url="http://evernight:8001", **kwargs)

    async def _session():
        return session

    client._get_session = _session  # type: ignore[method-assign]
    assert asyncio.iscoroutinefunction(client._get_session)
    return client


@pytest.mark.asyncio
async def test_dm_posts_are_signed_over_exact_body_bytes():
    session = _FakeDMSession()
    client = _dm_client(session)

    decision = await client.request_authorization(
        action="shell", command="echo hi", shell="bash", cwd="/x", timeout=20
    )

    assert decision.approved is True
    post = session.posts[0]
    assert post["url"] == "http://evernight:8001/dm"
    # The signature verifies against the exact transmitted bytes.
    peer = verify_a2a_request(
        headers=post["headers"],
        method="POST",
        path="/dm",
        body=post["data"],
        secret=A2A_SECRET,
        nonce_store=NonceStore(),
    )
    assert peer == "march7"
    payload = json.loads(post["data"].decode("utf-8"))
    assert payload["command"] == "echo hi"
    assert payload["shell"] == "bash"
    assert payload["user_id"] == int(OWNER)


@pytest.mark.asyncio
async def test_dm_forces_configured_owner_recipient():
    session = _FakeDMSession()
    client = _dm_client(session)

    assert await client.request_approval(command="echo hi", user_id=999) is True
    payload = json.loads(session.posts[0]["data"].decode("utf-8"))
    assert payload["user_id"] == int(OWNER)


@pytest.mark.asyncio
async def test_dm_missing_secret_is_unavailable(monkeypatch):
    monkeypatch.setattr(Config, "A2A_SHARED_SECRET", None)
    session = _FakeDMSession()
    client = _dm_client(session, secret=None)

    with pytest.raises(DMUnavailableError):
        await client.request_authorization(action="shell", command="echo hi")
    assert session.posts == []


@pytest.mark.asyncio
async def test_dm_unknown_owner_denies_without_http(monkeypatch):
    monkeypatch.setattr(Config, "EVERNIGHT_OWNER_USER_ID", None)
    session = _FakeDMSession()
    client = _dm_client(session, owner_user_id=None)

    decision = await client.request_authorization(action="shell", command="echo hi")

    assert decision.approved is False
    assert "owner" in decision.reason.lower()
    assert session.posts == []


@pytest.mark.asyncio
async def test_dm_approval_without_grant_is_denied():
    session = _FakeDMSession(
        _FakeDMResponse(payload={"approved": True, "grant": None, "reason": "x"})
    )
    client = _dm_client(session)

    decision = await client.request_authorization(action="shell", command="echo hi")
    assert decision.approved is False


# ---------------------------------------------------------------------------
# D. ApprovalGate host authorization (DM-only, fail closed)
# ---------------------------------------------------------------------------


class _StubDM:
    def __init__(self, decision=None, exc=None):
        self.decision = decision or ApprovalDecision.denied("nope")
        self.exc = exc
        self.calls: list[dict] = []

    async def request_authorization(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.decision

    async def request_approval(self, **kwargs):
        decision = await self.request_authorization(**kwargs)
        return decision.approved


class _ChannelSpy:
    def __init__(self):
        self.calls = 0

    async def request_channel_approval(self, context, command):
        self.calls += 1
        return True


def _host_context():
    return ApprovalRequestContext(
        platform="discord", user_id="non-owner-requester", conversation_id="c1"
    )


@pytest.mark.asyncio
async def test_authorize_host_denies_without_context():
    gate = ApprovalGate(dm_client=_StubDM(ApprovalDecision.approved_with_grant("g")))
    decision = await gate.authorize_host(action="shell", command="x", actor="march7")
    assert decision.approved is False


@pytest.mark.asyncio
async def test_authorize_host_denies_without_dm_client():
    set_current_approval_context(_host_context())
    decision = await ApprovalGate().authorize_host(
        action="shell", command="x", actor="march7"
    )
    assert decision.approved is False


@pytest.mark.asyncio
async def test_authorize_host_passes_grant_and_never_falls_back():
    dm = _StubDM(ApprovalDecision.approved_with_grant("grant-1"))
    spy = _ChannelSpy()
    context = ApprovalRequestContext(
        platform="discord", user_id="r", conversation_id="c1", approval_backend=spy
    )
    set_current_approval_context(context)

    decision = await ApprovalGate(dm_client=dm).authorize_host(
        action="shell", command="echo hi", shell="bash", cwd="/x", timeout=20,
        actor="march7",
    )

    assert decision.approved is True
    assert decision.grant == "grant-1"
    assert dm.calls[0]["command"] == "echo hi"
    assert dm.calls[0]["timeout"] == 20
    assert dm.calls[0]["actor"] == "march7"
    assert spy.calls == 0


@pytest.mark.asyncio
async def test_authorize_host_denial_does_not_fall_back_to_channel():
    dm = _StubDM(ApprovalDecision.denied("rejected_or_timeout"))
    spy = _ChannelSpy()
    context = ApprovalRequestContext(
        platform="discord", user_id="r", conversation_id="c1", approval_backend=spy
    )
    set_current_approval_context(context)

    decision = await ApprovalGate(dm_client=dm).authorize_host(
        action="shell", command="x", actor="march7"
    )

    assert decision.approved is False
    assert spy.calls == 0


@pytest.mark.asyncio
async def test_authorize_host_transport_failure_denies_without_channel():
    dm = _StubDM(exc=DMUnavailableError("down"))
    spy = _ChannelSpy()
    context = ApprovalRequestContext(
        platform="discord", user_id="r", conversation_id="c1", approval_backend=spy
    )
    set_current_approval_context(context)

    decision = await ApprovalGate(dm_client=dm).authorize_host(
        action="shell", command="x", actor="march7"
    )

    assert decision.approved is False
    assert spy.calls == 0


# ---------------------------------------------------------------------------
# E. HostSystemTool consumes the grant (never mints)
# ---------------------------------------------------------------------------


class _FakeGate:
    def __init__(self, decision):
        self.decision = decision
        self.calls: list[dict] = []

    async def authorize_host(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision

    async def check_approval(self, tool_name, command):
        return False


class _FakeGatewayClient:
    actor = "march7"

    def __init__(self):
        self.requests: list[Any] = []

    async def capabilities(self):
        from twin.shared.system_gateway import GatewayCapabilities

        return GatewayCapabilities(platform="linux", raw_shell=True)

    async def run_shell(self, request):
        self.requests.append(request)
        from twin.shared.system_gateway import GatewayActionResponse

        return GatewayActionResponse(ok=True, output="hi")


@pytest.mark.asyncio
async def test_host_tool_sends_returned_grant_with_exact_execution():
    from twin.shared.tools.modules.system.host_system_tool import HostSystemTool

    gate = _FakeGate(ApprovalDecision.approved_with_grant("grant-abc"))
    client = _FakeGatewayClient()
    tool = HostSystemTool(approval_gate=gate, host_gateway_client=client)

    result = await tool.execute(
        mode="shell", command="echo hi", shell="bash", cwd="/x", timeout=20
    )

    assert "✅" in result or "Thành công" in result
    assert gate.calls[0]["command"] == "echo hi"
    assert gate.calls[0]["shell"] == "bash"
    assert gate.calls[0]["cwd"] == "/x"
    assert gate.calls[0]["timeout"] == 20
    assert gate.calls[0]["actor"] == "march7"
    assert len(client.requests) == 1
    sent = client.requests[0]
    assert sent.approval_id == "grant-abc"
    assert sent.command == "echo hi"
    assert sent.timeout == 20


@pytest.mark.asyncio
async def test_host_tool_denial_or_timeout_never_executes():
    from twin.shared.tools.modules.system.host_system_tool import HostSystemTool

    for decision in (
        ApprovalDecision.denied("rejected_or_timeout"),
        ApprovalDecision.denied("dm_unavailable"),
        ApprovalDecision(approved=True, grant=None, reason="no-grant"),
    ):
        gate = _FakeGate(decision)
        client = _FakeGatewayClient()
        tool = HostSystemTool(approval_gate=gate, host_gateway_client=client)
        result = await tool.execute(mode="shell", command="echo hi")
        assert result.startswith("❌")
        assert client.requests == []


def test_host_tool_has_no_mint_or_key_path():
    assert not hasattr(
        __import__(
            "twin.shared.tools.modules.system.host_system_tool", fromlist=["x"]
        ).HostSystemTool,
        "_mint_token",
    )
    source = pathlib.Path(
        "twin/shared/tools/modules/system/host_system_tool.py"
    ).read_text(encoding="utf-8")
    assert "mint_approval_token" not in source
    assert "approval_secret" not in source
    assert "approval_issuer" not in source


# ---------------------------------------------------------------------------
# F. Issuer binds actor + execution; request key cannot issue
# ---------------------------------------------------------------------------


def test_shell_grant_binds_exact_execution_and_actor(approval_key_file, tmp_path):
    from twin.evernight.server.approval_issuer import issue_shell_grant

    grant = issue_shell_grant(
        actor="march7", command="echo hi", shell="bash", cwd="/x", timeout=20
    )
    expected = canonical_approval_action(
        "shell",
        {"command": "echo hi", "shell": "bash", "cwd": "/x", "timeout": 20},
    )
    ok = verify_approval_token(
        secret=APPROVAL_KEY, token=grant, action=expected, actor="march7"
    )
    assert ok.valid, ok.reason
    # Tampered execution, wrong actor, or request key all fail.
    tampered = canonical_approval_action(
        "shell", {"command": "echo BYE", "shell": "bash", "cwd": "/x", "timeout": 20}
    )
    assert (
        verify_approval_token(
            secret=APPROVAL_KEY, token=grant, action=tampered, actor="march7"
        ).valid
        is False
    )
    assert (
        verify_approval_token(
            secret=APPROVAL_KEY, token=grant, action=expected, actor="evernight"
        ).valid
        is False
    )
    assert (
        verify_approval_token(
            secret=REQUEST_KEY, token=grant, action=expected, actor="march7"
        ).valid
        is False
    )


def test_read_approval_secret_missing_file_returns_none(tmp_path, monkeypatch):
    from twin.evernight.server.approval_issuer import read_approval_secret

    monkeypatch.setattr(
        Config, "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(tmp_path / "absent")
    )
    assert read_approval_secret() is None


class _IssueBackend:
    def __init__(self, approved=True):
        self.approved = approved
        self.calls = 0

    async def request_owner_approval(self, **kwargs):
        self.calls += 1
        return self.approved

    async def send_owner_dm(self, **kwargs):
        return None

    async def notify_channel(self, **kwargs):
        return None


def _shell_body(**overrides):
    body = {
        "type": "approval",
        "user_id": int(OWNER),
        "command": "echo hi",
        "shell": "bash",
        "cwd": "/x",
        "timeout": 20,
        "channel_name": "#ops",
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_decide_binds_grant_actor_to_peer_not_caller(approval_key_file):
    from twin.evernight.server.approval_issuer import decide_host_approval

    backend = _IssueBackend(approved=True)
    decision = await decide_host_approval(
        backend=backend,
        owner_user_id=OWNER,
        peer="march7",
        body=_shell_body(actor="evernight"),  # caller assertion is ignored
    )

    assert decision.approved is True
    assert decision.grant
    expected = canonical_approval_action(
        "shell", {"command": "echo hi", "shell": "bash", "cwd": "/x", "timeout": 20}
    )
    assert (
        verify_approval_token(
            secret=APPROVAL_KEY, token=decision.grant, action=expected,
            actor="march7",
        ).valid
        is True
    )
    assert (
        verify_approval_token(
            secret=APPROVAL_KEY, token=decision.grant, action=expected,
            actor="evernight",
        ).valid
        is False
    )


@pytest.mark.asyncio
async def test_decide_denies_self_update_for_non_owner_peer(approval_key_file):
    from twin.evernight.server.approval_issuer import decide_host_approval

    backend = _IssueBackend(approved=True)
    decision = await decide_host_approval(
        backend=backend,
        owner_user_id=OWNER,
        peer="march7",
        body={"action": "self.update", "from_version": "0.1.0"},
    )
    assert decision.approved is False
    assert backend.calls == 0


@pytest.mark.asyncio
async def test_decide_fail_closed_matrix(tmp_path, monkeypatch):
    from twin.evernight.server.approval_issuer import decide_host_approval

    monkeypatch.setattr(
        Config, "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(tmp_path / "absent")
    )
    # Unknown owner.
    denied = await decide_host_approval(
        backend=_IssueBackend(), owner_user_id="", peer="march7",
        body=_shell_body(),
    )
    assert denied.approved is False
    # Missing DM path.
    denied = await decide_host_approval(
        backend=None, owner_user_id=OWNER, peer="march7", body=_shell_body()
    )
    assert denied.approved is False
    # Non-displayable payload never reaches the owner.
    backend = _IssueBackend()
    denied = await decide_host_approval(
        backend=backend, owner_user_id=OWNER, peer="march7",
        body=_shell_body(command="x" * 5000),
    )
    assert denied.approved is False
    assert backend.calls == 0
    # Missing approval key never bothers the owner.
    backend = _IssueBackend()
    denied = await decide_host_approval(
        backend=backend, owner_user_id=OWNER, peer="march7", body=_shell_body()
    )
    assert denied.approved is False
    assert "approval key" in denied.reason.lower()
    assert backend.calls == 0


# ---------------------------------------------------------------------------
# G. Evernight handler: auth, owner, structured decisions
# ---------------------------------------------------------------------------


class _FakeHandlerRequest:
    def __init__(self, body, peer="march7"):
        self._body = body
        self._peer = peer

    def get(self, key, default=None):
        if key == "a2a_peer":
            return self._peer
        return default

    async def json(self):
        return dict(self._body)


def _resp_json(resp):
    return json.loads(resp.text)


def _handler(backend=None, owner=OWNER):
    from twin.evernight.server.a2a_server import EvernightA2AHandler

    return EvernightA2AHandler(
        agent=object(), dm_backend=backend, owner_user_id=owner
    )


@pytest.mark.asyncio
async def test_handler_dm_requires_authenticated_peer():
    resp = await _handler(_IssueBackend()).handle_dm(_FakeHandlerRequest({}, peer=None))
    assert resp.status == 401


@pytest.mark.asyncio
async def test_handler_approval_happy_path_mints_bound_grant(approval_key_file):
    backend = _IssueBackend(approved=True)
    resp = await _handler(backend).handle_dm(
        _FakeHandlerRequest(_shell_body(), peer="march7")
    )
    assert resp.status == 200
    data = _resp_json(resp)
    assert data["approved"] is True
    assert data["grant"]
    expected = canonical_approval_action(
        "shell", {"command": "echo hi", "shell": "bash", "cwd": "/x", "timeout": 20}
    )
    assert (
        verify_approval_token(
            secret=APPROVAL_KEY, token=data["grant"], action=expected,
            actor="march7",
        ).valid
        is True
    )


@pytest.mark.asyncio
async def test_handler_reject_and_unknown_owner_deny(approval_key_file):
    backend = _IssueBackend(approved=False)
    resp = await _handler(backend).handle_dm(
        _FakeHandlerRequest(_shell_body(), peer="march7")
    )
    data = _resp_json(resp)
    assert data["approved"] is False
    assert data["grant"] is None

    resp = await _handler(_IssueBackend()).handle_dm(
        _FakeHandlerRequest(_shell_body(), peer="march7")
    )
    # Sanity: configured owner works; now unknown owner denies.
    assert _resp_json(resp)["approved"] is True
    resp = await _handler(_IssueBackend(), owner="").handle_dm(
        _FakeHandlerRequest(
            {"type": "approval", "user_id": 1, "command": "x"}, peer="march7"
        )
    )
    assert _resp_json(resp)["approved"] is False


@pytest.mark.asyncio
async def test_handler_rejects_non_owner_recipient():
    resp = await _handler(_IssueBackend()).handle_dm(
        _FakeHandlerRequest(_shell_body(user_id=999), peer="march7")
    )
    assert resp.status == 403
    assert _resp_json(resp)["approved"] is False


@pytest.mark.asyncio
async def test_handler_message_path_is_owner_only_and_needs_backend():
    sent: list[dict] = []

    class _Backend(_IssueBackend):
        async def send_owner_dm(self, *, owner_user_id, content):
            sent.append({"owner": owner_user_id, "content": content})

    resp = await _handler(_Backend()).handle_dm(
        _FakeHandlerRequest(
            {"user_id": int(OWNER), "content": "hi", "type": "message"},
            peer="march7",
        )
    )
    assert resp.status == 200
    assert sent and sent[0]["owner"] == OWNER

    resp = await _handler(None).handle_dm(
        _FakeHandlerRequest(
            {"user_id": int(OWNER), "content": "hi", "type": "message"},
            peer="march7",
        )
    )
    assert resp.status == 503


@pytest.mark.asyncio
async def test_handler_has_no_discord_import_or_hardcoded_owner():
    source = pathlib.Path("twin/evernight/server/a2a_server.py").read_text(
        encoding="utf-8"
    )
    assert "import discord" not in source
    assert "726302130318868500" not in source


# ---------------------------------------------------------------------------
# H. Bootstrap wires per-agent actors
# ---------------------------------------------------------------------------


def test_bootstrap_uses_per_agent_actors(monkeypatch):
    from twin.shared.tools.registry import bootstrap

    monkeypatch.setattr(bootstrap.Config, "SYSTEM_GATEWAY_URL", "http://gw.local")
    monkeypatch.setattr(
        bootstrap.Config, "EVERNIGHT_A2A_URL", "http://evernight.local"
    )

    class _LLM:
        pass

    march7 = bootstrap.build_tool_registry(
        agent_name="march7",
        core_manager=None,
        memory_manager=None,
        llm_service=_LLM(),
        base_memory_path="memories",
        use_evernight_dm_approval=True,
    )
    assert march7.approval_gate.dm_client.actor == "march7"
    assert march7.host_gateway_client.actor == "march7"

    evernight = bootstrap.build_tool_registry(
        agent_name="evernight",
        core_manager=None,
        memory_manager=None,
        llm_service=_LLM(),
        base_memory_path="memories",
        use_evernight_dm_approval=True,
    )
    assert evernight.approval_gate.dm_client.actor == "evernight"
    assert evernight.host_gateway_client.actor == "evernight"


# ---------------------------------------------------------------------------
# I. Deployment wiring: key in Evernight-only mount, never March7/.env
# ---------------------------------------------------------------------------


def test_approval_key_mount_is_evernight_only():
    evernight = pathlib.Path("docker/evernight/docker-compose.yml").read_text(
        encoding="utf-8"
    )
    # Long bind: safe, no auto-create, read-only, SELinux, configurable host path.
    assert "target: /run/secrets/system_gateway_approval" in evernight
    assert "read_only: true" in evernight
    assert "create_host_path: false" in evernight
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_HOST_PATH" in evernight
    assert (
        "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=/run/secrets/system_gateway_approval"
        in evernight
    )
    march7 = pathlib.Path("docker/march7/docker-compose.yml").read_text(
        encoding="utf-8"
    )
    # No approval mount/value reaches March7; explicit blanks defeat stray .env.
    assert "/run/secrets/system_gateway_approval" not in march7
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET=" in march7
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=" in march7
    for line in march7.splitlines():
        if "APPROVAL_SECRET" in line:
            assert line.strip().endswith("="), line


def test_no_inline_approval_secret_in_env_fixture():
    fixture = pathlib.Path(".env.example").read_text(encoding="utf-8")
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET=" not in fixture
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET =" not in fixture


def test_march7_side_modules_never_touch_approval_key():
    for rel in (
        "twin/shared/tools/dm_client.py",
        "twin/shared/tools/approval_gate.py",
        "twin/shared/tools/registry/bootstrap.py",
    ):
        source = pathlib.Path(rel).read_text(encoding="utf-8")
        assert "approval_issuer" not in source, rel
        assert "mint_approval_token" not in source, rel
