"""Gateway update approval fix: platform gate + structured DM + canonical pre-DM."""
from __future__ import annotations

import json
import pathlib

import pytest
from aiohttp.test_utils import TestClient, TestServer

from twin.shared.a2a.auth import make_a2a_headers
from twin.shared.config.settings import Config
from twin.shared.system_gateway.auth import (
    canonical_approval_action,
    verify_approval_token,
)
from twin.shared.tools.approval_context import clear_current_approval_context

APPROVAL_KEY = "owner-approval-key-fix-1"
A2A_SECRET = "a2a-fix-secret-2"
OWNER = "100000000000000001"


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


class _Backend:
    def __init__(self, approved=True):
        self.approved = approved
        self.calls = 0
        self.prompts: list[str] = []
        self.labels: list[str] = []

    async def request_owner_approval(self, *, owner_user_id, message_text, label, timeout):
        self.calls += 1
        self.prompts.append(message_text)
        self.labels.append(label)
        assert owner_user_id == OWNER
        assert "grant" not in message_text.lower() or "grant" in message_text.lower()
        return self.approved

    async def send_owner_dm(self, **kwargs):
        return None

    async def notify_channel(self, **kwargs):
        return None


def test_production_bootstrap_has_live_eve_gate(monkeypatch):
    from twin.shared.tools.registry import bootstrap

    monkeypatch.setattr(bootstrap.Config, "SYSTEM_GATEWAY_URL", "http://gw.local")
    monkeypatch.setattr(
        bootstrap.Config, "EVERNIGHT_A2A_URL", "http://evernight.local:8001"
    )

    class _LLM:
        pass

    result = bootstrap.build_tool_registry(
        agent_name="evernight",
        core_manager=None,
        memory_manager=None,
        llm_service=_LLM(),
        base_memory_path="memories",
        owner_user_id=OWNER,
        gateway_monitor=None,
        use_evernight_dm_approval=True,
    )
    gate = result.approval_gate
    assert gate.dm_client is not None
    assert gate.dm_client.actor == "evernight"
    assert gate.dm_client.base_url == "http://evernight.local:8001"
    proxy = result.registry.get_tool("gateway_admin")
    assert proxy is not None
    inner = getattr(proxy, "_tool", proxy)
    assert getattr(inner, "_approval_gate", None) is gate


def test_evernight_container_enables_self_dm():
    source = pathlib.Path("twin/evernight/container.py").read_text(encoding="utf-8")
    assert "use_evernight_dm_approval=True" in source


def test_production_dm_allows_eve_peer():
    from twin.evernight.server import a2a_server as server_module
    from twin.shared.a2a.types import AgentCard

    class _Agent:
        def get_agent_card(self):
            return AgentCard(name="Evernight", description="", url="", version="1")

    server = server_module.start_server(_Agent(), owner_user_id=OWNER)  # type: ignore[arg-type]
    assert "evernight" in server._dm_allowed_peers
    assert "march7" in server._dm_allowed_peers
    assert "owner" in server._dm_allowed_peers
    # Skill policy unchanged: evernight still cannot use its own chat skill.
    assert server._skill_peers["chat"] == frozenset({"march7", "owner"})


@pytest.mark.asyncio
async def test_signed_eve_self_dm_reaches_owner_backend(approval_key_file, monkeypatch):
    from twin.evernight.server import a2a_server as server_module
    from twin.shared.a2a.types import AgentCard

    monkeypatch.setattr(Config, "A2A_SHARED_SECRET", A2A_SECRET)

    class _Agent:
        def get_agent_card(self):
            return AgentCard(name="Evernight", description="", url="", version="1")

    backend = _Backend(approved=True)
    server = server_module.start_server(
        _Agent(),  # type: ignore[arg-type]
        owner_user_id=OWNER,
        dm_backend=backend,
        shared_secret=A2A_SECRET,
    )
    app = server.build_app()
    test_server = TestServer(app)
    client = TestClient(test_server)
    await client.start_server()
    try:
        payload = {
            "user_id": int(OWNER),
            "type": "approval",
            "action": "self.update",
            "actor": "evernight",
            "from_version": "0.1.0",
            "to_version": "0.2.0",
            "channel_name": "#ops",
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers = make_a2a_headers("evernight", "POST", "/dm", body, secret=A2A_SECRET)
        async with client.post("/dm", data=body, headers=headers) as resp:
            assert resp.status == 200, await resp.text()
            data = await resp.json()
        assert data["approved"] is True
        assert data["grant"]
        assert backend.calls == 1
        expected = canonical_approval_action(
            "self.update", {"from_version": "0.1.0", "to_version": "0.2.0"}
        )
        assert verify_approval_token(
            secret=APPROVAL_KEY, token=data["grant"], action=expected,
            actor="evernight",
        ).valid is True

        # March7 self.update still denied at issuer (not middleware 403).
        m7_payload = dict(payload)
        m7_body = json.dumps(m7_payload, separators=(",", ":")).encode()
        m7_headers = make_a2a_headers("march7", "POST", "/dm", m7_body, secret=A2A_SECRET)
        async with client.post("/dm", data=m7_body, headers=m7_headers) as resp:
            assert resp.status == 200
            denied = await resp.json()
        assert denied["approved"] is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_self_update_peers_and_owner_button(approval_key_file):
    from twin.evernight.server.approval_issuer import decide_host_approval

    # Evernight may request; owner approval mints evernight-bound grant.
    backend = _Backend(approved=True)
    decision = await decide_host_approval(
        backend=backend,
        owner_user_id=OWNER,
        peer="evernight",
        body={"action": "self.update", "from_version": "0.1.0",
              "to_version": "0.2.0", "actor": "evernight"},
    )
    assert decision.approved is True and decision.grant
    expected = canonical_approval_action(
        "self.update", {"from_version": "0.1.0", "to_version": "0.2.0"}
    )
    assert verify_approval_token(
        secret=APPROVAL_KEY, token=decision.grant, action=expected,
        actor="evernight",
    ).valid is True

    # March7 cannot request self.update; owner is never bothered.
    backend2 = _Backend(approved=True)
    denied = await decide_host_approval(
        backend=backend2, owner_user_id=OWNER, peer="march7",
        body={"action": "self.update", "from_version": "0.1.0"},
    )
    assert denied.approved is False
    assert backend2.calls == 0

    # Owner reject/timeout denies without minting.
    backend3 = _Backend(approved=False)
    rejected = await decide_host_approval(
        backend=backend3, owner_user_id=OWNER, peer="evernight",
        body={"action": "self.update", "from_version": "0.1.0"},
    )
    assert rejected.approved is False and rejected.grant is None


@pytest.mark.asyncio
async def test_nonowner_shell_requests_unchanged(approval_key_file):
    from twin.evernight.server.approval_issuer import decide_host_approval

    backend = _Backend(approved=True)
    decision = await decide_host_approval(
        backend=backend, owner_user_id=OWNER, peer="march7",
        body={"command": "echo hi", "shell": "bash", "cwd": "/x", "timeout": 20},
    )
    assert decision.approved is True and decision.grant


@pytest.mark.asyncio
async def test_canonical_predm_blank_denies_without_dm(approval_key_file):
    from twin.evernight.server.approval_issuer import (
        approve_local_shell,
        decide_host_approval,
    )

    for body in ({"command": "   "}, {"command": ""}, {}):
        backend = _Backend(approved=True)
        denied = await decide_host_approval(
            backend=backend, owner_user_id=OWNER, peer="march7", body=dict(body)
        )
        assert denied.approved is False
        assert backend.calls == 0

    backend = _Backend(approved=True)
    denied = await approve_local_shell(
        backend=backend, owner_user_id=OWNER, command="   \n\t "
    )
    assert denied.approved is False
    assert backend.calls == 0

    backend = _Backend(approved=True)
    denied = await decide_host_approval(
        backend=backend, owner_user_id=OWNER, peer="march7",
        body={"action": "delete-everything", "command": "x"},
    )
    assert denied.approved is False
    assert backend.calls == 0


@pytest.mark.asyncio
async def test_normalized_timeout_shown_and_bound(approval_key_file):
    from twin.evernight.server.approval_issuer import decide_host_approval

    backend = _Backend(approved=True)
    decision = await decide_host_approval(
        backend=backend, owner_user_id=OWNER, peer="march7",
        body={"command": "echo hi", "timeout": "abc"},
    )
    assert decision.approved is True
    assert "timeout: 30s" in backend.prompts[0]
    assert "abc" not in backend.prompts[0]
    expected = canonical_approval_action(
        "shell", {"command": "echo hi", "shell": None, "cwd": None, "timeout": 30}
    )
    assert verify_approval_token(
        secret=APPROVAL_KEY, token=decision.grant, action=expected,
        actor="march7",
    ).valid is True

    backend2 = _Backend(approved=True)
    decision2 = await decide_host_approval(
        backend=backend2, owner_user_id=OWNER, peer="march7",
        body={"command": "echo hi", "timeout": 9999},
    )
    assert decision2.approved is True
    assert "timeout: 300s" in backend2.prompts[0]
    expected2 = canonical_approval_action(
        "shell", {"command": "echo hi", "shell": None, "cwd": None, "timeout": 300}
    )
    assert verify_approval_token(
        secret=APPROVAL_KEY, token=decision2.grant, action=expected2,
        actor="march7",
    ).valid is True
