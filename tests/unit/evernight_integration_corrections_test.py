"""Integration corrections: compose isolation, explicit owner, neutral A2A validation.

Plan A: Evernight issuer+UI trusted, March7 untrusted as requester. Verifies:
- Neither agent holds the other's bot token or the full .env file.
- Owner authority has no hardcoded default; unknown owner fails closed.
- Evernight A2A validates sessions/scopes/entries, preserves shipped [],
  and sets a neutral approval context for chat (cleaned on completion).

Uses an explicit nonsecret test owner fixture; no real private values.
"""
from __future__ import annotations

import json
import pathlib

import pytest

TEST_OWNER = "100000000000000001"
OTHER_USER = "100000000000000002"
TEST_SCOPE = "100000000000000003"

REPO = pathlib.Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A. Compose isolation: no .env bind, peer token blanked, own token via env_file
# ---------------------------------------------------------------------------


def test_compose_has_no_env_file_bind():
    for rel in ("docker/march7/docker-compose.yml", "docker/evernight/docker-compose.yml"):
        text = _read(rel)
        assert "/app/.env" not in text, rel
        assert ".env:/app" not in text, rel


def test_compose_keeps_env_file_for_shared_config():
    for rel in ("docker/march7/docker-compose.yml", "docker/evernight/docker-compose.yml"):
        text = _read(rel)
        assert "env_file:" in text, rel
        assert "../../.env" in text, rel


def test_march7_blanks_evernight_token_only():
    text = _read("docker/march7/docker-compose.yml")
    assert "DISCORD_EVERNIGHT_TOKEN=" in text
    # Blank override (empty value) wins over env_file; own token comes via env_file.
    for line in text.splitlines():
        if "DISCORD_EVERNIGHT_TOKEN" in line:
            assert line.strip().endswith("DISCORD_EVERNIGHT_TOKEN="), line
    assert "DISCORD_MARCH7_TOKEN=" not in text


def test_evernight_blanks_march7_token_only():
    text = _read("docker/evernight/docker-compose.yml")
    assert "DISCORD_MARCH7_TOKEN=" in text
    for line in text.splitlines():
        if "DISCORD_MARCH7_TOKEN" in line:
            assert line.strip().endswith("DISCORD_MARCH7_TOKEN="), line
    assert "DISCORD_EVERNIGHT_TOKEN=" not in text


def test_march7_has_no_approval_mount_or_value():
    text = _read("docker/march7/docker-compose.yml")
    assert "/run/secrets/system_gateway_approval" not in text
    # Explicit blanks so a stray .env entry cannot reintroduce the key.
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET=" in text
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=" in text
    for line in text.splitlines():
        if "APPROVAL_SECRET" in line:
            assert line.strip().endswith("="), line


def test_compose_has_no_credential_literal():
    for rel in ("docker/march7/docker-compose.yml", "docker/evernight/docker-compose.yml"):
        text = _read(rel)
        for line in text.splitlines():
            stripped = line.strip()
            if "DISCORD_" in stripped and "_TOKEN=" in stripped:
                # Only blank peer overrides are allowed; no values in compose.
                assert stripped.endswith("_TOKEN="), f"{rel}: {line}"
            if "APPROVAL_SECRET=" in stripped and "HOST_PATH" not in stripped:
                # Evernight FILE env points at the mount path (nonsecret);
                # March7 blanks are empty; never an inline key value.
                if "SECRET_FILE=" in stripped:
                    continue
                assert stripped.endswith("="), f"{rel}: {line}"


def test_shared_env_still_wired():
    march7 = _read("docker/march7/docker-compose.yml")
    assert "EVERNIGHT_A2A_URL" in march7
    assert "REDIS_URL" in march7
    assert "SYSTEM_GATEWAY_URL" in march7
    evernight = _read("docker/evernight/docker-compose.yml")
    assert "MARCH7_URL" in evernight
    assert "REDIS_URL" in evernight
    assert "SYSTEM_GATEWAY_URL" in evernight
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=/run/secrets/system_gateway_approval" in evernight


def test_approval_mount_is_evernight_only_long_bind():
    evernight = _read("docker/evernight/docker-compose.yml")
    assert evernight.count("target: /run/secrets/system_gateway_approval") == 1
    assert "read_only: true" in evernight
    assert "create_host_path: false" in evernight
    assert "selinux: z" in evernight
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_HOST_PATH" in evernight
    # No short-syntax auto-creating bind for the key.
    assert "${HOME}/.config/system-gateway/approval_secret:/run/secrets" not in evernight
    march7 = _read("docker/march7/docker-compose.yml")
    assert "system_gateway_approval" not in march7 or "APPROVAL_SECRET" in march7
    assert "target: /run/secrets/system_gateway_approval" not in march7


# ---------------------------------------------------------------------------
# B. No baked .env defeats isolation
# ---------------------------------------------------------------------------


def test_dockerignore_excludes_env():
    # run-tests.sh mounts docker/ but not root dotfiles; check what exists.
    checked = []
    for rel in (".dockerignore", "docker/.dockerignore"):
        path = REPO / rel
        if not path.exists():
            # Fall back to relative CWD (container workdir /review).
            alt = pathlib.Path(rel)
            if not alt.exists():
                continue
            path = alt
        text = path.read_text(encoding="utf-8")
        assert ".env" in text, rel
        assert ".env.*" in text, rel
        checked.append(rel)
    # At least the docker build context ignore must be verifiable in-container.
    assert "docker/.dockerignore" in checked


def test_dockerfiles_do_not_copy_env():
    for rel in ("docker/march7/Dockerfile", "docker/evernight/Dockerfile", "docker/shared/Dockerfile.base"):
        text = _read(rel)
        assert ".env" not in text, rel


# ---------------------------------------------------------------------------
# C. Explicit owner config (no hardcoded authority default)
# ---------------------------------------------------------------------------


def test_evernight_config_defaults_to_unknown_owner():
    from twin.evernight.config import EvernightConfig

    assert EvernightConfig.owner_user_id is None


def test_evernight_config_from_env_requires_explicit_owner(monkeypatch):
    from twin.evernight.config import EvernightConfig

    monkeypatch.delenv("EVERNIGHT_OWNER_USER_ID", raising=False)
    assert EvernightConfig.from_env().owner_user_id is None
    monkeypatch.setenv("EVERNIGHT_OWNER_USER_ID", TEST_OWNER)
    assert EvernightConfig.from_env().owner_user_id == TEST_OWNER
    monkeypatch.setenv("EVERNIGHT_OWNER_USER_ID", "   ")
    assert EvernightConfig.from_env().owner_user_id is None


def test_no_hardcoded_owner_default_in_owned_sources():
    for rel in ("twin/evernight/config.py", "gateway/adapters/discord/evernight_adapter.py"):
        text = _read(rel)
        assert "DEFAULT_OWNER_USER_ID" not in text, rel
    from gateway.adapters.discord import evernight_adapter as adapter_module

    assert not hasattr(adapter_module, "DEFAULT_OWNER_USER_ID")


# ---------------------------------------------------------------------------
# D. Adapter explicit owner (fail closed, debounce path untouched)
# ---------------------------------------------------------------------------


def _make_adapter(monkeypatch, **kwargs):
    from gateway.adapters.discord import evernight_adapter as adapter_module

    monkeypatch.setattr(adapter_module.EvernightDiscordAdapter, "_build_bot", lambda self: object())
    from gateway.adapters.discord.evernight_adapter import EvernightDiscordAdapter

    return EvernightDiscordAdapter(token="fake-token", agent=object(), **kwargs)


def test_adapter_prefers_explicit_owner_over_env(monkeypatch):
    monkeypatch.setenv("EVERNIGHT_OWNER_USER_ID", OTHER_USER)
    adapter = _make_adapter(monkeypatch, owner_user_id=TEST_OWNER)
    assert adapter._owner_user_id == TEST_OWNER


def test_adapter_falls_back_to_env_when_no_explicit(monkeypatch):
    monkeypatch.setenv("EVERNIGHT_OWNER_USER_ID", TEST_OWNER)
    adapter = _make_adapter(monkeypatch)
    assert adapter._owner_user_id == TEST_OWNER


def test_adapter_unknown_owner_fails_closed(monkeypatch):
    monkeypatch.delenv("EVERNIGHT_OWNER_USER_ID", raising=False)
    adapter = _make_adapter(monkeypatch)
    assert adapter._owner_user_id == ""


@pytest.mark.asyncio
async def test_adapter_unknown_owner_ignores_everyone(monkeypatch):
    from gateway.adapters.discord import evernight_adapter as adapter_module

    monkeypatch.setattr(adapter_module.EvernightDiscordAdapter, "_build_bot", lambda self: object())
    from gateway.adapters.discord.evernight_adapter import EvernightDiscordAdapter

    monkeypatch.delenv("EVERNIGHT_OWNER_USER_ID", raising=False)
    adapter = EvernightDiscordAdapter.__new__(EvernightDiscordAdapter)
    adapter._owner_user_id = ""
    adapter._bot = type("B", (), {"user": None})()

    class _Author:
        id = int(TEST_OWNER)
        bot = False

    class _Channel:
        id = 1

        async def send(self, content):
            raise AssertionError("must not reply when owner unknown")

        def typing(self):
            raise AssertionError("must not type when owner unknown")

    message = type("M", (), {})()
    message.author = _Author()
    message.content = "hello"
    message.channel = _Channel()
    message.guild = None
    message.mentions = []

    await adapter._on_message(message)  # returns without reply or handler call


# ---------------------------------------------------------------------------
# E. Session validator (shared server contract, no bypass)
# ---------------------------------------------------------------------------


def test_validate_evernight_session():
    from twin.evernight.server.a2a_server import validate_evernight_session

    for skill in ("chat", "consolidate", "consolidate_discussion"):
        assert validate_evernight_session(skill, TEST_OWNER) is None
        assert validate_evernight_session(skill, "") is not None
        assert validate_evernight_session(skill, "   ") is not None
        assert validate_evernight_session(skill, None) is not None
        assert validate_evernight_session(skill, "abc") is not None
        assert validate_evernight_session(skill, "12ab") is not None
        assert validate_evernight_session(skill, 123) is not None


def test_start_server_wires_session_validator():
    from twin.evernight.server import a2a_server as server_module
    from twin.evernight.server.a2a_server import validate_evernight_session
    from twin.shared.a2a.types import AgentCard

    class _Agent:
        def get_agent_card(self):
            return AgentCard(name="Evernight", description="", url="", version="1")

    server = server_module.start_server(_Agent(), owner_user_id=TEST_OWNER)  # type: ignore[arg-type]
    assert server._session_validator is validate_evernight_session
    assert server._skill_peers["chat"] == frozenset({"march7", "owner"})
    assert server._skill_peers["consolidate_discussion"] == frozenset({"march7", "owner"})


# ---------------------------------------------------------------------------
# F. Consolidate_discussion: preserve [], reject malformed, no spoofing
# ---------------------------------------------------------------------------


class _ConsolidateAgent:
    def __init__(self):
        self.calls: list[dict] = []

    async def consolidate_via_tool(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok", "messages_summarized": len(kwargs.get("entries") or [])}


def _handler(agent=None, owner=TEST_OWNER):
    from twin.evernight.server.a2a_server import EvernightA2AHandler

    return EvernightA2AHandler(agent=agent or _ConsolidateAgent(), owner_user_id=owner)


async def _first_data(handler, params: dict) -> dict:
    messages = [m async for m in handler.handle_consolidate_discussion_task(params)]
    assert len(messages) == 1
    part = messages[0].parts[0]
    assert part.type == "data"
    assert isinstance(part.data, dict)
    return part.data


@pytest.mark.asyncio
async def test_shipped_empty_list_reaches_tool_as_empty():
    agent = _ConsolidateAgent()
    data = await _first_data(
        _handler(agent),
        {"sessionId": TEST_SCOPE, "payload": {"scope": "user", "scope_id": TEST_SCOPE, "entries": []}},
    )
    assert data["status"] == "ok"
    assert len(agent.calls) == 1
    assert agent.calls[0]["entries"] == []


@pytest.mark.asyncio
async def test_missing_entries_key_reads_local_t1():
    agent = _ConsolidateAgent()
    data = await _first_data(
        _handler(agent),
        {"sessionId": TEST_SCOPE, "payload": {"scope": "user", "scope_id": TEST_SCOPE}},
    )
    assert data["status"] == "ok"
    assert agent.calls[0]["entries"] is None


@pytest.mark.asyncio
async def test_malformed_entries_are_rejected():
    for bad in ("nope", 123, {"a": 1}, [1, 2], [{"ok": 1}, "bad"]):
        agent = _ConsolidateAgent()
        data = await _first_data(
            _handler(agent),
            {"sessionId": TEST_SCOPE, "payload": {"scope": "user", "scope_id": TEST_SCOPE, "entries": bad}},
        )
        assert data["status"] == "failed", bad
        assert agent.calls == []


@pytest.mark.asyncio
async def test_invalid_scope_scope_id_and_mismatch_are_rejected():
    cases = [
        {"sessionId": TEST_SCOPE, "payload": {"scope": "admin", "scope_id": TEST_SCOPE}},
        {"sessionId": TEST_SCOPE, "payload": {"scope": "user"}},
        {"sessionId": TEST_SCOPE, "payload": {"scope": "user", "scope_id": "abc"}},
        {"sessionId": TEST_SCOPE, "payload": {"scope": "channel", "scope_id": OTHER_USER}},
        {"sessionId": "abc", "payload": {"scope": "user", "scope_id": "abc"}},
        {"sessionId": TEST_SCOPE, "payload": "not-a-dict"},
        {"sessionId": TEST_SCOPE},
        {"sessionId": TEST_SCOPE, "payload": {"scope": "user", "scope_id": TEST_SCOPE, "max_messages": "200"}},
        {"sessionId": TEST_SCOPE, "payload": {"scope": "user", "scope_id": TEST_SCOPE, "reason": 123}},
    ]
    for params in cases:
        agent = _ConsolidateAgent()
        data = await _first_data(_handler(agent), params)
        assert data["status"] == "failed", params
        assert agent.calls == []


@pytest.mark.asyncio
async def test_channel_scope_with_shipped_entries_succeeds():
    agent = _ConsolidateAgent()
    entries = [{"entry_id": "e1", "content": "hi"}]
    data = await _first_data(
        _handler(agent),
        {"sessionId": TEST_SCOPE, "payload": {"scope": "channel", "scope_id": TEST_SCOPE, "entries": entries}},
    )
    assert data["status"] == "ok"
    assert data["scope"] == "channel"
    assert data["scope_id"] == TEST_SCOPE
    assert agent.calls[0]["entries"] == entries


# ---------------------------------------------------------------------------
# G. Chat sets neutral validated context and cleans it up
# ---------------------------------------------------------------------------


class _ChatAgent:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls: list[dict] = []
        self.seen_contexts: list = []

    async def handle_chat(self, user_id: str, content: str):
        from twin.shared.tools.approval_context import get_current_approval_context

        self.calls.append({"user_id": user_id, "content": content})
        self.seen_contexts.append(get_current_approval_context())
        return self.reply


def _chat_params(session_id: str, text: str = "hello") -> dict:
    return {
        "sessionId": session_id,
        "message": {"parts": [{"type": "text", "text": text}]},
    }


@pytest.mark.asyncio
async def test_chat_sets_neutral_context_and_clears_it():
    from twin.shared.tools.approval_context import get_current_approval_context

    agent = _ChatAgent()
    handler = _handler(agent)
    assert get_current_approval_context() is None
    messages = [m async for m in handler.handle_chat_task(_chat_params(TEST_OWNER))]
    assert messages[0].parts[0].text == "ok"
    assert agent.calls == [{"user_id": TEST_OWNER, "content": "hello"}]
    assert len(agent.seen_contexts) == 1
    context = agent.seen_contexts[0]
    assert context is not None
    assert context.platform == "a2a"
    assert context.user_id == TEST_OWNER
    assert context.native_message is None
    assert context.approval_backend is None
    assert get_current_approval_context() is None


@pytest.mark.asyncio
async def test_chat_invalid_session_or_message_never_sets_context():
    from twin.shared.tools.approval_context import get_current_approval_context

    bad_params = [
        {"sessionId": "abc", "message": {"parts": [{"type": "text", "text": "hi"}]}},
        {"sessionId": "", "message": {"parts": [{"type": "text", "text": "hi"}]}},
        {"message": {"parts": [{"type": "text", "text": "hi"}]}},
        {"sessionId": TEST_OWNER, "message": "not-a-dict"},
        {"sessionId": TEST_OWNER, "message": {"parts": "not-a-list"}},
        {"sessionId": TEST_OWNER, "message": {"parts": ["not-a-dict"]}},
        {"sessionId": TEST_OWNER, "message": {"parts": [{"type": "text", "text": 123}]}},
    ]
    for params in bad_params:
        agent = _ChatAgent()
        handler = _handler(agent)
        messages = [m async for m in handler.handle_chat_task(params)]
        assert messages[0].parts[0].text.startswith("Error:"), params
        assert agent.calls == []
        assert get_current_approval_context() is None


@pytest.mark.asyncio
async def test_chat_context_is_cleared_on_agent_error():
    from twin.shared.tools.approval_context import get_current_approval_context

    class _Failing:
        async def handle_chat(self, user_id: str, content: str):
            raise RuntimeError("boom")

    handler = _handler(_Failing())
    messages = [m async for m in handler.handle_chat_task(_chat_params(TEST_OWNER))]
    assert messages[0].parts[0].text.startswith("Error:")
    assert get_current_approval_context() is None


# ---------------------------------------------------------------------------
# H. Boot path: unknown owner denies, configured owner succeeds
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, body: dict, peer="march7"):
        self._body = body
        self._peer = peer

    def get(self, key, default=None):
        if key == "a2a_peer":
            return self._peer
        return default

    async def json(self):
        return dict(self._body)


class _ApproveBackend:
    def __init__(self, approved=True):
        self.approved = approved

    async def request_owner_approval(self, **kwargs):
        return self.approved

    async def send_owner_dm(self, **kwargs):
        return None

    async def notify_channel(self, **kwargs):
        return None


def _boot_handler(monkeypatch, owner_env: str | None, owner_arg):
    from twin.evernight.server.a2a_server import EvernightA2AHandler
    from twin.shared.config.settings import Config

    monkeypatch.setattr(Config, "EVERNIGHT_OWNER_USER_ID", owner_env)
    return EvernightA2AHandler(agent=object(), dm_backend=_ApproveBackend(), owner_user_id=owner_arg)


@pytest.mark.asyncio
async def test_boot_without_owner_denies_approval_and_dm(monkeypatch):
    handler = _boot_handler(monkeypatch, None, None)
    assert handler.owner_user_id == ""
    resp = await handler.handle_dm(_FakeRequest({"type": "approval", "user_id": 1, "command": "x"}))
    assert json.loads(resp.text)["approved"] is False
    resp = await handler.handle_dm(_FakeRequest({"type": "message", "user_id": 1, "content": "hi"}))
    assert resp.status == 503


@pytest.mark.asyncio
async def test_boot_with_explicit_owner_succeeds(monkeypatch, tmp_path):
    from twin.shared.config.settings import Config

    key_file = tmp_path / "approval_secret"
    key_file.write_text("test-approval-key", encoding="utf-8")
    monkeypatch.setattr(Config, "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(key_file))
    handler = _boot_handler(monkeypatch, None, TEST_OWNER)
    assert handler.owner_user_id == TEST_OWNER
    resp = await handler.handle_dm(
        _FakeRequest({"type": "approval", "user_id": int(TEST_OWNER), "command": "echo hi"})
    )
    assert resp.status == 200
    data = json.loads(resp.text)
    assert data["approved"] is True
    assert data["grant"]


@pytest.mark.asyncio
async def test_nonowner_recipient_still_denied_with_owner_configured(monkeypatch):
    handler = _boot_handler(monkeypatch, TEST_OWNER, None)
    assert handler.owner_user_id == TEST_OWNER
    resp = await handler.handle_dm(
        _FakeRequest({"type": "approval", "user_id": int(OTHER_USER), "command": "x"})
    )
    assert resp.status == 403
    assert json.loads(resp.text)["approved"] is False
