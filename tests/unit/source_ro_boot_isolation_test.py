"""Plan A source-RO + own-persona overlay + None-safe owner boot.

Verifies:
- ../../twin and ../../gateway are read-only in both agents; only the agent's
  OWN persona dir is a writable overlay (peer stays read-only).
- Models are read-only at runtime (host pull script only); shared memories
  and data mounts stay writable.
- No private issuer credential/path reaches March7; generic DISCORD_TOKEN
  aliases cannot leak peer authority.
- Persona dirs are data-only (.md, no executable); Eve imports no code from
  the March7 writable path.
- Owner-absent boot via the real config entry flow neither crashes
  (ValueError/TypeError) nor grants authority.

Uses explicit nonsecret test owner; no real keys.
"""
from __future__ import annotations

import pathlib

import pytest

TEST_OWNER = "100000000000000001"
OTHER_USER = "100000000000000002"

REPO = pathlib.Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _volume_lines(text: str, target: str) -> list[str]:
    return [ln for ln in text.splitlines() if target in ln and "- " in ln]


# ---------------------------------------------------------------------------
# A. Source binds read-only, own persona overlay writable only
# ---------------------------------------------------------------------------


def test_source_binds_are_readonly_both_agents():
    for rel in ("docker/march7/docker-compose.yml", "docker/evernight/docker-compose.yml"):
        text = _read(rel)
        for target in ("/app/twin:", "/app/gateway:"):
            # Parent source mount must be :ro; no writable twin/gateway bind.
            parent = [ln for ln in _volume_lines(text, target) if "personas" not in ln]
            assert parent, f"{rel}: missing {target} mount"
            for ln in parent:
                assert ":ro" in ln, f"{rel}: writable source bind: {ln}"
        # Regression: old writable short binds must be gone.
        assert "../../twin:/app/twin:z" not in text, rel
        assert "../../gateway:/app/gateway:z" not in text, rel


def test_own_persona_overlay_writable_only():
    march7 = _read("docker/march7/docker-compose.yml")
    evernight = _read("docker/evernight/docker-compose.yml")

    m_own = _volume_lines(march7, "/app/twin/march7/personas")
    assert len(m_own) == 1, march7
    assert ":ro" not in m_own[0], m_own[0]

    e_own = _volume_lines(evernight, "/app/twin/evernight/personas")
    assert len(e_own) == 1, evernight
    assert ":ro" not in e_own[0], e_own[0]

    # Overlay must shadow the read-only parent (order matters).
    assert march7.index("/app/twin:ro") < march7.index("/app/twin/march7/personas")
    assert evernight.index("/app/twin:ro") < evernight.index("/app/twin/evernight/personas")


def test_peer_persona_not_writable():
    march7 = _read("docker/march7/docker-compose.yml")
    evernight = _read("docker/evernight/docker-compose.yml")
    # Own-only: peer persona path must not appear as a writable overlay.
    # (It stays read-only via the parent ../../twin:ro mount.)
    assert "evernight/personas" not in march7
    assert "march7/personas" not in evernight


def test_models_readonly_memories_data_writable():
    for rel in ("docker/march7/docker-compose.yml", "docker/evernight/docker-compose.yml"):
        text = _read(rel)
        models = _volume_lines(text, "/app/models")
        assert models, f"{rel}: missing models mount"
        for ln in models:
            assert ":ro" in ln, f"{rel}: models must be read-only: {ln}"
        for target in ("/app/memories", "/app/data"):
            lines = _volume_lines(text, target)
            assert lines, f"{rel}: missing {target} mount"
            for ln in lines:
                assert ":ro" not in ln, f"{rel}: {target} must stay writable: {ln}"


def test_persona_overlay_matches_config_defaults():
    from twin.evernight.config import EvernightConfig
    from twin.march7.config import March7Config

    assert March7Config().persona_path == "twin/march7/personas"
    assert EvernightConfig().persona_path == "twin/evernight/personas"
    march7 = _read("docker/march7/docker-compose.yml")
    evernight = _read("docker/evernight/docker-compose.yml")
    assert "../../twin/march7/personas:/app/twin/march7/personas" in march7
    assert "../../twin/evernight/personas:/app/twin/evernight/personas" in evernight


# ---------------------------------------------------------------------------
# B. No private issuer credential/path in March7; no generic token alias
# ---------------------------------------------------------------------------


def test_no_private_issuer_credential_in_march7():
    march7 = _read("docker/march7/docker-compose.yml")
    assert "/run/secrets/system_gateway_approval" not in march7
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_HOST_PATH" not in march7
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET=" in march7
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE=" in march7
    for ln in march7.splitlines():
        if "APPROVAL_SECRET" in ln:
            assert ln.strip().endswith("="), ln
    evernight = _read("docker/evernight/docker-compose.yml")
    # Long bind preserved (no auto-create, read-only, SELinux, host override).
    assert "target: /run/secrets/system_gateway_approval" in evernight
    assert "read_only: true" in evernight
    assert "create_host_path: false" in evernight
    assert "SYSTEM_GATEWAY_APPROVAL_SECRET_HOST_PATH" in evernight


def test_no_generic_discord_token_alias_in_compose():
    for rel in ("docker/march7/docker-compose.yml", "docker/evernight/docker-compose.yml"):
        for ln in _read(rel).splitlines():
            if "DISCORD_" in ln and "_TOKEN=" in ln:
                assert "DISCORD_MARCH7_TOKEN" in ln or "DISCORD_EVERNIGHT_TOKEN" in ln, f"{rel}: {ln}"


def test_no_generic_discord_token_env_in_code():
    # Only scoped token env vars may be read; a generic DISCORD_TOKEN alias
    # would bypass the per-agent blanking in compose.
    hits: list[str] = []
    for base in ("twin", "gateway"):
        for path in (REPO / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if 'getenv("DISCORD_TOKEN"' in text or "getenv('DISCORD_TOKEN'" in text:
                hits.append(str(path))
            if 'getenv("DISCORD_BOT_TOKEN"' in text or "getenv('DISCORD_BOT_TOKEN'" in text:
                hits.append(str(path))
    assert hits == []


# ---------------------------------------------------------------------------
# C. Persona dirs are data-only; no executable import from writable path
# ---------------------------------------------------------------------------


def test_persona_dirs_are_data_only():
    for agent in ("march7", "evernight"):
        d = REPO / "twin" / agent / "personas"
        assert (d / "SOUL.md").exists()
        assert (d / "IDENTITY.md").exists()
        assert list(d.glob("*.py")) == []
        assert not (d / "__init__.py").exists()


def test_no_executable_import_from_persona_paths():
    offenders: list[str] = []
    for base in ("twin", "gateway"):
        for path in (REPO / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "personas" not in text:
                continue
            low = text.lower()
            if "from twin." in low and "personas" in low and "import" in low:
                offenders.append(f"{path}: from-import from personas")
            if "import_module" in text and "persona" in low:
                offenders.append(f"{path}: import_module persona")
            if "sys.path" in text and "persona" in low:
                offenders.append(f"{path}: sys.path persona")
    assert offenders == []
    # Persona content is loaded as prompt text, never executed.
    loader = _read("twin/shared/llm/base_llm_service.py")
    assert 'open(filepath, "r", encoding="utf-8")' in loader
    tool_src = _read("twin/shared/tools/modules/profile/update_personality_tool.py")
    assert ".md" in tool_src
    assert "import_module" not in tool_src


# ---------------------------------------------------------------------------
# D. Owner-absent boot via real config flow: no crash, no authority
# ---------------------------------------------------------------------------


def test_config_owner_absent_returns_none(monkeypatch):
    from twin.evernight.config import EvernightConfig

    monkeypatch.delenv("EVERNIGHT_OWNER_USER_ID", raising=False)
    assert EvernightConfig.from_env().owner_user_id is None


def test_notify_helper_is_none_safe():
    from twin.evernight.__main__ import _notify_user_id

    assert _notify_user_id(None) is None
    assert _notify_user_id("") is None
    assert _notify_user_id("   ") is None
    assert _notify_user_id("abc") is None
    assert _notify_user_id("12ab") is None
    assert _notify_user_id(True) is None
    assert _notify_user_id(-5) is None
    assert _notify_user_id(TEST_OWNER) == int(TEST_OWNER)
    assert _notify_user_id(f"  {TEST_OWNER}  ") == int(TEST_OWNER)
    assert _notify_user_id(int(TEST_OWNER)) == int(TEST_OWNER)
    # No hardcoded owner literal in the boot entry point.
    src = _read("twin/evernight/__main__.py")
    assert "726302130318868500" not in src
    assert "int(config.owner_user_id)" not in src


def test_boot_owner_absent_no_crash_no_authority(monkeypatch):
    """Real entry flow: from_env(None) -> helper(None) -> monitor + deny."""
    from twin.evernight.__main__ import _notify_user_id
    from twin.evernight.config import EvernightConfig
    from twin.evernight.self_heal.monitor import OwnerApprovalRecoveryExecutor, SelfHealMonitor
    from twin.evernight.server.a2a_server import EvernightA2AHandler

    monkeypatch.delenv("EVERNIGHT_OWNER_USER_ID", raising=False)
    config = EvernightConfig.from_env()
    assert config.owner_user_id is None

    notify = _notify_user_id(config.owner_user_id)
    assert notify is None

    monitor = SelfHealMonitor(notify_user_id=notify, gateway_monitor=object())
    assert monitor._notify_user_id is None
    assert isinstance(monitor._recovery_executor, OwnerApprovalRecoveryExecutor)
    assert monitor._recovery_executor._owner_user_id is None

    handler = EvernightA2AHandler(agent=object(), owner_user_id=config.owner_user_id)
    assert handler.owner_user_id == ""


@pytest.mark.asyncio
async def test_boot_owner_absent_denies_approval_and_dm(monkeypatch):
    import json

    from twin.evernight.config import EvernightConfig
    from twin.evernight.server.a2a_server import EvernightA2AHandler

    monkeypatch.delenv("EVERNIGHT_OWNER_USER_ID", raising=False)
    config = EvernightConfig.from_env()

    class _Req:
        def __init__(self, body):
            self._body = body

        def get(self, key, default=None):
            return "march7" if key == "a2a_peer" else default

        async def json(self):
            return dict(self._body)

    class _Backend:
        async def request_owner_approval(self, **kwargs):
            raise AssertionError("must not ask owner when unknown")

        async def send_owner_dm(self, **kwargs):
            raise AssertionError("must not DM when owner unknown")

        async def notify_channel(self, **kwargs):
            return None

    handler = EvernightA2AHandler(agent=object(), dm_backend=_Backend(), owner_user_id=config.owner_user_id)
    resp = await handler.handle_dm(_Req({"type": "approval", "user_id": 1, "command": "x"}))
    assert json.loads(resp.text)["approved"] is False
    resp = await handler.handle_dm(_Req({"type": "message", "user_id": 1, "content": "hi"}))
    assert resp.status == 503


@pytest.mark.asyncio
async def test_boot_owner_configured_preserves_authority(monkeypatch, tmp_path):
    import json

    from twin.evernight.__main__ import _notify_user_id
    from twin.evernight.config import EvernightConfig
    from twin.evernight.server.a2a_server import EvernightA2AHandler
    from twin.shared.config.settings import Config

    monkeypatch.setenv("EVERNIGHT_OWNER_USER_ID", TEST_OWNER)
    config = EvernightConfig.from_env()
    assert config.owner_user_id == TEST_OWNER
    assert _notify_user_id(config.owner_user_id) == int(TEST_OWNER)

    key_file = tmp_path / "approval_secret"
    key_file.write_text("test-approval-key", encoding="utf-8")
    monkeypatch.setattr(Config, "SYSTEM_GATEWAY_APPROVAL_SECRET_FILE", str(key_file))

    class _Req:
        def get(self, key, default=None):
            return "march7" if key == "a2a_peer" else default

        async def json(self):
            return {"type": "approval", "user_id": int(TEST_OWNER), "command": "echo hi"}

    class _Backend:
        async def request_owner_approval(self, **kwargs):
            assert kwargs["owner_user_id"] == TEST_OWNER
            return True

        async def send_owner_dm(self, **kwargs):
            return None

        async def notify_channel(self, **kwargs):
            return None

    handler = EvernightA2AHandler(agent=object(), dm_backend=_Backend(), owner_user_id=config.owner_user_id)
    resp = await handler.handle_dm(_Req())
    data = json.loads(resp.text)
    assert data["approved"] is True
    assert data["grant"]

    class _ReqOther:
        def get(self, key, default=None):
            return "march7" if key == "a2a_peer" else default

        async def json(self):
            return {"type": "approval", "user_id": int(OTHER_USER), "command": "x"}

    denied = await handler.handle_dm(_ReqOther())
    assert denied.status == 403
