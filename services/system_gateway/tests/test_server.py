from __future__ import annotations

from pathlib import Path

from aiohttp.test_utils import make_mocked_request
import pytest

from system_gateway.config import GatewayConfig
from system_gateway.server import CONFIG_KEY, STATE_KEY, capabilities, create_app, health


def _test_config(tmp_path: Path, **overrides) -> GatewayConfig:
    # Explicit private ledger; never the real host default.
    kwargs = dict(approval_ledger_file=tmp_path / "ledger.db")
    kwargs.update(overrides)
    return GatewayConfig(**kwargs)


@pytest.mark.asyncio
async def test_health_response(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path))
    request = make_mocked_request("GET", "/health")
    request._app = app

    response = await health(request)

    assert response.status == 200
    assert response.content_type == "application/json"
    payload = await response.json() if hasattr(response, "json") else None
    if payload is None:
        # Fall back to parsing text manually.
        import json as _json

        payload = _json.loads(response.text)
    assert payload["status"] == "ok"
    assert payload["service"] == "system_gateway"
    assert payload["version"]
    assert payload["platform"]
    assert payload["uptime"] >= 0


@pytest.mark.asyncio
async def test_capabilities_response(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path))
    request = make_mocked_request("GET", "/capabilities")
    request._app = app

    response = await capabilities(request)

    assert response.status == 200
    assert response.content_type == "application/json"
    assert "raw_shell" in response.text


def test_create_app_registers_routes(tmp_path: Path) -> None:
    app = create_app(_test_config(tmp_path, host="0.0.0.0", port=9999))

    route_paths = {
        route.resource.canonical
        for route in app.router.routes()
        if route.resource is not None
    }

    assert app[CONFIG_KEY] == _test_config(tmp_path, host="0.0.0.0", port=9999)
    assert app[STATE_KEY] is not None
    assert route_paths == {
        "/health",
        "/capabilities",
        "/shell/run",
        "/self/update",
    }
