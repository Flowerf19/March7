"""Tests for the platform-neutral gateway chat handler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from gateway.core.handler import GatewayChatHandler
from gateway.shared.model import UnifiedChannel, UnifiedMessage, UnifiedUser


class FakeMemory:
    def __init__(self) -> None:
        self.user_observations: list[dict] = []
        self.channel_observations: list[dict] = []

    async def observe_user_message(self, **kwargs):
        self.user_observations.append(kwargs)

    async def observe_channel_message(self, **kwargs):
        self.channel_observations.append(kwargs)


class FakeMarch7:
    def __init__(self) -> None:
        self.memory = FakeMemory()

    async def handle_chat(self, **kwargs) -> str:
        self.last_chat = kwargs
        return "core reply"


class FakeRouter:
    def __init__(self) -> None:
        self.march7 = FakeMarch7()
        self.calls: list[dict] = []

    async def route(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return await self.march7.handle_chat(**kwargs)


def _message(
    *,
    channel_type: str = "dm",
    extensions: dict | None = None,
    guild_id: str | None = None,
) -> UnifiedMessage:
    return UnifiedMessage(
        message_id="m1",
        user=UnifiedUser(
            platform_id="u1",
            platform_name="mock",
            display_name="User One",
        ),
        channel=UnifiedChannel(
            channel_id="c1",
            platform_name="mock",
            channel_type=channel_type,
            guild_id=guild_id,
        ),
        content="hello core",
        timestamp=datetime.now(timezone.utc),
        mentions=[
            UnifiedUser(
                platform_id="u2",
                platform_name="mock",
                display_name="User Two",
            )
        ],
        extensions=extensions,
    )


@pytest.mark.asyncio
async def test_clean_unified_message_routes_without_raw_discord(monkeypatch):
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)
    router = FakeRouter()
    handler = GatewayChatHandler(agent_router=router)

    response = await handler.handle_message(
        _message(
            extensions={
                "is_addressed": True,
                "assistant_id": "bot1",
                "assistant_name": "Core Bot",
            }
        )
    )

    assert response == "core reply"
    assert len(router.calls) == 1
    assert router.calls[0]["agent_name"] == "march7"
    assert router.calls[0]["user_id"] == "u1"
    assert router.calls[0]["content"] == "hello core"
    assert router.calls[0]["bot_id"] == "bot1"
    assert router.calls[0]["bot_name"] == "Core Bot"
    assert router.calls[0]["mentioned_users"] == [
        {"user_id": "u2", "display_name": "User Two", "is_bot": False}
    ]
    assert router.march7.memory.user_observations == [
        {"user_id": "u1", "role": "user", "content": "hello core"}
    ]


@pytest.mark.asyncio
async def test_observe_only_channel_observes_without_routing(monkeypatch):
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)
    router = FakeRouter()
    handler = GatewayChatHandler(agent_router=router)

    response = await handler.handle_message(
        _message(
            channel_type="guild",
            guild_id="g1",
            extensions={"should_respond": False, "is_addressed": False},
        )
    )

    assert response == ""
    assert router.calls == []
    assert router.march7.memory.channel_observations == [
        {
            "guild_id": "g1",
            "channel_id": "c1",
            "author_id": "u1",
            "author_name": "User One",
            "message_id": "m1",
            "content": "hello core",
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_non_dm_without_guild_still_uses_channel_scope(monkeypatch):
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)
    router = FakeRouter()
    handler = GatewayChatHandler(agent_router=router)

    response = await handler.handle_message(
        _message(
            channel_type="group",
            guild_id=None,
            extensions={"should_respond": False, "is_addressed": False},
        )
    )

    assert response == ""
    assert router.calls == []
    assert router.march7.memory.user_observations == []
    assert router.march7.memory.channel_observations == [
        {
            "guild_id": "",
            "channel_id": "c1",
            "author_id": "u1",
            "author_name": "User One",
            "message_id": "m1",
            "content": "hello core",
            "reply_to": None,
        }
    ]


@pytest.mark.asyncio
async def test_non_march7_route_does_not_observe_march7_memory(monkeypatch):
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)
    router = FakeRouter()
    handler = GatewayChatHandler(agent_router=router)

    response = await handler.handle_message(
        _message(
            extensions={
                "agent_name": "evernight",
                "is_addressed": True,
            }
        )
    )

    assert response == "core reply"
    assert router.calls[0]["agent_name"] == "evernight"
    assert router.march7.memory.user_observations == []
    assert router.march7.memory.channel_observations == []


class FakeEvernightMemory:
    def __init__(self) -> None:
        self.observed: list[dict] = []

    async def add_message(self, user_id: str, role: str, content: str) -> None:
        self.observed.append({"user_id": user_id, "role": role, "content": content})


class FakeEvernightAgent:
    def __init__(self) -> None:
        self.memory = FakeEvernightMemory()


class FakeEvernightRouter:
    march7 = None

    def __init__(self) -> None:
        self.evernight = FakeEvernightAgent()
        self.calls: list[dict] = []

    async def route(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "evernight reply"


def _evernight_message(message_id: str, content: str) -> UnifiedMessage:
    return UnifiedMessage(
        message_id=message_id,
        user=UnifiedUser(
            platform_id="u1",
            platform_name="mock",
            display_name="User One",
        ),
        channel=UnifiedChannel(
            channel_id="c1",
            platform_name="mock",
            channel_type="dm",
        ),
        content=content,
        timestamp=datetime.now(timezone.utc),
        extensions={
            "agent_name": "evernight",
            "is_addressed": True,
            "should_respond": True,
            "observe": True,
        },
    )


@pytest.mark.asyncio
async def test_evernight_burst_observes_every_input_replies_once(monkeypatch):
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)
    router = FakeEvernightRouter()
    handler = GatewayChatHandler(agent_router=router)

    messages = [_evernight_message(f"m{i}", f"burst {i}") for i in range(3)]
    results = await asyncio.gather(*(handler.handle_message(msg) for msg in messages))

    assert router.evernight.memory.observed == [
        {"user_id": "u1", "role": "user", "content": f"burst {i}"} for i in range(3)
    ]
    assert results == ["", "", "evernight reply"]
    assert len(router.calls) == 1
    assert router.calls[0]["content"] == "burst 2"
    assert router.calls[0]["observe_input"] is False


@pytest.mark.asyncio
async def test_evernight_without_local_memory_skips_local_observe(monkeypatch):
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)
    router = FakeRouter()
    handler = GatewayChatHandler(agent_router=router)

    response = await handler.handle_message(
        _message(
            extensions={
                "agent_name": "evernight",
                "is_addressed": True,
                "observe": True,
            }
        )
    )

    assert response == "core reply"
    assert router.march7.memory.user_observations == []
    assert router.march7.memory.channel_observations == []
