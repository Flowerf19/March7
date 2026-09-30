"""Regression tests for the Evernight Discord compatibility adapter."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from gateway.adapters.discord import evernight_adapter as adapter_module
from gateway.adapters.discord.evernight_adapter import (
    ERROR_MESSAGE,
    EvernightDiscordAdapter,
    _EvernightAgentRouter,
)
from gateway.core.handler import GatewayChatHandler
from gateway.shared.model import UnifiedChannel, UnifiedMessage, UnifiedUser
from twin.shared.tools.approval_context import (
    ApprovalRequestContext,
    get_current_approval_context,
)


OWNER_USER_ID = "726302130318868500"


class _FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAvatar:
    url = "https://example.invalid/avatar.png"


class _FakeUser:
    def __init__(
        self,
        user_id: str,
        *,
        display_name: str = "Owner",
        name: str = "owner",
        bot: bool = False,
    ) -> None:
        self.id = int(user_id)
        self.display_name = display_name
        self.name = name
        self.bot = bot
        self.avatar = _FakeAvatar()


class _FakeGuild:
    def __init__(self) -> None:
        self.id = 333
        self.name = "guild"


class _FakeChannel:
    def __init__(self) -> None:
        self.id = 222
        self.name = "general"
        self.sent: list[str] = []

    async def send(self, content: str):
        self.sent.append(content)

    def typing(self):
        return _FakeTyping()


class _FakeMessage:
    def __init__(
        self,
        *,
        content: str,
        guild: _FakeGuild | None,
        mentions: list[_FakeUser] | None = None,
    ) -> None:
        self.id = 111
        self.content = content
        self.author = _FakeUser(OWNER_USER_ID)
        self.channel = _FakeChannel()
        self.guild = guild
        self.mentions = mentions or []
        self.attachments = []
        self.reference = None
        self.created_at = datetime.now(timezone.utc)


class _FakeBot:
    def __init__(self, user: _FakeUser) -> None:
        self.user = user


class _RejectDirectAgent:
    async def handle_chat(self, **kwargs):
        raise AssertionError("Evernight agent must not be called directly by the adapter")


class _SpyHandler:
    def __init__(self) -> None:
        self.messages: list[UnifiedMessage] = []

    async def handle_message(self, msg: UnifiedMessage) -> str:
        self.messages.append(msg)
        return "reply from gate"


class _FakeEvernightAgent:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def handle_chat(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "evernight reply"


def _fake_to_unified(
    message: _FakeMessage,
    bot_user: _FakeUser | None = None,
    content_override: str | None = None,
) -> UnifiedMessage:
    guild_id = str(message.guild.id) if message.guild else None
    channel_type = "guild" if message.guild else "dm"
    return UnifiedMessage(
        message_id=str(message.id),
        user=UnifiedUser(
            platform_id=str(message.author.id),
            platform_name="discord",
            display_name=message.author.display_name,
        ),
        channel=UnifiedChannel(
            channel_id=str(message.channel.id),
            platform_name="discord",
            channel_type=channel_type,
            name=message.channel.name,
            guild_id=guild_id,
        ),
        content=content_override if content_override is not None else message.content,
        timestamp=message.created_at,
        mentions=[
            UnifiedUser(
                platform_id=str(user.id),
                platform_name="discord",
                display_name=user.display_name,
                is_bot=user.bot,
            )
            for user in message.mentions
        ],
        extensions={},
    )


def _fake_approval_context(message: _FakeMessage) -> ApprovalRequestContext:
    return ApprovalRequestContext(
        platform="discord",
        user_id=str(message.author.id),
        conversation_id=str(message.channel.id),
        channel_id=str(message.channel.id),
        message_id=str(message.id),
        channel_name=f"#{message.channel.name}",
        space_id=str(message.guild.id) if message.guild else None,
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(
        adapter_module.DiscordMessageConverter,
        "to_unified",
        staticmethod(_fake_to_unified),
    )
    monkeypatch.setattr(
        adapter_module,
        "build_discord_approval_context",
        _fake_approval_context,
    )

    bot_user = _FakeUser("999", display_name="Evernight", name="evernight", bot=True)
    instance = EvernightDiscordAdapter.__new__(EvernightDiscordAdapter)
    instance._agent = _RejectDirectAgent()
    instance._owner_user_id = OWNER_USER_ID
    instance._handler = _SpyHandler()
    instance._bot = _FakeBot(bot_user)
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "guild", "mentions_factory", "expected_content", "expected_mentioned"),
    [
        ("hello in dm", None, lambda bot_user: [], "hello in dm", False),
        ("!9 hello by prefix", _FakeGuild(), lambda bot_user: [], "hello by prefix", False),
        (
            "<@999> hello by tag",
            _FakeGuild(),
            lambda bot_user: [bot_user],
            "hello by tag",
            True,
        ),
    ],
)
async def test_evernight_discord_message_enters_gateway_contract(
    adapter,
    content: str,
    guild: _FakeGuild | None,
    mentions_factory,
    expected_content: str,
    expected_mentioned: bool,
):
    message = _FakeMessage(
        content=content,
        guild=guild,
        mentions=mentions_factory(adapter._bot.user),
    )

    await adapter._on_message(message)

    assert message.channel.sent == ["reply from gate"]
    assert len(adapter._handler.messages) == 1
    unified = adapter._handler.messages[0]
    assert unified.content == expected_content
    assert unified.user.platform_id == OWNER_USER_ID
    assert unified.extensions["agent_name"] == "evernight"
    assert unified.extensions["is_addressed"] is True
    assert unified.extensions["is_mentioned"] is expected_mentioned
    assert unified.extensions["should_respond"] is True
    assert unified.extensions["observe"] is True
    assert unified.extensions["assistant_id"] == "999"
    assert unified.extensions["assistant_name"] == "Evernight"
    assert get_current_approval_context() is None


@pytest.mark.asyncio
async def test_evernight_local_router_only_routes_evernight():
    agent = _FakeEvernightAgent()
    router = _EvernightAgentRouter(agent)

    response = await router.route(
        agent_name="evernight",
        user_id="u1",
        content="hello",
        channel_id="ignored",
    )
    wrong_route = await router.route(
        agent_name="march7",
        user_id="u1",
        content="hello",
    )

    assert response == "evernight reply"
    assert agent.calls == [{"user_id": "u1", "content": "hello", "observe_input": True}]
    assert wrong_route == ERROR_MESSAGE


@pytest.mark.asyncio
async def test_evernight_local_router_forwards_observe_input_flag():
    agent = _FakeEvernightAgent()
    router = _EvernightAgentRouter(agent)

    await router.route(
        agent_name="evernight",
        user_id="u1",
        content="already observed",
        observe_input=False,
    )

    assert agent.calls == [
        {"user_id": "u1", "content": "already observed", "observe_input": False}
    ]


class _FakeMemory:
    """Local T1 double with the agent's observe semantics."""

    def __init__(self) -> None:
        self.observed: list[dict] = []

    async def add_message(self, user_id: str, role: str, content: str) -> None:
        self.observed.append({"user_id": user_id, "role": role, "content": content})


class _FakeMemoryAgent:
    """Evernight agent double: observes on handle_chat unless already done."""

    def __init__(self) -> None:
        self.memory = _FakeMemory()
        self.calls: list[dict] = []

    async def handle_chat(
        self, user_id: str, content: str, observe_input: bool = True
    ) -> str:
        self.calls.append(
            {"user_id": user_id, "content": content, "observe_input": observe_input}
        )
        if observe_input:
            await self.memory.add_message(
                user_id=user_id, role="user", content=content
            )
        return "evernight reply"


@pytest.fixture
def live_adapter(monkeypatch):
    """Adapter wired to the real core handler and a memory-backed agent."""
    monkeypatch.setattr(
        adapter_module.DiscordMessageConverter,
        "to_unified",
        staticmethod(_fake_to_unified),
    )
    monkeypatch.setattr(
        adapter_module,
        "build_discord_approval_context",
        _fake_approval_context,
    )
    monkeypatch.setattr("gateway.core.handler.MESSAGE_DEBOUNCE_SECONDS", 0)

    bot_user = _FakeUser("999", display_name="Evernight", name="evernight", bot=True)
    agent = _FakeMemoryAgent()
    instance = EvernightDiscordAdapter.__new__(EvernightDiscordAdapter)
    instance._agent = agent
    instance._owner_user_id = OWNER_USER_ID
    instance._handler = GatewayChatHandler(
        agent_router=_EvernightAgentRouter(agent)
    )
    instance._bot = _FakeBot(bot_user)
    return instance


@pytest.mark.asyncio
async def test_evernight_burst_preserves_every_input_single_reply(live_adapter):
    messages = [_FakeMessage(content=f"burst {i}", guild=None) for i in range(3)]
    for i, message in enumerate(messages):
        message.id = 100 + i

    await asyncio.gather(*(live_adapter._on_message(message) for message in messages))

    agent = live_adapter._agent
    assert agent.memory.observed == [
        {"user_id": OWNER_USER_ID, "role": "user", "content": f"burst {i}"}
        for i in range(3)
    ]
    assert len(agent.calls) == 1
    assert agent.calls[0]["content"] == "burst 2"
    assert agent.calls[0]["observe_input"] is False
    assert messages[0].channel.sent == []
    assert messages[1].channel.sent == []
    assert messages[2].channel.sent == ["evernight reply"]


@pytest.mark.asyncio
async def test_evernight_ignores_bot_unauthorized_irrelevant(live_adapter):
    bot_msg = _FakeMessage(content="bot noise", guild=None)
    bot_msg.author = _FakeUser(OWNER_USER_ID, bot=True)
    stranger_msg = _FakeMessage(content="!9 hack", guild=None)
    stranger_msg.author = _FakeUser("123")
    irrelevant_msg = _FakeMessage(content="random chat", guild=_FakeGuild())
    empty_msg = _FakeMessage(content="   ", guild=None)

    for message in (bot_msg, stranger_msg, irrelevant_msg, empty_msg):
        await live_adapter._on_message(message)

    agent = live_adapter._agent
    assert agent.memory.observed == []
    assert agent.calls == []
    for message in (bot_msg, stranger_msg, irrelevant_msg, empty_msg):
        assert message.channel.sent == []
