from __future__ import annotations

import importlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest


TARGET_MODULES = (
    "gateway.adapters.discord.views.approve_view",
    "gateway.adapters.discord.dm_delivery",
)


class _HTTPException(Exception):
    pass


class _NotFound(_HTTPException):
    pass


class _Forbidden(_HTTPException):
    pass


class _View:
    def __init__(self, *, timeout=None):
        self.timeout = timeout
        self.children = []


def _button(**_kwargs):
    def decorator(func):
        return func

    return decorator


@pytest.fixture()
def fake_discord(monkeypatch):
    for module_name in TARGET_MODULES:
        sys.modules.pop(module_name, None)

    discord = ModuleType("discord")
    discord.NotFound = _NotFound
    discord.Forbidden = _Forbidden
    discord.HTTPException = _HTTPException
    discord.ButtonStyle = SimpleNamespace(green="green", red="red")
    discord.ui = SimpleNamespace(View=_View, button=_button, Button=object)

    discord_ext = ModuleType("discord.ext")
    commands = ModuleType("discord.ext.commands")
    commands.Bot = type("Bot", (), {})
    discord_ext.commands = commands

    monkeypatch.setitem(sys.modules, "discord", discord)
    monkeypatch.setitem(sys.modules, "discord.ext", discord_ext)
    monkeypatch.setitem(sys.modules, "discord.ext.commands", commands)

    yield discord

    for module_name in TARGET_MODULES:
        sys.modules.pop(module_name, None)


@pytest.mark.asyncio
async def test_approve_view_keeps_decision_when_message_update_disappears(
    fake_discord,
    caplog,
):
    module = importlib.import_module("gateway.adapters.discord.views.approve_view")
    view = module.ApproveView(command="printf ok", owner_user_id="111")

    class _Response:
        async def edit_message(self, **_kwargs):
            raise fake_discord.NotFound("message gone")

        async def send_message(self, *args, **_kwargs):
            return None

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=111), response=_Response()
    )

    with caplog.at_level(logging.WARNING):
        await view.approve_button(interaction, None)

    assert view._approved is True
    assert view._result.is_set()
    assert "Approval message disappeared" in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_dm_delivery_notify_skips_inaccessible_channel(
    fake_discord,
    caplog,
):
    module = importlib.import_module("gateway.adapters.discord.dm_delivery")

    class _Bot:
        def is_ready(self):
            return True

        def get_channel(self, _channel_id):
            return None

        async def fetch_channel(self, _channel_id):
            raise fake_discord.Forbidden("missing access")

    delivery = module.DiscordDMDelivery(_Bot())

    with caplog.at_level(logging.WARNING):
        await delivery.notify_channel(channel_id=123, approved=True, label="printf ok")

    assert (
        "Skipping approval result notification for inaccessible channel 123"
        in caplog.text
    )
    assert "Unexpected error" not in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
