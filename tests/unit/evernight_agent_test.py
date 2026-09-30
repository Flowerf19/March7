"""Focused tests for Evernight chat behavior."""
from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

from twin.evernight.agent import EvernightAgent, LLM_FAILURE_REPLY
from twin.shared.llm.base_llm_service import LLM_ERROR_RESPONSE
from twin.shared.llm.llm_response import LLMResponse


class FakeMemoryManager:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.get_context_calls: list[dict[str, Any]] = []

    async def add_message(self, user_id: str, role: str, content: str) -> None:
        self.messages.append({"user_id": user_id, "role": role, "content": content})

    async def get_context(self, user_id: str, current_query: str):
        self.get_context_calls.append({"user_id": user_id, "current_query": current_query})
        return "sys-prompt", list(self.messages)


class FakeLLM:
    def __init__(self, response: str | LLMResponse = "echo") -> None:
        self.response = response
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_system_prompt: str | None = None

    async def generate_response(
        self, messages, system_prompt=None, use_native_tools=False, include_tool_catalog=True, max_tokens=None, tool_choice=None
    ):
        self.last_messages = list(messages)
        self.last_system_prompt = system_prompt
        return self.response


def _make_agent(memory: FakeMemoryManager, response: str | LLMResponse = "echo") -> EvernightAgent:
    llm = FakeLLM(response=response)
    agent = EvernightAgent(
        memory_manager=memory,  # type: ignore[arg-type]
        llm_service=llm,  # type: ignore[arg-type]
        tool_registry=None,
        use_native_tools=False,
    )
    agent.handle_chat = MethodType(  # type: ignore[method-assign]
        EvernightAgent.handle_chat.__wrapped__,  # type: ignore[attr-defined]
        agent,
    )
    agent._fake_llm = llm  # type: ignore[attr-defined]
    return agent


@pytest.mark.asyncio
async def test_evernight_chat_saves_user_and_assistant_in_user_scope():
    memory = FakeMemoryManager()
    agent = _make_agent(memory, response="đã ghi nhận")

    response = await agent.handle_chat(user_id="u1", content="nhớ giúp tôi")

    assert response == "đã ghi nhận"
    assert memory.get_context_calls == [{"user_id": "u1", "current_query": "nhớ giúp tôi"}]
    assert memory.messages == [
        {"user_id": "u1", "role": "user", "content": "nhớ giúp tôi"},
        {"user_id": "u1", "role": "assistant", "content": "đã ghi nhận"},
    ]


@pytest.mark.asyncio
async def test_evernight_chat_llm_error_sentinel_not_relayed_or_saved():
    memory = FakeMemoryManager()
    agent = _make_agent(memory, response=LLM_ERROR_RESPONSE)

    response = await agent.handle_chat(user_id="u1", content="alo")

    assert response == LLM_FAILURE_REPLY
    assert response != LLM_ERROR_RESPONSE
    assert memory.messages == [{"user_id": "u1", "role": "user", "content": "alo"}]


@pytest.mark.asyncio
async def test_evernight_chat_reasoning_only_response_not_saved():
    memory = FakeMemoryManager()
    agent = _make_agent(
        memory,
        response=LLMResponse(content="đang suy luận", reasoning_only=True),
    )

    response = await agent.handle_chat(user_id="u1", content="phân tích đi")

    assert response == "đang suy luận"
    assert memory.messages == [
        {"user_id": "u1", "role": "user", "content": "phân tích đi"},
    ]


@pytest.mark.asyncio
async def test_evernight_chat_observe_input_true_persists_user_message():
    memory = FakeMemoryManager()
    agent = _make_agent(memory, response="rõ rồi")

    response = await agent.handle_chat(
        user_id="u1", content="nhớ giúp tôi", observe_input=True
    )

    assert response == "rõ rồi"
    assert memory.messages == [
        {"user_id": "u1", "role": "user", "content": "nhớ giúp tôi"},
        {"user_id": "u1", "role": "assistant", "content": "rõ rồi"},
    ]


@pytest.mark.asyncio
async def test_evernight_chat_observe_input_false_skips_user_persist():
    memory = FakeMemoryManager()
    agent = _make_agent(memory, response="rõ rồi")

    response = await agent.handle_chat(
        user_id="u1", content="đã observe trước", observe_input=False
    )

    assert response == "rõ rồi"
    assert memory.messages == [
        {"user_id": "u1", "role": "assistant", "content": "rõ rồi"},
    ]
