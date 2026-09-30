"""A2A HTTP client for task submission and SSE result consumption."""
from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator, Optional
from urllib.parse import urlsplit

import aiohttp

from twin.shared.a2a.auth import A2AAuthError, make_a2a_headers
from twin.shared.a2a.sse import A2AStreamError, SseParser
from twin.shared.a2a.types import A2AMessage, A2ATask, AgentCard, Part, TaskStatus
from twin.shared.config.settings import Config
from twin.shared.observability import a2a_parent_headers

logger = logging.getLogger(__name__)

__all__ = ["A2AClient", "A2AStreamError"]


class A2AClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        *,
        actor: str,
        secret: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.actor = actor
        self._secret = secret
        self._session: Optional[aiohttp.ClientSession] = None

    def _resolve_secret(self) -> str:
        resolved = self._secret or Config.A2A_SHARED_SECRET
        if not resolved:
            raise A2AAuthError("missing A2A shared secret")
        return resolved

    def _signed_headers(self, method: str, url: str, body: bytes) -> dict[str, str]:
        path = urlsplit(url).path or "/"
        return make_a2a_headers(
            self.actor, method, path, body, secret=self._resolve_secret()
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_agent_card(self) -> Optional[AgentCard]:
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/.well-known/agent.json") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return AgentCard(**data)
                return None
        except Exception as e:
            logger.error("Failed to get agent card from %s: %s", self.base_url, e)
            return None

    async def send_task(self, params: dict) -> A2ATask:
        session = await self._get_session()
        params = dict(params)
        trace_headers = params.pop("_langsmith_parent", None) or a2a_parent_headers()
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": params,
            "id": str(uuid.uuid4()),
        }
        url = f"{self.base_url}/"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        async with session.post(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                **trace_headers,
                **self._signed_headers("POST", url, body),
            },
        ) as resp:
            data = await resp.json()
            if "result" in data:
                return self._parse_task(data["result"])
            raise RuntimeError(f"A2A error: {data.get('error', data)}")

    async def get_task(self, task_id: str) -> Optional[A2ATask]:
        session = await self._get_session()
        trace_headers = a2a_parent_headers()
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }
        url = f"{self.base_url}/"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            async with session.post(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    **trace_headers,
                    **self._signed_headers("POST", url, body),
                },
            ) as resp:
                data = await resp.json()
                if "result" in data:
                    return self._parse_task(data["result"])
                return None
        except Exception as e:
            logger.error("Failed to get task %s from %s: %s", task_id, self.base_url, e)
            return None

    async def cancel_task(self, task_id: str) -> Optional[A2ATask]:
        session = await self._get_session()
        trace_headers = a2a_parent_headers()
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }
        url = f"{self.base_url}/"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            async with session.post(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    **trace_headers,
                    **self._signed_headers("POST", url, body),
                },
            ) as resp:
                data = await resp.json()
                if "result" in data:
                    return self._parse_task(data["result"])
                return None
        except Exception as e:
            logger.error("Failed to cancel task %s on %s: %s", task_id, self.base_url, e)
            return None

    async def subscribe_stream(self, task_id: str) -> AsyncIterator[A2AMessage]:
        session = await self._get_session()
        trace_headers = a2a_parent_headers()
        url = f"{self.base_url}/tasks/{task_id}/stream"
        async with session.get(
            url,
            headers={
                "Accept": "text/event-stream",
                **trace_headers,
                **self._signed_headers("GET", url, b""),
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout, sock_read=self.timeout),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Stream error: {resp.status}")

            parser = SseParser()
            async for chunk in resp.content.iter_chunked(1024):
                for payload in parser.feed(chunk):
                    try:
                        yield self._parse_message(payload)
                    except (AttributeError, TypeError, ValueError) as exc:
                        raise A2AStreamError(
                            f"A2A stream truncated: invalid message ({exc})"
                        ) from exc
            parser.finish()

    async def send_task_and_wait(self, params: dict) -> list[A2AMessage]:
        if "id" not in params:
            params = {**params, "id": str(uuid.uuid4())}

        task = await self.send_task(params)
        messages: list[A2AMessage] = []
        async for message in self.subscribe_stream(task.id):
            messages.append(message)

        final_task = await self.get_task(task.id)
        if final_task is None:
            raise RuntimeError(f"A2A task status unknown: {task.id}")
        if final_task.status == TaskStatus.FAILED:
            raise RuntimeError(self._last_text(messages) or f"A2A task failed: {task.id}")
        if final_task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"A2A task cancelled: {task.id}")
        if final_task.status != TaskStatus.COMPLETED:
            raise RuntimeError(
                f"A2A task incomplete: {task.id} status={final_task.status.value}"
            )
        return messages

    async def send_text_task(
        self,
        *,
        skill: str,
        session_id: str,
        text: str,
        task_id: str | None = None,
        trace_parent: dict[str, str] | None = None,
    ) -> str:
        task_params = {
            "id": task_id or str(uuid.uuid4()),
            "sessionId": session_id,
            "skill": skill,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": text}],
            },
        }
        if trace_parent:
            task_params["_langsmith_parent"] = trace_parent
        messages = await self.send_task_and_wait(task_params)
        return self._last_text(messages)

    async def send_data_task(
        self,
        *,
        skill: str,
        session_id: str,
        params: dict | None = None,
        task_id: str | None = None,
    ) -> dict:
        task_params = {
            "id": task_id or str(uuid.uuid4()),
            "sessionId": session_id,
            "skill": skill,
        }
        if params:
            task_params.update(params)
        if isinstance(task_params.get("payload"), dict):
            payload = dict(task_params["payload"])
            trace_parent = payload.pop("_langsmith_parent", None)
            task_params["payload"] = payload
            if trace_parent:
                task_params["_langsmith_parent"] = trace_parent

        messages = await self.send_task_and_wait(task_params)
        for message in reversed(messages):
            for part in reversed(message.parts):
                if part.type == "data" and part.data is not None:
                    return part.data
        return {}

    def _parse_task(self, data: dict) -> A2ATask:
        task = A2ATask(
            id=data.get("id", ""),
            session_id=data.get("sessionId"),
            skill=data.get("skill"),
            status=TaskStatus(data.get("status", "pending")),
            artifacts=data.get("artifacts", []),
            metadata=data.get("metadata", {}),
        )
        if "message" in data:
            task.message = self._parse_message(data["message"])
        return task

    def _parse_message(self, data: dict) -> A2AMessage:
        parts = []
        for p in data.get("parts", []):
            parts.append(Part(
                type=p.get("type", "text"),
                text=p.get("text"),
                data=p.get("data"),
                file_url=p.get("file_url"),
            ))
        return A2AMessage(
            role=data.get("role", "agent"),
            parts=parts,
            message_id=data.get("messageId"),
            context_id=data.get("contextId"),
        )

    @staticmethod
    def _last_text(messages: list[A2AMessage]) -> str:
        for message in reversed(messages):
            texts = [
                part.text
                for part in message.parts
                if part.type == "text" and part.text and part.text.strip() != "..."
            ]
            if texts:
                return "\n".join(texts)
        return ""
