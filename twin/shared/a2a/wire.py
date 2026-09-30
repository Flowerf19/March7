"""A2A wire-format helpers and SSE stream serving."""
from __future__ import annotations

import asyncio
import json

from aiohttp import web

from twin.shared.a2a.tasks import SubscriptionLimitExceeded, TaskStore
from twin.shared.a2a.types import A2AMessage, A2ATask, Part

SSE_WRITE_TIMEOUT_SECONDS = 5.0
STREAM_TRUNCATED_ERROR = "stream_truncated"


def parse_message_input(data: dict) -> A2AMessage:
    parts = []
    for p in data.get("parts", []):
        parts.append(Part(
            type=p.get("type", "text"),
            text=p.get("text"),
            data=p.get("data"),
            file_url=p.get("file_url"),
        ))
    return A2AMessage(
        role=data.get("role", "user"),
        parts=parts,
        message_id=data.get("messageId"),
        context_id=data.get("contextId"),
    )


def task_to_dict(task: A2ATask) -> dict:
    result = {
        "id": task.id,
        "sessionId": task.session_id,
        "skill": task.skill,
        "status": task.status.value,
        "artifacts": task.artifacts,
        "metadata": task.metadata,
    }
    if task.message:
        result["message"] = message_to_dict(task.message)
    return result


def message_to_dict(message: A2AMessage) -> dict:
    result: dict = {"role": message.role}
    if message.message_id:
        result["messageId"] = message.message_id
    if message.context_id:
        result["contextId"] = message.context_id
    result["parts"] = []
    for p in message.parts:
        part = {"type": p.type}
        if p.text is not None:
            part["text"] = p.text
        if p.data is not None:
            part["data"] = p.data
        if p.file_url is not None:
            part["file_url"] = p.file_url
        result["parts"].append(part)
    return result


def sse_bytes(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def sse_error_bytes(code: str, reason: str) -> bytes:
    payload = {"error": code, "reason": reason}
    return f"event: error\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


def sse_complete_bytes(message_count: int) -> bytes:
    payload = {"message_count": message_count}
    return f"event: complete\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


async def _write_bytes(
    response: web.StreamResponse, data: bytes, timeout: float
) -> None:
    await asyncio.wait_for(response.write(data), timeout=timeout)


async def _write_eof_quietly(
    response: web.StreamResponse, timeout: float
) -> None:
    try:
        await asyncio.wait_for(response.write_eof(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError, ConnectionError):
        pass
    except Exception:
        pass


async def serve_task_stream(
    store: TaskStore,
    task_id: str,
    request: web.Request,
    write_timeout: float = SSE_WRITE_TIMEOUT_SECONDS,
) -> web.StreamResponse:
    """Serve one SSE stream with exactly-once replay of buffered output.

    Reservation: subscribe synchronously before any await, so the per-task
    cap holds even with many slow concurrent replays (429 before headers).
    Snapshot + terminal state are captured synchronously with the
    reservation: publishes before are in the snapshot, publishes after land
    in the bounded queue (or evict) — no duplicate, loss, or count-index
    roll error. After replay, drain queued future/final until close;
    initially-terminal streams drain only available items (no hang).
    Slow peers: every write has a timeout; on timeout the stream closes
    and unsubscribes so terminal records can be TTL-purged.
    Clean close appends one `event: complete` with the count of normal
    frames actually written; any truncation, eviction, or write timeout
    omits it (or sends `event: error`) so clients fail closed.
    """
    try:
        queue = store.subscribe(task_id)
    except SubscriptionLimitExceeded:
        return web.json_response({"error": "too many subscribers"}, status=429)
    truncated_snapshot = bool(getattr(queue, "initial_truncated", False))
    try:
        # Both synchronous: atomic with reservation, no await interleaves.
        first = store.snapshot(task_id)
        initial_terminal = store.is_terminal(task_id)
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)
    except BaseException:
        store.unsubscribe(task_id, queue)
        raise
    if truncated_snapshot:
        try:
            await _write_bytes(
                response,
                sse_error_bytes(STREAM_TRUNCATED_ERROR, "history_truncated"),
                write_timeout,
            )
        except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            store.unsubscribe(task_id, queue)
        await _write_eof_quietly(response, write_timeout)
        return response
    try:
        written = 0
        for message in first:
            await _write_bytes(
                response, sse_bytes(message_to_dict(message)), write_timeout
            )
            written += 1
        if initial_terminal:
            # Close signal (if any) was broadcast before subscribe; never
            # block waiting for a future None that will not arrive.
            truncated_close = False
            while True:
                try:
                    message = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if message is None:
                    truncated_close = bool(getattr(queue, "close_reason", None))
                    break
                await _write_bytes(
                    response, sse_bytes(message_to_dict(message)), write_timeout
                )
                written += 1
            if truncated_close:
                await _write_bytes(
                    response,
                    sse_error_bytes(STREAM_TRUNCATED_ERROR, "evicted"),
                    write_timeout,
                )
            else:
                await _write_bytes(
                    response, sse_complete_bytes(written), write_timeout
                )
        else:
            # Live at subscribe: future messages + close reliably queued
            # (eviction also closes with None). No early return when the
            # task completes mid-replay; drain until the close signal.
            while True:
                message = await queue.get()
                if message is None:
                    if getattr(queue, "close_reason", None):
                        await _write_bytes(
                            response,
                            sse_error_bytes(STREAM_TRUNCATED_ERROR, "evicted"),
                            write_timeout,
                        )
                    else:
                        await _write_bytes(
                            response, sse_complete_bytes(written), write_timeout
                        )
                    break
                await _write_bytes(
                    response, sse_bytes(message_to_dict(message)), write_timeout
                )
                written += 1
    except (asyncio.CancelledError, asyncio.TimeoutError, ConnectionError):
        pass
    finally:
        store.unsubscribe(task_id, queue)

    await _write_eof_quietly(response, write_timeout)
    return response
