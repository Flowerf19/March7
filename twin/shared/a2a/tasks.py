"""Bounded in-memory lifecycle store for A2A tasks.

Owns task records, per-task output buffers, stream subscribers, and worker
handles. Bounds active tasks (including noncooperative workers that swallow
cancel), completed retention (count + TTL even when subscribed), per-task
output bytes, per-task subscriber count, and per-subscriber queued
messages/bytes so a busy or hostile peer cannot grow memory without limit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from twin.shared.a2a.types import A2AMessage, A2ATask, TaskStatus

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

MAX_ACTIVE_TASKS = 100
MAX_COMPLETED_TASKS = 100
MAX_TASK_BUFFER_BYTES = 256 * 1024
MAX_TASK_BUFFER_MESSAGES = 1000
COMPLETED_TTL_SECONDS = 3600.0
CANCEL_WAIT_SECONDS = 5.0
MAX_SUBSCRIBERS_PER_TASK = 16
MAX_SUBSCRIBER_QUEUE_MESSAGES = 64
MAX_SUBSCRIBER_QUEUE_BYTES = 64 * 1024
PURGE_INTERVAL_SECONDS = 60.0


class TaskNotFound(KeyError):
    """Raised when a task id is unknown (never existed or already evicted)."""


class TaskAccessDenied(PermissionError):
    """Raised when a peer touches a task created by another peer."""


class SubscriptionLimitExceeded(RuntimeError):
    """Raised when per-task subscriber cap denies a new stream (HTTP 429)."""


@dataclass
class _TaskRecord:
    task: A2ATask
    creator_peer: str
    buffer: list[A2AMessage] = field(default_factory=list)
    buffer_bytes: int = 0
    buffer_truncated: bool = False
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    worker: asyncio.Task | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


def _message_size(message: A2AMessage) -> int:
    size = 64 + len((message.role or "").encode("utf-8"))
    if message.message_id:
        size += len(message.message_id.encode("utf-8"))
    if message.context_id:
        size += len(message.context_id.encode("utf-8"))
    for part in message.parts:
        if part.text:
            size += len(part.text.encode("utf-8"))
        if part.data is not None:
            size += len(json.dumps(part.data, default=str).encode("utf-8"))
        if part.file_url:
            size += len(part.file_url.encode("utf-8"))
    return size


class _SubscriberQueue(asyncio.Queue):
    """Bounded per-subscriber queue with byte accounting."""

    def __init__(self, max_messages: int, max_bytes: int) -> None:
        super().__init__(maxsize=max_messages)
        self._max_bytes = max_bytes
        self.buffered_bytes = 0
        self.close_reason: str | None = None
        self.initial_truncated: bool = False

    def would_overflow(self, size: int) -> bool:
        return self.full() or self.buffered_bytes + size > self._max_bytes

    def put_nowait(self, item: A2AMessage | None) -> None:  # type: ignore[override]
        super().put_nowait(item)
        if item is not None:
            self.buffered_bytes += _message_size(item)

    def get_nowait(self) -> A2AMessage | None:  # type: ignore[override]
        item = super().get_nowait()
        if item is not None:
            # Single accounting point: asyncio.Queue.get() delegates to
            # get_nowait(), so no async-get override (would double-debit).
            self.buffered_bytes -= _message_size(item)
        return item


class SubscriberPolicy:
    """Denial/backpressure policy for per-task SSE subscribers.

    Denial: subscribe raises SubscriptionLimitExceeded at cap; existing
    streams unaffected (wire maps to HTTP 429).
    Backpressure: fanout never blocks the worker. A subscriber whose
    pending messages/bytes would exceed bounds is evicted (drained, closed
    with None + close_reason='evicted', removed); fast subscribers still
    receive every message including the final one, and the replay buffer
    retains it for re-subscribe. Close marker carries no byte charge;
    wire maps evicted/history-truncated closes to an SSE error event.
    """

    def __init__(
        self,
        *,
        max_subscribers: int = MAX_SUBSCRIBERS_PER_TASK,
        max_messages: int = MAX_SUBSCRIBER_QUEUE_MESSAGES,
        max_bytes: int = MAX_SUBSCRIBER_QUEUE_BYTES,
    ) -> None:
        self.max_subscribers = max_subscribers
        self.max_messages = max_messages
        self.max_bytes = max_bytes

    def make_queue(self) -> _SubscriberQueue:
        return _SubscriberQueue(self.max_messages, self.max_bytes)

    def check_capacity(self, current: int) -> None:
        if current >= self.max_subscribers:
            raise SubscriptionLimitExceeded(
                f"too many subscribers ({current}/{self.max_subscribers})"
            )

    def fanout(
        self, subscribers: list[asyncio.Queue], message: A2AMessage | None
    ) -> int:
        """Best-effort fanout; evict slow consumers. Never raises for slow."""
        if message is None:
            return self._fanout_close(subscribers)
        size = _message_size(message)
        evicted = 0
        for queue in list(subscribers):
            try:
                overflow = (
                    queue.would_overflow(size)
                    if isinstance(queue, _SubscriberQueue)
                    else queue.full()
                )
                if overflow:
                    self._evict(subscribers, queue)
                    evicted += 1
                else:
                    queue.put_nowait(message)
            except asyncio.QueueFull:
                self._evict(subscribers, queue)
                evicted += 1
            except Exception:
                logger.warning("subscriber fanout failed; evicting", exc_info=True)
                self._evict(subscribers, queue)
                evicted += 1
        return evicted

    def _fanout_close(self, subscribers: list[asyncio.Queue]) -> int:
        drained = 0
        for queue in list(subscribers):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                self._evict(subscribers, queue)
                drained += 1
            except Exception:
                self._evict(subscribers, queue)
                drained += 1
        return drained

    def _evict(
        self, subscribers: list[asyncio.Queue], queue: asyncio.Queue
    ) -> None:
        if queue in subscribers:
            subscribers.remove(queue)
        try:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        except Exception:
            pass
        try:
            queue.close_reason = "evicted"  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            queue.put_nowait(None)
        except Exception:
            pass

    def close_all(self, subscribers: list[asyncio.Queue]) -> None:
        """Signal close to all (preserving fast pending) and clear the set."""
        for queue in list(subscribers):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                try:
                    while True:
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except Exception:
                    pass
                try:
                    queue.close_reason = "evicted"  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass
            except Exception:
                pass
        subscribers.clear()


def _is_live_terminal(rec: _TaskRecord, current: object | None) -> bool:
    """True for terminal records whose worker still runs elsewhere."""
    if rec.task.status not in TERMINAL_STATUSES:
        return False
    if rec.worker is None or rec.worker.done():
        return False
    return rec.worker is not current


class TaskStore:
    """Owns A2A task lifecycle, buffers, subscribers, and workers."""

    def __init__(
        self,
        *,
        max_active_tasks: int = MAX_ACTIVE_TASKS,
        max_completed_tasks: int = MAX_COMPLETED_TASKS,
        max_task_buffer_bytes: int = MAX_TASK_BUFFER_BYTES,
        max_task_buffer_messages: int = MAX_TASK_BUFFER_MESSAGES,
        completed_ttl_seconds: float = COMPLETED_TTL_SECONDS,
        max_subscribers_per_task: int = MAX_SUBSCRIBERS_PER_TASK,
        max_subscriber_messages: int = MAX_SUBSCRIBER_QUEUE_MESSAGES,
        max_subscriber_bytes: int = MAX_SUBSCRIBER_QUEUE_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._max_active = max_active_tasks
        self._max_completed = max_completed_tasks
        self._max_buffer_bytes = max_task_buffer_bytes
        self._max_buffer_messages = max_task_buffer_messages
        self._completed_ttl = completed_ttl_seconds
        self._clock = clock
        self._sub_policy = SubscriberPolicy(
            max_subscribers=max_subscribers_per_task,
            max_messages=max_subscriber_messages,
            max_bytes=max_subscriber_bytes,
        )
        self._records: dict[str, _TaskRecord] = {}
        self._purge_task: asyncio.Task | None = None

    def __len__(self) -> int:
        return len(self._records)

    @property
    def max_subscribers_per_task(self) -> int:
        return self._sub_policy.max_subscribers

    def active_count(self) -> int:
        return sum(
            1 for r in self._records.values()
            if r.task.status not in TERMINAL_STATUSES
            or (r.worker is not None and not r.worker.done())
        )

    def get(self, task_id: str) -> A2ATask | None:
        rec = self._records.get(task_id)
        return rec.task if rec is not None else None

    def is_owner(self, task_id: str, peer: str) -> bool:
        rec = self._records.get(task_id)
        return rec is not None and rec.creator_peer == peer

    def check_owner(self, task_id: str, peer: str) -> A2ATask:
        """Return the task when *peer* created it, else raise."""
        rec = self._records.get(task_id)
        if rec is None:
            raise TaskNotFound(task_id)
        if rec.creator_peer != peer:
            raise TaskAccessDenied(task_id)
        return rec.task

    def is_terminal(self, task_id: str) -> bool:
        rec = self._records.get(task_id)
        return rec is None or rec.task.status in TERMINAL_STATUSES

    def create(
        self,
        *,
        task_id: str,
        skill: str | None,
        session_id: str | None,
        creator_peer: str,
        message: A2AMessage | None = None,
    ) -> A2ATask | None:
        """Create a task, or return None when the active-task bound is hit."""
        self.purge_expired()
        if self.active_count() >= self._max_active:
            return None
        task = A2ATask(
            id=task_id,
            session_id=session_id,
            skill=skill,
            message=message,
            status=TaskStatus.IN_PROGRESS,
        )
        self._records[task_id] = _TaskRecord(
            task=task, creator_peer=creator_peer, created_at=self._clock()
        )
        return task

    def set_worker(self, task_id: str, worker: asyncio.Task) -> None:
        rec = self._records.get(task_id)
        if rec is not None:
            rec.worker = worker

    def complete(self, task_id: str) -> bool:
        """Mark completed unless already terminal (a cancel always wins)."""
        return self._to_terminal(task_id, TaskStatus.COMPLETED)

    def fail(self, task_id: str) -> bool:
        return self._to_terminal(task_id, TaskStatus.FAILED)

    def mark_cancelled(self, task_id: str) -> bool:
        return self._to_terminal(task_id, TaskStatus.CANCELLED)

    def _to_terminal(self, task_id: str, status: TaskStatus) -> bool:
        rec = self._records.get(task_id)
        if rec is None or rec.task.status in TERMINAL_STATUSES:
            return False
        rec.task.status = status
        rec.completed_at = self._clock()
        return True

    def publish(self, task_id: str, message: A2AMessage) -> None:
        """Append to replay buffer and fan out; oversized fails explicitly."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        size = _message_size(message)
        if size > self._max_buffer_bytes:
            # Single output exceeds the whole task budget: never buffer,
            # deliver, or complete as if successful.
            self.fail(task_id)
            rec.buffer.clear()
            rec.buffer_bytes = 0
            self._sub_policy.close_all(rec.subscribers)
            logger.warning("task %s output %dB exceeds budget; failed", task_id, size)
            return
        rec.buffer.append(message)
        rec.buffer_bytes += size
        while (
            len(rec.buffer) > self._max_buffer_messages
            or rec.buffer_bytes > self._max_buffer_bytes
        ) and rec.buffer:
            rec.buffer_bytes -= _message_size(rec.buffer.pop(0))
            rec.buffer_truncated = True
        self._sub_policy.fanout(rec.subscribers, message)

    def broadcast(self, task_id: str, message: A2AMessage | None) -> None:
        """Fan out to subscribers without touching the replay buffer."""
        rec = self._records.get(task_id)
        if rec is None:
            return
        self._sub_policy.fanout(rec.subscribers, message)

    def snapshot(self, task_id: str) -> list[A2AMessage]:
        rec = self._records.get(task_id)
        return list(rec.buffer) if rec is not None else []

    def subscribe(self, task_id: str) -> asyncio.Queue:
        rec = self._records.get(task_id)
        if rec is None:
            return self._sub_policy.make_queue()
        self._sub_policy.check_capacity(len(rec.subscribers))
        queue = self._sub_policy.make_queue()
        queue.initial_truncated = rec.buffer_truncated  # type: ignore[attr-defined]
        rec.subscribers.append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        rec = self._records.get(task_id)
        if rec is not None and queue in rec.subscribers:
            rec.subscribers.remove(queue)

    def subscriber_count(self, task_id: str) -> int:
        rec = self._records.get(task_id)
        return len(rec.subscribers) if rec is not None else 0

    async def cancel_and_wait(
        self, task_id: str, timeout: float = CANCEL_WAIT_SECONDS
    ) -> A2ATask | None:
        """Cancel worker; unblock streams; wait briefly via wait (bounded)."""
        rec = self._records.get(task_id)
        if rec is None:
            return None
        self.mark_cancelled(task_id)
        try:
            self.broadcast(task_id, None)
        except Exception:
            pass
        worker = rec.worker
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                # wait() returns after timeout even when cancel is swallowed.
                await asyncio.wait([worker], timeout=timeout)
            except (asyncio.CancelledError, Exception):
                pass
        return rec.task

    def purge_expired(self, now: float | None = None) -> int:
        """Evict terminal tasks past TTL/count; close expired streams first."""
        current = self._clock() if now is None else now
        try:
            me = asyncio.current_task()
        except RuntimeError:
            me = None
        removed = 0
        for task_id, rec in list(self._records.items()):
            if rec.task.status not in TERMINAL_STATUSES:
                continue
            if _is_live_terminal(rec, me):
                continue
            if (
                rec.completed_at is not None
                and current - rec.completed_at >= self._completed_ttl
            ):
                self._sub_policy.close_all(rec.subscribers)
                rec.buffer.clear()
                rec.buffer_bytes = 0
                del self._records[task_id]
                removed += 1
        terminal = [
            (rec.completed_at or float("inf"), task_id)
            for task_id, rec in self._records.items()
            if rec.task.status in TERMINAL_STATUSES and not _is_live_terminal(rec, me)
        ]
        overflow = len(terminal) - self._max_completed
        if overflow > 0:
            terminal.sort()
            for _, task_id in terminal[:overflow]:
                rec = self._records.get(task_id)
                if rec is None:
                    continue
                self._sub_policy.close_all(rec.subscribers)
                rec.buffer.clear()
                rec.buffer_bytes = 0
                del self._records[task_id]
                removed += 1
        return removed

    def start_background_purge(
        self, interval: float = PURGE_INTERVAL_SECONDS
    ) -> None:
        """Start periodic TTL purge; idle/subscribed data cannot pin forever."""
        if self._purge_task is not None and not self._purge_task.done():
            return
        self._purge_task = asyncio.create_task(self._purge_loop(interval))

    async def stop_background_purge(self, timeout: float = 2.0) -> None:
        task = self._purge_task
        self._purge_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await asyncio.wait([task], timeout=timeout)
        except (asyncio.CancelledError, Exception):
            pass

    async def _purge_loop(self, interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    self.purge_expired()
                except Exception:
                    logger.exception("background purge failed")
        except asyncio.CancelledError:
            raise

    async def shutdown(self, timeout: float = CANCEL_WAIT_SECONDS) -> None:
        """Cancel workers + maintenance; always return within timeout."""
        try:
            await self.stop_background_purge()
        except Exception:
            pass
        workers = [
            rec.worker
            for rec in self._records.values()
            if rec.worker is not None and not rec.worker.done()
        ]
        for worker in workers:
            worker.cancel()
        if workers:
            try:
                await asyncio.wait(workers, timeout=timeout)
            except asyncio.CancelledError:
                pass
            pending = sum(1 for w in workers if not w.done())
            if pending:
                logger.warning(
                    "A2A shutdown: %d workers still running after %.1fs",
                    pending,
                    timeout,
                )
        for rec in self._records.values():
            if rec.task.status not in TERMINAL_STATUSES:
                rec.task.status = TaskStatus.CANCELLED
                rec.completed_at = self._clock()
        for rec in self._records.values():
            try:
                self._sub_policy.close_all(rec.subscribers)
            except Exception:
                pass
