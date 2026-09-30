"""A2A HTTP server with JSON-RPC endpoint and SSE streaming."""
import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Callable, Dict, Optional

from aiohttp import web

from twin.shared.a2a import auth as a2a_auth
from twin.shared.a2a.tasks import TaskAccessDenied, TaskNotFound, TaskStore
from twin.shared.a2a.types import (
    A2AMessage,
    AgentCard,
    Part,
)
from twin.shared.a2a.wire import (
    parse_message_input,
    serve_task_stream,
    task_to_dict,
)
from twin.shared.config.settings import Config
from twin.shared.observability import tracing_context_from_parent

logger = logging.getLogger(__name__)

_RPC_FORBIDDEN = -32003
_RPC_BUSY = -32004


class _HealthCheckFilter(logging.Filter):
    """Suppress access-log noise from healthcheck / agent-card polling."""

    _QUIET_PATHS = frozenset(["/.well-known/agent.json"])

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET_PATHS)

TaskHandler = Callable[[dict], AsyncIterator[A2AMessage]]
SessionValidator = Callable[[str, object], Optional[str]]


class _A2ARPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class A2AServer:
    def __init__(
        self,
        agent_card: AgentCard,
        skill_handlers: Dict[str, TaskHandler],
        host: str = "0.0.0.0",
        port: int = 8000,
        health_probe: Optional[Callable[[], bool]] = None,
        *,
        agent_name: Optional[str] = None,
        shared_secret: Optional[str] = None,
        skill_peers: Optional[Dict[str, frozenset]] = None,
        dm_allowed_peers: Optional[frozenset] = None,
        session_validator: Optional[SessionValidator] = None,
        task_store: Optional[TaskStore] = None,
        nonce_store: Optional[a2a_auth.NonceStore] = None,
    ):
        self.agent_card = agent_card
        self.skill_handlers = skill_handlers
        # Liveness of the platform connections behind the agent (e.g. Discord),
        # which the agent card cannot report: the A2A server stays up while the
        # bot is silently disconnected.
        self._health_probe = health_probe
        self.host = host
        self.port = port
        self.agent_name = a2a_auth.resolve_agent_name(agent_card, agent_name)
        self._shared_secret = shared_secret
        if skill_peers is None:
            skill_peers = a2a_auth.skill_peers_for_agent(self.agent_name)
        self._skill_peers = {
            skill: frozenset(peers) for skill, peers in skill_peers.items()
        }
        self._dm_allowed_peers = (
            frozenset(dm_allowed_peers)
            if dm_allowed_peers is not None
            else a2a_auth.DEFAULT_DM_ALLOWED_PEERS
        )
        self._session_validator = session_validator
        self._store = task_store if task_store is not None else TaskStore()
        self._nonce_store = (
            nonce_store if nonce_store is not None else a2a_auth.NonceStore()
        )
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    def _resolve_secret(self) -> Optional[str]:
        return self._shared_secret or Config.A2A_SHARED_SECRET

    def build_app(self) -> web.Application:
        app = web.Application(middlewares=[
            a2a_auth.a2a_auth_middleware(
                secret_resolver=self._resolve_secret,
                dm_allowed_peers=self._dm_allowed_peers,
                nonce_store=self._nonce_store,
            )
        ])
        app.router.add_get("/.well-known/agent.json", self._handle_agent_card)
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/", self._handle_jsonrpc)
        app.router.add_get("/tasks/{task_id}/stream", self._handle_stream)
        app.on_startup.append(self._start_maintenance)
        app.on_cleanup.append(self._stop_maintenance)
        return app

    async def _start_maintenance(self, _app: web.Application) -> None:
        self._store.start_background_purge()

    async def _stop_maintenance(self, _app: web.Application) -> None:
        await self._store.stop_background_purge()

    async def start(self):
        self._app = self.build_app()
        access_logger = logging.getLogger("aiohttp.access")
        access_logger.addFilter(_HealthCheckFilter())
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"A2A server listening on {self.host}:{self.port}")

    async def stop(self):
        await self._store.shutdown()
        if self._runner:
            await self._runner.cleanup()
            logger.info("A2A server stopped")

    async def wait_closed(self):
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    async def _handle_agent_card(self, request: web.Request) -> web.Response:
        data = {
            "name": self.agent_card.name,
            "description": self.agent_card.description,
            "url": self.agent_card.url,
            "version": self.agent_card.version,
            "provider": self.agent_card.provider,
            "capabilities": self.agent_card.capabilities,
            "skills": self.agent_card.skills,
        }
        return web.json_response(data)

    async def _handle_health(self, request: web.Request) -> web.Response:
        connected = True if self._health_probe is None else bool(self._health_probe())
        payload = {"status": "ok" if connected else "degraded", "connected": connected}
        return web.json_response(payload, status=200 if connected else 503)

    async def _handle_jsonrpc(self, request: web.Request) -> web.Response:
        peer = request.get("a2a_peer")
        if not peer:
            return web.json_response({"error": "unauthorized"}, status=401)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}},
                status=400,
            )

        method = body.get("method", "")
        params = body.get("params", {})
        rpc_id = body.get("id")

        try:
            if method == "tasks/send":
                result = await self._handle_send_task(params, request, peer)
            elif method == "tasks/get":
                result = await self._handle_get_task(params, peer)
            elif method == "tasks/cancel":
                result = await self._handle_cancel_task(params, peer)
            else:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                })

            return web.json_response({"jsonrpc": "2.0", "id": rpc_id, "result": result})

        except _A2ARPCError as exc:
            return web.json_response({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": exc.code, "message": exc.message},
            })
        except Exception as e:
            logger.exception(f"Error handling JSON-RPC method {method}")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32000, "message": str(e)},
            })

    async def _handle_send_task(
        self, params: dict, request: web.Request, peer: str
    ) -> dict:
        skill = params.get("skill", "chat")
        allowed = self._skill_peers.get(skill)
        if allowed is None or peer not in allowed:
            raise _A2ARPCError(
                _RPC_FORBIDDEN, f"forbidden: peer '{peer}' may not use skill '{skill}'"
            )
        if self._session_validator is not None:
            error = self._session_validator(skill, params.get("sessionId"))
            if error:
                raise _A2ARPCError(-32602, error)
        task_id = params.get("id", str(uuid.uuid4()))
        existing = self._store.get(task_id)
        if existing is not None:
            if not self._store.is_owner(task_id, peer):
                raise _A2ARPCError(_RPC_FORBIDDEN, "forbidden")
            return task_to_dict(existing)

        trace_parent = self._langsmith_parent_from_request(request)
        message = parse_message_input(params["message"]) if "message" in params else None
        task = self._store.create(
            task_id=task_id,
            skill=skill,
            session_id=params.get("sessionId"),
            creator_peer=peer,
            message=message,
        )
        if task is None:
            raise _A2ARPCError(_RPC_BUSY, "server busy: too many active tasks")

        handler = self.skill_handlers.get(skill)
        if handler is None:
            self._store.fail(task_id)
            return task_to_dict(task)

        worker_params = dict(params)
        worker_params["_a2a_peer"] = peer
        worker = asyncio.create_task(
            self._execute_handler(task_id, handler, worker_params, trace_parent)
        )
        self._store.set_worker(task_id, worker)
        return task_to_dict(task)

    async def _execute_handler(
        self,
        task_id: str,
        handler: TaskHandler,
        params: dict,
        trace_parent: dict[str, str] | None = None,
    ):
        try:
            with tracing_context_from_parent(trace_parent):
                async for message in handler(params):
                    self._store.publish(task_id, message)
            self._store.complete(task_id)
        except asyncio.CancelledError:
            self._store.mark_cancelled(task_id)
            raise
        except Exception as e:
            logger.exception(f"Task {task_id} handler failed")
            self._store.fail(task_id)
            err_msg = A2AMessage(
                role="agent",
                parts=[Part(type="text", text=f"Error: {e}")],
            )
            self._store.publish(task_id, err_msg)
        finally:
            # Signal stream closure on every exit path.
            self._store.broadcast(task_id, None)
            self._store.purge_expired()

    async def _handle_get_task(self, params: dict, peer: str) -> dict:
        task_id = params.get("id", "")
        try:
            task = self._store.check_owner(task_id, peer)
        except TaskNotFound:
            return {}
        except TaskAccessDenied:
            raise _A2ARPCError(_RPC_FORBIDDEN, "forbidden") from None
        return task_to_dict(task)

    async def _handle_cancel_task(self, params: dict, peer: str) -> dict:
        task_id = params.get("id", "")
        try:
            self._store.check_owner(task_id, peer)
        except TaskNotFound:
            return {}
        except TaskAccessDenied:
            raise _A2ARPCError(_RPC_FORBIDDEN, "forbidden") from None
        task = await self._store.cancel_and_wait(task_id)
        return task_to_dict(task) if task else {}

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        peer = request.get("a2a_peer")
        if not peer:
            return web.json_response({"error": "unauthorized"}, status=401)
        task_id = request.match_info["task_id"]
        try:
            self._store.check_owner(task_id, peer)
        except TaskNotFound:
            return web.json_response({"error": "not found"}, status=404)
        except TaskAccessDenied:
            return web.json_response({"error": "forbidden"}, status=403)
        return await serve_task_stream(self._store, task_id, request)

    @staticmethod
    def _langsmith_parent_from_request(request: web.Request) -> dict[str, str] | None:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower().startswith("langsmith")
            or key.lower() in {"baggage", "traceparent"}
        }
        return headers or None
