import json
import logging
from typing import Dict, List, Optional, Union

import aiohttp

from twin.shared.config.settings import Config
from twin.shared.observability.langsmith import traceable
from .base_llm_service import (
    BaseLLMService,
    LLM_ERROR_BAD_FORMAT,
    LLM_ERROR_RESPONSE,
)
from .llm_response import LLMResponse


def _coerce_message_text(value: object) -> str:
    """Flatten OpenAI-compat content/reasoning fields to a string.

    Providers may return a string, a list of text parts, or
    ``reasoning_details`` as a list of objects. ``LLMResponse.content``
    must stay a str so later ``in frozenset`` checks do not crash.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item:
                    parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("summary") or item.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


class OpenAIService(BaseLLMService):
    """
    Generic OpenAI-compatible chat completions service.

    Works with OpenAI and compatible routers/endpoints such as 9Router,
    OpenRouter, LM Studio, and other /v1/chat/completions providers.
    """

    def __init__(self, persona_path: str = "memories"):
        super().__init__(persona_path=persona_path)

        self.api_key = Config.OPENAI_API_KEY
        self.api_url = Config.OPENAI_API_URL.rstrip("/")
        self.model = Config.OPENAI_MODEL
        self.session = None
        self.logger = logging.getLogger("discord_bot.OpenAIService")

    async def _get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(
                total=Config.LLM_REQUEST_TIMEOUT,
                connect=Config.LLM_CONNECT_TIMEOUT,
            )
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    @traceable(
        name="llm.openai_compatible.generate",
        run_type="llm",
        tags=["openai_compatible", "generation"],
        metadata={"provider": "openai_compatible"},
    )
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        use_native_tools: bool = False,
        include_tool_catalog: bool = True,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        include_persona: bool = True,
    ) -> Union[str, LLMResponse]:
        session = await self._get_session()
        # Native Function Calling carries tool descriptions in the `tools` payload,
        # so the system-prompt catalog would only duplicate them. Inject the catalog
        # only when native tools are OFF (fallback / non-native providers) so the
        # model still knows the tool surface. Hot path (native on) stays deduped.
        final_system_prompt = self._build_final_system_prompt(
            system_prompt,
            include_tool_catalog=(include_tool_catalog and not use_native_tools),
            include_persona=include_persona,
        )

        # Strict chat templates (Qwen-derived, e.g. LM Studio) raise
        # "No user query found in messages" when the first non-system message
        # is an assistant turn. A channel's active-context window can start
        # with a "Bot:" turn, so drop any leading assistant messages. Lenient
        # providers (9Router, OpenRouter) are unaffected by the trim.
        first_user = next(
            (i for i, m in enumerate(messages) if m.get("role") == "user"), None
        )
        convo = messages[first_user:] if first_user is not None else messages
        # Strict providers (e.g. commandcode) reject an empty system message
        # with 400 "system message must have content". Utility calls like
        # consolidation pass include_persona=False with no system_prompt, so
        # omit the system role entirely instead of sending content="".
        if final_system_prompt and final_system_prompt.strip():
            api_messages = [{"role": "system", "content": final_system_prompt}] + convo
        else:
            api_messages = list(convo)

        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": max_tokens or Config.LLM_MAX_TOKENS,
            "top_p": Config.LLM_TOP_P,
            "frequency_penalty": Config.LLM_FREQUENCY_PENALTY,
            "presence_penalty": Config.LLM_PRESENCE_PENALTY,
        }
        # Ollama OpenAI-compat bridge maps this to native `think` param.
        # See Ollama openai.go: "none" -> think=false, "low|medium|high|max"
        # -> think="<level>". Only include when explicitly set, so providers
        # that don't understand the field keep working. A per-call override
        # (e.g. consolidation) beats the global default.
        effective_effort = reasoning_effort or Config.LLM_REASONING_EFFORT
        if effective_effort:
            payload["reasoning_effort"] = effective_effort

        if use_native_tools:
            tool_schemas = []
            if self.tool_registry:
                tool_schemas = self.tool_registry.get_all_openai_schemas()
            if tool_schemas:
                payload["tools"] = tool_schemas
                self.logger.debug("Native tools enabled: %s tools", len(tool_schemas))

        if tool_choice and "tools" in payload:
            payload["tool_choice"] = tool_choice

        try:
            async with session.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error("OpenAI-compatible API error: %s", error_text)
                    return LLM_ERROR_RESPONSE

                # Some OpenAI-compatible gateways return valid JSON without
                # a proper JSON content-type header. Keep parsing tolerant so
                # existing providers continue to work unchanged.
                response_data = await response.json(content_type=None)
                usage = response_data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

                if "choices" not in response_data or not response_data["choices"]:
                    return LLM_ERROR_BAD_FORMAT

                choice = response_data["choices"][0]
                message = choice.get("message", {})
                content = _coerce_message_text(message.get("content"))
                # Ollama proxy returns `reasoning` (OpenAI-compat surface).
                # `reasoning_details` appears when reasoning_split=True.
                # Legacy Ollama field `reasoning_content` is also accepted as
                # a fallback so older provider quirks don't drop the trace.
                reasoning_content = (
                    _coerce_message_text(message.get("reasoning"))
                    or _coerce_message_text(message.get("reasoning_details"))
                    or _coerce_message_text(message.get("reasoning_content"))
                    or None
                )
                reasoning_only = False

                if not content and reasoning_content:
                    content = reasoning_content
                    reasoning_only = True

                tool_calls = None
                if message.get("tool_calls"):
                    tool_calls = []
                    for tc in message["tool_calls"]:
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        try:
                            args_dict = json.loads(args_str)
                        except json.JSONDecodeError:
                            self.logger.warning("Failed to parse tool arguments: %s", args_str)
                            args_dict = {}

                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": args_dict,
                        })

                    self.logger.info(
                        "OpenAI-compatible endpoint returned %s tool calls: %s",
                        len(tool_calls),
                        [tc["name"] for tc in tool_calls],
                    )

                self.logger.info(
                    "OpenAI-compatible API - Input tokens: %s, Output tokens: %s, Total: %s",
                    input_tokens,
                    output_tokens,
                    total_tokens,
                )

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    model=response_data.get("model", self.model),
                    finish_reason=choice.get("finish_reason"),
                    raw_response=response_data,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                    reasoning_only=reasoning_only,
                )

        except Exception as e:
            self.logger.error("Error communicating with OpenAI-compatible API: %s", e)
            return LLM_ERROR_RESPONSE

    async def close(self):
        if self.session:
            await self.session.close()
