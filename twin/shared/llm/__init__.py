"""LLM wrapper services module."""

from .base_llm_service import BaseLLMService
from .llm_response import LLMResponse


def __getattr__(name):
    _imports = {
        "BaseEmbeddingService": ".embedding",
        "HarrierEmbeddingService": ".embedding",
        "create_embedding_service": ".embedding",
        "GeminiService": ".gemini_service",
        "OpenAIService": ".openai_service",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseLLMService",
    "BaseEmbeddingService",
    "HarrierEmbeddingService",
    "create_embedding_service",
    "GeminiService",
    "LLMResponse",
    "OpenAIService",
]
