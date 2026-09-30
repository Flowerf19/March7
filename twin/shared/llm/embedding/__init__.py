"""Embedding services package (local Harrier ONNX only, no API).

Public API:
    create_embedding_service()  — factory, builds HarrierEmbeddingService
    BaseEmbeddingService        — ABC (cache, dim-fit, trace)
    HarrierEmbeddingService     — local Harrier q4 ONNX embeddings
"""

from .base_embedding_service import BaseEmbeddingService
from .embedding_factory import create_embedding_service
from .harrier_embedding_service import HarrierEmbeddingService

__all__ = [
    "BaseEmbeddingService",
    "HarrierEmbeddingService",
    "create_embedding_service",
]
