"""Factory for the embedding service (local Harrier ONNX only).

There is exactly one embedding backend: HarrierEmbeddingService. The factory
exists so containers keep a single construction point and tests keep a seam.
"""

from __future__ import annotations

from pathlib import Path

from twin.shared.config.settings import Config
from .base_embedding_service import BaseEmbeddingService
from .embedding_trace_logger import EmbeddingTraceLogger
from .harrier_embedding_service import HarrierEmbeddingService
from .harrier_manifest import DIMENSIONS


def _create_trace_logger() -> EmbeddingTraceLogger | None:
    """Create the trace logger when enabled, otherwise return None."""
    if not getattr(Config, "EMBEDDING_TRACE_LOG_ENABLED", False):
        return None
    return EmbeddingTraceLogger(
        log_path=Config.EMBEDDING_TRACE_LOG_PATH,
        enabled=True,
    )


def create_embedding_service(
    *,
    model_dir: str | Path | None = None,
    expected_dim: int | None = None,
) -> BaseEmbeddingService:
    """Construct the local Harrier embedding service.

    Rejects any configured dimension other than 640 before inference —
    legacy 1024-dim vectors must be migrated, never padded/trimmed.
    """
    dim = expected_dim or Config.EMBEDDING_VECTOR_SIZE
    if dim != DIMENSIONS:
        raise ValueError(
            f"Harrier emits exactly {DIMENSIONS} dims; refusing "
            f"EMBEDDING_VECTOR_SIZE={dim}. Migrate legacy vectors with "
            "scripts/migrate_t2_harrier.py."
        )
    return HarrierEmbeddingService(
        model_dir=model_dir or Config.HARRIER_MODEL_DIR,
        expected_dim=dim,
        trace_logger=_create_trace_logger(),
    )
