"""HarrierEmbeddingService — local Harrier q4 ONNX embeddings, no API.

Runs the locked Harrier model (see harrier_manifest.py) in-process via raw
ONNX Runtime on CPU: no Torch, no SentenceTransformers, no HTTP. Drop-in for
BaseEmbeddingService — callers keep applying Config.EMBEDDING_QUERY_PREFIX /
EMBEDDING_PASSAGE_PREFIX exactly as with the API providers.

Load is lazy and thread-safe: the first get_embedding initializes tokenizer
+ session under a lock; failures are sticky (no retry storm) and surface as
[] per the base-class error contract. After repairing the install with
scripts/pull_harrier_model.py, restart the process or call
reset_load_error() (close() also clears it) — see the recovery contract there.

Dimension policy: Harrier emits exactly 640 dims. Any other configured
EMBEDDING_VECTOR_SIZE is rejected before inference (ValueError) — legacy
1024-dim Qwen vectors live in a different embedding space and must be
migrated (scripts/migrate_t2_harrier.py), never padded/trimmed into 640.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from twin.shared.config.settings import Config
from .base_embedding_service import BaseEmbeddingService
from .harrier_budgets import check_budget
from .harrier_manifest import (
    DIMENSIONS,
    FILES_SHA256,
    GRAPH_REL_PATH,
    MARKER_NAME,
    OUTPUT_NAME,
    TOKENIZER_REL_PATH,
)

if TYPE_CHECKING:
    from .embedding_trace_logger import EmbeddingTraceLogger

NORM_TOLERANCE = 1e-3


class HarrierEmbeddingService(BaseEmbeddingService):
    """Local ONNX embeddings. Sync inference runs in a worker thread."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        expected_dim: int | None = None,
        trace_logger: EmbeddingTraceLogger | None = None,
        cache_size: int = 100,
    ):
        resolved_dim = expected_dim or Config.EMBEDDING_VECTOR_SIZE
        if resolved_dim != DIMENSIONS:
            raise ValueError(
                f"Harrier emits exactly {DIMENSIONS} dims; refusing "
                f"EMBEDDING_VECTOR_SIZE={resolved_dim}. Legacy 1024-dim Qwen "
                "vectors live in a different embedding space — migrate with "
                "scripts/migrate_t2_harrier.py instead of padding/trimming."
            )
        super().__init__(
            model_name="harrier-oss-v1-270m",
            expected_dim=resolved_dim,
            cache_size=cache_size,
            trace_logger=trace_logger,
            provider="harrier",
            api_url="local",
        )
        self.model_dir = Path(
            model_dir or getattr(Config, "HARRIER_MODEL_DIR", "models/harrier-q4")
        )
        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._load_lock = threading.Lock()
        self._load_error: str | None = None  # sticky: set once, never retried

    async def initialize(self) -> None:
        if self.expected_dim != DIMENSIONS:
            raise ValueError(
                f"Harrier emits exactly {DIMENSIONS} dims; refusing "
                f"expected_dim={self.expected_dim} — migrate legacy vectors "
                "with scripts/migrate_t2_harrier.py."
            )
        problem = self.install_problem()
        if problem is not None:
            self.logger.warning(
                "Harrier model unavailable at %s: %s; embeddings will be empty",
                self.model_dir, problem,
            )

    async def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []

        cached = self._cache_get(text)
        if cached is not None:
            self._trace_embedding_event(
                input_text=text, vector=cached, raw_dim=None,
                latency_ms=0.0, cache_hit=True,
            )
            return cached

        started_at = time.perf_counter()
        try:
            vector = await asyncio.to_thread(self._embed_sync, text)
        except Exception as exc:
            self.logger.error("Harrier embed failed: %s", exc)
            return []
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        fitted = self._fit_vector(vector)
        self._cache_put(text, fitted)
        self._trace_embedding_event(
            input_text=text, vector=fitted, raw_dim=len(vector),
            latency_ms=latency_ms, cache_hit=False,
        )
        return fitted

    async def close(self) -> None:
        with self._load_lock:
            self._session = None
            self._tokenizer = None
            self._load_error = None  # release sticky failure; next use reloads
        await super().close()

    def reset_load_error(self) -> None:
        """Clear a sticky load failure so the next get_embedding retries.

        Recovery contract: a failed install stays failed in-process (no retry
        storm). After repairing with scripts/pull_harrier_model.py — which
        invalidates the stale marker before fetching and publishes a new one
        only after verifying every file — either restart the process or call
        this (close() also clears it). Returns nothing; the retry happens
        lazily inside the next get_embedding.
        """
        with self._load_lock:
            self._load_error = None

    def install_problem(self) -> str | None:
        """Return None when the install marker exists and matches this build,
        else a human-readable reason. A corrupt/stale marker counts as not
        installed — the loader never trusts a partial or foreign install."""
        marker = self.model_dir / MARKER_NAME
        if not marker.exists():
            return (
                f"marker {MARKER_NAME} missing — "
                "run scripts/pull_harrier_model.py"
            )
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return (
                f"marker unreadable/corrupt ({exc}) — "
                "re-run scripts/pull_harrier_model.py"
            )
        if not isinstance(data, dict):
            return "marker is not a JSON object — re-pull the model"
        if data.get("dimensions") != DIMENSIONS:
            return (
                f"marker dimensions={data.get('dimensions')} != {DIMENSIONS} — "
                "stale install, re-run scripts/pull_harrier_model.py"
            )
        if sorted(data.get("files") or []) != sorted(FILES_SHA256):
            return (
                "marker file list does not match this build — stale install, "
                "re-run scripts/pull_harrier_model.py"
            )
        return None

    def _fit_vector(self, vector: List[float]) -> List[float]:
        """Harrier never pads/trims: any dim mismatch is a loud error."""
        if self.expected_dim != DIMENSIONS:
            raise ValueError(
                f"Harrier expected_dim={self.expected_dim} != {DIMENSIONS}; "
                "refusing to pad/trim across embedding spaces."
            )
        if len(vector) != DIMENSIONS:
            raise ValueError(
                f"Harrier output dim {len(vector)} != {DIMENSIONS}; refusing "
                "to pad/trim across embedding spaces."
            )
        return vector

    # ------------------------------------------------------------ internals

    def _embed_sync(self, text: str) -> List[float]:
        tokenizer, session = self._ensure_loaded()
        encoding = tokenizer.encode(text)
        query_prefix = Config.EMBEDDING_QUERY_PREFIX or ""
        warning = check_budget(
            tokens=len(encoding.ids),
            is_query=bool(query_prefix) and text.startswith(query_prefix),
        )
        if warning:
            self.logger.warning(warning)
        feed: dict[str, Any] = {"input_ids": np.array([encoding.ids], dtype=np.int64)}
        input_names = {i.name for i in session.get_inputs()}
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.array(
                [encoding.attention_mask], dtype=np.int64
            )
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)
        outputs = session.run([OUTPUT_NAME], feed)[0]
        return self._validate_outputs(outputs)

    def _ensure_loaded(self) -> tuple[Tokenizer, ort.InferenceSession]:
        if self._session is not None:
            return self._tokenizer, self._session  # type: ignore[return-value]
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        with self._load_lock:
            if self._session is None and self._load_error is None:
                self._load()
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            return self._tokenizer, self._session  # type: ignore[return-value]

    def _load(self) -> None:
        problem = self.install_problem()
        if problem is not None:
            self._load_error = (
                f"harrier model not installed at {self.model_dir}: {problem}"
            )
            return
        try:
            tokenizer = Tokenizer.from_file(
                str(self.model_dir / TOKENIZER_REL_PATH)
            )
            session = ort.InferenceSession(
                str(self.model_dir / GRAPH_REL_PATH),
                providers=["CPUExecutionProvider"],
            )
            self._assert_graph_shape(session)
        except Exception as exc:
            self._load_error = f"harrier model load failed: {exc}"
            self.logger.error(self._load_error)
            return
        self._tokenizer = tokenizer
        self._session = session
        self.logger.info("Harrier ONNX session ready (%s)", self.model_dir)

    @staticmethod
    def _assert_graph_shape(session: ort.InferenceSession) -> None:
        match = next(
            (o for o in session.get_outputs() if o.name == OUTPUT_NAME), None
        )
        if match is None:
            raise RuntimeError(
                f"{OUTPUT_NAME} output missing from graph; got "
                f"{[o.name for o in session.get_outputs()]}"
            )
        shape = list(match.shape)
        dim = shape[1] if len(shape) == 2 else None
        if dim != DIMENSIONS:
            raise RuntimeError(
                f"graph {OUTPUT_NAME} dim {dim} != {DIMENSIONS} (shape {shape})"
            )

    @staticmethod
    def _validate_outputs(outputs: Any) -> List[float]:
        if not isinstance(outputs, np.ndarray) or outputs.dtype != np.float32:
            raise RuntimeError(f"expected FLOAT32 output, got {type(outputs).__name__}")
        if outputs.ndim != 2 or outputs.shape[1] != DIMENSIONS:
            raise RuntimeError(f"expected [batch, {DIMENSIONS}], got {outputs.shape}")
        if not np.all(np.isfinite(outputs)):
            raise RuntimeError("non-finite harrier output")
        norms = np.linalg.norm(outputs, axis=1)
        if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
            raise RuntimeError(f"harrier output not unit norm: {norms}")
        return outputs[0].tolist()
