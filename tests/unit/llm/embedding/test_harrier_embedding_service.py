"""Unit tests for HarrierEmbeddingService (mocked ONNX, no model needed)."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from twin.shared.llm.embedding.embedding_factory import create_embedding_service
from twin.shared.llm.embedding.harrier_budgets import (
    BUDGET_DOCUMENT_TOKENS,
    BUDGET_QUERY_TOKENS,
    check_budget,
)
from twin.shared.llm.embedding.harrier_embedding_service import (
    HarrierEmbeddingService,
)
from twin.shared.llm.embedding.harrier_manifest import DIMENSIONS


def _unit_vector(dim: int = DIMENSIONS) -> np.ndarray:
    raw = np.arange(1, dim + 1, dtype=np.float32)
    return (raw / np.linalg.norm(raw)).reshape(1, dim)


def _make_service(tmp_path: Path, **kwargs) -> HarrierEmbeddingService:
    return HarrierEmbeddingService(model_dir=tmp_path / "harrier", **kwargs)


def _mock_loaded(service: HarrierEmbeddingService, outputs: np.ndarray) -> None:
    """Pretend the ONNX session is loaded with canned outputs."""
    encoding = MagicMock()
    encoding.ids = [1, 2, 3]
    encoding.attention_mask = [1, 1, 1]
    encoding.type_ids = [0, 0, 0]
    tokenizer = MagicMock()
    tokenizer.encode.return_value = encoding
    session = MagicMock()
    session.get_inputs.return_value = []
    session.run.return_value = [outputs]
    service._tokenizer = tokenizer
    service._session = session


class TestHarrierEmbeddingService:
    async def test_get_embedding_returns_unit_vector(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path, expected_dim=640)
        _mock_loaded(service, _unit_vector())

        vector = await service.get_embedding("hello")

        assert len(vector) == 640
        assert math.isclose(
            math.sqrt(sum(x * x for x in vector)), 1.0, abs_tol=1e-3
        )

    async def test_empty_text_returns_empty(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path)
        assert await service.get_embedding("") == []
        assert await service.get_embedding("   ") == []

    async def test_cache_hit_skips_session(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path)
        _mock_loaded(service, _unit_vector())

        first = await service.get_embedding("hello")
        service._session.run.reset_mock()
        second = await service.get_embedding("hello")

        assert second == first
        service._session.run.assert_not_called()

    async def test_bad_output_returns_empty(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path)
        _mock_loaded(service, np.zeros((1, 640), dtype=np.float32))  # zero norm

        assert await service.get_embedding("hello") == []

    async def test_missing_model_returns_empty(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path)  # dir has no .installed.json marker

        assert await service.get_embedding("hello") == []

    async def test_dim_mismatch_rejected_before_inference(self, tmp_path: Path) -> None:
        import pytest as _pytest

        with _pytest.raises(ValueError, match="1024"):
            _make_service(tmp_path, expected_dim=1024)
        with _pytest.raises(ValueError, match="refusing"):
            _make_service(tmp_path, expected_dim=8)

    async def test_fit_vector_never_pads_or_trims(self, tmp_path: Path) -> None:
        import pytest as _pytest

        service = _make_service(tmp_path, expected_dim=640)
        with _pytest.raises(ValueError, match="refusing"):
            service._fit_vector([0.0] * 8)
        with _pytest.raises(ValueError, match="refusing"):
            service._fit_vector([0.0] * 1024)
        assert service._fit_vector([0.0] * 640) == [0.0] * 640

    async def test_initialize_rejects_mutated_dim(self, tmp_path: Path) -> None:
        import pytest as _pytest

        service = _make_service(tmp_path, expected_dim=640)
        service.expected_dim = 1024  # type: ignore[assignment]
        with _pytest.raises(ValueError, match="refusing"):
            await service.initialize()

    async def test_trace_event(self, tmp_path: Path) -> None:
        from twin.shared.llm.embedding.embedding_trace_logger import (
            EmbeddingTraceLogger,
        )

        log_path = tmp_path / "trace.jsonl"
        service = _make_service(
            tmp_path,
            expected_dim=640,
            trace_logger=EmbeddingTraceLogger(
                log_path=log_path, enabled=True
            ),
        )
        _mock_loaded(service, _unit_vector())

        await service.get_embedding("hello")

        import json

        record = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert record["event_type"] == "EMBED"
        assert record["provider"] == "harrier"
        assert record["vector_dim"] == 640
        assert record["cache_hit"] is False


class TestHarrierTraceBestEffort:
    """Optional trace I/O must never break valid inference (fresh or cached)."""

    def _failing_logger(self):
        logger = MagicMock()
        logger.log.side_effect = OSError("disk full")
        return logger

    async def test_trace_failure_on_fresh_call_still_returns(self, tmp_path: Path) -> None:
        service = _make_service(
            tmp_path, expected_dim=640, trace_logger=self._failing_logger()
        )
        _mock_loaded(service, _unit_vector())

        vector = await service.get_embedding("hello")

        assert len(vector) == 640

    async def test_trace_failure_on_cached_call_still_returns(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path, expected_dim=640)  # no logger: prime cache
        _mock_loaded(service, _unit_vector())
        first = await service.get_embedding("hello")
        assert len(first) == 640

        service.trace_logger = self._failing_logger()  # breaks only tracing now
        service._session.run.reset_mock()

        second = await service.get_embedding("hello")

        assert second == first
        service._session.run.assert_not_called()


class TestHarrierMarkerAndRecovery:
    """Install marker validation + sticky-failure recovery contract."""

    def _write_marker(self, model_dir: Path, payload: object) -> None:
        import json

        from twin.shared.llm.embedding.harrier_manifest import MARKER_NAME

        model_dir.mkdir(parents=True, exist_ok=True)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        (model_dir / MARKER_NAME).write_text(text, encoding="utf-8")

    def _valid_marker(self) -> dict:
        from twin.shared.llm.embedding.harrier_manifest import FILES_SHA256

        return {
            "repo": "x", "revision": "y", "profile": "q4",
            "dimensions": 640, "files": sorted(FILES_SHA256),
        }

    async def test_corrupt_marker_counts_as_not_installed(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "harrier"
        self._write_marker(model_dir, "{not-json")
        service = HarrierEmbeddingService(model_dir=model_dir, expected_dim=640)

        assert "corrupt" in (service.install_problem() or "")
        assert await service.get_embedding("hello") == []

    async def test_stale_marker_dimensions_rejected(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "harrier"
        marker = self._valid_marker()
        marker["dimensions"] = 1024
        self._write_marker(model_dir, marker)
        service = HarrierEmbeddingService(model_dir=model_dir, expected_dim=640)

        assert "stale" in (service.install_problem() or "")
        assert await service.get_embedding("hello") == []

    def _mock_loader(self, monkeypatch, outputs: np.ndarray) -> None:
        """Mock Tokenizer/ONNX constructors so _load() succeeds with canned outputs."""
        import types

        import twin.shared.llm.embedding.harrier_embedding_service as mod

        encoding = types.SimpleNamespace(
            ids=[1, 2, 3], attention_mask=[1, 1, 1], type_ids=[0, 0, 0]
        )
        tokenizer = MagicMock()
        tokenizer.encode.return_value = encoding
        graph_out = types.SimpleNamespace(
            name=mod.OUTPUT_NAME, shape=(1, mod.DIMENSIONS)
        )
        session = MagicMock()
        session.get_inputs.return_value = []
        session.get_outputs.return_value = [graph_out]
        session.run.return_value = [outputs]
        monkeypatch.setattr(
            mod, "Tokenizer", types.SimpleNamespace(from_file=lambda p: tokenizer)
        )
        monkeypatch.setattr(
            mod, "ort", types.SimpleNamespace(InferenceSession=lambda *a, **k: session)
        )

    async def test_sticky_failure_needs_reset_after_repair(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        model_dir = tmp_path / "harrier"  # no marker yet: install missing
        service = HarrierEmbeddingService(model_dir=model_dir, expected_dim=640)
        self._mock_loader(monkeypatch, _unit_vector())

        assert await service.get_embedding("hello") == []  # sticky failure set

        self._write_marker(model_dir, self._valid_marker())  # repair lands
        assert await service.get_embedding("hello") == []  # still sticky: no retry storm

        service.reset_load_error()
        vector = await service.get_embedding("hello")
        assert len(vector) == 640

    async def test_close_clears_sticky_failure(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        model_dir = tmp_path / "harrier"
        service = HarrierEmbeddingService(model_dir=model_dir, expected_dim=640)
        self._mock_loader(monkeypatch, _unit_vector())

        assert await service.get_embedding("hello") == []
        self._write_marker(model_dir, self._valid_marker())
        await service.close()  # restart-equivalent releases the sticky error
        assert len(await service.get_embedding("hello")) == 640


class TestHarrierFactory:
    def test_factory_builds_harrier(self, tmp_path: Path) -> None:
        service = create_embedding_service(
            model_dir=str(tmp_path / "harrier"), expected_dim=640
        )
        assert isinstance(service, HarrierEmbeddingService)
        assert service.expected_dim == 640

    def test_factory_rejects_legacy_dim(self, tmp_path: Path, monkeypatch) -> None:
        from twin.shared.config.settings import Config

        with pytest.raises(ValueError, match="refusing"):
            create_embedding_service(
                model_dir=str(tmp_path / "harrier"), expected_dim=1024
            )
        monkeypatch.setattr(Config, "EMBEDDING_VECTOR_SIZE", 1024)
        with pytest.raises(ValueError, match="refusing"):
            create_embedding_service(model_dir=str(tmp_path / "harrier"))


class TestHarrierBudgets:
    def test_limits_match_another_brain(self) -> None:
        assert BUDGET_DOCUMENT_TOKENS == 256
        assert BUDGET_QUERY_TOKENS == 128

    def test_check_budget(self) -> None:
        assert check_budget(tokens=10, is_query=True) is None
        assert check_budget(tokens=200, is_query=False) is None
        assert "128" in (check_budget(tokens=129, is_query=True) or "")
        assert "256" in (check_budget(tokens=300, is_query=False) or "")
