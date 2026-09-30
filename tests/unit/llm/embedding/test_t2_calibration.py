"""T2 gate calibration regression tests (finding #16).

No-model tests assert the calibrated defaults and the recorded measurement
zones. The real-model test embeds a compact labeled subset with the read-only
Harrier weights (skipped when HARRIER_MODEL_DIR has no valid install) and
asserts the live separation supports both gates.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from twin.shared.config.settings import (
    T2_MERGE_MIN_COSINE_DEFAULT,
    T2_MIN_COSINE_DEFAULT,
)

# Recorded 2026-09-30 on real harrier-oss-v1-270m q4 ONNX (deployed prefixes;
# full probes in scripts/calibrate_t2.py --offline): 13 clear positives +
# 7 negatives, 8 same-event follow-ups vs 140 cross-topic pairs.
RECORDED_POS_MIN = 0.603
RECORDED_NEG_MAX = 0.577
RECORDED_FOLLOWUP_MIN = 0.821
RECORDED_CROSS_MAX = 0.710


class TestGateDefaults:
    def test_retrieval_default_calibrated(self) -> None:
        assert T2_MIN_COSINE_DEFAULT == 0.60

    def test_merge_default_calibrated(self) -> None:
        assert T2_MERGE_MIN_COSINE_DEFAULT == 0.75

    def test_recorded_zones_separated_by_defaults(self) -> None:
        # Retrieval: every recorded negative peaks below the gate, every
        # clear positive clears it (precision 1.00, recall 1.00 on the set).
        assert RECORDED_NEG_MAX < T2_MIN_COSINE_DEFAULT <= RECORDED_POS_MIN
        # Merge: every cross-topic pair stays below, every same-event
        # follow-up merges (8/8 merge, 0/140 false merges).
        assert RECORDED_CROSS_MAX < T2_MERGE_MIN_COSINE_DEFAULT <= RECORDED_FOLLOWUP_MIN

    def test_legacy_retrieval_values_all_below_default(self) -> None:
        for legacy in (0.0, 0.35, 0.42, 0.45):
            assert legacy < T2_MIN_COSINE_DEFAULT

    def test_legacy_merge_value_below_default(self) -> None:
        assert 0.60 < T2_MERGE_MIN_COSINE_DEFAULT


class TestLegacyOverrideWarning:
    def test_below_default_warns(self, monkeypatch, caplog) -> None:
        import twin.shared.config.settings as settings

        monkeypatch.setenv("T2_MIN_COSINE", "0.35")
        with caplog.at_level(logging.WARNING, logger="twin.shared.config.settings"):
            settings._warn_if_legacy_t2_gate(
                "T2_MIN_COSINE", T2_MIN_COSINE_DEFAULT, "legacy Qwen-era"
            )
        assert "T2_MIN_COSINE=0.35" in caplog.text
        assert "0.60" in caplog.text

    def test_unset_or_stricter_stays_quiet(self, monkeypatch, caplog) -> None:
        import twin.shared.config.settings as settings

        monkeypatch.delenv("T2_MERGE_MIN_COSINE", raising=False)
        with caplog.at_level(logging.WARNING, logger="twin.shared.config.settings"):
            settings._warn_if_legacy_t2_gate(
                "T2_MERGE_MIN_COSINE", T2_MERGE_MIN_COSINE_DEFAULT, "legacy"
            )
        assert "T2_MERGE_MIN_COSINE" not in caplog.text

        monkeypatch.setenv("T2_MERGE_MIN_COSINE", "0.80")
        with caplog.at_level(logging.WARNING, logger="twin.shared.config.settings"):
            settings._warn_if_legacy_t2_gate(
                "T2_MERGE_MIN_COSINE", T2_MERGE_MIN_COSINE_DEFAULT, "legacy"
            )
        assert "T2_MERGE_MIN_COSINE" not in caplog.text


# Compact labeled subset of scripts/calibrate_t2.py OFFLINE_* for live checks.
_DOCS = {
    "hiking1": ("hiking", "Cuối tuần Hòa đi leo núi Bà Đen với nhóm bạn, mang theo lều và đồ ăn nhẹ, ngắm bình minh trên đỉnh."),
    "hiking2": ("hiking", "Hòa lên kế hoạch leo Tà Năng Phan Dũng tháng sau, chuẩn bị giày trekking và balo chống nước."),
    "food1": ("food", "Hòa nấu bún bò Huế sáng nay, nước dùng ngọt xương, sả ớt chanh, thịt bò bắp với chả cua."),
    "food2": ("food", "Tối qua Hòa thử quán phở mới gần nhà, nước lèo trong, bánh phở mềm, thịt tái ngon."),
    "pet1": ("pet", "Mèo Mun nhà Hòa hôm nay leo lên mái nhà làm cả nhà hết hồn, gọi mãi mới chịu xuống ăn tối."),
    "pet2": ("pet", "Hòa mới nhận nuôi một con mèo tên Mun, lông đen, rất quậy, hay leo trèo khắp nơi."),
    "work1": ("work", "Hòa vật lộn với bug state trong app todo React TypeScript, nghi do Zustand store update mà component không re-render."),
    "work2": ("work", "Dự án web của Hòa dùng React với TypeScript, đang debug lỗi build và viết thêm unit test."),
    "lifestyle1": ("lifestyle", "Tối nay Hòa coi phim trên Netflix, đang phân vân giữa phim hành động và hài tình cảm."),
    "psych1": ("psychological", "Đêm nay Hòa thấy cô đơn và trống trải, nhắn tin tâm sự liên tiếp, cần được lắng nghe."),
}
_QUERIES = [
    ("cuối tuần này tui nên leo núi nào gần Sài Gòn?", "hiking"),
    ("tui mê ăn phở với bún chả lắm", "food"),
    ("con mèo nhà tui dạo này quậy lắm", "pet"),
    ("tui đang debug cái project React hoài không xong", "work"),
    ("Bảy ơi thời tiết hôm nay thế nào?", None),
    ("1 cộng 1 bằng mấy", None),
    ("alo bé", None),
    ("giá vàng hôm nay bao nhiêu", None),
]
_FOLLOWUPS = [
    ("hiking1", "Hòa kể thêm về chuyến leo Bà Đen cuối tuần rồi, ngắm bình minh trên đỉnh với nhóm bạn, mang lều và đồ ăn nhẹ."),
    ("pet1", "Mèo Mun sau vụ leo mái nhà thì bị Hòa la, giờ nằm cuộn tròn trên đùi đòi vuốt ve."),
]


def _model_dir() -> Path | None:
    from twin.shared.llm.embedding.harrier_manifest import MARKER_NAME

    raw = os.environ.get("HARRIER_MODEL_DIR", "models/harrier-q4")
    model_dir = Path(raw)
    if not (model_dir / MARKER_NAME).exists():
        return None
    return model_dir


class TestLiveHarrierSeparation:
    @pytest.mark.asyncio
    async def test_live_gates_separate_labeled_probes(self) -> None:
        model_dir = _model_dir()
        if model_dir is None:
            pytest.skip("no Harrier install (HARRIER_MODEL_DIR marker missing)")
        from twin.shared.config.settings import Config
        from twin.shared.llm.embedding.embedding_trace_logger import cosine_similarity
        from twin.shared.llm.embedding.harrier_embedding_service import (
            HarrierEmbeddingService,
        )

        svc = HarrierEmbeddingService(model_dir=model_dir, expected_dim=640)
        try:
            demb = {}
            for doc_id, (topic, text) in _DOCS.items():
                vec = await svc.get_embedding(Config.EMBEDDING_PASSAGE_PREFIX + text)
                assert len(vec) == 640, f"{doc_id}: no embedding"
                demb[doc_id] = (topic, vec)

            for query, expected in _QUERIES:
                qv = await svc.get_embedding(Config.EMBEDDING_QUERY_PREFIX + query)
                assert len(qv) == 640, f"{query!r}: no embedding"
                scored = sorted(
                    ((cosine_similarity(qv, v), t) for _, (t, v) in demb.items()),
                    key=lambda x: -x[0],
                )
                top_cos, top_topic = scored[0]
                if expected is None:
                    assert top_cos < T2_MIN_COSINE_DEFAULT, (
                        f"negative {query!r} leaks past retrieval gate: {top_cos:.3f}"
                    )
                else:
                    rel = max(c for c, t in scored if t == expected)
                    assert rel >= T2_MIN_COSINE_DEFAULT, (
                        f"positive {query!r} gated out: relevant={rel:.3f}"
                    )
                    assert top_topic == expected, (
                        f"positive {query!r} ranks {top_topic}, want {expected}"
                    )

            for base_id, text in _FOLLOWUPS:
                fv = await svc.get_embedding(Config.EMBEDDING_PASSAGE_PREFIX + text)
                base_topic, base_vec = demb[base_id]
                fu_cos = cosine_similarity(fv, base_vec)
                assert fu_cos >= T2_MERGE_MIN_COSINE_DEFAULT, (
                    f"followup for {base_id} would not merge: {fu_cos:.3f}"
                )
                for did, (topic, vec) in demb.items():
                    if topic == base_topic:
                        continue
                    c = cosine_similarity(fv, vec)
                    assert c < T2_MERGE_MIN_COSINE_DEFAULT, (
                        f"false merge: followup({base_id})<->{did}={c:.3f}"
                    )
            ids = list(demb)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if demb[ids[i]][0] == demb[ids[j]][0]:
                        continue
                    c = cosine_similarity(demb[ids[i]][1], demb[ids[j]][1])
                    assert c < T2_MERGE_MIN_COSINE_DEFAULT, (
                        f"false merge: {ids[i]}<->{ids[j]}={c:.3f}"
                    )
        finally:
            await svc.close()
