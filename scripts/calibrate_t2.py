#!/usr/bin/env python3
"""calibrate_t2.py — P1 measurement for the T2 diary/retrieval plan.

Measures the REAL cosine distribution between labeled queries (query-side,
with Config.EMBEDDING_QUERY_PREFIX applied — the thing P0 fixed) and the
T2 summaries actually stored in Redis, then proposes values for:

  - T2_MIN_COSINE        (search gate — query vs doc)
  - T2_MERGE_MIN_COSINE  (P2 diary same-day merge gate — incoming summary
                          vs stored doc, passage vs passage)

Run MANUALLY on the host (outside the container), read-only against Redis:

    python3 scripts/calibrate_t2.py
    python3 scripts/calibrate_t2.py --redis-url redis://localhost:6379

It never writes to Redis (SCAN + HGETALL only) and never touches the
production trace file. Merge probes are embedded in-memory only.

--offline needs no Redis at all: it embeds the built-in labeled probe set
below with the local read-only Harrier model and reports both gate
separations plus a precision/recall sweep. Docs whose stored embedding dim
does not match the configured size are skipped and counted (mixed legacy
1024 / Harrier 640 data — migrate with scripts/migrate_t2_harrier.py).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

# Same load the real entrypoints do (twin/march7/__main__.py) — must happen
# BEFORE Config is imported, since settings reads os.getenv at import time.
load_dotenv(ROOT / ".env", override=True)

import redis.asyncio as aioredis  # noqa: E402

from twin.shared.config.settings import (  # noqa: E402
    Config,
    T2_MERGE_MIN_COSINE_DEFAULT,
    T2_MIN_COSINE_DEFAULT,
)
from twin.shared.llm.embedding import HarrierEmbeddingService  # noqa: E402
from twin.shared.llm.embedding.embedding_trace_logger import (  # noqa: E402
    cosine_similarity,
    token_overlap,
)
from twin.shared.memory.diary.codec import unpack_embedding  # noqa: E402
from twin.shared.memory.diary import TimelineSummaryStore  # noqa: E402

# --------------------------------------------------------------------------
# Labeled query set (built 2026-07-03 from data/embedding_trace.jsonl real
# queries + casual/pronoun variants + negatives). Scopes match the docs
# present in Redis at calibration time.
# --------------------------------------------------------------------------

PERSONA = "1487427038280421456"  # test persona "Hòa": food/hiking/lifestyle/pet/work
REAL = "726302130318868500"      # real user: habit/psychological/work

# (scope, query, expected_topics) — empty set = negative (nothing relevant).
QUERIES: list[tuple[str, str, set[str]]] = [
    # persona / food
    (PERSONA, "Bảy ơi cuối tuần này tui nên nấu món gì ngon?", {"food"}),
    (PERSONA, "tui mê ăn phở với bún chả lắm", {"food"}),
    (PERSONA, "món bún đậu mắm tôm ăn ở đâu ngon nhỉ", {"food"}),
    # persona / pet
    (PERSONA, "tui mới nhận nuôi một con mèo tên Mun, nó đen thui à", {"pet"}),
    (PERSONA, "con mèo nhà tui dạo này quậy lắm", {"pet"}),
    (PERSONA, "con Mun nó cứ leo trèo khắp nơi", {"pet"}),
    # persona / hiking
    (PERSONA, "cuối tuần tui hay đi leo núi ở Bà Đen cho khỏe", {"hiking"}),
    (PERSONA, "leo Tà Năng có cực không ta", {"hiking"}),
    # persona / work
    (PERSONA, "tui đang làm một dự án web bằng React với TypeScript, hơi khó", {"work"}),
    (PERSONA, "Bảy ơi tui đang debug cái project hoài không xong", {"work"}),
    (PERSONA, "cái app todo list của tui bị lỗi state hoài", {"work"}),
    (PERSONA, "ê mà cái con hàm bữa t với m fix ấy nó sao rồi nhỉ", {"work"}),
    # persona / lifestyle
    (PERSONA, "tối nay coi gì trên Netflix đây ta", {"lifestyle"}),
    (PERSONA, "gợi ý tui mấy bài nhạc indie Việt đi", {"lifestyle"}),
    (PERSONA, "dạo này tui nghiện nghe Ngọt với Chillies", {"lifestyle"}),
    # persona / negatives (no relevant doc in scope)
    (PERSONA, "Bảy ơi thời tiết hôm nay thế nào?", set()),
    (PERSONA, "mai nắng hay mưa thế m. cả Phú Thọ với luôn", set()),
    (PERSONA, "alo bé", set()),
    (PERSONA, "1 cộng 1 bằng mấy", set()),
    (PERSONA, "giá vàng hôm nay bao nhiêu", set()),
    (PERSONA, "trận chung kết tối qua tỉ số bao nhiêu", set()),
    # real user / psychological
    (REAL, "Bảy ơi an ủi tui mệt quá", {"psychological"}),
    (REAL, "tối nay tự nhiên thấy trống trải ghê", {"psychological"}),
    (REAL, "hic hoài", {"psychological"}),
    # real user / work (embedding eval)
    (REAL, "cái vụ embedding model tiếng Việt hôm bữa tới đâu rồi", {"work"}),
    (REAL, "e5-small với qwen3 cái nào ngon hơn cho tiếng Việt", {"work"}),
    # real user / habit
    (REAL, "sao tui cứ nhắn đi nhắn lại một câu vậy trời", {"habit", "psychological"}),
    # real user / negatives (food doc belongs to the OTHER scope)
    (REAL, "công thức nấu phở bò sao cho ngọt nước", set()),
    (REAL, "M search hộ t cái coi", set()),
]

# Merge probes: synthetic "incoming consolidation summary" texts (diary-entry
# style), embedded PASSAGE-side in-memory. same-topic pair = should merge on
# the same day; vs other topics = must never merge.
MERGE_PROBES: list[tuple[str, str, str]] = [
    (PERSONA, "food",
     "Hòa nấu bún bò Huế sáng nay, cay xé lưỡi, sả ớt chanh, thịt bò bắp với "
     "chả cua. Vẫn mê nấu món Việt như mọi khi."),
    (PERSONA, "pet",
     "Mèo Mun hôm nay leo lên mái nhà làm Hòa hết hồn, gọi mãi mới chịu "
     "xuống, xong nhảy vào đùi đòi ăn."),
    (PERSONA, "work",
     "Hòa vẫn vật lộn với bug state trong app todo React/TypeScript, nghi do "
     "Zustand store update mà component không re-render."),
    (PERSONA, "hiking",
     "Cuối tuần này Hòa tính đổi gió leo núi Chứa Chan thay vì Bà Đen, rủ "
     "thêm hai đứa bạn cùng đi."),
    (REAL, "psychological",
     "Đêm nay Hòa lại thấy cô đơn, nhắn liên tiếp mấy tin tâm sự, cần được "
     "lắng nghe hơn là lời khuyên."),
    (REAL, "work",
     "Hòa chốt dùng qwen3-embedding 0.6b thay cho e5-small cho semantic "
     "search tiếng Việt, reindex 1024 chiều đã chạy xong."),
]


# --------------------------------------------------------------------------
# Offline labeled probes (no Redis). Calibrated 2026-09-30 on the real
# harrier-oss-v1-270m q4 ONNX with the deployed query/passage prefixes:
# retrieval 13 positives + 7 negatives, merge 8 same-event follow-ups +
# 6 same-topic paraphrases vs 60+ cross-topic pairs. Expected separation:
# negatives peak ~0.58, clear positives >= ~0.63 (gate 0.60); follow-ups
# 0.82-0.93, cross-topic max ~0.71 (gate 0.75).
# --------------------------------------------------------------------------

# (doc_id, topic, passage-side summary text)
OFFLINE_DOCS: list[tuple[str, str, str]] = [
    ("hiking1", "hiking", "Cuối tuần Hòa đi leo núi Bà Đen với nhóm bạn, mang theo lều và đồ ăn nhẹ, ngắm bình minh trên đỉnh."),
    ("hiking2", "hiking", "Hòa lên kế hoạch leo Tà Năng Phan Dũng tháng sau, chuẩn bị giày trekking và balo chống nước."),
    ("food1", "food", "Hòa nấu bún bò Huế sáng nay, nước dùng ngọt xương, sả ớt chanh, thịt bò bắp với chả cua."),
    ("food2", "food", "Tối qua Hòa thử quán phở mới gần nhà, nước lèo trong, bánh phở mềm, thịt tái ngon."),
    ("pet1", "pet", "Mèo Mun nhà Hòa hôm nay leo lên mái nhà làm cả nhà hết hồn, gọi mãi mới chịu xuống ăn tối."),
    ("pet2", "pet", "Hòa mới nhận nuôi một con mèo tên Mun, lông đen, rất quậy, hay leo trèo khắp nơi."),
    ("work1", "work", "Hòa vật lộn với bug state trong app todo React TypeScript, nghi do Zustand store update mà component không re-render."),
    ("work2", "work", "Dự án web của Hòa dùng React với TypeScript, đang debug lỗi build và viết thêm unit test."),
    ("lifestyle1", "lifestyle", "Tối nay Hòa coi phim trên Netflix, đang phân vân giữa phim hành động và hài tình cảm."),
    ("lifestyle2", "lifestyle", "Dạo này Hòa nghiện nghe nhạc indie Việt, Ngọt với Chillies, mở playlist mỗi tối."),
    ("psych1", "psychological", "Đêm nay Hòa thấy cô đơn và trống trải, nhắn tin tâm sự liên tiếp, cần được lắng nghe."),
    ("psych2", "psychological", "Hòa mệt mỏi sau tuần làm việc dài, tâm trạng xuống, muốn nghỉ ngơi và được an ủi."),
]

# (query, expected_topic or None for negatives)
OFFLINE_QUERIES: list[tuple[str, str | None]] = [
    ("cuối tuần này tui nên leo núi nào gần Sài Gòn?", "hiking"),
    ("leo Bà Đen có cực không, cần chuẩn bị gì?", "hiking"),
    ("tui mê ăn phở với bún chả lắm", "food"),
    ("món bún bò nấu sao cho ngọt nước?", "food"),
    ("công thức nấu phở bò sao cho ngọt nước", "food"),
    ("con mèo nhà tui dạo này quậy lắm", "pet"),
    ("con Mun nó cứ leo trèo khắp nơi", "pet"),
    ("tui đang debug cái project React hoài không xong", "work"),
    ("cái app todo của tui bị lỗi state hoài", "work"),
    ("tối nay coi gì trên Netflix đây ta", "lifestyle"),
    ("gợi ý tui mấy bài nhạc indie Việt đi", "lifestyle"),
    ("Bảy ơi an ủi tui, mệt quá", "psychological"),
    ("tối nay tự nhiên thấy trống trải ghê", "psychological"),
    ("Bảy ơi thời tiết hôm nay thế nào?", None),
    ("mai nắng hay mưa thế, cả Phú Thọ nữa", None),
    ("1 cộng 1 bằng mấy", None),
    ("giá vàng hôm nay bao nhiêu", None),
    ("trận chung kết tối qua tỉ số bao nhiêu", None),
    ("alo bé", None),
    ("M search hộ t cái coi", None),
]

# (base_doc_id, followup_text): same-event continuations that SHOULD merge.
OFFLINE_FOLLOWUPS: list[tuple[str, str]] = [
    ("hiking1", "Hòa kể thêm về chuyến leo Bà Đen cuối tuần rồi, ngắm bình minh trên đỉnh với nhóm bạn, mang lều và đồ ăn nhẹ."),
    ("hiking1", "Chuyến leo núi Bà Đen của Hòa vui lắm, cả nhóm cắm lều qua đêm rồi dậy sớm ngắm bình minh."),
    ("food1", "Hòa vẫn còn thòm thèm món bún bò Huế sáng nay, nước dùng ngọt xương sả ớt, thịt bò bắp với chả cua."),
    ("food1", "Bún bò Huế Hòa nấu sáng nay được cả nhà khen, nước lèo đậm đà sả ớt chanh."),
    ("pet1", "Mèo Mun sau vụ leo mái nhà thì bị Hòa la, giờ nằm cuộn tròn trên đùi đòi vuốt ve."),
    ("pet1", "Hòa vẫn chưa hết hồn vụ Mun leo lên mái nhà hôm nay, may mà nó tự xuống ăn tối."),
    ("work1", "Hòa nghi bug state app todo là do Zustand store update nhưng component không re-render, đang thử fix."),
    ("work1", "Vụ bug React TypeScript app todo của Hòa vẫn chưa xong, đang đọc docs Zustand để tìm cách fix re-render."),
]


def fmt(v: float) -> str:
    return f"{v:.3f}"


def _dist(name: str, xs: list[tuple[float, str]]) -> None:
    if not xs:
        print(f"{name}: (empty)")
        return
    vs = sorted(v for v, _ in xs)
    print(f"{name}: n={len(vs)} min={fmt(vs[0])} p25={fmt(vs[len(vs)//4])} "
          f"med={fmt(vs[len(vs)//2])} max={fmt(vs[-1])}")


async def load_docs(r: aioredis.Redis, store: TimelineSummaryStore) -> list[dict]:
    docs = []
    skipped_dim = 0
    async for key in r.scan_iter(match=f"{store.prefix}:*".encode(), count=200):
        h = await r.hgetall(key)
        if not h or b"embedding" not in h:
            continue
        embedding = unpack_embedding(h[b"embedding"])
        if len(embedding) != Config.EMBEDDING_VECTOR_SIZE:
            skipped_dim += 1  # legacy 1024 / mixed-space data: migrate first
            continue
        docs.append({
            "key": key.decode(),
            "user_id": h[b"user_id"].decode(),
            "topic": h.get(b"topic", b"").decode(),
            "summary": h[b"summary"].decode(),
            "embedding": embedding,
        })
    docs.sort(key=lambda d: (d["user_id"], d["topic"]))
    if skipped_dim:
        print(f"skipped {skipped_dim} doc(s) with dim != {Config.EMBEDDING_VECTOR_SIZE} "
              "(legacy vectors — run scripts/migrate_t2_harrier.py)")
    return docs


async def offline_main() -> int:
    """Calibrate both gates with the read-only local model, no Redis.

    Returns 0 when the measured separation supports the calibrated defaults,
    1 when the model/prefixes regressed (bars below reference the calibrated
    DEFAULTS, not the effective env-overridden gates — an owner override is
    reported but never fails this check).
    """
    print("=" * 78)
    print("T2 offline calibration — labeled probes, no Redis")
    print(f"  model=harrier-oss-v1-270m (local ONNX) dim={Config.EMBEDDING_VECTOR_SIZE}")
    print(f"  calibrated defaults: T2_MIN_COSINE={T2_MIN_COSINE_DEFAULT} "
          f"T2_MERGE_MIN_COSINE={T2_MERGE_MIN_COSINE_DEFAULT}")
    print(f"  effective gates:     T2_MIN_COSINE={Config.T2_MIN_COSINE} "
          f"T2_MERGE_MIN_COSINE={Config.T2_MERGE_MIN_COSINE}")
    print("=" * 78)

    svc = HarrierEmbeddingService(
        model_dir=Config.HARRIER_MODEL_DIR,
        expected_dim=Config.EMBEDDING_VECTOR_SIZE,
    )
    try:
        demb = {}
        for doc_id, topic, text in OFFLINE_DOCS:
            vec = await svc.get_embedding(Config.EMBEDDING_PASSAGE_PREFIX + text)
            if not vec:
                print(f"!! embed failed for doc {doc_id}; aborting")
                return 1
            demb[doc_id] = (topic, vec)
        print(f"embedded {len(demb)} passage docs")

        by_id = {doc_id: vec for doc_id, (_, vec) in demb.items()}
        topics = {doc_id: topic for doc_id, (topic, _) in demb.items()}

        # Review key cases, printed explicitly for the record.
        q_math = await svc.get_embedding(Config.EMBEDDING_QUERY_PREFIX + "1 cộng 1 bằng mấy")
        q_wx = await svc.get_embedding(Config.EMBEDDING_QUERY_PREFIX + "Bảy ơi thời tiết hôm nay thế nào?")
        print(f"pet1<->hiking1 (passage): {fmt(cosine_similarity(by_id['pet1'], by_id['hiking1']))}")
        print(f"math-query<->hiking1:     {fmt(cosine_similarity(q_math, by_id['hiking1']))}")
        print(f"weather-query<->hiking1:  {fmt(cosine_similarity(q_wx, by_id['hiking1']))}")

        pos, irr, neg, qdata = [], [], [], []
        for query, expected in OFFLINE_QUERIES:
            qv = await svc.get_embedding(Config.EMBEDDING_QUERY_PREFIX + query)
            if not qv:
                print(f"!! embed failed for {query!r}; aborting")
                return 1
            scored = sorted(
                ((cosine_similarity(qv, vec), did) for did, (_, vec) in demb.items()),
                key=lambda x: -x[0],
            )
            top_cos, top_id = scored[0]
            qdata.append((expected, top_cos, topics[top_id]))
            if expected:
                rel = [c for c, did in scored if topics[did] == expected]
                oth = [c for c, did in scored if topics[did] != expected]
                pos.append((max(rel), query))
                if oth:
                    irr.append((max(oth), query))
            else:
                neg.append((top_cos, query))

        same, cross, follow = [], [], []
        ids = list(demb)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                c = cosine_similarity(by_id[ids[i]], by_id[ids[j]])
                label = f"{ids[i]}<->{ids[j]}"
                (same if topics[ids[i]] == topics[ids[j]] else cross).append((c, label))
        for base_id, text in OFFLINE_FOLLOWUPS:
            fv = await svc.get_embedding(Config.EMBEDDING_PASSAGE_PREFIX + text)
            if not fv:
                print(f"!! embed failed for followup {base_id}; aborting")
                return 1
            follow.append((cosine_similarity(fv, by_id[base_id]), base_id))
            for did in ids:
                if topics[did] != topics[base_id]:
                    cross.append((cosine_similarity(fv, by_id[did]), f"fu({base_id})<->{did}"))

        print("\nDISTRIBUTIONS")
        _dist("POS (query vs relevant)      ", pos)
        _dist("IRR (pos query vs other)     ", irr)
        _dist("NEG (negative vs best)        ", neg)
        _dist("FOLLOWUP (should merge)       ", follow)
        _dist("SAME (paraphrase, may append) ", same)
        _dist("CROSS (must never merge)      ", cross)

        print("\nRETRIEVAL gate sweep (top1-topic-correct & >= gate)")
        for gate in (0.35, 0.45, 0.55, 0.60, 0.65):
            tp = sum(1 for exp, c, t in qdata if exp and t == exp and c >= gate)
            fn = sum(1 for exp, c, t in qdata if exp and not (t == exp and c >= gate))
            fp = sum(1 for exp, c, t in qdata if (not exp and c >= gate) or (exp and t != exp and c >= gate))
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            print(f"  gate={gate:.2f} tp={tp} fn={fn} fp={fp} precision={prec:.3f} recall={rec:.3f}")

        print("MERGE gate sweep (followup>=gate -> merge; cross>=gate -> FALSE merge)")
        for gate in (0.60, 0.70, 0.75, 0.80):
            tm = sum(1 for c, _ in follow if c >= gate)
            fm = sum(1 for c, _ in cross if c >= gate)
            print(f"  gate={gate:.2f} followup-merges={tm}/{len(follow)} false-merges={fm}/{len(cross)}")

        failures = []
        neg_hi = max((v for v, _ in neg), default=0.0)
        if neg_hi >= T2_MIN_COSINE_DEFAULT:
            failures.append(f"negative peak {neg_hi:.3f} >= retrieval default {T2_MIN_COSINE_DEFAULT}")
        pos_lo = min((v for v, _ in pos), default=1.0)
        if pos_lo < T2_MIN_COSINE_DEFAULT:
            failures.append(f"clear-positive min {pos_lo:.3f} < retrieval default {T2_MIN_COSINE_DEFAULT}")
        fu_lo = min((v for v, _ in follow), default=1.0)
        if fu_lo < T2_MERGE_MIN_COSINE_DEFAULT:
            failures.append(f"followup min {fu_lo:.3f} < merge default {T2_MERGE_MIN_COSINE_DEFAULT}")
        cross_hi = max((v for v, _ in cross), default=0.0)
        if cross_hi >= T2_MERGE_MIN_COSINE_DEFAULT:
            failures.append(f"cross-topic max {cross_hi:.3f} >= merge default {T2_MERGE_MIN_COSINE_DEFAULT}")
        if failures:
            print("\nOFFLINE REGRESSION FAIL:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print(f"\nOFFLINE OK: separation supports retrieval={T2_MIN_COSINE_DEFAULT} "
              f"merge={T2_MERGE_MIN_COSINE_DEFAULT}")
        return 0
    finally:
        await svc.close()


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redis-url", default="redis://localhost:6379")
    ap.add_argument("--limit", type=int, default=8, help="live search() limit")
    ap.add_argument("--offline", action="store_true",
                    help="no Redis: embed built-in labeled probes locally")
    args = ap.parse_args()
    if args.offline:
        return await offline_main()

    print("=" * 78)
    print("T2 calibration — query prefix under test:")
    print(f"  EMBEDDING_QUERY_PREFIX   = {Config.EMBEDDING_QUERY_PREFIX!r}")
    print(f"  EMBEDDING_PASSAGE_PREFIX = {Config.EMBEDDING_PASSAGE_PREFIX!r}")
    print(f"  model=harrier-oss-v1-270m (local ONNX) dim={Config.EMBEDDING_VECTOR_SIZE}")
    print(f"  current T2_MIN_COSINE={Config.T2_MIN_COSINE} (gate DISABLED for this run)")
    print("=" * 78)

    # Gate off in-process so live search() shows everything it fused.
    Config.T2_MIN_COSINE = 0.0

    r = aioredis.Redis.from_url(args.redis_url, db=Config.TIMELINE_REDIS_DB,
                                password=Config.REDIS_PASSWORD, decode_responses=False)
    store = TimelineSummaryStore(r, embedding_dim=Config.EMBEDDING_VECTOR_SIZE)
    svc = HarrierEmbeddingService(
        model_dir=Config.HARRIER_MODEL_DIR,
        expected_dim=Config.EMBEDDING_VECTOR_SIZE,
    )

    try:
        docs = await load_docs(r, store)
        print(f"\nT2 docs in Redis (read-only): {len(docs)}")
        for d in docs:
            print(f"  [{d['user_id']}|{d['topic']}] {d['summary'][:72]}")
        if not docs:
            print("No docs — nothing to calibrate against.")
            return 0

        by_scope: dict[str, list[dict]] = {}
        for d in docs:
            by_scope.setdefault(d["user_id"], []).append(d)

        # ---------------- SEARCH gate: query (prefixed) vs doc ----------------
        pos, irr, neg = [], [], []   # (cosine, label) buckets
        mismatches = 0
        print("\n" + "-" * 78)
        print("SEARCH calibration — cosine per (query, doc), matrix + live search()")
        print("-" * 78)
        for scope, query, expected in QUERIES:
            emb = await svc.get_embedding(Config.EMBEDDING_QUERY_PREFIX + query)
            if not emb:
                print(f"!! embed failed for {query!r}")
                continue
            scored = sorted(
                ((cosine_similarity(emb, d["embedding"]), d) for d in by_scope.get(scope, [])),
                key=lambda x: -x[0],
            )
            kind = "NEG" if not expected else "POS"
            top_line = ", ".join(
                f"{d['topic']}={fmt(c)}{'✓' if d['topic'] in expected else ''}"
                for c, d in scored[:4]
            )
            print(f"[{kind}] {query[:52]!r:<56} {top_line}")
            if expected:
                rel = [c for c, d in scored if d["topic"] in expected]
                oth = [c for c, d in scored if d["topic"] not in expected]
                if rel:
                    pos.append((max(rel), query))
                if oth:
                    irr.append((max(oth), query))
                to = token_overlap(query, scored[0][1]["summary"]) if scored else 0.0
                # live production path (hybrid RRF + fusion), gate disabled
                live = await store.search(scope, emb, limit=args.limit, query_text=query)
                live_top = live[0] if live else None
                live_topic = (live_top or {}).get("topic")
                matrix_topic = scored[0][1]["topic"] if scored else None
                if live_topic != matrix_topic:
                    mismatches += 1
                    print(f"      live search() top1={live_topic} != matrix top1={matrix_topic} "
                          f"(RRF/BM25 reorder — check if expected) tok_overlap={fmt(to)}")
            else:
                if scored:
                    neg.append((scored[0][0], query))

        # ---------------- MERGE gate: incoming summary vs docs ----------------
        same, cross = [], []
        print("\n" + "-" * 78)
        print("MERGE calibration — synthetic incoming summaries (passage-side, in-memory)")
        print("-" * 78)
        for scope, topic, text in MERGE_PROBES:
            emb = await svc.get_embedding(Config.EMBEDDING_PASSAGE_PREFIX + text)
            if not emb:
                print(f"!! embed failed for merge probe {topic}")
                continue
            for d in by_scope.get(scope, []):
                c = cosine_similarity(emb, d["embedding"])
                if d["topic"] == topic:
                    same.append((c, f"{topic}↔{topic}"))
                    print(f"[SAME ] {topic:<13} vs {d['topic']:<13} {fmt(c)}")
                else:
                    cross.append((c, f"{topic}↔{d['topic']}"))
        # stored doc↔doc, same scope, different topic (must never merge)
        for scope, ds in by_scope.items():
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    if ds[i]["topic"] != ds[j]["topic"]:
                        c = cosine_similarity(ds[i]["embedding"], ds[j]["embedding"])
                        cross.append((c, f"{ds[i]['topic']}↔{ds[j]['topic']}"))
        cross.sort(key=lambda x: -x[0])
        print("[CROSS] top-5 highest cross-topic (must stay BELOW merge gate):")
        for c, label in cross[:5]:
            print(f"        {label:<30} {fmt(c)}")

        # ---------------- Summary & suggestions ----------------
        def dist(name: str, xs: list[tuple[float, str]]) -> None:
            if not xs:
                print(f"{name}: (empty)")
                return
            vs = sorted(v for v, _ in xs)
            lo, hi = xs[min(range(len(xs)), key=lambda i: xs[i][0])], xs[max(range(len(xs)), key=lambda i: xs[i][0])]
            print(f"{name}: n={len(vs)} min={fmt(vs[0])} p25={fmt(vs[len(vs)//4])} "
                  f"med={fmt(vs[len(vs)//2])} max={fmt(vs[-1])}")
            print(f"    min ← {lo[1][:60]!r}")
            print(f"    max ← {hi[1][:60]!r}")

        print("\n" + "=" * 78)
        print("DISTRIBUTIONS")
        dist("POS (query vs its relevant doc)      ", pos)
        dist("IRR (pos query vs best OTHER-topic)  ", irr)
        dist("NEG (negative query vs best doc)     ", neg)
        dist("SAME (merge probe vs same-topic doc) ", same)
        dist("CROSS (probe/doc vs other-topic doc) ", cross)

        noise_hi = max([v for v, _ in irr + neg], default=0.0)
        pos_lo = min([v for v, _ in pos], default=1.0)
        print(f"\nSEARCH gate: highest noise={fmt(noise_hi)}  lowest relevant={fmt(pos_lo)}"
              f"  {'SEPARATED' if pos_lo > noise_hi else '!! OVERLAP — pick by trade-off'}")
        print(f"  suggestion zone: ({fmt(noise_hi)}, {fmt(pos_lo)}) → "
              f"midpoint {fmt((noise_hi + pos_lo) / 2)}")
        cross_hi = max([v for v, _ in cross], default=0.0)
        same_lo = min([v for v, _ in same], default=1.0)
        print(f"MERGE gate: highest cross-topic={fmt(cross_hi)}  lowest same-topic={fmt(same_lo)}"
              f"  {'SEPARATED' if same_lo > cross_hi else '!! OVERLAP — bias HIGH (false merge is worse)'}")
        print(f"  suggestion zone: ({fmt(cross_hi)}, {fmt(same_lo)}) → "
              f"upper-third {fmt(cross_hi + (same_lo - cross_hi) * 2 / 3)}")
        if mismatches:
            print(f"\nNOTE: live search() top1 differed from matrix top1 on {mismatches} "
                  f"queries (RRF/BM25 fusion) — inspect above.")
    finally:
        await svc.close()
        await r.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
