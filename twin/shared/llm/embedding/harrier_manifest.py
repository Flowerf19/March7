"""Locked Harrier q4 ONNX manifest (mirrors another-brain).

Single source of truth for the local embedding model: upstream repo +
revision, the exact five runtime files and their SHA-256 hashes, output
contract (sentence_embedding, 640 dims, unit L2), and install marker name.

Pure standard library — no onnxruntime import here, so config/factory code
can read it without loading the model.
"""
from __future__ import annotations

REPO = "onnx-community/harrier-oss-v1-270m-ONNX"
REVISION = "d59c919d0159aea2c19ed7d04288fcdd048d0f9c"

FILES_SHA256: dict[str, str] = {
    "onnx/model_q4.onnx": "228dca2603b907d673dd99cf89c309c0ca68baeed127416a5e027a48e62b0f49",
    "onnx/model_q4.onnx_data": "b5a15487360f5341659480ae4b5ad60028d5f865bd329196ec8d5708bbed3118",
    "config.json": "5366f9919a82aaeceb6707bf218c5769f414d60f5dbaf781fa07e5465487fd7c",
    "tokenizer.json": "ec95be298bea26f90370854faa650744c9fb0a04ca5e5ff95dd3913393ac5e45",
    "tokenizer_config.json": "135405f3479eaebc473e2e78593f2195c7598948a215ee748758def426b30f59",
}

GRAPH_REL_PATH = "onnx/model_q4.onnx"
TOKENIZER_REL_PATH = "tokenizer.json"
OUTPUT_NAME = "sentence_embedding"
DIMENSIONS = 640
MARKER_NAME = ".installed.json"

# Asymmetric instruction prompt for the QUERY side only (documents embed raw).
# Callers apply Config.EMBEDDING_QUERY_PREFIX — this constant documents the
# locked upstream wording new deployments should use.
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query"
    "\nQuery: "
)
