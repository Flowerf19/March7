"""Pull the locked Harrier ONNX embedding model into ./models/harrier-q4.

Mirrors the another-brain q4 manifest (onnx-community/harrier-oss-v1-270m-ONNX):
exactly five runtime files, SHA-256 verified, resume-safe. Stdlib-only so it
runs on the host without the container venv.

Usage:
    python scripts/pull_harrier_model.py [--model-dir models/harrier-q4]

The profile directory becomes visible to the provider only after the
.installed.json marker is written last — a partial pull is never loadable.
Repair invalidates any stale marker before fetching, and publishes the new
marker atomically (tmp + os.replace) only after every installed file
re-verifies. After a repair, restart running services or call
HarrierEmbeddingService.reset_load_error() to clear their sticky failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "onnx-community/harrier-oss-v1-270m-ONNX"
REVISION = "d59c919d0159aea2c19ed7d04288fcdd048d0f9c"

FILES_SHA256: dict[str, str] = {
    "onnx/model_q4.onnx": "228dca2603b907d673dd99cf89c309c0ca68baeed127416a5e027a48e62b0f49",
    "onnx/model_q4.onnx_data": "b5a15487360f5341659480ae4b5ad60028d5f865bd329196ec8d5708bbed3118",
    "config.json": "5366f9919a82aaeceb6707bf218c5769f414d60f5dbaf781fa07e5465487fd7c",
    "tokenizer.json": "ec95be298bea26f90370854faa650744c9fb0a04ca5e5ff95dd3913393ac5e45",
    "tokenizer_config.json": "135405f3479eaebc473e2e78593f2195c7598948a215ee748758def426b30f59",
}

CHUNK_BYTES = 1 << 20
MARKER_NAME = ".installed.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, expected_sha: str) -> None:
    """Download with resume; dest is a .part file, renamed by the caller.

    A .part left behind by an interrupted run is checksum-verified BEFORE any
    Range fetch: a complete part skips the download, and a 416 (range beyond
    EOF — the part already holds the whole file) is treated the same way.
    A corrupt/oversized part is deleted and fetched from scratch.
    """
    if dest.exists() and sha256_file(dest) == expected_sha:
        print("  part already complete (verified, skipping fetch)")
        return
    existing = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "march7-harrier-pull/1.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and dest.exists() and sha256_file(dest) == expected_sha:
            print("  server reports range complete (416); part verified")
            return
        if exc.code == 416:
            # Stale or oversized part the server cannot resume from; restart.
            print("  stale part (416, checksum mismatch); restarting from scratch")
            dest.unlink(missing_ok=True)
            _download(url, dest, expected_sha)
            return
        raise RuntimeError(f"download failed: {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"download failed: {url}: {exc}") from exc
    with resp:
        if existing and resp.status != 206:
            existing = 0  # server ignored Range; restart from scratch
        mode = "ab" if existing else "wb"
        total = resp.headers.get("Content-Length")
        total = (int(total) + existing) if total else None
        with dest.open(mode) as fh:
            downloaded = existing
            while True:
                chunk = resp.read(CHUNK_BYTES)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()


def _needs_fetch(model_dir: Path) -> list[str]:
    """Files whose installed copy is missing or fails its checksum."""
    missing = []
    for rel, expected in FILES_SHA256.items():
        final = model_dir / rel
        if not final.exists() or sha256_file(final) != expected:
            missing.append(rel)
    return missing


def pull(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = model_dir.parent / f"{model_dir.name}.tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    marker_path = model_dir / MARKER_NAME

    missing = _needs_fetch(model_dir)
    if missing:
        # Repair in progress: invalidate the old marker FIRST so a loader
        # never sees a partial/corrupt install as complete. A failure below
        # therefore leaves NO marker (honest not-installed); running services
        # keep their sticky load error until restart/reset_load_error().
        marker_path.unlink(missing_ok=True)
        print(f"repair: {len(missing)} file(s) need fetching, marker invalidated")
    else:
        print("all files cached and verified")

    for rel, expected in FILES_SHA256.items():
        final = model_dir / rel
        if final.exists() and sha256_file(final) == expected:
            print(f"ok (cached): {rel}")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{rel}"
        part = tmp_dir / (rel.replace("/", "__") + ".part")
        print(f"pull: {rel}")
        _download(url, part, expected)
        actual = sha256_file(part)
        if actual != expected:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"hash mismatch for {rel}: {actual} != {expected}")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, final)
        print(f"verified: {rel}")

    # Publish the marker atomically ONLY after every installed file verifies.
    for rel, expected in FILES_SHA256.items():
        final = model_dir / rel
        if not final.exists() or sha256_file(final) != expected:
            raise RuntimeError(
                f"incomplete installation: {rel} missing/corrupt; "
                f"marker {MARKER_NAME} NOT published"
            )
    marker = {
        "repo": REPO,
        "revision": REVISION,
        "profile": "q4",
        "dimensions": 640,
        "files": sorted(FILES_SHA256),
    }
    staging = model_dir / (MARKER_NAME + ".tmp")
    staging.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, marker_path)
    print(f"installed: {model_dir} (marker {MARKER_NAME} written)")
    return model_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pull Harrier q4 ONNX model.")
    parser.add_argument(
        "--model-dir", default="models/harrier-q4",
        help="Target directory (default: models/harrier-q4)",
    )
    args = parser.parse_args(argv)
    try:
        pull(Path(args.model_dir))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
