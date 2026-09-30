"""Unit tests for scripts/pull_harrier_model.py (mocked HTTP, no network)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "pull_harrier_model.py"


def _load_pull_module():
    spec = importlib.util.spec_from_file_location("pull_harrier_model", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeResp:
    """Minimal urllib response: status, headers.get, read, context manager."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, f"HTTP {code}", {}, io.BytesIO(b""))


def _install_files(pull, model_dir: Path, files: dict[str, bytes]) -> None:
    pull.FILES_SHA256.clear()
    pull.FILES_SHA256.update({rel: _sha(data) for rel, data in files.items()})


class TestPartRecovery:
    def test_complete_part_skips_fetch(self, tmp_path: Path, monkeypatch) -> None:
        pull = _load_pull_module()
        files = {"a.txt": b"aaa-bytes"}
        _install_files(pull, tmp_path, files)
        model_dir = tmp_path / "harrier"
        tmp_dir = tmp_path / "harrier.tmp"
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "a.txt.part").write_bytes(b"aaa-bytes")  # complete, interrupted before rename

        def _no_network(req, timeout=60):
            raise AssertionError("must not fetch: part already complete")

        monkeypatch.setattr(pull.urllib.request, "urlopen", _no_network)
        out = pull.pull(model_dir)

        assert out == model_dir
        assert (model_dir / "a.txt").read_bytes() == b"aaa-bytes"
        marker = json.loads((model_dir / pull.MARKER_NAME).read_text(encoding="utf-8"))
        assert marker["dimensions"] == 640

    def test_416_with_concurrently_completed_part_recovers(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pull = _load_pull_module()
        dest = tmp_path / "f.part"
        dest.write_bytes(b"done")
        expected = _sha(b"done")
        calls = {"n": 0}

        def _flaky_sha(path: Path) -> str:
            calls["n"] += 1
            return "mismatch" if calls["n"] == 1 else expected  # completed mid-flight

        def _raise_416(req, timeout=60):
            raise _http_error(req.full_url, 416)

        monkeypatch.setattr(pull, "sha256_file", _flaky_sha)
        monkeypatch.setattr(pull.urllib.request, "urlopen", _raise_416)
        pull._download("http://x/f", dest, expected)  # must not raise

    def test_416_with_stale_part_restarts_from_scratch(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pull = _load_pull_module()
        dest = tmp_path / "f.part"
        dest.write_bytes(b"stale-oversized-junk")
        expected = _sha(b"fresh-full-body")
        seen = []

        def _urlopen(req, timeout=60):
            seen.append(req.get_header("Range"))
            if len(seen) == 1:
                raise _http_error(req.full_url, 416)
            return _FakeResp(b"fresh-full-body", status=200)

        monkeypatch.setattr(pull.urllib.request, "urlopen", _urlopen)
        pull._download("http://x/f", dest, expected)

        assert dest.read_bytes() == b"fresh-full-body"
        assert seen[0] is not None and seen[1] is None  # resume, then fresh

    def test_resume_appends_remaining_bytes(self, tmp_path: Path, monkeypatch) -> None:
        pull = _load_pull_module()
        dest = tmp_path / "f.part"
        dest.write_bytes(b"0123456789")
        full = b"0123456789ABCDEF"
        expected = _sha(full)

        def _urlopen(req, timeout=60):
            assert req.get_header("Range") == "bytes=10-"
            return _FakeResp(b"ABCDEF", status=206)

        monkeypatch.setattr(pull.urllib.request, "urlopen", _urlopen)
        pull._download("http://x/f", dest, expected)

        assert dest.read_bytes() == full

    def test_hash_mismatch_deletes_part_and_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pull = _load_pull_module()
        files = {"a.txt": b"good"}
        _install_files(pull, tmp_path, files)
        model_dir = tmp_path / "harrier"

        def _urlopen(req, timeout=60):
            return _FakeResp(b"corrupt", status=200)

        monkeypatch.setattr(pull.urllib.request, "urlopen", _urlopen)
        with pytest.raises(RuntimeError, match="hash mismatch"):
            pull.pull(model_dir)

        assert not (tmp_path / "harrier.tmp" / "a.txt.part").exists()
        assert not (model_dir / pull.MARKER_NAME).exists()  # never published


class TestMarkerLifecycle:
    def test_failed_repair_invalidates_old_marker(self, tmp_path: Path, monkeypatch) -> None:
        pull = _load_pull_module()
        files = {"a.txt": b"good", "b.txt": b"also-good"}
        _install_files(pull, tmp_path, files)
        model_dir = tmp_path / "harrier"
        (model_dir / "a.txt").parent.mkdir(parents=True, exist_ok=True)
        (model_dir / "a.txt").write_bytes(b"good")
        (model_dir / "b.txt").write_bytes(b"CORRUPT")  # triggers repair
        (model_dir / pull.MARKER_NAME).write_text('{"dimensions": 640}', encoding="utf-8")

        def _down(req, timeout=60):
            raise RuntimeError("network down")

        monkeypatch.setattr(pull.urllib.request, "urlopen", _down)
        with pytest.raises(RuntimeError, match="network down"):
            pull.pull(model_dir)

        # Loader must see honest not-installed, not a stale complete marker.
        assert not (model_dir / pull.MARKER_NAME).exists()

    def test_interrupted_repair_resumes_and_publishes_marker(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pull = _load_pull_module()
        files = {"a.txt": b"good", "sub/b.txt": b"0123456789ABCDEF"}
        _install_files(pull, tmp_path, files)
        model_dir = tmp_path / "harrier"
        model_dir.mkdir(parents=True)
        (model_dir / "a.txt").write_bytes(b"good")
        tmp_dir = tmp_path / "harrier.tmp"
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "sub__b.txt.part").write_bytes(b"0123456789")  # partial

        def _urlopen(req, timeout=60):
            assert req.full_url.endswith("sub/b.txt")
            assert req.get_header("Range") == "bytes=10-"
            return _FakeResp(b"ABCDEF", status=206)

        monkeypatch.setattr(pull.urllib.request, "urlopen", _urlopen)
        pull.pull(model_dir)

        assert (model_dir / "sub" / "b.txt").read_bytes() == b"0123456789ABCDEF"
        marker = json.loads((model_dir / pull.MARKER_NAME).read_text(encoding="utf-8"))
        assert marker["dimensions"] == 640
        assert sorted(marker["files"]) == ["a.txt", "sub/b.txt"]
        assert not (model_dir / (pull.MARKER_NAME + ".tmp")).exists()  # atomic swap done

    def test_all_cached_publishes_valid_marker_without_network(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        pull = _load_pull_module()
        files = {"a.txt": b"good"}
        _install_files(pull, tmp_path, files)
        model_dir = tmp_path / "harrier"
        model_dir.mkdir(parents=True)
        (model_dir / "a.txt").write_bytes(b"good")

        def _no_network(req, timeout=60):
            raise AssertionError("must not fetch: all cached")

        monkeypatch.setattr(pull.urllib.request, "urlopen", _no_network)
        pull.pull(model_dir)

        marker = json.loads((model_dir / pull.MARKER_NAME).read_text(encoding="utf-8"))
        assert marker["dimensions"] == 640
        assert marker["files"] == ["a.txt"]
