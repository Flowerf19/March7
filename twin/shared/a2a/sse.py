"""Incremental SSE parsing for A2A streams with completion contract."""
from __future__ import annotations

import codecs
import json


class A2AStreamError(RuntimeError):
    """SSE stream truncated or errored; partial output never counts as success."""


class SseParser:
    """Incremental UTF-8 SSE parser enforcing a single complete marker.

    Feed raw bytes; returns message dicts for normal events. Raises
    A2AStreamError on error events, invalid UTF-8/JSON, malformed or
    duplicate complete, and data after complete. Call finish() at EOF to
    require the marker, exact count match, and no leftover bytes.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buf = ""
        self._normal = 0
        self._complete: int | None = None

    def feed(self, chunk: bytes) -> list[dict]:
        try:
            self._buf += self._decoder.decode(chunk, False)
        except UnicodeDecodeError as exc:
            raise A2AStreamError(
                f"A2A stream truncated: invalid utf-8 ({exc})"
            ) from exc
        out: list[dict] = []
        while "\n\n" in self._buf:
            event_str, self._buf = self._buf.split("\n\n", 1)
            event_name: str | None = None
            data_lines: list[str] = []
            for raw_line in event_str.split("\n"):
                line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    payload = line[5:]
                    if payload.startswith(" "):
                        payload = payload[1:]
                    data_lines.append(payload)
            if event_name == "error":
                reason = "stream_truncated"
                if data_lines:
                    try:
                        err = json.loads("\n".join(data_lines))
                        if isinstance(err, dict):
                            reason = str(
                                err.get("reason") or err.get("error") or reason
                            )
                    except (json.JSONDecodeError, AttributeError):
                        pass
                raise A2AStreamError(f"A2A stream truncated: {reason}")
            if event_name == "complete":
                if self._complete is not None:
                    raise A2AStreamError(
                        "A2A stream truncated: duplicate complete"
                    )
                if not data_lines:
                    raise A2AStreamError(
                        "A2A stream truncated: malformed complete"
                    )
                try:
                    obj = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as exc:
                    raise A2AStreamError(
                        f"A2A stream truncated: malformed complete ({exc})"
                    ) from exc
                count = obj.get("message_count") if isinstance(obj, dict) else None
                if type(count) is not int or count < 0:
                    raise A2AStreamError(
                        "A2A stream truncated: malformed complete count"
                    )
                self._complete = count
                continue
            if not data_lines:
                continue
            if self._complete is not None:
                raise A2AStreamError(
                    "A2A stream truncated: data after complete"
                )
            try:
                obj = json.loads("\n".join(data_lines))
            except json.JSONDecodeError as exc:
                raise A2AStreamError(
                    f"A2A stream truncated: invalid message json ({exc})"
                ) from exc
            if not isinstance(obj, dict):
                raise A2AStreamError(
                    "A2A stream truncated: invalid message payload"
                )
            self._normal += 1
            out.append(obj)
        return out

    def finish(self) -> None:
        try:
            self._buf += self._decoder.decode(b"", True)
        except UnicodeDecodeError as exc:
            raise A2AStreamError(
                f"A2A stream truncated: invalid utf-8 ({exc})"
            ) from exc
        if self._buf.strip() != "":
            raise A2AStreamError(
                "A2A stream truncated: incomplete event at EOF"
            )
        if self._complete is None:
            raise A2AStreamError("A2A stream truncated: missing complete")
        if self._complete != self._normal:
            raise A2AStreamError(
                f"A2A stream truncated: complete count "
                f"{self._complete} != messages {self._normal}"
            )
