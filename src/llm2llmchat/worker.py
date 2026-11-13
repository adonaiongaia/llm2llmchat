"""Cancellable worker for streamed OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal

from .core import (
    EndpointSettings,
    ProtocolError,
    SSEParser,
    build_chat_payload,
    chat_completions_url,
    models_url,
    parse_non_streaming_response,
    parse_stream_event,
)

# Local models can take minutes to ingest a growing prompt before the first token arrives,
# so this is a per-socket-operation safety net rather than a deadline for the whole reply.
DEFAULT_TIMEOUT_SECONDS = 600.0


def _response_socket(response):
    """Return the socket behind an HTTPResponse so a reader can be woken on cancel."""
    try:
        return response.fp.raw._sock
    except AttributeError:
        return None


class ModelRequestThread(QThread):
    text_received = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        endpoint: EndpointSettings,
        messages: list[dict[str, str]] | None = None,
        *,
        list_models: bool = False,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.endpoint = endpoint
        self.messages = messages or []
        self.list_models = list_models
        self.timeout = timeout
        self._cancel_event = threading.Event()
        self._connection_lock = threading.Lock()
        self._connection = None
        self._socket = None

    def cancel(self) -> None:
        # The worker still owns and closes the response object; shutting the socket down
        # here only wakes a read that is blocked waiting for the model's next token. Unlike
        # closing the descriptor from the GUI thread, this cannot race the worker's reader.
        self._cancel_event.set()
        with self._connection_lock:
            sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                # Already finished or never connected: the cancel flag is enough.
                pass

    def run(self) -> None:
        try:
            result = self._list_models() if self.list_models else self._complete_chat()
            if not self._cancel_event.is_set():
                self.completed.emit(result)
        except Exception as exc:
            if not self._cancel_event.is_set():
                self.failed.emit(str(exc))
        finally:
            with self._connection_lock:
                self._socket = None
                self._connection = None

    def _request(self, url: str, body: bytes | None = None) -> None:
        headers = {"Accept": "application/json"}
        if self.endpoint.api_key.strip():
            headers["Authorization"] = f"Bearer {self.endpoint.api_key.strip()}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
        try:
            connection = urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            # Display a bounded, credential-free server error, never request headers or URLs with tokens.
            try:
                detail = exc.read(2048).decode("utf-8", "replace").strip()
            except OSError:
                detail = ""
            if self.endpoint.api_key:
                detail = detail.replace(self.endpoint.api_key, "[redacted]")
            if detail:
                try:
                    payload = json.loads(detail)
                    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
                        detail = str(payload["error"].get("message", detail))
                    elif isinstance(payload, dict) and isinstance(payload.get("error"), str):
                        detail = payload["error"]
                except (json.JSONDecodeError, ValueError):
                    pass
                detail = detail[:400]
            raise ProtocolError(f"Model server returned HTTP {exc.code}" + (f": {detail}" if detail else ".")) from None
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProtocolError(f"Could not reach model server: {reason}") from None
        with self._connection_lock:
            if self._cancel_event.is_set():
                connection.close()
                return
            self._connection = connection
            self._socket = _response_socket(connection)

    def _read_complete_body(self, connection, *, limit: int, oversized: str) -> bytes:
        """Read a non-streamed body, translating a stalled socket into a clear error."""
        try:
            body = connection.read(limit + 1)
        except TimeoutError:
            raise ProtocolError(
                f"The model server sent nothing for {self.timeout:g} seconds. "
                "Check that it is still running, or raise the request timeout."
            ) from None
        except OSError as exc:
            if self._cancel_event.is_set():
                return b""
            raise ProtocolError(f"Lost the connection to the model server: {exc}") from None
        if len(body) > limit:
            raise ProtocolError(oversized)
        return body

    def _list_models(self) -> str:
        self._request(models_url(self.endpoint.base_url))
        with self._connection_lock:
            connection = self._connection
        if connection is None:
            return ""
        with connection:
            if self._cancel_event.is_set():
                return ""
            raw = self._read_complete_body(
                connection,
                limit=2 * 1024 * 1024,
                oversized="The model server returned an unexpectedly large model list.",
            )
        if not raw and self._cancel_event.is_set():
            return ""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("The model server returned an invalid model list.") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ProtocolError("The model server's /models response did not include a data list.")
        ids = [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]
        if not ids:
            raise ProtocolError("The model server reported no available model IDs.")
        return "\n".join(ids)

    def _complete_chat(self) -> str:
        body = json.dumps(
            build_chat_payload(self.endpoint, self.messages),
            ensure_ascii=False,
        ).encode("utf-8")
        self._request(chat_completions_url(self.endpoint.base_url), body)
        with self._connection_lock:
            connection = self._connection
        if connection is None:
            return ""
        with connection:
            content_type = connection.headers.get("Content-Type", "").lower()
            if "text/event-stream" not in content_type:
                response = self._read_complete_body(
                    connection,
                    limit=32 * 1024 * 1024,
                    oversized="The model server's response exceeded the 32 MiB safety limit.",
                )
                if not response:
                    if self._cancel_event.is_set():
                        return ""
                    raise ProtocolError("The model server returned an empty response.")
                result = parse_non_streaming_response(response)
                if result:
                    self.text_received.emit(result)
                return result
            parser = SSEParser()
            parts: list[str] = []
            total_characters = 0
            # Block for up to the configured timeout instead of polling with a short
            # timeout: Python latches a timed-out socket, so a poll loop that continues
            # after socket.timeout makes every later read fail with
            # "cannot read from timed out object". A stop request wakes this read by
            # shutting the socket down (see cancel).
            while True:
                try:
                    chunk = connection.read1(4096)
                except TimeoutError:
                    if self._cancel_event.is_set():
                        break
                    raise ProtocolError(
                        f"The model server sent no data for {self.timeout:g} seconds, "
                        "so the request was abandoned. Check the server or raise the timeout."
                    ) from None
                except OSError as exc:
                    if self._cancel_event.is_set():
                        break
                    raise ProtocolError(f"Lost the connection to the model server: {exc}") from None
                for event in parser.feed(chunk, final=not chunk):
                    delta = parse_stream_event(event)
                    if delta:
                        total_characters += len(delta)
                        if total_characters > 32 * 1024 * 1024:
                            raise ProtocolError("The model server's streamed response exceeded the 32 MiB safety limit.")
                        parts.append(delta)
                        self.text_received.emit(delta)
                    if event == "[DONE]":
                        return "".join(parts)
                if not chunk:
                    break
            return "".join(parts)
