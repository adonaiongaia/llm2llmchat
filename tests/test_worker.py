import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from llm2llmchat.app import MainWindow
from llm2llmchat.core import EndpointSettings
from llm2llmchat.worker import ModelRequestThread


class MockModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    next_stream_deltas = ["Hello", " ", "friend", "!"]
    seen: list[dict] = []
    seen_auth: list[str | None] = []
    use_non_streaming = False
    stream_delay = 0.0
    initial_gate: threading.Event | None = None
    streaming_started: threading.Event | None = None
    advertised_models = ["test-model", "test-model-b"]

    def log_message(self, _format, *_args):
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/v1/models":
            self._send_json(404, {"error": "route not found"})
            return
        self._send_json(200, {"data": [{"id": item} for item in self.advertised_models]})

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        type(self).seen.append(payload)
        type(self).seen_auth.append(self.headers.get("Authorization"))
        if self.use_non_streaming:
            self._send_json(200, {"choices": [{"message": {"content": "non-streaming answer"}}]})
            return
        wire = b"".join(
            (f'data: {json.dumps({"choices": [{"delta": {"content": delta}}]})}\r\n\r\n').encode()
            for delta in self.next_stream_deltas
        ) + b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Content-Length", str(len(wire)))
        self.end_headers()
        started = type(self).streaming_started
        if started is not None:
            started.set()
        gate = type(self).initial_gate
        if gate is not None:
            # Hold the response open with no bytes, like a local model ingesting a prompt.
            gate.wait(15)
        # Deliberately split protocol records across separate writes.
        try:
            for index in range(0, len(wire), 9):
                self.wfile.write(wire[index : index + 9])
                self.wfile.flush()
                if self.stream_delay:
                    time.sleep(self.stream_delay)
        except OSError:
            # The client cancelled and closed its end of the socket.
            self.close_connection = True


@pytest.fixture
def model_server():
    handler = type(f"TestHandler-{id(threading.current_thread())}", (MockModelHandler,), {})
    handler.seen = []
    handler.seen_auth = []
    handler.use_non_streaming = False
    handler.stream_delay = 0.0
    handler.initial_gate = None
    handler.streaming_started = None
    handler.next_stream_deltas = ["Hello", " ", "friend", "!"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_worker(worker, qt_app, timeout=5):
    completed = []
    failed = []
    chunks = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)
    worker.text_received.connect(chunks.append)
    worker.start()
    deadline = time.monotonic() + timeout
    while worker.isRunning() and time.monotonic() < deadline:
        qt_app.processEvents()
        worker.wait(10)
    qt_app.processEvents()
    assert not worker.isRunning(), "model worker did not finish before timeout"
    return completed, failed, chunks


def wait_until(qt_app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)
    qt_app.processEvents()
    return predicate()


def test_fetch_models_over_local_http_server(model_server, qt_app):
    base, _ = model_server
    worker = ModelRequestThread(EndpointSettings("Atlas", base, "configured"), list_models=True)
    completed, failed, _chunks = run_worker(worker, qt_app)
    assert failed == []
    assert completed == ["test-model\ntest-model-b"]


def test_worker_streams_fragmented_server_events_and_sends_bearer_token(model_server, qt_app):
    base, handler = model_server
    endpoint = EndpointSettings("Atlas", base, "exact-id", "secret-token")
    messages = [{"role": "user", "content": "open prompt"}]
    worker = ModelRequestThread(endpoint, messages)
    completed, failed, chunks = run_worker(worker, qt_app)
    assert failed == []
    assert completed == ["Hello friend!"]
    assert "".join(chunks) == "Hello friend!"
    assert handler.seen == [{"model": "exact-id", "messages": messages, "stream": True, "temperature": 0.7}]
    assert handler.seen_auth == ["Bearer secret-token"]
    assert "secret-token" not in repr(handler.seen[0])


def test_worker_waits_through_a_long_silent_gap_before_the_first_token(model_server, qt_app):
    """A slow local model can be silent for a while; that must not abort the request."""
    base, handler = model_server
    gate = threading.Event()
    handler.initial_gate = gate
    worker = ModelRequestThread(EndpointSettings("Atlas", base, "model-id"), [{"role": "user", "content": "hi"}])
    completed, failed, chunks = [], [], []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)
    worker.text_received.connect(chunks.append)
    worker.start()
    try:
        # Longer than a second of silence: previously the socket polling timeout latched
        # and the next read raised "cannot read from timed out object".
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        assert worker.isRunning(), "worker gave up while the model was still thinking"
    finally:
        gate.set()
    deadline = time.monotonic() + 5
    while worker.isRunning() and time.monotonic() < deadline:
        qt_app.processEvents()
        worker.wait(10)
    qt_app.processEvents()
    assert not worker.isRunning()
    assert failed == []
    assert completed == ["Hello friend!"]
    assert "".join(chunks) == "Hello friend!"


def test_cancel_interrupts_a_blocked_stream_read_promptly(model_server, qt_app):
    base, handler = model_server
    handler.initial_gate = threading.Event()
    handler.streaming_started = threading.Event()
    worker = ModelRequestThread(EndpointSettings("Atlas", base, "model-id"), [{"role": "user", "content": "hi"}])
    completed, failed, chunks = [], [], []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)
    worker.text_received.connect(chunks.append)
    worker.start()
    try:
        assert handler.streaming_started.wait(5), "server never started the streamed response"
        # The server is now holding the socket open without sending bytes.
        time.sleep(0.05)
        cancelled_at = time.monotonic()
        worker.cancel()
        deadline = cancelled_at + 2
        while worker.isRunning() and time.monotonic() < deadline:
            qt_app.processEvents()
            worker.wait(10)
        qt_app.processEvents()
        assert not worker.isRunning(), "cancel did not interrupt the blocked read"
        assert time.monotonic() - cancelled_at < 2
        assert completed == []
        assert failed == []
        assert chunks == []
    finally:
        handler.initial_gate.set()


def test_worker_supports_non_streaming_compat_fallback(model_server, qt_app):
    base, handler = model_server
    handler.use_non_streaming = True
    worker = ModelRequestThread(EndpointSettings("Birch", base, "model-id"), [{"role": "user", "content": "hi"}])
    completed, failed, chunks = run_worker(worker, qt_app)
    assert failed == []
    assert completed == ["non-streaming answer"]
    assert chunks == ["non-streaming answer"]


def test_desktop_gui_runs_two_alternating_streamed_turns(model_server, qt_app, tmp_path):
    base, handler = model_server
    qt_app.setOrganizationName("LLM2LLMChat-Tests")
    qt_app.setApplicationName("Temporary-GUI-Test")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    window = MainWindow()
    try:
        window.endpoints[0].name.setText("Atlas")
        window.endpoints[0].url.setText(base)
        window.endpoints[0].system_prompt.setPlainText("Answer in concise bullet points.")
        window.endpoints[0].temperature.setValue(35)
        window.endpoints[1].name.setText("Birch")
        window.endpoints[1].url.setText(base)
        window.endpoints[1].system_prompt.setPlainText("Respond like a curious botanist.")
        window.endpoints[1].temperature.setValue(145)
        for index in range(2):
            editor = window.endpoints[index]
            editor.check.click()
            assert editor in window._probe_workers
            deadline = time.monotonic() + 3
            while editor in window._probe_workers and time.monotonic() < deadline:
                qt_app.processEvents()
                time.sleep(0.003)
            qt_app.processEvents()
            assert editor not in window._probe_workers
            assert [editor.model.itemText(item) for item in range(editor.model.count())] == [
                "test-model",
                "test-model-b",
            ]
        window.endpoints[0].model.setCurrentText("test-model")
        window.endpoints[1].model.setCurrentText("test-model-b")
        window.max_turns.setValue(3)
        window.request_timeout.setValue(120)
        window.history_limit.setValue(5_000)
        window.prompt.setPlainText("Talk about gardens.")
        window.show()
        window.run_button.click()
        assert window._worker is not None
        assert window._worker.timeout == 120
        assert window._session is not None
        assert window._session.max_history_characters == 5_000
        assert not window.request_timeout.isEnabled()
        assert not window.history_limit.isEnabled()
        assert not window.run_button.isEnabled()
        assert window.stop_button.isEnabled()
        deadline = time.monotonic() + 5
        while window._session is not None and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.005)
        qt_app.processEvents()
        assert window._session is None, "GUI did not finish its configured three-turn conversation"
        assert window.run_button.isEnabled()
        assert not window.stop_button.isEnabled()
        transcript = window.transcript.toPlainText()
        assert "YOU\nTalk about gardens." in transcript
        assert "Atlas · test-model\nHello friend!" in transcript
        assert "Birch · test-model-b\nHello friend!" in transcript
        assert len(handler.seen) == 3
        assert [payload["model"] for payload in handler.seen] == ["test-model", "test-model-b", "test-model"]
        assert [payload["temperature"] for payload in handler.seen] == [0.35, 1.45, 0.35]
        assert [payload["messages"][0] for payload in handler.seen] == [
            {"role": "system", "content": "Answer in concise bullet points."},
            {"role": "system", "content": "Respond like a curious botanist."},
            {"role": "system", "content": "Answer in concise bullet points."},
        ]
        second_history = handler.seen[1]["messages"]
        assert {"role": "assistant", "content": "Hello friend!", "name": "atlas"} in second_history
        # The third request replays each earlier turn once, with unique participant IDs.
        third_history = handler.seen[2]["messages"]
        assistant_messages = [message for message in third_history if message["role"] == "assistant"]
        assert [(message["name"], message["content"]) for message in assistant_messages] == [
            ("atlas", "Hello friend!"),
            ("birch", "Hello friend!"),
        ]
        assert len(window._history_turns) == 3
        assert window._player.state.value == "idle"
        assert window.export_button.isEnabled()
        reopened = MainWindow()
        try:
            assert len(reopened.endpoints) == 2
            assert [editor.order.value() for editor in reopened.endpoints] == [1, 2]
            assert reopened.endpoints[0].name.text() == "Atlas"
            assert reopened.endpoints[0].url.text() == f"{base}/v1"
            assert reopened.endpoints[0].model.currentText() == "test-model"
            assert reopened.endpoints[1].name.text() == "Birch"
            assert reopened.endpoints[1].model.currentText() == "test-model-b"
            assert reopened.endpoints[0].system_prompt.toPlainText() == "Answer in concise bullet points."
            assert reopened.endpoints[1].system_prompt.toPlainText() == "Respond like a curious botanist."
            assert reopened.endpoints[0].temperature.value() == 35
            assert reopened.endpoints[1].temperature.value() == 145
            assert reopened.max_turns.value() == 3
            assert reopened.request_timeout.value() == 120
            assert reopened.history_limit.value() == 5_000
            assert reopened.request_timeout.isEnabled()
            assert reopened.history_limit.isEnabled()
        finally:
            reopened.close()
            reopened.deleteLater()
            qt_app.processEvents()
    finally:
        window.close()
        qt_app.processEvents()
        window.deleteLater()
        qt_app.processEvents()


def test_desktop_gui_adds_participant_columns_and_runs_in_numeric_order(model_server, qt_app, tmp_path):
    base, handler = model_server
    qt_app.setOrganizationName("LLM2LLMChat-Tests")
    qt_app.setApplicationName("Temporary-Order-Test")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    window = MainWindow()
    try:
        window.add_participant_button.click()
        assert len(window.endpoints) == 3
        names = ("Atlas", "Birch", "Cedar")
        models = ("model-a", "model-b", "model-c")
        orders = (3, 1, 2)
        for index, (name, model, order) in enumerate(zip(names, models, orders)):
            editor = window.endpoints[index]
            editor.name.setText(name)
            editor.url.setText(base)
            editor.model.setCurrentText(model)
            editor.order.setValue(order)
        assert [editor.order.value() for editor in window.endpoints] == [3, 1, 2]
        window.max_turns.setValue(3)
        window.prompt.setPlainText("Ordered arena prompt.")
        window.show()
        window.run_button.click()
        assert wait_until(qt_app, lambda: window._session is None, timeout=8)
        assert [payload["model"] for payload in handler.seen] == ["model-b", "model-c", "model-a"]
        transcript = window.transcript.toPlainText()
        positions = [transcript.index(f"{name} · {model}") for name, model in zip(
            ("Birch", "Cedar", "Atlas"), ("model-b", "model-c", "model-a")
        )]
        assert positions == sorted(positions)
        assert len(window._history_turns) == 3
        assert [turn.speaker for turn in window._history_turns] == ["Birch", "Cedar", "Atlas"]
        assert [turn.speaker_id for turn in window._history_turns] == ["birch", "cedar", "atlas"]
        markdown_path = window._write_export(tmp_path / "arena.md")
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "### Birch (`birch`)" in markdown
        assert "### Cedar (`cedar`)" in markdown
        assert markdown.index("### Birch") < markdown.index("### Cedar") < markdown.index("### Atlas")
        text_path = window._write_export(tmp_path / "arena.txt")
        plain_text = text_path.read_text(encoding="utf-8")
        assert "Atlas [atlas]:" in plain_text
        assert "# Model Arena" not in plain_text
        reopened = MainWindow()
        try:
            assert len(reopened.endpoints) == 3
            assert [editor.name.text() for editor in reopened.endpoints] == list(names)
            assert [editor.order.value() for editor in reopened.endpoints] == list(orders)
            assert [editor.model.currentText() for editor in reopened.endpoints] == list(models)
        finally:
            reopened.close()
            reopened.deleteLater()
            qt_app.processEvents()
    finally:
        window.close()
        qt_app.processEvents()
        window.deleteLater()
        qt_app.processEvents()


def test_desktop_gui_pauses_finishes_active_turn_injects_and_resumes(model_server, qt_app, tmp_path):
    base, handler = model_server
    handler.stream_delay = 0.02
    handler.next_stream_deltas = ["alpha ", "beta ", "gamma ", "delta"]
    qt_app.setOrganizationName("LLM2LLMChat-Tests")
    qt_app.setApplicationName("Temporary-Pause-Test")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    window = MainWindow()
    try:
        for index, model in enumerate(("model-a", "model-b")):
            editor = window.endpoints[index]
            editor.name.setText(("Atlas", "Birch")[index])
            editor.url.setText(base)
            editor.model.setCurrentText(model)
        window.max_turns.setValue(4)
        window.prompt.setPlainText("Pause and inject prompt.")
        window.show()
        window.run_button.click()
        assert wait_until(qt_app, lambda: "alpha" in window.transcript.toPlainText(), timeout=4)
        assert window.pause_button.isEnabled()
        window.pause_button.click()
        assert window._player.state.value == "pausing"
        assert not window.interject_edit.isEnabled()
        assert wait_until(qt_app, lambda: window._player.is_paused, timeout=6)
        assert len(handler.seen) == 1
        assert "alpha beta gamma delta" in window.transcript.toPlainText()
        assert len(window._history_turns) == 1
        assert window.pause_button.isEnabled() is False
        assert window.resume_button.isEnabled()
        assert window.interject_edit.isEnabled()
        assert window.inject_button.isEnabled()
        # Frozen means no further request is launched while paused.
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.005)
        assert len(handler.seen) == 1
        window.interject_edit.setPlainText("Please discuss the moon.")
        window.inject_button.click()
        assert window._session is not None
        injected = window._session.turns[-1]
        assert injected.role == "user"
        assert injected.speaker_id == "user"
        assert injected.content == "Please discuss the moon."
        assert "YOU · Interjection\nPlease discuss the moon." in window.transcript.toPlainText()
        window.resume_button.click()
        assert wait_until(qt_app, lambda: len(handler.seen) >= 2, timeout=5)
        assert window._player.state.value == "running"
        assert handler.seen[1]["model"] == "model-b"
        assert {"role": "user", "content": "Please discuss the moon.", "name": "user"} in handler.seen[1]["messages"]
        assert [
            message["name"] for message in handler.seen[1]["messages"] if message["role"] == "assistant"
        ] == ["atlas"]
        window.stop_button.click()
        assert wait_until(qt_app, lambda: window._session is None and window._worker is None, timeout=5)
        assert window.run_button.isEnabled()
        assert not window.stop_button.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()
        window.deleteLater()
        qt_app.processEvents()


def test_desktop_gui_stop_cancels_active_turn_and_restores_controls(model_server, qt_app, tmp_path):
    base, handler = model_server
    handler.stream_delay = 0.01
    handler.next_stream_deltas = ["first", " second", " third", " fourth"]
    qt_app.setOrganizationName("LLM2LLMChat-Tests")
    qt_app.setApplicationName("Temporary-Cancel-Test")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    window = MainWindow()
    try:
        for index, model in enumerate(("test-model", "test-model-b")):
            window.endpoints[index].name.setText(("Atlas", "Birch")[index])
            window.endpoints[index].url.setText(base)
            window.endpoints[index].model.setCurrentText(model)
        window.prompt.setPlainText("Talk about gardens.")
        window.show()
        window.run_button.click()
        deadline = time.monotonic() + 2
        while "first" not in window.transcript.toPlainText() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.003)
        assert "first" in window.transcript.toPlainText()
        window.stop_button.click()
        assert not window.run_button.isEnabled()
        deadline = time.monotonic() + 3
        while (window._worker is not None or window.run_button.isEnabled() is False) and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.005)
        qt_app.processEvents()
        assert window._session is None
        assert window._worker is None
        assert window.run_button.isEnabled()
        assert not window.stop_button.isEnabled()
        assert len(handler.seen) == 1
        assert "Conversation stopped." in window.status.currentMessage()
    finally:
        window.close()
        qt_app.processEvents()
        window.deleteLater()
        qt_app.processEvents()


def test_desktop_gui_close_button_removes_individual_participant(qt_app, tmp_path):
    qt_app.setOrganizationName("LLM2LLMChat-Tests")
    qt_app.setApplicationName("Temporary-Close-Test")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    window = MainWindow()
    try:
        assert len(window.endpoints) == 2
        assert all(not editor.close.isEnabled() for editor in window.endpoints)
        window.add_participant_button.click()
        assert len(window.endpoints) == 3
        for editor, name in zip(window.endpoints, ("Atlas", "Birch", "Cedar")):
            editor.name.setText(name)
        assert all(editor.close.isEnabled() for editor in window.endpoints)
        window.endpoints[1].close.click()
        assert [editor.name.text() for editor in window.endpoints] == ["Atlas", "Cedar"]
        assert [editor.group.title() for editor in window.endpoints] == ["● PARTICIPANT 01", "● PARTICIPANT 02"]
        assert all(not editor.close.isEnabled() for editor in window.endpoints)
    finally:
        window.close()
        qt_app.processEvents()
        window.deleteLater()
        qt_app.processEvents()


def test_cancel_terminates_streaming_worker_and_suppresses_success_signal(model_server, qt_app):
    base, handler = model_server
    handler.next_stream_deltas = ["one", "two", "three"]
    worker = ModelRequestThread(EndpointSettings("Atlas", base, "model-id"), [{"role": "user", "content": "hi"}])
    completed, failed, chunks = [], [], []
    worker.completed.connect(completed.append)
    worker.failed.connect(failed.append)
    worker.text_received.connect(chunks.append)
    worker.start()
    # The local response arrives immediately; wait briefly and cancel if it is still streaming.
    deadline = time.monotonic() + 0.05
    while worker.isRunning() and time.monotonic() < deadline:
        qt_app.processEvents()
        worker.wait(5)
    if worker.isRunning():
        worker.cancel()
    worker.wait(3000)
    qt_app.processEvents()
    assert not worker.isRunning()
    assert failed == []
    assert completed == ["onetwothree"]
