"""PySide6 desktop interface for Model Arena conversations."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

try:
    import keyring
    from keyring.errors import KeyringError
except ImportError:  # Optional: API keys still work for the current app session.
    keyring = None
    KeyringError = Exception

from PySide6.QtCore import QPointF, QSettings, QSize, Qt
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QStyle,
    QStyleOptionSpinBox,
    QVBoxLayout,
    QWidget,
)

from .core import (
    DEFAULT_MAX_HISTORY_CHARACTERS,
    DialogueSession,
    EndpointSettings,
    Participant,
    PlaybackController,
    PlaybackState,
    ProtocolError,
    normalize_base_url,
    render_transcript,
)
from .worker import DEFAULT_TIMEOUT_SECONDS, ModelRequestThread


APP_NAME = "LLM2LLMChat"
KEYRING_SERVICE = "com.freebuff.llm2llmchat"


def _secret_backend_available() -> bool:
    if keyring is None:
        return False
    try:
        return keyring.get_keyring().priority > 0
    except Exception:
        return False


def _load_secret(participant_id: str, legacy_slot: int) -> str:
    if not _secret_backend_available():
        return ""
    for account in (f"endpoint-{participant_id}", f"endpoint-{legacy_slot}"):
        try:
            value = keyring.get_password(KEYRING_SERVICE, account)
        except Exception:
            value = None
        if value:
            return value
    return ""


def _forget_secret(participant_id: str) -> None:
    if not _secret_backend_available():
        return
    try:
        keyring.delete_password(KEYRING_SERVICE, f"endpoint-{participant_id}")
    except Exception:
        pass


def hud_font(
    point_size: float,
    weight: QFont.Weight = QFont.Weight.Medium,
    spacing: float = 1.0,
) -> QFont:
    """Return the monospaced terminal font used for the HUD headings."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    label.setFont(hud_font(9.0, QFont.Weight.DemiBold, 1.4))
    return label


def participant_title(index: int) -> str:
    return f"● PARTICIPANT {index + 1:02d}"


class HudSpinBox(QSpinBox):
    """Spin box whose stepper arrows are drawn in the HUD palette."""

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#9fe6fb") if self.isEnabled() else QColor("#3d5a68"))
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        for sub_control, direction in (
            (QStyle.SubControl.SC_SpinBoxUp, -1.0),
            (QStyle.SubControl.SC_SpinBoxDown, 1.0),
        ):
            centre = self.style().subControlRect(
                QStyle.ComplexControl.CC_SpinBox, option, sub_control, self
            ).center()
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(centre.x() - 3.8, centre.y() - direction * 2.2),
                        QPointF(centre.x() + 3.8, centre.y() - direction * 2.2),
                        QPointF(centre.x(), centre.y() + direction * 2.2),
                    ]
                )
            )
        painter.end()


def _dark_palette() -> QPalette:
    """Dark palette so dialogs and native-drawn details match the window chrome."""
    palette = QPalette()
    for role, value in (
        (QPalette.ColorRole.Window, "#050d18"),
        (QPalette.ColorRole.WindowText, "#d7ecff"),
        (QPalette.ColorRole.Base, "#04101b"),
        (QPalette.ColorRole.AlternateBase, "#0a1e30"),
        (QPalette.ColorRole.ToolTipBase, "#07182a"),
        (QPalette.ColorRole.ToolTipText, "#d7ecff"),
        (QPalette.ColorRole.Text, "#d7ecff"),
        (QPalette.ColorRole.Button, "#0a1e30"),
        (QPalette.ColorRole.ButtonText, "#d7ecff"),
        (QPalette.ColorRole.BrightText, "#ffffff"),
        (QPalette.ColorRole.Link, "#4fd7f5"),
        (QPalette.ColorRole.Highlight, "#156b8c"),
        (QPalette.ColorRole.HighlightedText, "#f2feff"),
        (QPalette.ColorRole.PlaceholderText, "#5f7f94"),
    ):
        palette.setColor(role, QColor(value))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#5d7a8d"))
    return palette


class ArenaBackdrop(QWidget):
    """Painted backdrop: deep-space gradient, circuit traces and neural nodes.

    The artwork is rendered once per size into a pixmap, so resizing costs nothing
    during a conversation and the gradient never competes with the transcript text.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("root")
        self._decoration: QPixmap | None = None

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._decoration = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._decoration is None or self._decoration.size() != self._pixel_size():
            self._decoration = self._render()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._decoration)

    def _pixel_size(self) -> QSize:
        ratio = self.devicePixelRatioF()
        return QSize(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))

    def _render(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(self._pixel_size())
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QColor("#04070f"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            painter.end()
            return pixmap

        base = QLinearGradient(0, 0, 0, height)
        base.setColorAt(0.0, QColor("#04070f"))
        base.setColorAt(0.42, QColor("#071a2b"))
        base.setColorAt(0.74, QColor("#050f1c"))
        base.setColorAt(1.0, QColor("#02050a"))
        painter.fillRect(0, 0, width, height, base)

        for cx, cy, radius, color, alpha in (
            (width * 0.5, height * 0.06, max(width, height) * 0.5, QColor(255, 150, 60), 58),
            (width * 0.02, height * 0.95, max(width, height) * 0.4, QColor(53, 224, 255), 34),
            (width * 0.99, height * 0.8, max(width, height) * 0.34, QColor(60, 130, 255), 30),
        ):
            glow = QRadialGradient(QPointF(cx, cy), radius)
            glow.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), alpha))
            glow.setColorAt(0.45, QColor(color.red(), color.green(), color.blue(), alpha // 3))
            glow.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
            painter.fillRect(0, 0, width, height, glow)

        painter.setPen(QPen(QColor(38, 120, 158, 26), 1))
        for x in range(52, width, 52):
            painter.drawLine(x, 0, x, height)
        for y in range(52, height, 52):
            painter.drawLine(0, y, width, y)

        rng = random.Random(11)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for _ in range(22):
            x = float(rng.randrange(0, max(1, width)))
            y = float(rng.randrange(int(height * 0.62), max(int(height * 0.62) + 2, height)))
            points = [QPointF(x, y)]
            horizontal = rng.random() < 0.5
            for _ in range(rng.randint(3, 5)):
                span = rng.randrange(34, 150)
                if horizontal:
                    x += span if rng.random() < 0.5 else -span
                else:
                    y += span if rng.random() < 0.5 else -span
                    y = max(y, height * 0.55)
                points.append(QPointF(x, y))
                horizontal = not horizontal
            painter.setPen(QPen(QColor(34, 130, 178, 34), 1.1))
            for start, end in zip(points, points[1:]):
                painter.drawLine(start, end)
            painter.setPen(QPen(QColor(94, 232, 255, 90), 1.0))
            painter.drawEllipse(points[-1], 1.8, 1.8)

        for fx, fy, tint, offsets in (
            (0.88, 0.1, QColor(90, 225, 255), ((0, 0), (-58, 44), (34, 72), (-96, 96), (72, -38), (-24, -58))),
            (0.04, 0.7, QColor(190, 120, 255), ((0, 0), (62, 38), (-44, 66), (98, 84), (18, -52))),
            (0.2, 0.86, QColor(255, 154, 60), ((0, 0), (54, -30), (-30, 48), (86, 40))),
        ):
            nodes = [QPointF(width * fx + dx, height * fy + dy) for dx, dy in offsets]
            painter.setPen(QPen(QColor(tint.red(), tint.green(), tint.blue(), 46), 1.1))
            for index, start in enumerate(nodes):
                for end in nodes[index + 1 :]:
                    if abs(start.x() - end.x()) + abs(start.y() - end.y()) < 150:
                        painter.drawLine(start, end)
            painter.setPen(Qt.PenStyle.NoPen)
            for node in nodes:
                painter.setBrush(QColor(tint.red(), tint.green(), tint.blue(), 26))
                painter.drawEllipse(node, 5.0, 5.0)
                painter.setBrush(QColor(tint.red(), tint.green(), tint.blue(), 175))
                painter.drawEllipse(node, 2.2, 2.2)

        painter.end()
        return pixmap


class ParticipantBoard(QWidget):
    """Grid board that reports only the width its participant columns need.

    QScrollArea sizes a resizable widget from its sizeHint; a plain QWidget with a
    form-heavy grid reports a hint wider than the viewport even when every column
    fits, leaving an empty horizontal scrollbar under the arena.
    """

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        hint = self.minimumSizeHint()
        return QSize(hint.width(), hint.height())


@dataclass(eq=False)
class EndpointEditor:
    participant_id: str
    name: QLineEdit
    url: QLineEdit
    model: QComboBox
    api_key: QLineEdit
    check: QPushButton
    system_prompt: QPlainTextEdit
    temperature: QSlider
    temperature_value: QLabel
    order: QSpinBox
    remove: QPushButton
    close: QPushButton
    group: QGroupBox | None = field(default=None, repr=False)

    def settings(self, slot: int, *, require_model: bool = True) -> EndpointSettings:
        name = self.name.text().strip() or f"Model {slot + 1}"
        model = self.model.currentText().strip()
        if require_model and not model:
            raise ValueError(f"Enter a model ID for {name}.")
        url = normalize_base_url(self.url.text())
        self.url.setText(url)
        return EndpointSettings(
            name,
            url,
            model,
            self.api_key.text(),
            self.system_prompt.toPlainText(),
            self.temperature.value() / 100.0,
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings_store = QSettings()
        self.setWindowTitle("LLM2LLMChat — Model Arena")
        self.resize(1180, 980)
        self.setMinimumSize(980, 940)
        self._probe_workers: dict[EndpointEditor, ModelRequestThread] = {}
        self.endpoints: list[EndpointEditor] = []
        self._worker: ModelRequestThread | None = None
        self._session: DialogueSession | None = None
        self._active_endpoints: list[EndpointSettings] = []
        self._history_turns = []
        self._starter_prompt = ""
        self._worker_result: str | None = None
        self._worker_failure: str | None = None
        self._player = PlaybackController()
        self._closing = False
        self._stop_pending = False

        root = ArenaBackdrop(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 20, 22, 16)
        layout.setSpacing(8)

        eyebrow = QLabel("MULTI-AGENT PROTOCOL // LOCAL INFERENCE NODES")
        eyebrow.setObjectName("eyebrow")
        eyebrow.setFont(hud_font(8.4, QFont.Weight.DemiBold, 1.9))
        title = QLabel("MODEL ARENA")
        title.setObjectName("title")
        title_font = QFont("DejaVu Sans")
        title_font.setPointSize(24)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.2)
        title.setFont(title_font)
        subtitle = QLabel(
            "Add a column for every model, number their speaking order, and let them continue one shared transcript."
        )
        subtitle.setObjectName("subtitle")
        rule = QFrame()
        rule.setObjectName("accentRule")
        rule.setFrameShape(QFrame.Shape.NoFrame)
        rule.setFixedHeight(2)
        rule.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(rule)

        self.notice = QLabel(
            "SECURITY // only connect to model servers you trust. Unauthenticated HTTP server endpoints "
            "should stay behind a trusted LAN/VPN and must never be exposed to the public internet. "
            "Ollama's local API does not authenticate requests by default."
        )
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)

        participant_heading = QHBoxLayout()
        participant_label = section_label("// PARTICIPANTS")
        self.add_participant_button = QPushButton("+ Add Participant")
        self.add_participant_button.setObjectName("quietButton")
        self.add_participant_button.setToolTip("Add another model configuration column to the arena.")
        self.add_participant_button.clicked.connect(self._add_participant)
        participant_heading.addWidget(participant_label)
        participant_heading.addStretch(1)
        participant_heading.addWidget(self.add_participant_button)
        layout.addLayout(participant_heading)

        self.participant_container = ParticipantBoard()
        self.participant_layout = QGridLayout(self.participant_container)
        self.participant_layout.setContentsMargins(0, 0, 0, 0)
        self.participant_layout.setHorizontalSpacing(12)
        self.participant_layout.setVerticalSpacing(12)
        self.participant_scroll = QScrollArea()
        self.participant_scroll.setObjectName("participantScroll")
        self.participant_scroll.setWidgetResizable(True)
        self.participant_scroll.setWidget(self.participant_container)
        self.participant_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.participant_scroll.setMinimumHeight(348)
        self.participant_scroll.setMaximumHeight(360)
        self.participant_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.participant_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.participant_scroll)

        conversation = QHBoxLayout()
        conversation.addWidget(QLabel("Max turns"))
        self.max_turns = HudSpinBox()
        self.max_turns.setRange(1, 1000)
        self.max_turns.setValue(6)
        self.max_turns.setToolTip("The total number of model responses before the dialogue stops automatically.")
        conversation.addWidget(self.max_turns)
        conversation.addSpacing(12)
        conversation.addWidget(QLabel("Request timeout"))
        self.request_timeout = HudSpinBox()
        self.request_timeout.setRange(30, 3600)
        self.request_timeout.setSingleStep(30)
        self.request_timeout.setValue(int(DEFAULT_TIMEOUT_SECONDS))
        self.request_timeout.setSuffix(" s")
        self.request_timeout.setToolTip(
            "How long one network read may wait for the model server before the request is abandoned. "
            "Local models can stay silent for minutes while they ingest a growing prompt."
        )
        conversation.addWidget(self.request_timeout)
        conversation.addSpacing(12)
        conversation.addWidget(QLabel("History"))
        self.history_limit = HudSpinBox()
        self.history_limit.setRange(1_000, 200_000)
        self.history_limit.setSingleStep(1_000)
        self.history_limit.setValue(DEFAULT_MAX_HISTORY_CHARACTERS)
        self.history_limit.setSuffix(" chars")
        self.history_limit.setToolTip(
            "How much earlier conversation is replayed with each request (about 4 characters per token). "
            "When the limit is reached, the oldest turns are dropped first."
        )
        conversation.addWidget(self.history_limit)
        conversation.addStretch(1)
        self.run_button = QPushButton("Start conversation")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._start_conversation)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.setToolTip("Let the active model finish, then freeze the conversation loop.")
        self.pause_button.clicked.connect(self._pause_conversation)
        self.resume_button = QPushButton("Resume")
        self.resume_button.setEnabled(False)
        self.resume_button.setToolTip("Continue from the next queued participant.")
        self.resume_button.clicked.connect(self._resume_conversation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_conversation)
        for button in (self.run_button, self.pause_button, self.resume_button, self.stop_button):
            conversation.addWidget(button)
        layout.addLayout(conversation)

        transcript_heading = QHBoxLayout()
        transcript_label = section_label("// DIALOGUE STREAM")
        self.export_button = QPushButton("Export")
        self.export_button.setObjectName("quietButton")
        self.export_button.setToolTip("Export the chat history as a Markdown or plain-text file.")
        self.export_button.clicked.connect(self._export_transcript)
        self.clear_button = QPushButton("Clear transcript")
        self.clear_button.setObjectName("quietButton")
        self.clear_button.clicked.connect(self._clear_transcript)
        transcript_heading.addWidget(transcript_label)
        transcript_heading.addStretch(1)
        transcript_heading.addWidget(self.export_button)
        transcript_heading.addWidget(self.clear_button)
        layout.addLayout(transcript_heading)

        self.transcript = QPlainTextEdit()
        self.transcript.setObjectName("transcript")
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("Start a conversation to see the arena respond here…")
        self.transcript.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.transcript.setFont(hud_font(9.8, QFont.Weight.Normal, 0.4))
        layout.addWidget(self.transcript, 1)

        interject_label = section_label("// INTERJECT CONVERSATION")
        layout.addWidget(interject_label)
        interject_row = QHBoxLayout()
        self.interject_edit = QPlainTextEdit()
        self.interject_edit.setObjectName("interjectEdit")
        self.interject_edit.setMaximumHeight(54)
        self.interject_edit.setPlaceholderText("Pause the arena, type a message for every model, then inject it…")
        self.interject_edit.setMaximumHeight(44)
        self.interject_edit.setEnabled(False)
        self.inject_button = QPushButton("Inject")
        self.inject_button.setObjectName("quietButton")
        self.inject_button.setEnabled(False)
        self.inject_button.setToolTip("Append this message to the shared history as the User.")
        self.inject_button.clicked.connect(self._inject_message)
        interject_row.addWidget(self.interject_edit, 1)
        interject_row.addWidget(self.inject_button)
        layout.addLayout(interject_row)

        composer_heading = section_label("// OPENING PROMPT")
        layout.addWidget(composer_heading)
        self.prompt = QPlainTextEdit()
        self.prompt.setMaximumHeight(64)
        self.prompt.setPlaceholderText("What should the models discuss?")
        self.prompt.setPlainText("Discuss one practical way AI could make everyday life better.")
        layout.addWidget(self.prompt)

        self.setCentralWidget(root)
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self._set_status("Ready — configure participants, set their order, and press Fetch models to check them.")
        self._load_settings()
        self._apply_style()
        self._refresh_controls()

    def _make_endpoint(self) -> EndpointEditor:
        order = HudSpinBox()
        order.setRange(1, 999)
        order.setValue(1)
        order.setToolTip("Lower numbers speak first. Equal orders use the column position as a tie-breaker.")
        remove = QPushButton("Remove")
        remove.setObjectName("quietButton")
        remove.setToolTip("Remove this participant column.")
        close = QPushButton("✕")
        close.setObjectName("participantCloseButton")
        close.setFixedSize(20, 18)
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setToolTip("Remove this participant window.")
        return EndpointEditor(
            participant_id=uuid4().hex,
            name=QLineEdit(),
            url=QLineEdit(),
            model=QComboBox(),
            api_key=QLineEdit(),
            check=QPushButton("Fetch models"),
            system_prompt=QPlainTextEdit(),
            temperature=QSlider(Qt.Orientation.Horizontal),
            temperature_value=QLabel(),
            order=order,
            remove=remove,
            close=close,
            group=None,
        )

    def _endpoint_group(self, index: int, editor: EndpointEditor) -> QGroupBox:
        group = QGroupBox(participant_title(index))
        group.setObjectName("endpointGroup")
        group.setMinimumWidth(326)
        editor.close.setParent(group)
        editor.close.move(6, 4)
        editor.close.setAccessibleName(f"Remove participant {index + 1}")
        editor.close.clicked.connect(lambda _checked=False, current=editor: self._remove_participant(current))
        editor.close.raise_()
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setContentsMargins(16, 16, 16, 12)
        form.setSpacing(5)
        editor.name.setPlaceholderText(f"Model {index + 1}")
        editor.name.setClearButtonEnabled(True)
        editor.url.setPlaceholderText("http://192.168.1.20:1234/v1")
        editor.url.setClearButtonEnabled(True)
        editor.model.setEditable(True)
        editor.model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        editor.model.lineEdit().setPlaceholderText("Type a model ID or fetch the list")
        editor.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        editor.api_key.setPlaceholderText("Optional server token")
        editor.api_key.setClearButtonEnabled(True)
        editor.api_key.setToolTip(
            "Sent as Authorization: Bearer … when non-empty. Never stored in ordinary application settings."
        )
        editor.system_prompt.setPlaceholderText("Optional instructions that guide this participant's replies")
        editor.system_prompt.setMaximumHeight(58)
        editor.temperature.setRange(0, 200)
        editor.temperature.setSingleStep(10)
        editor.temperature.setPageStep(10)
        editor.temperature.setValue(70)
        editor.temperature.setToolTip("Choose a temperature from 0.00 (focused) to 2.00 (more varied).")
        editor.temperature_value.setMinimumWidth(38)
        editor.temperature_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        editor.temperature_value.setText("0.70")
        editor.temperature.valueChanged.connect(
            lambda value, label=editor.temperature_value: label.setText(f"{value / 100:.2f}")
        )
        editor.check.setObjectName("quietButton")
        editor.check.clicked.connect(lambda _checked=False, current=editor: self._fetch_models(current))
        editor.remove.clicked.connect(lambda _checked=False, current=editor: self._remove_participant(current))
        form.addRow("Name", editor.name)
        form.addRow("Server URL", editor.url)
        model_row = QHBoxLayout()
        model_row.addWidget(editor.model, 1)
        model_row.addWidget(editor.check)
        form.addRow("Model ID", model_row)
        form.addRow("API token", editor.api_key)
        order_row = QHBoxLayout()
        order_row.addWidget(editor.order)
        order_row.addStretch(1)
        order_row.addSpacing(8)
        order_row.addWidget(editor.remove)
        form.addRow("Order", order_row)
        form.addRow("System prompt", editor.system_prompt)
        temperature_row = QHBoxLayout()
        temperature_row.addWidget(editor.temperature, 1)
        temperature_row.addWidget(editor.temperature_value)
        form.addRow("Temperature", temperature_row)
        editor.group = group
        return group

    def _rebuild_participant_layout(self) -> None:
        while self.participant_layout.count():
            item = self.participant_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.participant_container)
        for index in range(32):
            self.participant_layout.setColumnStretch(index, 0)
        for index, editor in enumerate(self.endpoints):
            if editor.group is None:
                editor.group = self._endpoint_group(index, editor)
            editor.group.setTitle(participant_title(index))
            editor.name.setPlaceholderText(f"Model {index + 1}")
            self.participant_layout.addWidget(editor.group, 0, index)
            self.participant_layout.setColumnStretch(index, 1)
        self.participant_layout.setRowStretch(0, 1)
        minimum_width = max(0, len(self.endpoints) * 326 + max(0, len(self.endpoints) - 1) * 12 + 8)
        self.participant_container.setMinimumWidth(minimum_width)

    def _load_settings(self) -> None:
        try:
            count = int(self.settings_store.value("endpoints/count", 2))
        except (TypeError, ValueError):
            count = 2
        count = max(2, count)
        for index in range(count):
            editor = self._make_endpoint()
            editor.group = self._endpoint_group(index, editor)
            prefix = f"endpoints/{index}/"
            participant_id = str(self.settings_store.value(prefix + "id", "")).strip()
            if participant_id:
                editor.participant_id = participant_id
            editor.name.setText(str(self.settings_store.value(prefix + "name", "")))
            editor.url.setText(str(self.settings_store.value(prefix + "url", "")))
            model = str(self.settings_store.value(prefix + "model", ""))
            if model:
                editor.model.addItem(model)
            editor.api_key.setText(_load_secret(editor.participant_id, index))
            editor.system_prompt.setPlainText(str(self.settings_store.value(prefix + "systemPrompt", "")))
            editor.temperature.setValue(round(float(self.settings_store.value(prefix + "temperature", 0.7)) * 100))
            try:
                order = int(self.settings_store.value(prefix + "order", index + 1))
            except (TypeError, ValueError):
                order = index + 1
            editor.order.setValue(max(1, order))
            self.endpoints.append(editor)
        self._rebuild_participant_layout()
        self.max_turns.setValue(int(self.settings_store.value("conversation/maxTurns", 6)))
        self.request_timeout.setValue(
            int(self.settings_store.value("conversation/timeoutSeconds", int(DEFAULT_TIMEOUT_SECONDS)))
        )
        self.history_limit.setValue(
            int(self.settings_store.value("conversation/maxHistoryCharacters", DEFAULT_MAX_HISTORY_CHARACTERS))
        )
        if not _secret_backend_available():
            self._set_status(
                "Settings persist, but no Linux Secret Service/keyring is available: API tokens stay in memory only."
            )

    def _save_settings(self) -> None:
        self.settings_store.beginGroup("endpoints")
        try:
            self.settings_store.setValue("count", len(self.endpoints))
            for index, editor in enumerate(self.endpoints):
                self.settings_store.setValue(f"{index}/id", editor.participant_id)
                self.settings_store.setValue(f"{index}/name", editor.name.text().strip())
                self.settings_store.setValue(f"{index}/url", editor.url.text().strip())
                self.settings_store.setValue(f"{index}/model", editor.model.currentText().strip())
                self.settings_store.setValue(f"{index}/systemPrompt", editor.system_prompt.toPlainText())
                self.settings_store.setValue(f"{index}/temperature", editor.temperature.value() / 100.0)
                self.settings_store.setValue(f"{index}/order", editor.order.value())
                if keyring is not None and _secret_backend_available():
                    token = editor.api_key.text()
                    try:
                        if token:
                            keyring.set_password(KEYRING_SERVICE, f"endpoint-{editor.participant_id}", token)
                        else:
                            try:
                                keyring.delete_password(KEYRING_SERVICE, f"endpoint-{editor.participant_id}")
                            except Exception:
                                pass
                    except (KeyringError, Exception):
                        self._set_status("Token not persisted in the OS keyring; it remains available this session only.")
        finally:
            self.settings_store.endGroup()
        self.settings_store.setValue("conversation/maxTurns", self.max_turns.value())
        self.settings_store.setValue("conversation/timeoutSeconds", self.request_timeout.value())
        self.settings_store.setValue("conversation/maxHistoryCharacters", self.history_limit.value())
        self.settings_store.sync()

    def _add_participant(self) -> None:
        if self._session is not None or self._worker is not None or self._probe_workers:
            return
        editor = self._make_endpoint()
        try:
            highest = max(editor.order.value() for editor in self.endpoints)
        except ValueError:
            highest = 0
        editor.order.setValue(min(999, max(highest + 1, len(self.endpoints) + 1)))
        self.endpoints.append(editor)
        editor.group = self._endpoint_group(len(self.endpoints) - 1, editor)
        self._rebuild_participant_layout()
        self._save_settings()
        self._refresh_controls()
        self.participant_scroll.ensureWidgetVisible(editor.group)
        editor.name.setFocus()
        self._set_status(f"Added Participant {len(self.endpoints)}. Set its turn order to place it in the queue.")

    def _remove_participant(self, editor: EndpointEditor) -> None:
        if self._session is not None or self._worker is not None or self._probe_workers:
            return
        if editor not in self.endpoints or len(self.endpoints) <= 2:
            return
        _forget_secret(editor.participant_id)
        self.endpoints.remove(editor)
        self._rebuild_participant_layout()
        if editor.group is not None:
            editor.group.setParent(None)
            editor.group.deleteLater()
        self._save_settings()
        self._refresh_controls()
        self._set_status(f"Removed a participant — {len(self.endpoints)} remain in the arena.")

    def _editor_label(self, editor: EndpointEditor) -> str:
        if editor in self.endpoints:
            fallback = f"Model {self.endpoints.index(editor) + 1}"
        else:
            fallback = "Model"
        return editor.name.text().strip() or fallback

    def _fetch_models(self, editor: EndpointEditor) -> None:
        if editor in self._probe_workers or editor not in self.endpoints:
            return
        try:
            endpoint = editor.settings(self.endpoints.index(editor), require_model=False)
        except ValueError as exc:
            QMessageBox.warning(self, "Check endpoint settings", str(exc))
            return
        worker = ModelRequestThread(
            endpoint,
            list_models=True,
            timeout=float(self.request_timeout.value()),
            parent=self,
        )
        self._probe_workers[editor] = worker
        editor.check.setEnabled(False)
        editor.check.setText("Connecting…")
        worker.completed.connect(lambda result, current=editor: self._models_loaded(current, result))
        worker.failed.connect(lambda message, current=editor: self._model_load_failed(current, message))
        worker.finished.connect(lambda current=editor, active=worker: self._probe_finished(current, active))
        worker.start()
        self._save_settings()
        self._refresh_controls()
        self._set_status(f"Checking {endpoint.name}…")

    def _models_loaded(self, editor: EndpointEditor, model_ids: str) -> None:
        if editor not in self.endpoints:
            return
        combo = editor.model
        previous = combo.currentText().strip()
        ids = model_ids.splitlines()
        combo.clear()
        combo.addItems(ids)
        editor.model.setCurrentText(previous if previous in ids else ids[0])
        self._set_status(f"Found {len(ids)} model ID(s) for {self._editor_label(editor)}.")

    def _model_load_failed(self, editor: EndpointEditor, message: str) -> None:
        self._set_status(f"Could not connect to {self._editor_label(editor)}: {message}")
        QMessageBox.warning(self, "Model server unavailable", message)

    def _probe_finished(self, editor: EndpointEditor, worker: ModelRequestThread) -> None:
        self._probe_workers.pop(editor, None)
        editor.check.setText("Fetch models")
        worker.deleteLater()
        self._refresh_controls()

    def _start_conversation(self) -> None:
        if self._worker is not None or self._session is not None:
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "Add an opening prompt", "Type a short opening prompt before starting.")
            self.prompt.setFocus()
            return
        try:
            endpoint_settings = [editor.settings(index) for index, editor in enumerate(self.endpoints)]
            paired = [
                (Participant(endpoint.name, editor.order.value(), endpoint.system_prompt), endpoint, index)
                for index, (editor, endpoint) in enumerate(zip(self.endpoints, endpoint_settings))
            ]
            paired.sort(key=lambda item: (item[0].turn_order, item[2]))
            participants = [item[0] for item in paired]
            session = DialogueSession(participants, prompt, self.max_turns.value(), self.history_limit.value())
        except ValueError as exc:
            QMessageBox.warning(self, "Check conversation settings", str(exc))
            return
        self._save_settings()
        self._active_endpoints = [item[1] for item in paired]
        self._session = session
        self._history_turns = []
        self._starter_prompt = prompt
        self._worker_result = None
        self._worker_failure = None
        self._stop_pending = False
        self._player.start()
        self._append_text(f"YOU\n{prompt}\n\n")
        self._refresh_controls()
        self._launch_next_turn()

    def _launch_next_turn(self) -> None:
        session = self._session
        if session is None or not session.running or self._closing:
            return
        if not self._player.can_advance():
            self._refresh_controls()
            if self._player.is_paused:
                self._set_status("Paused — add an interjection or press Resume.")
            return
        try:
            plan = session.next_turn()
        except RuntimeError:
            return
        endpoint = self._active_endpoints[plan.endpoint_index]
        self._append_text(f"{plan.speaker} · {endpoint.model}\n")
        worker = ModelRequestThread(
            endpoint,
            plan.messages,
            timeout=float(self.request_timeout.value()),
            parent=self,
        )
        self._worker = worker
        self._worker_result = None
        self._worker_failure = None
        worker.text_received.connect(self._append_text)
        worker.completed.connect(self._model_completed)
        worker.failed.connect(self._model_failed)
        worker.finished.connect(lambda current=worker: self._turn_finished(current))
        self._refresh_controls()
        self._set_status(
            f"Turn {session.model_turns + 1} of {session.max_turns} — waiting for {endpoint.name}…"
        )
        worker.start()

    def _model_completed(self, response: str) -> None:
        self._worker_result = response

    def _model_failed(self, message: str) -> None:
        self._worker_failure = message

    def _turn_finished(self, worker: ModelRequestThread) -> None:
        if worker is not self._worker:
            worker.deleteLater()
            return
        self._worker = None
        worker.deleteLater()
        self._append_text("\n\n")
        if self._closing:
            return
        session = self._session
        if session is None:
            self._refresh_controls()
            return
        if self._stop_pending:
            self._finalize("Conversation stopped.")
            return
        if self._worker_failure:
            message = self._worker_failure
            self._finalize(f"Conversation stopped: {message}")
            QMessageBox.warning(self, "Model request failed", message)
            return
        if self._worker_result is None:
            self._finalize("Conversation stopped.")
            return
        try:
            should_continue = session.record_response(self._worker_result)
        except ProtocolError as exc:
            self._finalize(f"Conversation stopped: {exc}")
            QMessageBox.warning(self, "Empty model response", str(exc))
            return
        self._history_turns = list(session.turns)
        if not should_continue:
            self._finalize(f"Conversation complete — reached the {session.max_turns}-turn limit.")
            return
        if self._player.is_pausing:
            self._player.mark_paused()
        if self._player.is_paused:
            self._refresh_controls()
            self._set_status("Paused — add an interjection or press Resume.")
            return
        if self._player.state is PlaybackState.RUNNING:
            self._launch_next_turn()
            return
        self._finalize("Conversation stopped.")

    def _pause_conversation(self) -> None:
        if not self._player.pause():
            return
        if self._worker is None:
            self._player.mark_paused()
            self._set_status("Paused — add an interjection or press Resume.")
        else:
            self._set_status("Pausing after the current response finishes…")
        self._refresh_controls()

    def _resume_conversation(self) -> None:
        if not self._player.resume():
            return
        self._refresh_controls()
        if self._session is not None and self._worker is None and self._session.running:
            self._launch_next_turn()
        else:
            self._set_status("Resuming — the next queued participant will start after the active response.")

    def _inject_message(self) -> None:
        session = self._session
        if session is None or not self._player.is_paused or not session.running:
            return
        text = self.interject_edit.toPlainText().strip()
        if not text:
            self.interject_edit.setFocus()
            self._set_status("Type an interjection before pressing Inject.")
            return
        try:
            session.inject_user_message(text)
        except (RuntimeError, ValueError) as exc:
            self._set_status(str(exc))
            return
        self._history_turns = list(session.turns)
        self._append_text(f"YOU · Interjection\n{text}\n\n")
        self.interject_edit.clear()
        self._refresh_controls()
        self._set_status("Interjection added to shared history. Press Resume to let the models respond.")

    def _stop_conversation(self) -> None:
        if self._session is None and self._worker is None:
            return
        if self._session is not None:
            self._session.stop()
        self._player.stop()
        self._stop_pending = True
        if self._worker is not None:
            self._worker.cancel()
            self._refresh_controls()
            self._set_status("Stopping the current model request…")
        else:
            self._finalize("Conversation stopped.")

    def _finalize(self, message: str) -> None:
        if self._session is not None:
            self._history_turns = list(self._session.turns)
        self._session = None
        self._active_endpoints = []
        self._stop_pending = False
        self._worker_result = None
        self._worker_failure = None
        self._player.finish()
        self._refresh_controls()
        self._set_status(message)

    def _refresh_controls(self) -> None:
        state = self._player.state
        session_active = self._session is not None
        configuration_enabled = self._session is None and self._worker is None
        self.run_button.setEnabled(configuration_enabled)
        self.pause_button.setEnabled(state is PlaybackState.RUNNING and session_active)
        self.resume_button.setEnabled(state in (PlaybackState.PAUSING, PlaybackState.PAUSED) and session_active)
        self.stop_button.setEnabled(session_active and state in (PlaybackState.RUNNING, PlaybackState.PAUSING, PlaybackState.PAUSED))
        interject_enabled = self._player.is_paused and session_active and bool(self._session and self._session.running)
        self.interject_edit.setEnabled(interject_enabled)
        self.inject_button.setEnabled(interject_enabled)
        self.max_turns.setEnabled(configuration_enabled)
        self.request_timeout.setEnabled(configuration_enabled)
        self.history_limit.setEnabled(configuration_enabled)
        self.prompt.setEnabled(configuration_enabled)
        self.clear_button.setEnabled(configuration_enabled)
        self.add_participant_button.setEnabled(configuration_enabled and not self._probe_workers)
        for editor in self.endpoints:
            for field_widget in (
                editor.name,
                editor.url,
                editor.model,
                editor.api_key,
                editor.system_prompt,
                editor.temperature,
                editor.order,
            ):
                field_widget.setEnabled(configuration_enabled)
            editor.check.setEnabled(configuration_enabled and editor not in self._probe_workers)
            removable = configuration_enabled and len(self.endpoints) > 2 and editor not in self._probe_workers
            editor.remove.setEnabled(removable)
            editor.close.setEnabled(removable)
            if len(self.endpoints) <= 2:
                editor.close.setToolTip("The arena needs at least two participants.")
            elif not configuration_enabled:
                editor.close.setToolTip("Stop the conversation before removing participants.")
            else:
                editor.close.setToolTip("Remove this participant window.")
        has_history = bool(self._history_turns) or bool(self._session and self._session.turns)
        self.export_button.setEnabled(has_history)

    def _write_export(self, filename: str) -> Path:
        turns = list(self._session.turns) if self._session is not None else list(self._history_turns)
        if not turns:
            raise ValueError("No completed conversation turns to export yet.")
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix not in {".md", ".markdown", ".txt"}:
            path = Path(f"{filename}.md")
            suffix = ".md"
        path.write_text(
            render_transcript(turns, self._starter_prompt, markdown=suffix != ".txt"),
            encoding="utf-8",
        )
        return path

    def _export_transcript(self) -> None:
        turns = list(self._session.turns) if self._session is not None else list(self._history_turns)
        if not turns:
            self._set_status("No completed conversation turns to export yet.")
            return
        default_path = Path.cwd() / f"model-arena-{datetime.now():%Y%m%d-%H%M%S}.md"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export conversation",
            str(default_path),
            "Markdown (*.md *.markdown);;Plain text (*.txt)",
        )
        if not filename:
            return
        if Path(filename).suffix.lower() not in {".md", ".markdown", ".txt"}:
            filename += ".txt" if "Plain text" in selected_filter else ".md"
        try:
            path = self._write_export(filename)
        except (OSError, ValueError) as exc:
            self._set_status(f"Could not export conversation: {exc}")
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._set_status(f"Exported conversation to {path}.")

    def _clear_transcript(self) -> None:
        if self._session is not None or self._worker is not None:
            return
        self.transcript.clear()
        self._history_turns = []
        self._starter_prompt = ""
        self._refresh_controls()
        self._set_status("Transcript cleared.")

    def _append_text(self, text: str) -> None:
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def _set_status(self, message: str) -> None:
        self.status.showMessage(message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            "QWidget { color: #d7ecff; }"
            "QLabel { background: transparent; }"
            "QLabel#eyebrow { color: #46c8e6; }"
            "QLabel#title { color: #f2feff; }"
            "QLabel#subtitle { color: #8fb3cc; font-size: 13px; }"
            "QLabel#sectionTitle { color: #6fe3ff; }"
            "QFrame#accentRule { border: none; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #35e0ff, stop:0.34 #1f7bd6, stop:0.62 #ff9a3c, stop:1 rgba(255, 154, 60, 0)); }"
            "QLabel#notice { background: rgba(46, 26, 8, 190); color: #ffcf9b; border: 1px solid "
            "rgba(255, 154, 60, 120); border-radius: 8px; padding: 9px 12px; }"
            "QGroupBox#endpointGroup { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
            "stop:0 rgba(9, 30, 49, 225), stop:1 rgba(5, 15, 26, 225)); border: 1px solid #14506b; "
            "border-radius: 12px; margin-top: 13px; padding-top: 10px; }"
            "QGroupBox#endpointGroup::title { subcontrol-origin: margin; left: 31px; padding: 0 6px; "
            "color: #63e6ff; font-family: 'DejaVu Sans Mono'; font-size: 9pt; }"
            "QScrollArea#participantScroll { border: none; background: transparent; }"
            "QScrollArea#participantScroll > QWidget > QWidget { background: transparent; }"
            "QLineEdit, QComboBox, QSpinBox, QPlainTextEdit { background: rgba(4, 16, 27, 235); "
            "border: 1px solid #164a63; border-radius: 6px; padding: 6px 8px; color: #d8f2ff; "
            "selection-background-color: #156b8c; selection-color: #f2feff; }"
            "QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus { border: 1px solid "
            "#3fd9ff; background: rgba(6, 26, 42, 245); }"
            "QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QPlainTextEdit:disabled { "
            "color: #5d7d92; border-color: #103042; background: rgba(4, 12, 20, 200); }"
            "QComboBox QAbstractItemView { background: #04121f; color: #d8f2ff; border: 1px solid "
            "#1b526e; selection-background-color: #156b8c; selection-color: #f2feff; outline: none; }"
            "QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 16px; "
            "margin: 1px 1px 0 0; border: none; border-left: 1px solid #164a63; border-top-right-radius: 5px; "
            "background: rgba(16, 46, 66, 215); }"
            "QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; "
            "margin: 0 1px 1px 0; border: none; border-left: 1px solid #164a63; border-bottom-right-radius: 5px; "
            "background: rgba(16, 46, 66, 215); }"
            "QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: rgba(28, 78, 104, 235); }"
            "QSlider::groove:horizontal { height: 4px; background: #0e2c3f; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #1f6be0, stop:1 #35e0ff); border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 14px; margin: -6px 0; background: #bff4ff; "
            "border: 2px solid #35b7dd; border-radius: 8px; }"
            "QSlider::handle:horizontal:hover { background: #ffffff; border-color: #ff9a3c; }"
            "QSlider::handle:horizontal:disabled { background: #46606f; border-color: #23414f; }"
            "QPlainTextEdit#transcript { background: rgba(3, 10, 19, 232); border: 1px solid #13455f; "
            "border-radius: 10px; padding: 14px; color: #c9ecff; selection-background-color: #176b8b; }"
            "QPlainTextEdit#interjectEdit { background: rgba(6, 21, 34, 220); border: 1px dashed #2b7d9e; }"
            "QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0d283c, "
            "stop:1 #081724); border: 1px solid #1b526e; border-radius: 7px; padding: 8px 14px; "
            "color: #cfeeff; }"
            "QPushButton:hover { border-color: #3fd9ff; color: #f2feff; background: qlineargradient("
            "x1:0, y1:0, x2:0, y2:1, stop:0 #11405c, stop:1 #0a2233); }"
            "QPushButton:pressed { background: #06121e; }"
            "QPushButton:focus { border-color: #35e0ff; }"
            "QPushButton:disabled { color: #4f6a7c; border-color: #123346; background: #06111c; }"
            "QPushButton#primaryButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            "stop:0 #ffc271, stop:0.45 #ff9a3c, stop:1 #ff7038); border: 1px solid #ffdca6; "
            "color: #1b0a02; font-weight: 700; padding: 9px 18px; }"
            "QPushButton#primaryButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            "stop:0 #ffd79a, stop:0.45 #ffab55, stop:1 #ff8350); }"
            "QPushButton#primaryButton:disabled { background: #241a12; border-color: #4b3722; color: #806a55; }"
            "QPushButton#stopButton { background: #1c0d18; border: 1px solid #7a2f4a; color: #ff9fb6; }"
            "QPushButton#stopButton:hover { background: #2b1020; border-color: #ff5f8a; color: #ffd0dd; }"
            "QPushButton#stopButton:disabled { background: #12070d; border-color: #3a1a26; color: #6d4553; }"
            "QPushButton#quietButton { padding: 6px 10px; background: rgba(8, 24, 38, 200); "
            "border: 1px solid #16455f; }"
            "QPushButton#quietButton:hover { background: rgba(14, 44, 66, 230); border-color: #3fd9ff; }"
            "QPushButton#participantCloseButton { background: transparent; border: none; border-radius: 5px; "
            "padding: 0; color: #7fd4ea; font-size: 12px; font-weight: 700; }"
            "QPushButton#participantCloseButton:hover { background: #4a1530; color: #ff9fba; }"
            "QPushButton#participantCloseButton:disabled { color: #3d5a68; background: transparent; }"
            "QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }"
            "QScrollBar::handle:vertical { background: #1b5f7d; min-height: 26px; border-radius: 4px; }"
            "QScrollBar::handle:vertical:hover { background: #35b7dd; }"
            "QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }"
            "QScrollBar::handle:horizontal { background: #1b5f7d; min-width: 26px; border-radius: 4px; }"
            "QScrollBar::handle:horizontal:hover { background: #35b7dd; }"
            "QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; border: none; background: none; }"
            "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }"
            "QStatusBar { background: rgba(3, 9, 16, 235); color: #79cfe8; border-top: 1px solid #123c53; }"
            "QStatusBar::item { border: none; }"
            "QToolTip { background: #07182a; color: #d7ecff; border: 1px solid #2b7d9e; padding: 6px 8px; }"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        active = [*self._probe_workers.values()]
        if self._worker is not None:
            active.append(self._worker)
        if not active:
            self._save_settings()
            event.accept()
            return
        self._closing = True
        self._stop_pending = False
        if self._session is not None:
            self._session.stop()
        self._session = None
        self._player.stop()
        for worker in active:
            worker.cancel()
        if all(worker.wait(1500) for worker in active):
            self._save_settings()
            event.accept()
        else:
            event.ignore()
            self._set_status("Waiting for network requests to stop before closing…")
            for worker in active:
                if worker.isRunning():
                    worker.finished.connect(self.close, Qt.ConnectionType.SingleShotConnection)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Freebuff")
    app.setOrganizationDomain("freebuff.com")
    window = MainWindow()
    window.show()
    return app.exec()
