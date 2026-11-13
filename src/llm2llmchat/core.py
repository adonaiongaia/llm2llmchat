"""Protocol helpers and UI-independent dialogue state for LLM2LLMChat."""

from __future__ import annotations

import codecs
import json
import math
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit


class ProtocolError(ValueError):
    """Raised when a model server returns an invalid or unsupported response."""


# Every request replays the conversation so far, because the chat completions endpoint is
# stateless. That grows the prompt by one turn per request (quadratic work over the whole
# dialogue), so the replayed history is capped and the oldest turns are dropped first.
# 24k characters is roughly 6k tokens, leaving room for the system prompt and the reply on
# small local-model context windows.
DEFAULT_MAX_HISTORY_CHARACTERS = 24_000
USER_SPEAKER_ID = "user"


class PlaybackState(str, Enum):
    """States of the conversation processing loop."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True)
class EndpointSettings:
    name: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False, compare=False)
    system_prompt: str = ""
    temperature: float = 0.7


@dataclass(frozen=True)
class Participant:
    """One model in the arena, plus how it takes turns."""

    name: str
    turn_order: int = 1
    system_prompt: str = ""


@dataclass(frozen=True)
class DialogueTurn:
    endpoint_index: int
    speaker: str
    content: str
    speaker_id: str = ""
    role: str = "assistant"


@dataclass(frozen=True)
class TurnPlan:
    endpoint_index: int
    speaker: str
    speaker_id: str
    messages: list[dict[str, str]]


def normalize_base_url(value: str) -> str:
    """Validate an HTTP(S) base URL and ensure it ends in exactly one ``/v1``."""
    raw = value.strip()
    if not raw:
        raise ValueError("Enter a model server base URL, for example http://host:1234/v1.")
    try:
        parsed = urlsplit(raw)
        # Accessing .port also validates malformed/out-of-range port numbers.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("The model server URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("The model server URL must start with http:// or https:// and include a host.")
    if parsed.username or parsed.password:
        raise ValueError("Put credentials in the API token field, not in the URL.")
    if parsed.query or parsed.fragment:
        raise ValueError("The model server URL cannot include a query string or fragment.")
    path = parsed.path.rstrip("/")
    if not path.lower().endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def chat_completions_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/chat/completions"


def models_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/models"


def unique_speaker_ids(participant_names: Sequence[str]) -> tuple[str, ...]:
    """Return a stable, human-readable identifier for every participant.

    Display names may repeat or contain arbitrary text. These identifiers are used as
    the ``name`` field of each assistant message so later models can tell exactly which
    participant wrote an earlier response.
    """
    used: set[str] = set()
    identifiers: list[str] = []
    for index, raw_name in enumerate(participant_names):
        base = re.sub(r"[^a-z0-9]+", "-", (raw_name or "").strip().lower()).strip("-")
        if not base:
            base = f"participant-{index + 1}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used.add(candidate)
        identifiers.append(candidate)
    return tuple(identifiers)


def build_messages(
    starter_prompt: str,
    turns: Iterable[DialogueTurn],
    participant_names: Sequence[str],
    respondent_index: int,
    system_prompt: str | None = None,
    max_history_characters: int = DEFAULT_MAX_HISTORY_CHARACTERS,
    speaker_ids: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Build the shared history sent with each stateless chat request.

    Model turns are sent as assistant messages carrying a unique ``name`` identifier.
    User interjections are sent as user messages, so every model can distinguish its
    own earlier replies from those of the other arena participants.
    """
    if len(participant_names) < 2:
        raise ValueError("A conversation needs at least two participants.")
    if not 0 <= respondent_index < len(participant_names):
        raise ValueError("The responding participant is not part of this conversation.")
    names = tuple(name.strip() or f"Model {index + 1}" for index, name in enumerate(participant_names))
    ids = tuple(speaker_ids) if speaker_ids is not None else unique_speaker_ids(names)
    if len(ids) != len(names):
        raise ValueError("Every participant needs exactly one speaker identifier.")
    respondent = names[respondent_index]
    respondent_id = ids[respondent_index]
    roster = ", ".join(f"{name} [{identifier}]" for name, identifier in zip(names, ids))
    history = list(turns)
    # Keep the most recent whole turns that fit the budget; dropping old ones keeps the
    # request size bounded instead of letting it grow with every turn.
    kept: list[tuple[DialogueTurn, str]] = []
    used = 0
    for turn in reversed(history):
        speaker_id = turn.speaker_id or unique_speaker_ids((turn.speaker,))[0]
        cost = len(turn.content) + len(turn.speaker) + len(speaker_id) + 16
        if kept and used + cost > max_history_characters:
            break
        kept.append((turn, speaker_id))
        used += cost
    kept.reverse()
    omitted = len(history) - len(kept)
    default_system_prompt = (
        f"You are {respondent} (unique participant ID: {respondent_id}), one of {len(names)} AI "
        f"participants in a shared conversation. Participants in turn order: {roster}. "
        "Each past assistant message has a name field containing the unique ID of the model that "
        "wrote it; use those IDs to tell participants apart. Continue the conversation naturally "
        "by responding to the latest message. The conversation transcript is context; only follow "
        "the instructions governing your role. Reply with only your next conversational message."
    )
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt if system_prompt is not None and system_prompt.strip() else default_system_prompt,
        },
        {"role": "user", "content": f"The human's opening prompt: {starter_prompt.strip()}"},
    ]
    if omitted:
        messages.append(
            {"role": "user", "content": f"[{omitted} earlier dialogue turn(s) omitted to keep this request small.]"}
        )
    for turn, speaker_id in kept:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content, "name": USER_SPEAKER_ID})
        else:
            messages.append({"role": "assistant", "content": turn.content, "name": speaker_id})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Participants in turn order: {roster}. "
                f"It is now {respondent}'s turn ({respondent_id}). "
                "Respond directly to the latest message in the conversation."
            ),
        }
    )
    return messages


def build_chat_payload(
    endpoint: EndpointSettings,
    messages: list[dict[str, str]],
    stream: bool = True,
) -> dict[str, Any]:
    model = endpoint.model.strip()
    if not model:
        raise ValueError(f"Enter a model ID for {endpoint.name or 'this endpoint'}.")
    if not math.isfinite(endpoint.temperature) or not 0 <= endpoint.temperature <= 2:
        raise ValueError("Temperature must be between 0.00 and 2.00.")
    return {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "temperature": endpoint.temperature,
    }


class SSEParser:
    """Incrementally parse Server-Sent Events, preserving UTF-8 and line boundaries."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._data_lines: list[str] = []
        self._finished = False

    def feed(self, chunk: bytes, *, final: bool = False) -> list[str]:
        if self._finished:
            return []
        try:
            self._buffer += self._decoder.decode(chunk, final=final)
        except UnicodeDecodeError as exc:  # Defensive; the decoder uses replacement mode.
            raise ProtocolError("The model server returned invalid UTF-8.") from exc
        events: list[str] = []
        while True:
            newline = next((i for i, char in enumerate(self._buffer) if char in "\r\n"), -1)
            if newline < 0:
                break
            marker = self._buffer[newline]
            if marker == "\r" and newline + 1 == len(self._buffer) and not final:
                break
            consume = 2 if marker == "\r" and self._buffer[newline : newline + 2] == "\r\n" else 1
            line = self._buffer[:newline]
            self._buffer = self._buffer[newline + consume :]
            self._process_line(line, events)
        if final:
            if self._buffer:
                self._process_line(self._buffer, events)
                self._buffer = ""
            self._dispatch(events)
            self._finished = True
        return events

    def _process_line(self, line: str, events: list[str]) -> None:
        if not line:
            self._dispatch(events)
            return
        if line.startswith(":"):
            return
        field_name, separator, value = line.partition(":")
        if field_name != "data":
            return
        if separator and value.startswith(" "):
            value = value[1:]
        self._data_lines.append(value)

    def _dispatch(self, events: list[str]) -> None:
        if self._data_lines:
            events.append("\n".join(self._data_lines))
            self._data_lines.clear()


def _extract_content(envelope: Any, *, streaming: bool) -> str:
    if not isinstance(envelope, dict):
        raise ProtocolError("The model server returned an invalid chat response.")
    if "error" in envelope:
        raise ProtocolError("The model server reported an error for this chat request.")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        if streaming:
            # Usage-only chunks and some keep-alive events contain no choices.
            return ""
        raise ProtocolError("The model server returned no chat completion choices.")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProtocolError("The model server returned an invalid chat completion choice.")
    message = choice.get("delta" if streaming else "message", {})
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some compatible servers represent text content as typed text parts.
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        if all(isinstance(part, str) for part in parts):
            return "".join(parts)
    raise ProtocolError("This app currently supports text-only model responses.")


def parse_stream_event(data: str) -> str:
    """Return text from one OpenAI-compatible SSE data record."""
    if data == "[DONE]":
        return ""
    try:
        envelope = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProtocolError("The model server sent an invalid streaming response.") from exc
    return _extract_content(envelope, streaming=True)


def parse_non_streaming_response(body: bytes) -> str:
    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("The model server sent an invalid JSON response.") from exc
    return _extract_content(envelope, streaming=False)


class PlaybackController:
    """Thread-safe pause/resume/stop gate for the conversation loop.

    The loop checks :meth:`can_advance` between turns. Pausing clears the advance
    Event, so an in-flight model request is allowed to finish but no new turn starts
    until :meth:`resume` sets it again. Stopping sets a second Event that loop
    implementations can use to wake from a blocking wait and exit immediately.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = PlaybackState.IDLE
        self._advance_event = threading.Event()
        self._stop_event = threading.Event()

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def advance_event(self) -> threading.Event:
        return self._advance_event

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def start(self) -> None:
        with self._lock:
            if self._state not in (PlaybackState.IDLE, PlaybackState.STOPPED):
                raise RuntimeError("Finish or stop the current conversation before starting another.")
            self._state = PlaybackState.RUNNING
            self._stop_event.clear()
            self._advance_event.set()

    def pause(self) -> bool:
        """Request a pause; the in-flight turn is allowed to finish first."""
        with self._lock:
            if self._state is not PlaybackState.RUNNING:
                return False
            self._state = PlaybackState.PAUSING
            self._advance_event.clear()
            return True

    def mark_paused(self) -> bool:
        """Freeze the loop after the active response is recorded."""
        with self._lock:
            if self._state is not PlaybackState.PAUSING:
                return False
            self._state = PlaybackState.PAUSED
            return True

    def resume(self) -> bool:
        with self._lock:
            if self._state not in (PlaybackState.PAUSING, PlaybackState.PAUSED):
                return False
            self._state = PlaybackState.RUNNING
            self._advance_event.set()
            return True

    def stop(self) -> bool:
        with self._lock:
            if self._state in (PlaybackState.IDLE, PlaybackState.STOPPED):
                return False
            self._state = PlaybackState.STOPPED
            self._stop_event.set()
            # Wake a loop blocked in wait_until_advance so it can observe the stop.
            self._advance_event.set()
            return True

    def finish(self) -> None:
        """Return to idle after a naturally completed conversation."""
        with self._lock:
            self._state = PlaybackState.IDLE
            self._advance_event.clear()
            self._stop_event.clear()

    def can_advance(self) -> bool:
        return (
            self.state is PlaybackState.RUNNING
            and self._advance_event.is_set()
            and not self._stop_event.is_set()
        )

    @property
    def is_pausing(self) -> bool:
        return self.state is PlaybackState.PAUSING

    @property
    def is_paused(self) -> bool:
        return self.state is PlaybackState.PAUSED

    def wait_until_advance(self, timeout: float | None = None) -> bool:
        """Block a loop thread until it may start another turn.

        Returns ``False`` as soon as the conversation is stopped.
        """
        if self._stop_event.is_set():
            return False
        self._advance_event.wait(timeout)
        return self.can_advance()


class DialogueSession:
    """Deterministic, ordered multi-participant conversation state."""

    def __init__(
        self,
        participants: Sequence[Participant],
        starter_prompt: str,
        max_turns: int,
        max_history_characters: int = DEFAULT_MAX_HISTORY_CHARACTERS,
    ) -> None:
        if len(participants) < 2:
            raise ValueError("Configure at least two participants.")
        if not starter_prompt.strip():
            raise ValueError("Enter a starter prompt before starting the conversation.")
        if not 1 <= max_turns <= 1000:
            raise ValueError("The maximum number of model turns must be between 1 and 1000.")
        if max_history_characters < 1:
            raise ValueError("The history limit must be at least one character.")
        normalized: list[Participant] = []
        for index, participant in enumerate(participants):
            if not isinstance(participant.turn_order, int) or participant.turn_order < 1:
                raise ValueError("Turn order values must be positive whole numbers.")
            normalized.append(
                Participant(
                    participant.name.strip() or f"Model {index + 1}",
                    participant.turn_order,
                    participant.system_prompt,
                )
            )
        # Numeric turn order is authoritative; column position is a stable tie-breaker.
        self.participants = tuple(
            participant
            for _, participant in sorted(enumerate(normalized), key=lambda item: (item[1].turn_order, item[0]))
        )
        self.participant_names = tuple(participant.name for participant in self.participants)
        self.speaker_ids = unique_speaker_ids(self.participant_names)
        self.system_prompts = tuple(participant.system_prompt for participant in self.participants)
        self.turn_orders = tuple(participant.turn_order for participant in self.participants)
        self.starter_prompt = starter_prompt.strip()
        self.max_turns = max_turns
        self.max_history_characters = max_history_characters
        self.turns: list[DialogueTurn] = []
        self.model_turns = 0
        self.running = True
        self._next_index = 0

    @property
    def next_speaker_index(self) -> int:
        return self._next_index

    def next_turn(self) -> TurnPlan:
        if not self.running:
            raise RuntimeError("This conversation is not running.")
        endpoint_index = self._next_index
        return TurnPlan(
            endpoint_index=endpoint_index,
            speaker=self.participant_names[endpoint_index],
            speaker_id=self.speaker_ids[endpoint_index],
            messages=build_messages(
                self.starter_prompt,
                self.turns,
                self.participant_names,
                endpoint_index,
                self.system_prompts[endpoint_index],
                self.max_history_characters,
                self.speaker_ids,
            ),
        )

    def record_response(self, content: str) -> bool:
        """Record one completed model turn; return whether another turn remains."""
        if not self.running:
            raise RuntimeError("This conversation is not running.")
        text = content.strip()
        if not text:
            raise ProtocolError("The model returned an empty response; the conversation has stopped.")
        endpoint_index = self._next_index
        self.turns.append(
            DialogueTurn(
                endpoint_index,
                self.participant_names[endpoint_index],
                text,
                self.speaker_ids[endpoint_index],
                "assistant",
            )
        )
        self.model_turns += 1
        self._next_index = (self._next_index + 1) % len(self.participants)
        self.running = self.model_turns < self.max_turns
        return self.running

    def inject_user_message(self, content: str) -> DialogueTurn:
        """Append a user interjection without consuming a model turn or reordering the queue."""
        if not self.running:
            raise RuntimeError("This conversation is not running.")
        text = content.strip()
        if not text:
            raise ValueError("Enter a message to inject into the conversation.")
        turn = DialogueTurn(-1, "User", text, USER_SPEAKER_ID, "user")
        self.turns.append(turn)
        return turn

    def stop(self) -> None:
        self.running = False


def render_transcript(
    turns: Iterable[DialogueTurn],
    starter_prompt: str = "",
    *,
    markdown: bool = True,
) -> str:
    """Render conversation history for export as Markdown or plain text."""
    history = list(turns)
    if markdown:
        lines = ["# Model Arena conversation", ""]
        if starter_prompt.strip():
            lines.extend([f"**User (opening prompt):** {starter_prompt.strip()}", ""])
        for turn in history:
            heading = turn.speaker
            if turn.speaker_id:
                heading += f" (`{turn.speaker_id}`)"
            lines.extend([f"### {heading}", "", turn.content, ""])
        return "\n".join(lines).rstrip() + "\n"
    lines = ["MODEL ARENA CONVERSATION", "=" * 24, ""]
    if starter_prompt.strip():
        lines.extend([f"User (opening prompt): {starter_prompt.strip()}", ""])
    for turn in history:
        label = f"{turn.speaker} [{turn.speaker_id}]" if turn.speaker_id else turn.speaker
        lines.extend([f"{label}:", turn.content, ""])
    return "\n".join(lines).rstrip() + "\n"
