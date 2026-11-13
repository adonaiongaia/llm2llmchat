import json
import threading
import time

import pytest

from llm2llmchat.core import (
    DialogueSession,
    DialogueTurn,
    EndpointSettings,
    Participant,
    PlaybackController,
    PlaybackState,
    ProtocolError,
    SSEParser,
    build_chat_payload,
    build_messages,
    chat_completions_url,
    models_url,
    normalize_base_url,
    parse_non_streaming_response,
    parse_stream_event,
    render_transcript,
    unique_speaker_ids,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
        ("https://ai.example.test/proxy", "https://ai.example.test/proxy/v1"),
        ("https://ai.example.test/PROXY/V1", "https://ai.example.test/PROXY/V1"),
    ],
)
def test_normalize_base_url(value, expected):
    assert normalize_base_url(value) == expected


@pytest.mark.parametrize("value", ["", "hello", "ftp://server/models", "https://", "http://user:secret@host/v1", "http://host/v1?token=x", "http://host:bad/v1"])
def test_normalize_base_url_rejects_malformed_or_credential_urls(value):
    with pytest.raises(ValueError):
        normalize_base_url(value)


def test_route_helpers_and_payload_use_exact_model_and_not_api_key():
    endpoint = EndpointSettings("Mira", "http://gpu:11434/", "llama3.2", "do-not-send-in-json")
    assert chat_completions_url(endpoint.base_url) == "http://gpu:11434/v1/chat/completions"
    assert models_url(endpoint.base_url) == "http://gpu:11434/v1/models"
    messages = [{"role": "user", "content": "hi"}]
    assert build_chat_payload(endpoint, messages) == {
        "model": "llama3.2",
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }
    assert "do-not-send-in-json" not in repr(endpoint)


def test_unique_speaker_ids_are_readable_and_collision_free():
    identifiers = unique_speaker_ids(("Atlas", "Atlas", "Café!", "", "Atlas-2"))
    assert identifiers == ("atlas", "atlas-2", "caf", "participant-4", "atlas-2-2")
    assert len(set(identifiers)) == len(identifiers)


def test_build_messages_includes_shared_turns_and_names_current_respondent():
    prior = [DialogueTurn(0, "Atlas", "I like gardening.", "atlas")]
    messages = build_messages("What is a relaxing hobby?", prior, ("Atlas", "Birch"), 1)
    assert messages[0]["role"] == "system"
    assert "You are Birch" in messages[0]["content"]
    assert "birch" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "The human's opening prompt: What is a relaxing hobby?"}
    assert {"role": "assistant", "content": "I like gardening.", "name": "atlas"} in messages
    final_instruction = messages[-1]["content"]
    assert "Birch's turn (birch)" in final_instruction
    assert "Atlas [atlas]" in final_instruction
    assert "Birch [birch]" in final_instruction


def test_build_messages_gives_duplicate_display_names_unique_message_ids():
    prior = [
        DialogueTurn(0, "Echo", "first voice", "echo"),
        DialogueTurn(1, "Echo", "second voice", "echo-2"),
    ]
    messages = build_messages("Compare yourselves.", prior, ("Echo", "Echo"), 0)
    names = [message.get("name") for message in messages if message["role"] == "assistant"]
    assert names == ["echo", "echo-2"]
    assert "Echo [echo]" in messages[-1]["content"]
    assert "Echo [echo-2]" in messages[-1]["content"]


def test_build_messages_sends_each_turn_once_so_payload_grows_linearly():
    turns = [
        DialogueTurn(i % 2, "Atlas" if i % 2 == 0 else "Birch", f"msg-{i:02d}-" + "x" * 20, f"speaker-{i % 2}")
        for i in range(10)
    ]
    small = build_messages("Topic.", turns[:4], ("Atlas", "Birch"), 0)
    large = build_messages("Topic.", turns, ("Atlas", "Birch"), 0)
    assert small.count({"role": "user", "content": "The human's opening prompt: Topic."}) == 1
    for turn in turns:
        assert sum(message["content"].count(turn.content) for message in large) == 1
    assert len([message for message in large if message["role"] == "assistant"]) == 10
    assert len(json.dumps(large)) > len(json.dumps(small))


def test_build_messages_caps_history_and_keeps_the_most_recent_turns():
    turns = [DialogueTurn(0, "Atlas", f"turn-{index:02d} " + "y" * 200, "atlas") for index in range(50)]
    messages = build_messages("Keep it short.", turns, ("Atlas", "Birch"), 1, max_history_characters=2_000)
    transcript = "\n".join(message["content"] for message in messages)
    assert "turn-49" in transcript
    assert "turn-00" not in transcript
    assert "earlier dialogue turn(s) omitted" in transcript
    assert "Keep it short." in transcript
    replay = [
        message
        for message in messages
        if message["role"] == "assistant" and message["name"] == "atlas"
    ]
    assert 1 <= len(replay) < 50
    assert sum(len(message["content"]) for message in replay) < 2_000


def test_custom_system_prompt_is_the_system_role_content_for_each_ordered_participant():
    session = DialogueSession(
        [
            Participant("Atlas", 1, "Answer with concise bullet points."),
            Participant("Birch", 2, "Respond like a curious botanist."),
        ],
        "Talk about gardens.",
        2,
    )
    assert session.next_turn().messages[0] == {
        "role": "system",
        "content": "Answer with concise bullet points.",
    }
    session.record_response("Gardens are peaceful.")
    assert session.next_turn().messages[0] == {
        "role": "system",
        "content": "Respond like a curious botanist.",
    }


def test_dialogue_session_sorts_turns_by_numeric_order_and_wraps_the_queue():
    session = DialogueSession(
        [
            Participant("Atlas", 30),
            Participant("Birch", 10),
            Participant("Cedar", 20),
        ],
        "start",
        4,
    )
    assert session.participant_names == ("Birch", "Cedar", "Atlas")
    assert session.turn_orders == (10, 20, 30)
    assert session.speaker_ids == ("birch", "cedar", "atlas")
    speakers = []
    for index in range(4):
        plan = session.next_turn()
        speakers.append(plan.speaker)
        assert plan.endpoint_index == index % 3
        assert plan.speaker_id == session.speaker_ids[index % 3]
        assert session.record_response(f"reply {index}") is (index < 3)
    assert speakers == ["Birch", "Cedar", "Atlas", "Birch"]
    assert session.model_turns == 4
    assert session.running is False


def test_equal_turn_orders_use_column_position_as_a_stable_tie_breaker():
    session = DialogueSession(
        [Participant("First", 5), Participant("Second", 5), Participant("Third", 1)],
        "start",
        3,
    )
    assert session.participant_names == ("Third", "First", "Second")


def test_user_interjections_are_user_messages_and_do_not_consume_model_turns():
    session = DialogueSession(
        [Participant("Atlas", 1), Participant("Cedar", 2)],
        "start",
        2,
    )
    assert session.record_response("first model reply") is True
    injected = session.inject_user_message("Please discuss water.")
    assert injected.role == "user"
    assert injected.speaker_id == "user"
    assert session.model_turns == 1
    assert session.running is True
    plan = session.next_turn()
    assert plan.speaker == "Cedar"
    assert {"role": "assistant", "content": "first model reply", "name": "atlas"} in plan.messages
    assert {"role": "user", "content": "Please discuss water.", "name": "user"} in plan.messages
    session.inject_user_message("And sunlight.")
    assert session.next_turn().speaker == "Cedar"
    assert session.running is True
    assert session.record_response("second model reply") is False
    assert session.model_turns == 2
    with pytest.raises(RuntimeError):
        session.inject_user_message("too late")
    with pytest.raises(ValueError):
        DialogueSession([Participant("A", 1), Participant("B", 2)], "start", 2).inject_user_message("   ")


def test_dialogue_session_replays_only_the_configured_history_budget():
    session = DialogueSession(
        [Participant("Atlas", 1), Participant("Birch", 2)],
        "start",
        6,
        max_history_characters=150,
    )
    for index in range(4):
        assert session.record_response(f"reply-{index} " + "z" * 40) is True
    messages = session.next_turn().messages
    transcript = "\n".join(message["content"] for message in messages)
    assert "reply-3" in transcript
    assert "reply-0" not in transcript
    assert "earlier dialogue turn(s) omitted" in transcript


def test_chat_payload_rejects_out_of_range_or_non_finite_temperature():
    messages = [{"role": "user", "content": "hi"}]
    for temperature in (-0.01, 2.01, float("nan"), float("inf")):
        endpoint = EndpointSettings("Atlas", "http://localhost:1234/v1", "model", temperature=temperature)
        with pytest.raises(ValueError, match="Temperature"):
            build_chat_payload(endpoint, messages)


def test_playback_controller_pauses_after_active_turn_then_resumes_exactly():
    player = PlaybackController()
    assert player.state is PlaybackState.IDLE
    assert player.wait_until_advance(0.01) is False
    player.start()
    assert player.state is PlaybackState.RUNNING
    assert player.can_advance()
    assert player.pause() is True
    assert player.state is PlaybackState.PAUSING
    assert not player.can_advance()
    with pytest.raises(RuntimeError):
        player.start()
    assert player.pause() is False
    assert player.mark_paused() is True
    assert player.state is PlaybackState.PAUSED
    assert player.is_paused

    results: list[bool] = []
    waiter = threading.Thread(target=lambda: results.append(player.wait_until_advance(2)))
    waiter.start()
    time.sleep(0.05)
    assert results == []
    assert player.resume() is True
    assert player.state is PlaybackState.RUNNING
    assert player.can_advance()
    waiter.join(1)
    assert results == [True]

    assert player.stop() is True
    assert player.state is PlaybackState.STOPPED
    assert not player.can_advance()
    assert player.wait_until_advance(0.01) is False
    assert player.stop() is False
    player.start()
    assert player.state is PlaybackState.RUNNING


def test_playback_controller_can_resume_while_a_turn_is_still_pausing():
    player = PlaybackController()
    player.start()
    player.pause()
    assert player.state is PlaybackState.PAUSING
    assert player.resume() is True
    assert player.state is PlaybackState.RUNNING
    assert player.can_advance()


def test_render_transcript_supports_markdown_and_plain_text_with_speaker_ids():
    turns = [
        DialogueTurn(0, "Atlas", "Hello from Atlas.", "atlas"),
        DialogueTurn(1, "User", "Please continue.", "user", "user"),
        DialogueTurn(0, "Birch", "Hello from Birch.", "birch"),
    ]
    markdown = render_transcript(turns, "Talk about the weather.", markdown=True)
    assert markdown.startswith("# Model Arena conversation")
    assert "**User (opening prompt):** Talk about the weather." in markdown
    assert "### Atlas (`atlas`)" in markdown
    assert "### User (`user`)" in markdown
    assert "Hello from Birch." in markdown
    text = render_transcript(turns, "Talk about the weather.", markdown=False)
    assert "MODEL ARENA CONVERSATION" in text
    assert "Atlas [atlas]:" in text
    assert "User [user]:" in text
    assert "Birch [birch]:" in text


def test_sse_parser_handles_fragmented_utf8_fields_and_boundaries():
    parser = SSEParser()
    wire = (
        ': keepalive\r\n'
        'event: chat.completion.chunk\r\n'
        'data: ' + json.dumps({"choices": [{"delta": {"content": "Olá friend"}}]}, ensure_ascii=False) + '\r\n'
        '\r\n'
        'data: [DONE]\r\n\r\n'
    ).encode("utf-8")
    utf8_char = "á".encode("utf-8")
    split_at = wire.index(utf8_char) + 1
    chunks = [wire[:split_at], wire[split_at : split_at + 4], wire[split_at + 4 :]]
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.feed(b"", final=True))
    assert events == ['{"choices": [{"delta": {"content": "Olá friend"}}]}', "[DONE]"]
    assert parse_stream_event(events[0]) == "Olá friend"
    assert parse_stream_event(events[1]) == ""


def test_sse_parser_preserves_multiline_data_and_ignores_comments():
    parser = SSEParser()
    assert parser.feed(b": comment\ndata: first\ndata: second\n\n", final=True) == ["first\nsecond"]


def test_non_streaming_response_supports_plain_text_and_text_parts():
    assert parse_non_streaming_response(
        b'{"choices":[{"message":{"content":"hi"}}]}'
    ) == "hi"
    parts = {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "!"}]}}]}
    assert parse_non_streaming_response(json.dumps(parts).encode()) == "hi!"


@pytest.mark.parametrize("value", [b"not json", b"{}", b'{"choices":[]}', b'{"error":"bad request"}'])
def test_non_streaming_response_rejects_invalid_or_failed_response(value):
    with pytest.raises(ProtocolError):
        parse_non_streaming_response(value)


def test_parse_stream_delta_tolerates_empty_choice_and_role_only_chunks():
    assert parse_stream_event('{"choices":[]}') == ""
    assert parse_stream_event('{"choices":[{"delta":{"role":"assistant"}}]}') == ""
    with pytest.raises(ProtocolError):
        parse_stream_event("definitely not json")


def test_dialogue_alternates_tracks_shared_history_and_stops_at_turn_limit():
    session = DialogueSession(
        [Participant("Atlas", 1), Participant("Birch", 2)],
        "Talk about art.",
        3,
    )
    first = session.next_turn()
    assert first.endpoint_index == 0
    assert first.speaker == "Atlas"
    assert "Talk about art." in first.messages[1]["content"]
    assert session.record_response("  Paint with watercolor.  ") is True
    second = session.next_turn()
    assert second.endpoint_index == 1
    assert second.speaker == "Birch"
    assert {"role": "assistant", "content": "Paint with watercolor.", "name": "atlas"} in second.messages
    assert session.record_response("Try drawing together.") is True
    assert session.next_turn().endpoint_index == 0
    assert session.record_response("Good idea!") is False
    assert len(session.turns) == 3
    with pytest.raises(RuntimeError):
        session.next_turn()


def test_dialogue_validation_and_cancellation():
    with pytest.raises(ValueError):
        DialogueSession([Participant("Only one", 1)], "hello", 2)
    with pytest.raises(ValueError):
        DialogueSession([Participant("a", 1), Participant("b", 2)], " \n", 2)
    with pytest.raises(ValueError):
        DialogueSession([Participant("a", 1), Participant("b", 2)], "hello", 0)
    with pytest.raises(ValueError):
        DialogueSession([Participant("a", 0), Participant("b", 2)], "hello", 2)
    session = DialogueSession([Participant("a", 1), Participant("b", 2)], "hello", 2)
    session.stop()
    with pytest.raises(RuntimeError):
        session.record_response("no longer in the loop")
    with pytest.raises(RuntimeError):
        session.inject_user_message("nope")


def test_dialogue_rejects_empty_completed_response():
    session = DialogueSession([Participant("a", 1), Participant("b", 2)], "hello", 2)
    with pytest.raises(ProtocolError, match="empty"):
        session.record_response(" \n ")
