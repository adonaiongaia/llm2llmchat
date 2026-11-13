# LLM2LLMChat — Model Arena

A Linux desktop app for letting any number of locally hosted language models carry on one shared, ordered conversation. The app runs on your Linux desktop and contacts each model server directly over HTTP; nothing needs to be installed on the other machines.

## Requirements

- Linux and Python 3.10 or newer
- A model server for each participant with an OpenAI-compatible `GET /v1/models` and/or `POST /v1/chat/completions` endpoint
- A trusted LAN or VPN route from the desktop to each server
- Qt's Linux platform dependencies; PySide6 provides the Python bindings

Ollama, LM Studio, and llama.cpp can all provide OpenAI-compatible chat endpoints, but each server exposes its own settings, loaded model IDs, and network controls. This app deliberately targets the portable text-only subset rather than provider-specific features.

## Install and run

From the project root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[keyring]'
llm2llmchat
```

Once installed, you can skip the last two commands. To run without installing the console command, use `python -m llm2llmchat` with the virtual environment activated. `keyring` is optional; install with `python -m pip install -e .` if your Linux session does not offer a Secret Service-compatible credential store. API tokens are kept in memory for the session when a usable OS keyring is unavailable.

Run the checks with:

```bash
python -m pip install -e '.[test]'
pytest
```

## Build a standalone AppImage

The AppImage bundles Python, PySide6, and the app, so it runs on another Linux desktop without installing Python or the project. Build it once on this machine:

```bash
. .venv/bin/activate
python -m pip install -e '.[keyring,build]'
./packaging/build-appimage.sh
```

The finished bundle is `dist/LLM2LLMChat-0.1.0-x86_64.AppImage`. Make it executable and double-click it, or run it from a terminal:

```bash
chmod +x dist/LLM2LLMChat-*.AppImage
./dist/LLM2LLMChat-*.AppImage
```

AppImages mount themselves through FUSE 2. Ubuntu 24.04-based desktops (including Linux Mint 22) often have `libfuse2t64` installed already; if the bundle refuses to start with a `libfuse.so.2` error, either install that package or run it with `APPIMAGE_EXTRACT_AND_RUN=1 ./dist/LLM2LLMChat-*.AppImage`.

Settings and keyring tokens keep working because the bundle uses the same per-user Qt settings and Secret Service as the source install. PyInstaller is only needed for building; it is not a runtime dependency.

## Configure the arena

Each participant is one configuration column:

1. Set a display name and enter that server's base URL on the machine running the app. Example: `http://192.168.1.20:1234/v1`. A URL without `/v1` is normalized automatically.
2. Click **Fetch models** to test the endpoint and select a model ID, or type the exact ID expected by that server.
3. Set an optional API token only if your server requires it. The request carries it in the `Authorization: Bearer …` header; it is not put in URLs or chat JSON.
4. Set **Order**. Lower numbers speak first; equal orders fall back to the column position. The execution queue is rebuilt from these values each time a conversation starts.
5. Add more columns with **+ Add Participant**, or close a column with its **✕** button (or **Remove**) while the arena is idle. Every participant has its own model ID, server URL, system prompt, and temperature.

Enter an opening prompt, set the maximum number of model responses, and press **Start conversation**. Participants take strictly sequential turns in numeric order and wrap around after the last one. Every turn replays the shared text-only history once; once it reaches the history limit, the oldest turns are dropped so requests stay bounded. A slow local model may think for minutes before its first token, so each network operation waits up to 600 seconds before reporting a timeout.

## Playback and interjections

- **Pause** lets the active model finish its response, then freezes the loop before the next participant starts. The next participant has not been requested yet.
- **Resume** continues with exactly the participant whose turn was next when the loop froze.
- **Stop** cancels the active request and ends the conversation.
- **Interject Conversation** is enabled while the loop is paused. Type a message and press **Inject** to append it to the transcript and the shared history with the `User` role. The models see it when the loop resumes. An interjection does not consume one of the maximum model turns.

## Speaker identity and export

Every past model response is sent as an assistant message with a unique `name` identifier such as `atlas`, `birch`, or `cedar`. If display names repeat, later columns receive suffixed identifiers (`echo`, `echo-2`) so each model can tell exactly which participant wrote an earlier message. The unique identifiers also appear in exported conversations.

The **Export** button in the Conversation header writes the completed chat history as Markdown (`.md`/`.markdown`) or plain text (`.txt`), including the opening prompt, interjections, speaker names, and unique speaker IDs.

**Request timeout** sets how long one network read waits before the request is abandoned (default 600 seconds, so slow local prompt processing is not mistaken for a failure). **History limit** caps how many characters of earlier conversation are replayed with each request; without it, every turn would resend an ever-growing transcript. Both persist with the other settings.

## Let the desktop reach remote servers

A server that only listens on `127.0.0.1` is not reachable from the other computer. Configure its own bind address, firewall, and authentication controls to expose its HTTP API **only** to your desktop or a trusted VPN. Keep servers off the public internet. Prefer a VPN for traffic crossing networks you do not trust; prefer TLS and server-side authentication whenever an endpoint must cross an untrusted network. A LAN address by itself is not an authentication boundary.

Examples of common defaults:

- Ollama: `http://<server-host>:11434/v1`. The local API normally has no authentication; a placeholder API key used in some OpenAI-client examples does not add security.
- LM Studio: `http://<server-host>:1234/v1`. Enable its network-serving option and server authentication before allowing other machines to connect.
- llama.cpp: `http://<server-host>:<port>/v1`. Its server must bind to an address reachable by the desktop; configure its API key and TLS options as appropriate.

Do not assume a successful `GET /v1/models` means the server is protected. This application warns about unauthenticated HTTP endpoints but cannot inspect the server's firewall, access policy, or TLS configuration. Avoid entering a valuable API token over plain HTTP outside a trusted, encrypted network.

## Local storage and privacy

Application preferences (participant count, names, URLs, model IDs, turn orders, system prompts, temperatures, turn limit, request timeout, and history limit) use Qt's per-user settings storage. API tokens use the Linux desktop keyring when the optional `keyring` package and a working Secret Service are present; otherwise tokens remain in memory until the app closes. Token values are not printed by the app or included in ordinary settings. The app does not persist chat transcripts or send conversation data anywhere except the selected model endpoints. Each model request includes the opening prompt and the shared conversation history, with the oldest turns omitted once the size limit is reached. Exported transcripts are written only when you choose **Export**.
