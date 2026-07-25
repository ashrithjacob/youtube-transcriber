# yt-transcriber

Transcribe any YouTube video to a `.txt` file — with or without timestamps — using NVIDIA **Parakeet TDT 0.6B v3**, a lightweight, high-accuracy speech-to-text model. Everything runs locally on your CPU via ONNX Runtime: no API keys, no cloud inference, no cost per transcription.

## How it works

```
YouTube URL ──▶ [ acquire ]  yt-dlp + ffmpeg → 16 kHz mono WAV
                    │
                    ▼
               [ asr ]       Silero VAD splits speech → Parakeet TDT (ONNX) transcribes
                    │             each segment with start/end times
                    ▼
               [ formats ]   Segments → plain text  or  [MM:SS]-prefixed lines
```

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`
- ~2 GB free disk for the model cache (int8, downloaded once from Hugging Face)
- `ffmpeg` is **not** required system-wide — a static binary is bundled via `imageio-ffmpeg` (system ffmpeg is used if present)

## Setup

With `uv`:

```bash
git clone <this-repo>
cd youtube-transcriber
uv venv
uv pip install -e .
```

With plain `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Plain transcript → "<video title>.txt"
uv run yt-transcriber "https://www.youtube.com/watch?v=VIDEO_ID"

# With [MM:SS] timestamps on every line
uv run yt-transcriber -t "https://www.youtube.com/watch?v=VIDEO_ID"

# Choose the output file
uv run yt-transcriber -t -o transcript.txt "https://www.youtube.com/watch?v=VIDEO_ID"

# Highest fidelity (larger download, slower; default is int8)
uv run yt-transcriber --quantization fp32 "https://www.youtube.com/watch?v=VIDEO_ID"
```

(Without `uv`, activate the venv and call `yt-transcriber ...` directly.)

The first run downloads the Parakeet ONNX weights from Hugging Face and caches them (`~/.cache/huggingface`); subsequent runs start immediately.

### Output formats

**Plain** (default) — one paragraph per speech segment:

```
All right, so here we are in front of the elephants.

The cool thing about these guys is that they have really, really, really long trunks.
```

**Timestamped** (`-t` / `--timestamps`):

```
[00:00] All right, so here we are in front of the elephants.
[00:05] The cool thing about these guys is that they have really, really, really long trunks.
```

## Architecture — designed for the web deployment that comes next

This codebase is deliberately split along the same boundary the production web app will use, so productionizing is a port, not a rewrite:

| Module | Today (local CLI) | Production (website) |
|---|---|---|
| [`acquire.py`](yt_transcriber/acquire.py) | yt-dlp + ffmpeg → 16 kHz mono PCM on disk | **Thin acquisition server** (containerized yt-dlp + ffmpeg on Cloud Run / Fly.io) streaming PCM to the browser. I/O + light CPU only — cheap, stateless, horizontally scalable. |
| [`asr.py`](yt_transcriber/asr.py) | Parakeet TDT 0.6B v3 ONNX on local CPU, Silero VAD chunking | **Client-side** [parakeet.js](https://github.com/ysdede/parakeet.js) in a Web Worker — WebGPU encoder + WASM decoder, WASM-only fallback for browsers without WebGPU. Same model family, zero inference cost to the operator. |
| [`segments.py`](yt_transcriber/segments.py) | `Segment(start, end, text)` — the data contract between the two stages | Unchanged. Any backend that emits `{start, end, text}` plugs in. |
| [`formats.py`](yt_transcriber/formats.py) | Pure functions: segments → plain / timestamped text | Ported 1:1 to the frontend (trivial JS). |

Rules that keep the boundary clean:

- **`acquire` never imports `asr`** (and vice versa). They only meet in `cli.py`, which plays the role the frontend will play.
- The ASR stage consumes **16 kHz mono PCM** and produces **`Segment` lists** — exactly the interface parakeet.js exposes (`transcribe(pcm, 16000, { returnTimestamps: true })`).
- Formatting is pure and stateless, so "with/without timestamps" stays a client-side toggle over the same segment data.

**The full productionization plan — phases, endpoints, cost model, ops, and legal posture — lives in [PRODUCTION.md](PRODUCTION.md).**

## Troubleshooting

- **`Sign in to confirm you're not a bot` / 403 from YouTube** — yt-dlp is in a constant cat-and-mouse with YouTube. Update it: `uv pip install -U yt-dlp`.
- **Slow first run** — that's the one-time model download (~0.7 GB int8). Watch stderr for progress.
- **Out-of-memory on very long videos** — VAD already caps segments at 60 s, so memory stays flat; if you still hit limits, close other apps or use `--quantization int8` (the default).
# youtube-transcriberr
