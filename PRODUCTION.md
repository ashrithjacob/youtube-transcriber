# Productionizing yt-transcriber: from local CLI to a free-to-use website

Goal: a public web app where a user pastes a YouTube URL and gets a transcript (with or without timestamps), at near-zero marginal cost to the operator — **no paywalls** — by running inference on the **user's own device** (WebGPU/WASM) instead of a GPU server.

This plan maps each module of the current CLI onto its production counterpart. The code was structured for exactly this migration, so each phase replaces one module without touching the others.

## Target architecture

```
                 ┌────────────────────────────────────────────────┐
                 │                  Browser (client)              │
  YouTube URL ──▶│  React/Vite UI                                 │
                 │    │                                           │
                 │    ├─ 1. ask server: cached transcript? ───────┼──▶ KV / object store
                 │    ├─ 2. ask server: existing YT captions? ────┼──▶ thin server
                 │    └─ 3. else: fetch 16 kHz PCM audio ─────────┼──▶ thin server ──▶ yt-dlp + ffmpeg
                 │            │                                   │        (Cloud Run / Fly.io container)
                 │            ▼                                   │
                 │   Web Worker: parakeet.js                      │
                 │   (WebGPU encoder + WASM decoder,              │
                 │    WASM-only fallback, IndexedDB model cache)  │
                 │            │                                   │
                 │            ▼                                   │
                 │   {start, end, text} segments                  │
                 │            │                                   │
                 │            ▼                                   │
                 │   format toggle: plain / [MM:SS] ──▶ .txt      │
                 └────────────────────────────────────────────────┘

  Static assets + model weights served from a CDN. No inference servers anywhere.
```

### Module → production mapping

| CLI module | Production component | What changes |
|---|---|---|
| `yt_transcriber/acquire.py` | **Thin acquisition service**: containerized FastAPI + yt-dlp + ffmpeg | Same logic, wrapped in an HTTP endpoint that streams PCM instead of writing a file |
| `yt_transcriber/asr.py` | **parakeet.js in a Web Worker** (client-side) | Same model family (Parakeet TDT 0.6B v3), same 16 kHz PCM input, same segment output |
| `yt_transcriber/segments.py` | `{start: number, end: number, text: string}` in TypeScript | The contract is the whole point — it survives the port unchanged |
| `yt_transcriber/formats.py` | ~30 lines of TypeScript in the frontend | 1:1 port of two pure functions |
| `yt_transcriber/cli.py` | The React UI | Orchestration role moves from CLI to frontend |

### Cost model (why this stays paywall-free)

- **Inference: $0.** The user's GPU/CPU does the work via parakeet.js.
- **Static hosting + model weights: ~$0.** CDN (Cloudflare Pages/R2 has free egress).
- **The only metered cost is acquisition egress** — audio bytes streamed from the thin server (~1 MB/min of video as 16 kbps Opus, more if you ship WAV; see Phase 1). If anything ever needs limiting, limit this — never the transcription.
- **Captions fast path (Phase 3) makes most requests cost ~nothing** — a caption fetch is a few KB.

---

## Phase 1 — Acquisition service (the thin server)

Turn `acquire.py` into a stateless container.

**Endpoints**

- `GET /api/meta?v=<video_id>` → `{title, duration, hasCaptions, captionLangs}` (yt-dlp `extract_info` without download)
- `GET /api/audio?v=<video_id>` → streams audio, transcoded to **16 kHz mono**. Ship **Opus/WebM (~16–24 kbps)** rather than raw WAV — ~10× less egress; the browser decodes to Float32 PCM for free with `AudioContext.decodeAudioData`/`OfflineAudioContext`.
- `GET /api/captions?v=<video_id>&lang=en` → existing YouTube caption track normalized to `{start, end, text}[]` (Phase 3, but design the route now)

**Implementation notes**

- FastAPI (or Hono/Node — anything thin), `yt-dlp` as a library, ffmpeg subprocess with `-ac 1 -ar 16000 -c:a libopus -b:a 24k -f webm` piped straight to the response — no temp files, start streaming as soon as ffmpeg produces bytes.
- Stateless: no disk writes that outlive a request. Horizontal scale = more containers.
- **Runtime, not edge**: yt-dlp + ffmpeg need real processes and a filesystem — Cloud Run / Fly.io / Railway / cheap VPS. Not Cloudflare Workers or Vercel edge functions.
- Validate input as a YouTube video ID (11-char regex), not a raw URL — kills SSRF and open-proxy abuse in one move.
- Timeouts + max-duration cap (e.g. refuse > 4 h videos initially).

**Ops reality: yt-dlp is a treadmill.** YouTube changes extraction regularly. Mitigations:
- Pin yt-dlp in the image but rebuild/redeploy on a **weekly schedule** (GitHub Action: bump yt-dlp → build → canary test against 3 known videos → deploy).
- Alert on extraction-failure rate; a spike means "update yt-dlp today".
- Consider a fallback resolver (e.g. a second extraction library) behind the same endpoint.

**Deliverable**: `server/` directory with Dockerfile; CLI gains a `--remote <url>` flag so the existing Python pipeline can consume the service — proving the boundary before any frontend exists.

## Phase 2 — Frontend with client-side inference

**Stack**: React + Vite + TypeScript, [parakeet.js](https://github.com/ysdede/parakeet.js) (ONNX Runtime Web).

**Inference setup**

```ts
// inside a Web Worker — never on the main thread
import { fromHub } from 'parakeet.js';
const model = await fromHub('parakeet-tdt-0.6b-v3', {
  backend: hasWebGPU ? 'webgpu' : 'wasm',
  encoderQuant: hasWebGPU ? 'fp16' : 'int8',   // shrink first-load download
  decoderQuant: 'int8',
});
const result = await model.transcribe(pcm, 16000, { returnTimestamps: true });
```

- **Backend selection**: feature-detect WebGPU (`navigator.gpu`), fall back to WASM. Chrome/Edge get WebGPU; Firefox/Safari get WASM-only (slower but working). Show which mode is active.
- **Model download UX is where the product lives or dies**: one-time "preparing model" progress bar with real byte counts; weights cached in **IndexedDB** so it's paid once per user. Serve ONNX files from your own CDN (R2/Pages), not HuggingFace hub — HF rate-limits and you want cache-control headers you own.
- **Long videos**: chunk PCM into ~60–120 s windows with 2–5 s overlap (client-side VAD via silero-onnx in the same worker, mirroring what `asr.py` does today), dedup at seams, and **stream results into the UI chunk-by-chunk** — the user reads the beginning while the rest transcribes.
- Keep the tab responsive: all decode + inference in the Worker; transferables for PCM buffers.

**Required headers** (multithreaded WASM needs cross-origin isolation):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Set on the static host; note this forces same-origin (or CORP-tagged) subresources — another reason to self-host model weights.

**UI**: URL input → progress (fetch / model / transcribe) → transcript view with a **timestamps toggle** (same segments, two renderings — port of `formats.py`) → copy / download `.txt`.

**Deliverable**: deployed static site talking to the Phase 1 service; works on Chrome/Edge (WebGPU) and acceptably everywhere else (WASM).

## Phase 3 — Captions fast path + transcript cache

Two features that make the median request instant and nearly free:

1. **Captions first**: on URL submit, hit `/api/captions`. If a human or ASR track exists, render it immediately and offer "Re-transcribe with Parakeet for higher quality" as an explicit button. Most videos never need inference at all.
2. **Transcript cache**: after a client finishes transcribing, POST segments to the server keyed by `video_id + model_version + quant`; store in KV/R2. Next visitor gets it instantly. Cache is content-addressed and immutable — trivial to serve from CDN.
   - Guard against poisoning: server spot-checks (duration sanity, language detection) or only caches after N identical submissions; or accept the risk initially and version the cache so it can be flushed.

## Phase 4 — Hardening, ops, launch

- **Abuse control**: rate-limit `/api/audio` by IP (it's the only metered cost); optional Turnstile on abuse spikes. No accounts, no paywall.
- **Observability**: track extraction failure rate (yt-dlp health), P50/P95 transcribe time by backend (webgpu vs wasm), model download abandonment. No transcript content in logs.
- **Browser matrix testing**: Chrome/Edge desktop (WebGPU), Firefox (WASM), Safari macOS (WASM), iOS Safari (WASM, memory-constrained — cap chunk size; this is the weakest target, revisit as WebGPU ships there).
- **Legal posture, decided before launch**:
  - Parakeet TDT v3 is **CC-BY-4.0** → attribution in the footer. parakeet.js/ONNX Runtime are permissively licensed.
  - **YouTube ToS prohibit stream extraction outside the official API.** The captions fast path is lower-risk; the audio path is where exposure sits. Decide the stance (personal tool vs. public product), keep the acquisition service isolated so it can be geo-fenced, throttled, or swapped for user-side file upload if it ever must be.
- **Graceful degradation ladder**: cached transcript → YouTube captions → client Parakeet (WebGPU) → client Parakeet (WASM) → "paste an audio file instead" (upload path needs no acquisition server at all and is ToS-clean).

## Suggested layout after migration

```
youtube-transcriber/
├── yt_transcriber/     # existing Python CLI (kept working throughout)
├── server/             # Phase 1: FastAPI + yt-dlp + ffmpeg, Dockerfile
├── web/                # Phase 2+: Vite app, worker/, parakeet.js
└── infra/              # deploy configs, weekly yt-dlp bump workflow
```

## Milestones & exit criteria

| Phase | Exit criterion |
|---|---|
| 1 | CLI transcribes via `--remote` against the deployed container; weekly yt-dlp bump pipeline green |
| 2 | Stranger on a random Chrome laptop pastes a URL and downloads a transcript with zero server inference |
| 3 | ≥ 70 % of requests served from captions or cache; P50 time-to-first-text < 3 s for cached/captioned videos |
| 4 | Browser matrix passes; egress cost per 1000 transcriptions known and acceptable; attribution + ToS stance documented |
