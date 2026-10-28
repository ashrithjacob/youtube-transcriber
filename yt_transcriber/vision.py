"""Visual track: keyframes + speech -> VisualNotes describing the screen.

`VisionClient` is the provider boundary (one adapter per VLM provider);
`ScreenDescriber` is provider-agnostic and owns prompting, the speech
context window for each frame, and concurrency.
"""

from __future__ import annotations

import base64
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from yt_transcriber.keyframes import Keyframe
from yt_transcriber.segments import Segment, VisualNote
from yt_transcriber.settings import TranscriberSettings

NOTHING = "NONE"

_PROMPT = """\
This is a frame from a video. While it was on screen, the speaker said:
\"\"\"{speech}\"\"\"

Transcribe and describe what is on screen so someone reading only the
transcript doesn't miss it:
- Slide or page text: transcribe it faithfully, keeping the structure (titles, bullets).
- Code, terminal output, or config: reproduce it verbatim in a fenced code block.
- Diagrams, charts, UI: describe the structure and what it shows in a few sentences,
  using the speech above to name things correctly.
Be concise. No preamble, and don't repeat the speech.
If the frame only shows a person, camera shot, or nothing informative, reply exactly: {nothing}"""

_MAX_SPEECH_CHARS = 1500


class VisionClient(ABC):
    """Sends one image plus a prompt to a vision-language model."""

    @abstractmethod
    def describe(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class OpenRouterVisionClient(VisionClient):
    _RETRY_STATUSES = {408, 429, 500, 502, 503, 504}

    def __init__(self, settings: TranscriberSettings, *, max_attempts: int = 4) -> None:
        if settings.openrouter_api_key is None:
            raise ValueError("OPENROUTER_API_KEY is not set (add it to .env)")
        self._model = settings.openrouter_vision_model
        self._max_attempts = max_attempts
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}",
        }
        if settings.openrouter_http_referer:
            headers["HTTP-Referer"] = settings.openrouter_http_referer
        if settings.openrouter_x_title:
            headers["X-Title"] = settings.openrouter_x_title
        self._http = httpx.Client(
            base_url=settings.openrouter_base_url,
            headers=headers,
            timeout=settings.openrouter_timeout_s,
        )

    def describe(self, image_path: Path, prompt: str) -> str:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
        }
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._http.post("/chat/completions", json=payload)
            except httpx.TransportError:
                if attempt == self._max_attempts:
                    raise
            else:
                if resp.status_code not in self._RETRY_STATUSES or attempt == self._max_attempts:
                    resp.raise_for_status()
                    return (resp.json()["choices"][0]["message"]["content"] or "").strip()
            time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()


def _speech_between(segments: list[Segment], start: float, end: float) -> str:
    text = " ".join(s.text for s in segments if s.end > start and s.start < end)
    return text[:_MAX_SPEECH_CHARS] or "(no speech)"


class ScreenDescriber:
    def __init__(self, client: VisionClient, *, concurrency: int = 8) -> None:
        self._client = client
        self._concurrency = concurrency

    def describe(
        self,
        frames: list[Keyframe],
        segments: list[Segment],
        duration: float,
    ) -> list[VisualNote]:
        """One note per informative frame; each spans until the next frame."""
        ends = [f.time for f in frames[1:]] + [max(duration, frames[-1].time if frames else 0)]

        def run(i: int) -> VisualNote | None:
            frame, end = frames[i], ends[i]
            prompt = _PROMPT.format(
                speech=_speech_between(segments, frame.time, end), nothing=NOTHING
            )
            text = self._client.describe(frame.path, prompt)
            if not text or text.strip().strip(".").upper() == NOTHING:
                return None
            return VisualNote(start=frame.time, end=end, text=text)

        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            notes = list(pool.map(run, range(len(frames))))
        return [n for n in notes if n is not None]
