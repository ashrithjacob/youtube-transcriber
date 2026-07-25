"""Transcription: 16 kHz mono WAV -> list of timestamped Segments.

Runs NVIDIA Parakeet TDT 0.6B v3 via ONNX Runtime on the local CPU, with
Silero VAD splitting long audio into speech segments. This is the stage
that moves to the user's browser (parakeet.js: WebGPU encoder + WASM
decoder) in production — the same model family, the same Segment output
contract, so everything around it stays put.
"""

from __future__ import annotations

from pathlib import Path

import onnx_asr

from yt_transcriber.segments import Segment

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"

# Parakeet's attention window comfortably covers a few minutes of audio;
# capping VAD segments keeps peak memory flat on hour-long videos.
_MAX_SEGMENT_S = 60.0


class ParakeetTranscriber:
    def __init__(self, quantization: str | None = "int8") -> None:
        model = onnx_asr.load_model(MODEL_NAME, quantization=quantization)
        vad = onnx_asr.load_vad("silero")
        self._pipeline = model.with_vad(
            vad, max_speech_duration_s=_MAX_SEGMENT_S
        ).with_timestamps()

    def transcribe(self, wav_path: Path) -> list[Segment]:
        results = self._pipeline.recognize(str(wav_path))
        return [
            Segment(start=r.start, end=r.end, text=r.text.strip())
            for r in results
            if r.text.strip()
        ]
