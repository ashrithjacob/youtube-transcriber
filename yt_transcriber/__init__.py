"""Transcribe YouTube videos locally with NVIDIA Parakeet TDT (ONNX).

The package is split along the same boundary as the planned production
deployment:

- acquire:  YouTube URL -> 16 kHz mono PCM audio   (becomes the thin server)
- asr:      PCM audio   -> timestamped segments    (becomes parakeet.js in the browser)
- formats:  segments    -> plain / timestamped txt (shared, pure functions)
"""

from yt_transcriber.segments import Segment

__all__ = ["Segment"]
