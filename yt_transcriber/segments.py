"""The data contract between the ASR stage and the output stage.

Any transcription backend (local ONNX today, parakeet.js in the browser
later) only has to produce a list of these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """One contiguous stretch of speech."""

    start: float
    """Start time in seconds."""
    end: float
    """End time in seconds."""
    text: str
    """Transcribed text for this stretch."""


@dataclass(frozen=True)
class VisualNote:
    """What was on screen for a stretch of the video."""

    start: float
    """Time in seconds the frame was captured."""
    end: float
    """Time in seconds the next distinct frame appears (or the video ends)."""
    text: str
    """VLM transcription/description of the frame."""
