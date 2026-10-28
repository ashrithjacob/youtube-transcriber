"""Output formatting: Segments -> plain or timestamped text.

Pure functions over the Segment contract; usable verbatim on a server or
ported 1:1 to the browser frontend.
"""

from __future__ import annotations

from yt_transcriber.segments import Segment, VisualNote


def format_timestamp(seconds: float) -> str:
    """Render seconds as MM:SS, or HH:MM:SS once the hour mark is passed."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def to_plain(segments: list[Segment]) -> str:
    """One paragraph per speech segment."""
    return "\n\n".join(s.text for s in segments) + "\n"


def to_timestamped(segments: list[Segment]) -> str:
    """One `[MM:SS] text` line per speech segment."""
    return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments) + "\n"


def to_interleaved(
    segments: list[Segment], notes: list[VisualNote], *, timestamps: bool = True
) -> str:
    """Speech and on-screen notes merged in time order.

    A screen note sorts before speech starting at the same moment, so the
    reader sees the slide, then hears it explained.
    """
    events: list[tuple[float, int, str]] = []
    for n in notes:
        body = "\n".join(f"    {line}" if line else "" for line in n.text.splitlines())
        events.append((n.start, 0, f"[SCREEN]\n{body}"))
    for s in segments:
        events.append((s.start, 1, s.text))
    events.sort(key=lambda e: (e[0], e[1]))

    if timestamps:
        lines = [f"[{format_timestamp(t)}] {text}" for t, _, text in events]
    else:
        lines = [text for _, _, text in events]
    return "\n\n".join(lines) + "\n"
