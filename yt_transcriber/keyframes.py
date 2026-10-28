"""Keyframe extraction: video file -> the few frames where the screen changed.

Talks and screencasts are mostly static, so instead of sampling at a fixed
rate we decode only codec keyframes, keep the ones where ffmpeg's scene
score jumps (plus one every `max_gap_s` as a fallback), then drop
near-duplicates with a perceptual hash so a speaker moving in front of the
same slide doesn't produce ten identical VLM calls.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from yt_transcriber.acquire import _ffmpeg_exe

_PTS_RE = re.compile(r"pts_time:\s*([0-9.]+)")


@dataclass(frozen=True)
class Keyframe:
    time: float
    """Capture time in seconds."""
    path: Path
    """JPEG on disk."""


def _dhash(path: Path) -> int:
    """64-bit difference hash: robust to compression noise and small motion."""
    with Image.open(path) as img:
        px = list(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def _subsample(frames: list[Keyframe], limit: int) -> list[Keyframe]:
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    return [frames[int(i * step)] for i in range(limit)]


def extract_keyframes(
    video_path: Path,
    out_dir: Path,
    *,
    scene_threshold: float = 0.3,
    max_gap_s: float = 60.0,
    dedup_distance: int = 6,
    max_frames: int = 300,
) -> list[Keyframe]:
    out_dir.mkdir(parents=True, exist_ok=True)
    select = (
        f"isnan(prev_selected_t)"
        f"+gt(scene\\,{scene_threshold})"
        f"+gte(t-prev_selected_t\\,{max_gap_s})"
    )
    cmd = [
        _ffmpeg_exe(),
        "-hide_banner",
        "-skip_frame", "nokey",
        "-i", str(video_path),
        "-vf", f"select='{select}',showinfo",
        "-fps_mode", "vfr",
        "-q:v", "3",
        str(out_dir / "frame_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    times = [float(t) for t in _PTS_RE.findall(proc.stderr)]
    paths = sorted(out_dir.glob("frame_*.jpg"))

    kept: list[Keyframe] = []
    last_hash: int | None = None
    for time, path in zip(times, paths):
        h = _dhash(path)
        if last_hash is not None and bin(h ^ last_hash).count("1") <= dedup_distance:
            continue
        kept.append(Keyframe(time=time, path=path))
        last_hash = h
    return _subsample(kept, max_frames)
