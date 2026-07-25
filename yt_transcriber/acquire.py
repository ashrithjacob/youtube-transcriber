"""Audio acquisition: YouTube URL -> 16 kHz mono WAV.

This module is the local stand-in for the production "thin acquisition
server". It does I/O and light CPU only (download + transcode) and knows
nothing about transcription. When productionizing, this logic moves into a
containerized endpoint (yt-dlp + ffmpeg) that streams PCM to the browser;
nothing downstream of it changes.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AcquiredAudio:
    wav_path: Path
    """Path to 16 kHz mono 16-bit PCM WAV."""
    video_id: str
    title: str
    duration: float
    """Video duration in seconds (0 if unknown)."""


def _ffmpeg_exe() -> str:
    """Prefer system ffmpeg; fall back to the static binary shipped by imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch_audio(url: str, workdir: Path, *, quiet: bool = False) -> AcquiredAudio:
    """Download the audio-only stream for `url` and transcode it to 16 kHz mono WAV.

    Intermediate files are written to `workdir`, which the caller owns.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(workdir / "source.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": quiet,
        "no_warnings": quiet,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        source = Path(ydl.prepare_filename(info))

    wav_path = workdir / "audio.wav"
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)

    return AcquiredAudio(
        wav_path=wav_path,
        video_id=info.get("id", "unknown"),
        title=info.get("title", "untitled"),
        duration=float(info.get("duration") or 0),
    )
