"""Command-line entry point: URL in, .txt out."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from yt_transcriber import formats
from yt_transcriber.acquire import fetch_audio
from yt_transcriber.segments import Segment, VisualNote

if TYPE_CHECKING:
    from yt_transcriber.settings import TranscriberSettings


def _safe_filename(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "", title).strip()
    return re.sub(r"\s+", " ", name) or "transcript"


def _describe_visuals(
    url: str,
    workdir: Path,
    segments: list[Segment],
    duration: float,
    settings: TranscriberSettings,
) -> list[VisualNote]:
    """Fetch the video, pick keyframes, and have the VLM describe each one."""
    from yt_transcriber.acquire import fetch_video
    from yt_transcriber.keyframes import extract_keyframes
    from yt_transcriber.vision import OpenRouterVisionClient, ScreenDescriber

    print("Fetching video (<=720p) ...", file=sys.stderr)
    video = fetch_video(url, workdir, quiet=True)

    print("Extracting keyframes ...", file=sys.stderr)
    frames = extract_keyframes(
        video,
        workdir / "frames",
        scene_threshold=settings.vision_scene_threshold,
        max_gap_s=settings.vision_max_gap_s,
        dedup_distance=settings.vision_dedup_distance,
        max_frames=settings.vision_max_frames,
    )
    if not frames:
        return []

    print(
        f"Describing {len(frames)} frames with {settings.openrouter_vision_model} ...",
        file=sys.stderr,
    )
    client = OpenRouterVisionClient(settings)
    try:
        describer = ScreenDescriber(client, concurrency=settings.vision_concurrency)
        return describer.describe(frames, segments, duration)
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yt-transcriber",
        description="Transcribe a YouTube video to a .txt file with Parakeet TDT (runs locally).",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="prefix each line with its [MM:SS] start time (default: plain text)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="output .txt path (default: <video title>.txt in the current directory)",
    )
    parser.add_argument(
        "--visuals",
        action="store_true",
        help="also capture on-screen content (slides, code, diagrams) with an OpenRouter "
        "vision model and interleave it with the speech; needs OPENROUTER_API_KEY in .env",
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "fp32"],
        default="int8",
        help="model precision: int8 is smaller and faster, fp32 is highest fidelity (default: int8)",
    )
    args = parser.parse_args(argv)

    if args.visuals:
        from yt_transcriber.settings import TranscriberSettings

        settings = TranscriberSettings()
        if settings.openrouter_api_key is None:
            parser.error("--visuals needs OPENROUTER_API_KEY (set it in .env)")

    notes: list[VisualNote] = []
    with tempfile.TemporaryDirectory(prefix="yt-transcriber-") as tmp:
        print(f"Fetching audio for {args.url} ...", file=sys.stderr)
        audio = fetch_audio(args.url, Path(tmp), quiet=True)
        mins, secs = divmod(int(audio.duration), 60)
        print(f'Got "{audio.title}" ({mins}:{secs:02d}).', file=sys.stderr)

        print("Loading Parakeet model (downloads once, then cached) ...", file=sys.stderr)
        # Imported here so --help and URL errors stay instant.
        from yt_transcriber.asr import ParakeetTranscriber

        quant = None if args.quantization == "fp32" else args.quantization
        transcriber = ParakeetTranscriber(quantization=quant)

        print("Transcribing ...", file=sys.stderr)
        t0 = time.monotonic()
        segments = transcriber.transcribe(audio.wav_path)
        elapsed = time.monotonic() - t0

        if args.visuals and segments:
            notes = _describe_visuals(args.url, Path(tmp), segments, audio.duration, settings)

        title = audio.title

    if not segments:
        print("No speech detected in this video.", file=sys.stderr)
        return 1

    if args.visuals:
        text = formats.to_interleaved(segments, notes, timestamps=args.timestamps)
    elif args.timestamps:
        text = formats.to_timestamped(segments)
    else:
        text = formats.to_plain(segments)
    out_path = args.output or Path(f"{_safe_filename(title)}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(
        f"Done in {elapsed:.1f}s — {len(segments)} segments"
        + (f" and {len(notes)} screen notes" if args.visuals else "")
        + f" written to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
