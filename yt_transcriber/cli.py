"""Command-line entry point: URL in, .txt out."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

from yt_transcriber import formats
from yt_transcriber.acquire import fetch_audio


def _safe_filename(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "", title).strip()
    return re.sub(r"\s+", " ", name) or "transcript"


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
        "--quantization",
        choices=["int8", "fp32"],
        default="int8",
        help="model precision: int8 is smaller and faster, fp32 is highest fidelity (default: int8)",
    )
    args = parser.parse_args(argv)

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

        title = audio.title

    if not segments:
        print("No speech detected in this video.", file=sys.stderr)
        return 1

    text = formats.to_timestamped(segments) if args.timestamps else formats.to_plain(segments)
    out_path = args.output or Path(f"{_safe_filename(title)}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(
        f"Done in {elapsed:.1f}s — {len(segments)} segments written to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
