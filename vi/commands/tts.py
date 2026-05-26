"""`tts` — synthesize voice.wav + narration_subtitle.srt from narration_aligned.txt."""
import argparse

from .. import config, paths
from ..core.tts_engine import run_sync


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--voice", help="edge-tts voice (default from config)")
    p.add_argument("--volume", help="Voice volume, e.g. +0%%")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing voice.wav")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)

    aligned = pdir / "narration_aligned.txt"
    if not aligned.exists():
        print(f"narration_aligned.txt not found in {pdir}.")
        print("Write one manually, or run `align` / `llm-narrate --align`.")
        return 2

    out_wav = pdir / "voice.wav"
    out_srt = pdir / "narration_subtitle.srt"
    if out_wav.exists() and not args.force:
        print(f"{out_wav} already exists (use --force to overwrite)")
        return 0

    run_sync(
        aligned,
        out_wav,
        out_srt,
        pdir / "_tts_tmp",
        voice=args.voice or config.get(cfg, "tts.voice"),
        volume=args.volume or config.get(cfg, "tts.volume", "+0%"),
        max_rate_boost=config.get(cfg, "tts.max_rate_boost", 40),
    )
    return 0
