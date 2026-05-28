"""`clean` — delete intermediate artifacts from a project."""
import argparse
import shutil

from .. import paths

INTERMEDIATE = ["_tts_tmp", "_temp_video.mp4", "_orig_sub.srt", "_narr_sub.srt",
                "voice.wav", "narration_subtitle.srt", "narration_preview.srt"]
DEEP_EXTRA = ["subtitle.srt", "subtitle_zh.srt", "source.json", "output.mp4"]


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--all", action="store_true",
                   help="Also delete subtitle.srt, source.json, output.mp4 (keeps narration*)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    targets = list(INTERMEDIATE)
    if args.all:
        targets.extend(DEEP_EXTRA)

    for name in targets:
        p = pdir / name
        if not p.exists():
            continue
        if args.dry_run:
            print(f"would remove {p}")
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"removed {p}")
    return 0
