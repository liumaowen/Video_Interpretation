"""`refine` — re-split subtitle.srt from source.json using sentence-aware cuts."""
import argparse

from .. import paths
from ..core.refine import refine


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing subtitle.srt")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    json_path = pdir / "source.json"
    if not json_path.exists():
        print(f"source.json not found in {pdir}.")
        print("Run `transcribe --engine exe` first, or place a whisper JSON at source.json.")
        return 2
    out_srt = pdir / "subtitle.srt"
    if out_srt.exists() and not args.force:
        print(f"{out_srt} already exists (use --force to overwrite)")
        return 0
    refine(json_path, out_srt)
    return 0
