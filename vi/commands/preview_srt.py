"""`preview-srt` — char-proportional subtitle preview without running TTS."""
import argparse

from .. import paths
from ..core.preview_srt import build


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("-o", "--output", default="narration_preview.srt",
                   help="Output filename inside the project dir")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    aligned = pdir / "narration_aligned.txt"
    if not aligned.exists():
        print(f"narration_aligned.txt not found in {pdir}")
        return 2
    out = pdir / args.output
    build(aligned, out)
    return 0
