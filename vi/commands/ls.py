"""`ls` — list projects and their completion stages."""
import argparse

from .. import paths

STAGES = [
    ("transcribe", "subtitle.srt"),
    ("refine", "source.json"),
    ("narration", "narration.txt"),
    ("aligned", "narration_aligned.txt"),
    ("tts", "voice.wav"),
    ("build", "output.mp4"),
]


def add_parser(sub):
    p = sub.add_parser("ls", help="List projects")
    p.add_argument("-l", "--long", action="store_true", help="Show stage completion")
    p.set_defaults(func=run)
    return p


def run(args: argparse.Namespace) -> int:
    projs = paths.list_projects()
    if not projs:
        print("No projects yet. Create one with: python main.py init <name> --video <path>")
        return 0
    if not args.long:
        for p in projs:
            print(p.name)
        return 0
    width = max(len(p.name) for p in projs)
    for p in projs:
        marks = []
        for label, fname in STAGES:
            ok = (p / fname).exists()
            marks.append(f"{label}{'+' if ok else '-'}")
        print(f"{p.name:<{width}}  [{' '.join(marks)}]")
    return 0
