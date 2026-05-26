"""`init` — create project skeleton, copy/link source video, generate project.toml."""
import argparse
import datetime
import re
import shutil
from pathlib import Path

from .. import paths


def add_parser(sub):
    p = sub.add_parser("init", help="Create a new project")
    p.add_argument("name", help="Project name (ASCII letters/digits/_-)")
    p.add_argument("--video", required=True, help="Path to source video")
    p.add_argument("--title", default="", help="Display title (Chinese ok)")
    p.add_argument(
        "--link", action="store_true",
        help="Symlink source video instead of copying (saves disk)"
    )
    p.set_defaults(func=run)
    return p


NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def run(args: argparse.Namespace) -> int:
    if not NAME_RE.match(args.name):
        print(f"Invalid project name: {args.name!r}")
        print("Use only ASCII letters, digits, _ and -")
        return 2

    src = Path(args.video).expanduser().resolve()
    if not src.exists():
        print(f"Source video not found: {src}")
        return 2

    pdir = paths.project_path(args.name)
    if pdir.exists():
        print(f"Project already exists: {pdir}")
        return 2
    pdir.mkdir(parents=True)

    target = pdir / f"source{src.suffix}"
    if args.link:
        try:
            target.symlink_to(src)
        except OSError as e:
            print(f"Symlink failed ({e}); falling back to copy")
            shutil.copy2(src, target)
    else:
        shutil.copy2(src, target)

    title = args.title or args.name
    created = datetime.date.today().isoformat()
    toml = (
        f"[project]\n"
        f'name = "{args.name}"\n'
        f'title = "{title}"\n'
        f'created = "{created}"\n'
        f"\n"
        f"[source]\n"
        f'video = "{target.name}"\n'
        f"\n"
        f"[output]\n"
        f'filename = "output.mp4"\n'
    )
    (pdir / "project.toml").write_text(toml, encoding="utf-8")

    print(f"Created project: {pdir}")
    print(f"Source video: {target.name}")
    print()
    print("Next steps:")
    print(f"  python main.py transcribe {args.name}")
    print(f"  # edit projects/{args.name}/narration.txt")
    print(f"  # edit projects/{args.name}/narration_aligned.txt")
    print(f"  python main.py tts {args.name}")
    print(f"  python main.py build {args.name}")
    return 0
