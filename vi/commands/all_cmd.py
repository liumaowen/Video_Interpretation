"""`all` — chain transcribe → refine → tts → build for one project."""
import argparse
import importlib

from .. import paths

STEPS = ["transcribe", "refine", "tts", "build"]
MODULES = {
    "transcribe": "vi.commands.transcribe",
    "refine": "vi.commands.refine",
    "tts": "vi.commands.tts",
    "build": "vi.commands.build",
}


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", help="Project name (required)")
    p.add_argument("--from", dest="from_step", choices=STEPS,
                   help="Start from this step (default: transcribe)")
    p.add_argument("--skip", default="",
                   help="Comma-separated steps to skip (e.g. transcribe,refine)")
    p.add_argument("-f", "--force", action="store_true", help="Pass --force to each step")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    started = args.from_step is None

    for step in STEPS:
        if not started:
            if step == args.from_step:
                started = True
            else:
                continue
        if step in skip:
            print(f"Skipping {step}")
            continue

        # refine needs source.json (only produced by --engine exe). Skip
        # cleanly for py/funasr engines so the pipeline doesn't abort.
        if step == "refine" and not (pdir / "source.json").exists():
            print("Skipping refine (no source.json — engine doesn't need it)")
            continue

        if step == "tts" and not (pdir / "narration_aligned.txt").exists():
            print(f"\nStopping: {pdir}/narration_aligned.txt missing.")
            print(f"Write it manually or run: python main.py llm-narrate {args.name} --align")
            return 1

        print(f"\n=== {step} ===")
        mod = importlib.import_module(MODULES[step])
        # Build a fresh Namespace each module's configure expects
        sub_parser = argparse.ArgumentParser()
        mod.configure(sub_parser)
        ns = sub_parser.parse_args([args.name] + (["--force"] if args.force else []))
        rc = mod.run(ns)
        if rc != 0:
            print(f"\nStep {step} failed (exit {rc})")
            return rc

    print(f"\nDone. → {pdir / 'output.mp4'}")
    return 0
