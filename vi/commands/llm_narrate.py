"""`llm-narrate` — generate Chinese narration via Claude API."""
import argparse

from .. import config, paths
from ..core.llm import generate_narration


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--style", help="Style keywords (default from config)")
    p.add_argument("--length", type=int, help="Target character count (default from config)")
    p.add_argument("--model", help="Claude model (default from config)")
    p.add_argument("--align", action="store_true",
                   help="Also generate narration_aligned.txt draft (requires human review)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing narration.txt")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)

    subtitle = pdir / "subtitle.srt"
    if not subtitle.exists():
        print(f"subtitle.srt not found in {pdir}. Run `transcribe` first.")
        return 2

    out = pdir / "narration.txt"
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)")
        return 0

    generate_narration(
        subtitle_path=subtitle,
        out_path=out,
        style=args.style or config.get(cfg, "llm.default_style"),
        length=args.length or config.get(cfg, "llm.default_length", 800),
        model=args.model or config.get(cfg, "llm.model", "claude-opus-4-7"),
        api_key_env=config.get(cfg, "llm.api_key_env", "ANTHROPIC_API_KEY"),
    )

    if args.align:
        from ..core.llm import align_narration
        align_out = pdir / "narration_aligned.txt"
        align_narration(
            narration_path=out,
            subtitle_path=subtitle,
            out_path=align_out,
            n_segments=config.get(cfg, "llm.n_segments", 3),
            model=args.model or config.get(cfg, "llm.model", "claude-opus-4-7"),
            api_key_env=config.get(cfg, "llm.api_key_env", "ANTHROPIC_API_KEY"),
        )
    return 0
