"""`translate` — Translate English subtitle.srt into Chinese subtitle_zh.srt via LLM."""
import argparse

from .. import config, paths


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--model", help="LLM model id (default from config)")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite existing subtitle_zh.srt")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)

    src_srt = pdir / "subtitle.srt"
    if not src_srt.exists():
        print(f"subtitle.srt missing: {src_srt}")
        print("Run `transcribe` first.")
        return 2

    out_srt = pdir / "subtitle_zh.srt"
    if out_srt.exists() and not args.force:
        print(f"{out_srt} already exists (use --force to overwrite)")
        return 0

    from ..core.llm import translate_srt
    translate_srt(
        src_srt, out_srt,
        model=args.model or config.get(cfg, "llm.model", "qwen2.5:7b"),
        api_key=config.get(cfg, "llm.api_key", ""),
        api_key_env=config.get(cfg, "llm.api_key_env", "OPENAI_API_KEY"),
        provider=config.get(cfg, "llm.provider", "openai_compatible"),
        base_url=config.get(cfg, "llm.base_url", ""),
    )
    return 0
