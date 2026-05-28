"""argparse router."""
import argparse
import sys

from .commands import init as cmd_init
from .commands import ls as cmd_ls


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vi", description="Video Interpretation — turn a movie trailer into a B站 commentary video"
    )
    sub = p.add_subparsers(dest="command", required=True)

    cmd_init.add_parser(sub)
    cmd_ls.add_parser(sub)

    # Lazy-loaded heavy commands (avoid importing whisper/moviepy at startup)
    for name, helptext in [
        ("transcribe", "Transcribe source video to English SRT (whisper)"),
        ("refine", "Re-split SRT from whisper JSON for finer cuts"),
        ("translate", "Translate subtitle.srt to Chinese subtitle_zh.srt via LLM"),
        ("llm-narrate", "Generate Chinese narration via Claude API"),
        ("align", "Add timestamps to narration.txt"),
        ("preview-srt", "Build a preview SRT from narration_aligned.txt (no TTS)"),
        ("tts", "Synthesize voice.wav + narration_subtitle.srt with edge-tts"),
        ("build", "Mix audio and burn dual subtitles into output.mp4"),
        ("all", "Run transcribe→refine→tts→build in one shot"),
        ("clean", "Delete intermediate artifacts"),
    ]:
        sp = sub.add_parser(name, help=helptext, add_help=False)
        sp.set_defaults(_lazy=name.replace("-", "_"))

    return p


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    # First pass: only parse the subcommand name; defer lazy parsing
    if not argv:
        parser.print_help()
        return 2
    cmd = argv[0]
    if cmd in {"-h", "--help"}:
        parser.print_help()
        return 0

    if cmd in {"init", "ls"}:
        args = parser.parse_args(argv)
        return args.func(args)

    # Lazy-load
    lazy_map = {
        "transcribe": "vi.commands.transcribe",
        "refine": "vi.commands.refine",
        "translate": "vi.commands.translate",
        "llm-narrate": "vi.commands.llm_narrate",
        "align": "vi.commands.align",
        "preview-srt": "vi.commands.preview_srt",
        "tts": "vi.commands.tts",
        "build": "vi.commands.build",
        "all": "vi.commands.all_cmd",
        "clean": "vi.commands.clean",
    }
    if cmd not in lazy_map:
        parser.print_help()
        return 2
    import importlib
    mod = importlib.import_module(lazy_map[cmd])
    # Build a fresh subparser owned by the lazy module so it controls all its flags
    sub_parser = argparse.ArgumentParser(prog=f"vi {cmd}")
    mod.configure(sub_parser)
    args = sub_parser.parse_args(argv[1:])
    return mod.run(args)
