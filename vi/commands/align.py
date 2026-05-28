"""`align` — produce narration_aligned.txt. Manual edit or LLM-assisted."""
import argparse

from .. import config, paths


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--llm", action="store_true", help="Use Claude to draft the alignment")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing aligned file")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)

    narration = pdir / "narration.txt"
    if not narration.exists():
        print(f"narration.txt not found in {pdir}. Write one or run `llm-narrate` first.")
        return 2

    out = pdir / "narration_aligned.txt"
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)")
        return 0

    if args.llm:
        subtitle = pdir / "subtitle.srt"
        if not subtitle.exists():
            print("subtitle.srt required for --llm alignment. Run `transcribe` first.")
            return 2
        from ..core.llm import align_narration
        from ..core.srt import parse_srt, ms_to_ts
        # Calculate per-segment character limits based on duration (~3.5 chars/sec for Chinese)
        entries = parse_srt(subtitle)
        total_ms = entries[-1][2] if entries else 60000
        n = config.get(cfg, "llm.n_segments", 3)
        window = total_ms // n
        char_limits = []
        for i in range(n):
            end_ms = (i + 1) * window if i < n - 1 else total_ms
            duration_sec = (end_ms - i * window) // 1000
            char_limits.append(max(10, int(duration_sec * 3.5)))
        align_narration(
            narration_path=narration,
            subtitle_path=subtitle,
            out_path=out,
            n_segments=n,
            char_limits=char_limits,
            model=config.get(cfg, "llm.model", "qwen2.5:7b"),
            api_key=config.get(cfg, "llm.api_key", ""),
            api_key_env=config.get(cfg, "llm.api_key_env", "OPENAI_API_KEY"),
            provider=config.get(cfg, "llm.provider", "openai_compatible"),
            base_url=config.get(cfg, "llm.base_url", ""),
        )
        return 0

    # Manual: write a template the user can fill in
    from ..core.srt import parse_srt, ms_to_ts
    entries = parse_srt(pdir / "subtitle.srt") if (pdir / "subtitle.srt").exists() else []
    total_ms = entries[-1][2] if entries else 60000
    n = config.get(cfg, "llm.n_segments", 3)
    window = total_ms // n
    template = []
    for i in range(n):
        start = i * window
        end = (i + 1) * window if i < n - 1 else total_ms
        template.append(str(i + 1))
        template.append(f"{ms_to_ts(start)} --> {ms_to_ts(end)}")
        template.append("（在此填入第 {} 段中文解说）".format(i + 1))
        template.append("")
    out.write_text("\n".join(template), encoding="utf-8")
    print(f"Wrote template to {out}")
    print(f"Edit it manually, then run: python main.py tts {pdir.name}")
    return 0
