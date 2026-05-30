"""`align` — produce narration_aligned.txt. Manual edit or LLM-assisted."""
import argparse

from .. import config, paths


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--llm", action="store_true", help="Use LLM to draft the alignment")
    p.add_argument("--vision", action="store_true",
                   help="Analyze video frames with a vision model and use visual events as time windows")
    p.add_argument("--vision-model", help="Vision model name (default from config or glm-4v-flash)")
    p.add_argument("--frame-interval", type=int,
                   help="Seconds between extracted key frames (default from config or 1)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing aligned file")
    p.add_argument("--force-vision", action="store_true",
                   help="Force re-analyze video frames even if visual_description.txt exists")


def _get_visual_description(pdir, cfg, args):
    """Get visual description: from cache or by running vision analysis."""
    cache_path = pdir / "visual_description.txt"
    if cache_path.exists() and not args.force_vision:
        print(f"Using cached visual description from {cache_path}")
        return cache_path.read_text(encoding="utf-8").strip()

    source_video = config.get(cfg, "source.video", "source.mp4")
    video_path = pdir / source_video
    if not video_path.exists():
        print(f"Source video not found: {video_path}. Skipping vision analysis.")
        return ""

    from ..core.vision import describe_video
    video_title = config.get(cfg, "project.title", "") or config.get(cfg, "project.name", "")
    api_key = config.get(cfg, "llm.api_key", "")
    api_key_env = config.get(cfg, "llm.api_key_env", "OPENAI_API_KEY")
    resolved_key = api_key or __import__("os").environ.get(api_key_env, "")
    if not resolved_key:
        print("API key not available for vision analysis. Skipping.")
        return ""

    base_url = config.get(cfg, "llm.base_url", "")
    vision_model = args.vision_model or config.get(cfg, "llm.vision_model", "glm-4v-flash")
    return describe_video(
        video_path=video_path,
        video_title=video_title,
        base_url=base_url,
        api_key=resolved_key,
        vision_model=vision_model,
        interval_sec=args.frame_interval or config.get(cfg, "vision.frame_interval", 1),
    )


def run(args: argparse.Namespace) -> int:
    import sys

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

    # Vision analysis: extract visual events for time-window alignment
    visual_description = ""
    visual_description_path = pdir / "visual_description.txt"
    if args.vision:
        visual_description = _get_visual_description(pdir, cfg, args)
        if visual_description and not visual_description_path.exists():
            visual_description_path.write_text(visual_description + "\n", encoding="utf-8")
            print(f"Wrote visual description to {visual_description_path}")

    if args.llm:
        subtitle = pdir / "subtitle.srt"
        if not subtitle.exists():
            print("subtitle.srt required for --llm alignment. Run `transcribe` first.")
            return 2
        from ..core.llm import align_narration
        from ..core.srt import parse_srt, ms_to_ts
        entries = parse_srt(subtitle)
        total_ms = entries[-1][2] if entries else 60000
        n = config.get(cfg, "llm.n_segments", 3)
        window = total_ms // n
        char_limits = []
        for i in range(n):
            end_ms = (i + 1) * window if i < n - 1 else total_ms
            duration_sec = (end_ms - i * window) // 1000
            char_limits.append(max(10, int(duration_sec * 3.5)))

        # Pre-flight check: is narration too long?
        narration_text = narration.read_text(encoding="utf-8").strip()
        narration_chars = sum(1 for c in narration_text if c.strip() and c not in "（）\n")
        total_limit = sum(char_limits)
        if narration_chars > total_limit:
            print(f"\n  ⚠ 解说稿 {narration_chars} 字，但 {total_ms//1000} 秒最多容纳 {total_limit} 字（{narration_chars - total_limit} 字超出）")
            print(f"  LLM 会尝试精简压缩，但效果可能不佳。建议重新生成更短的解说稿：\n")
            print(f"  python main.py llm-narrate {pdir.name} --length {total_limit}\n")

        align_narration(
            narration_path=narration,
            subtitle_path=subtitle,
            out_path=out,
            n_segments=n,
            char_limits=char_limits,
            video_duration_ms=total_ms,
            visual_description_path=visual_description_path if visual_description_path.exists() else None,
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
