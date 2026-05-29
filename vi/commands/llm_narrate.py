"""`llm-narrate` — generate Chinese narration via LLM."""
import argparse

from .. import config, paths
from ..core.llm import generate_narration, generate_narration_srt


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--style", help="Style keywords (default from config)")
    p.add_argument("--length", type=int, help="Target character count (default from config)")
    p.add_argument("--model", help="LLM model name (default from config)")
    p.add_argument("--provider", choices=["anthropic", "openai_compatible"],
                   help="LLM provider (default from config)")
    p.add_argument("--base-url", help="OpenAI-compatible API base URL")
    p.add_argument("--align", action="store_true",
                   help="Also generate narration_aligned.txt draft (requires human review)")
    p.add_argument("--srt", action="store_true",
                   help="One-step mode: directly generate narration_aligned.txt (SRT)")
    p.add_argument("--visual", action="store_true",
                   help="Visual-anchored mode: generate narration per visual event "
                        "(requires visual_description.txt)")
    p.add_argument("--vision", action="store_true",
                   help="Analyze video frames with a vision model and include in prompt")
    p.add_argument("--vision-model", help="Vision model name (default from config or glm-4v-flash)")
    p.add_argument("--frame-interval", type=int, default=3,
                   help="Seconds between extracted key frames (default: 3)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing narration files")
    p.add_argument("--force-vision", action="store_true",
                   help="Force re-analyze video frames even if visual_description.txt exists")


def _get_video_duration_ms(subtitle_path) -> int:
    """Estimate video duration from subtitle.srt end time."""
    from ..core.srt import parse_srt
    entries = parse_srt(subtitle_path)
    if entries:
        return entries[-1][2]
    return 0


def _get_visual_description(pdir, cfg, video_title, base_url, api_key, args):
    """Get visual description: from cache or by running vision analysis."""
    # Check for cached visual_description.txt first
    cache_path = pdir / "visual_description.txt"
    if cache_path.exists() and not args.force_vision:
        print(f"Using cached visual description from {cache_path}", file=__import__("sys").stderr)
        return cache_path.read_text(encoding="utf-8").strip()

    # Find source video
    source_video = config.get(cfg, "source.video", "source.mp4")
    video_path = pdir / source_video
    if not video_path.exists():
        print(f"Source video not found: {video_path}. Skipping vision analysis.", file=__import__("sys").stderr)
        return ""

    from ..core.vision import describe_video
    vision_model = args.vision_model or config.get(cfg, "llm.vision_model", "glm-4v-flash")
    return describe_video(
        video_path=video_path,
        video_title=video_title,
        base_url=base_url,
        api_key=api_key,
        vision_model=vision_model,
        interval_sec=args.frame_interval,
    )


def _load_visual_description_file(pdir) -> str:
    """Load visual_description.txt if it exists."""
    vd_path = pdir / "visual_description.txt"
    if vd_path.exists():
        content = vd_path.read_text(encoding="utf-8").strip()
        if content:
            print(f"Loaded visual description from {vd_path}", file=__import__("sys").stderr)
            return content
    return ""


def run(args: argparse.Namespace) -> int:
    import sys

    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)

    subtitle = pdir / "subtitle.srt"
    if not subtitle.exists():
        print(f"subtitle.srt not found in {pdir}. Run `transcribe` first.")
        return 2

    provider = args.provider or config.get(cfg, "llm.provider", "openai_compatible")
    base_url = args.base_url or config.get(cfg, "llm.base_url", "")
    api_key = config.get(cfg, "llm.api_key", "")
    api_key_env = config.get(cfg, "llm.api_key_env", "OPENAI_API_KEY")
    model = args.model or config.get(cfg, "llm.model", "qwen2.5:7b")
    style = args.style or config.get(cfg, "llm.default_style", "")
    length = args.length or config.get(cfg, "llm.default_length", 800)
    video_title = config.get(cfg, "project.title", "") or config.get(cfg, "project.name", "")

    # Load visual description: from cache file or vision analysis
    visual_description = ""
    visual_description_path = pdir / "visual_description.txt"
    if visual_description_path.exists():
        visual_description = _load_visual_description_file(pdir)

    # Vision analysis (only if --vision flag)
    if args.vision:
        resolved_key = api_key or __import__("os").environ.get(api_key_env, "")
        if resolved_key:
            visual_description = _get_visual_description(
                pdir, cfg, video_title, base_url, resolved_key, args,
            )
        else:
            print("API key not available for vision analysis. Skipping.", file=sys.stderr)

    video_duration_ms = _get_video_duration_ms(subtitle)
    if video_duration_ms:
        print(f"Estimated video duration: {video_duration_ms / 1000:.1f}s", file=sys.stderr)

    # Visual-anchored mode: generate narration per visual event
    if args.visual:
        if not visual_description_path.exists():
            print("visual_description.txt not found. Run `llm-narrate --vision` first.", file=sys.stderr)
            return 2
        out = pdir / "narration_aligned.txt"
        if out.exists() and not args.force:
            print(f"{out} already exists (use --force to overwrite)")
            return 0
        from ..core.llm import generate_narration_visual
        generate_narration_visual(
            subtitle_path=subtitle,
            visual_description_path=visual_description_path,
            out_path=out,
            video_duration_ms=video_duration_ms,
            style=style,
            video_title=video_title,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            provider=provider,
            base_url=base_url,
        )
        return 0

    # One-step SRT mode
    if args.srt:
        out = pdir / "narration_aligned.txt"
        if out.exists() and not args.force:
            print(f"{out} already exists (use --force to overwrite)")
            return 0
        generate_narration_srt(
            subtitle_path=subtitle,
            out_path=out,
            style=style,
            video_title=video_title,
            visual_description=visual_description,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            provider=provider,
            base_url=base_url,
        )
        return 0

    # Legacy two-step mode
    out = pdir / "narration.txt"
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)")
        return 0

    generate_narration(
        subtitle_path=subtitle,
        out_path=out,
        style=style,
        length=length,
        video_title=video_title,
        visual_description=visual_description,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        provider=provider,
        base_url=base_url,
    )

    if args.align:
        from ..core.llm import align_narration
        align_out = pdir / "narration_aligned.txt"
        align_narration(
            narration_path=out,
            subtitle_path=subtitle,
            out_path=align_out,
            n_segments=config.get(cfg, "llm.n_segments", 3),
            video_duration_ms=video_duration_ms,
            visual_description_path=visual_description_path,
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            provider=provider,
            base_url=base_url,
        )
    return 0
