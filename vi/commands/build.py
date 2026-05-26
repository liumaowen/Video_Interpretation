"""`build` — mix voice+ambient+bgm, burn dual subtitles into output.mp4."""
import argparse

from .. import config, paths
from ..core.ffmpeg import build_video


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--bgm", help="Override BGM path (relative to root or absolute)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing output.mp4")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)

    video_name = config.get(cfg, "source.video")
    video_path = pdir / video_name
    voice_path = pdir / "voice.wav"
    srt_orig = pdir / "subtitle.srt"
    srt_narr = pdir / "narration_subtitle.srt"

    missing = [p for p in [video_path, voice_path, srt_orig, srt_narr] if not p.exists()]
    if missing:
        print("Missing required files:")
        for p in missing:
            print(f"  {p}")
        return 2

    bgm_arg = args.bgm or config.get(cfg, "mix.bgm_path", "shared/bgm/default.mp3")
    bgm_path = (paths.root() / bgm_arg).resolve()
    if not bgm_path.exists():
        print(f"BGM not found: {bgm_path}")
        return 2

    out_name = config.get(cfg, "output.filename", "output.mp4")
    output_path = pdir / out_name
    if output_path.exists() and not args.force:
        print(f"{output_path} already exists (use --force to overwrite)")
        return 0

    build_video(
        project_dir=pdir,
        video_path=video_path,
        voice_path=voice_path,
        bgm_path=bgm_path,
        srt_original=srt_orig,
        srt_narration=srt_narr,
        output_path=output_path,
        style_original=cfg["subtitle_original"],
        style_narration=cfg["subtitle_narration"],
        voice_gain=config.get(cfg, "mix.voice_gain", 1.4),
        ambient_gain=config.get(cfg, "mix.ambient_gain", 0.15),
        bgm_gain=config.get(cfg, "mix.bgm_gain", 0.15),
        crf=config.get(cfg, "encode.crf", 18),
        codec_v=config.get(cfg, "encode.codec_v", "libx264"),
        codec_a=config.get(cfg, "encode.codec_a", "aac"),
    )
    return 0
