"""`transcribe` — ASR on source video."""
import argparse
from pathlib import Path

from .. import config, paths


def configure(p: argparse.ArgumentParser):
    p.add_argument("name", nargs="?", help="Project name (omit if in project dir)")
    p.add_argument("--engine", choices=["py", "exe", "funasr"],
                   help="ASR engine (default from config)")
    p.add_argument("--model",
                   help="Whisper size (tiny/base/.../large-v3) or FunASR id "
                        "(e.g. iic/SenseVoiceSmall)")
    p.add_argument("--language", help="Source language code (default en)")
    p.add_argument("-f", "--force", action="store_true", help="Overwrite existing subtitle.srt")


def run(args: argparse.Namespace) -> int:
    pdir = paths.resolve_project(args.name)
    cfg = config.merged(pdir)
    engine = args.engine or config.get(cfg, "asr.engine", "py")
    language = args.language or config.get(cfg, "asr.language", "en")

    video_name = config.get(cfg, "source.video")
    if not video_name:
        print("project.toml missing [source].video")
        return 2
    video_path = pdir / video_name
    if not video_path.exists():
        print(f"Source video not found: {video_path}")
        return 2

    out_srt = pdir / "subtitle.srt"
    if out_srt.exists() and not args.force:
        print(f"{out_srt} already exists (use --force to overwrite)")
        return 0

    if engine == "py":
        from ..core.whisper_py import transcribe
        model = args.model or config.get(cfg, "asr.model", "tiny")
        transcribe(
            video_path, out_srt,
            model_size=model,
            language=language,
            device=config.get(cfg, "asr.device", "cpu"),
            compute_type=config.get(cfg, "asr.compute_type", "int8"),
        )
    elif engine == "funasr":
        from ..core.asr_funasr import transcribe
        model = args.model or config.get(cfg, "asr.funasr_model", "iic/SenseVoiceSmall")
        transcribe(
            video_path, out_srt,
            model_size=model,
            language=language,
            device=config.get(cfg, "asr.device", "cpu"),
            vad_model=config.get(cfg, "asr.funasr_vad_model", "fsmn-vad"),
            punc_model=config.get(cfg, "asr.funasr_punc_model", "ct-punc"),
        )
    else:
        from ..core.whisper_exe import transcribe
        model = args.model or config.get(cfg, "asr.model", "tiny")
        exe = paths.root() / config.get(cfg, "asr.exe_path")
        model_dir = paths.root() / config.get(cfg, "asr.exe_model_dir")
        json_path = transcribe(exe, video_path, pdir, model=model, language=language, model_dir=model_dir)
        # Normalize the JSON filename to source.json for the refine step
        target = pdir / "source.json"
        if json_path != target:
            if target.exists():
                target.unlink()
            json_path.rename(target)
        # Also produce a subtitle.srt from the JSON using refine for consistency
        from ..core.refine import refine
        refine(target, out_srt)

    return 0
