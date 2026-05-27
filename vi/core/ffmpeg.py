"""Audio mixing (moviepy) + dual-subtitle burn-in (ffmpeg).
Preserves the make.py Windows workaround: copy SRTs to colon-free relative paths."""
import os
import shutil
import subprocess
from pathlib import Path

from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip


def _resolve_font(font_name: str, cfg: dict) -> str:
    """Use Linux fallback fonts when Windows-only fonts aren't available."""
    if os.name == "nt":
        return font_name
    fonts_section = cfg.get("fonts", {})
    linux_map = {
        "Arial": fonts_section.get("linux_subtitle_original", "DejaVu Sans"),
        "Microsoft YaHei": fonts_section.get("linux_subtitle_narration", "Noto Sans CJK SC Regular"),
    }
    return linux_map.get(font_name, font_name)


def _style(cfg: dict, font_cfg: dict | None = None) -> str:
    effective_font = _resolve_font(cfg["font"], font_cfg or cfg)
    parts = [
        f"FontName={effective_font}",
        f"FontSize={cfg['size']}",
        f"PrimaryColour={cfg['primary']}",
        f"OutlineColour={cfg['outline_color']}",
        f"Outline={cfg['outline']}",
        f"Shadow={cfg['shadow']}",
        f"MarginV={cfg['margin_v']}",
        f"Alignment={cfg['alignment']}",
    ]
    if cfg.get("margin_l") is not None:
        parts.append(f"MarginL={cfg['margin_l']}")
    if cfg.get("margin_r") is not None:
        parts.append(f"MarginR={cfg['margin_r']}")
    if cfg.get("bold"):
        parts.append(f"Bold={cfg['bold']}")
    return ",".join(parts)


def build_video(
    project_dir: Path,
    video_path: Path,
    voice_path: Path,
    bgm_path: Path,
    srt_original: Path,
    srt_narration: Path,
    output_path: Path,
    style_original: dict,
    style_narration: dict,
    voice_gain: float = 1.4,
    ambient_gain: float = 0.15,
    bgm_gain: float = 0.15,
    crf: int = 18,
    codec_v: str = "libx264",
    codec_a: str = "aac",
) -> None:
    video = VideoFileClip(str(video_path))
    duration = video.duration

    voice_clip = AudioFileClip(str(voice_path))
    voice_end = min(voice_clip.duration, duration)
    voice = voice_clip.with_volume_scaled(voice_gain).subclipped(0, voice_end)

    ambient = video.audio.with_volume_scaled(ambient_gain) if video.audio else None

    bgm_clip = AudioFileClip(str(bgm_path))
    bgm_end = min(bgm_clip.duration, duration)
    bgm = bgm_clip.with_volume_scaled(bgm_gain).subclipped(0, bgm_end)

    tracks = [voice, bgm]
    if ambient is not None:
        tracks.append(ambient)
    audio = CompositeAudioClip(tracks)
    final = video.with_audio(audio).with_duration(duration)

    temp_out = project_dir / "_temp_video.mp4"
    final.write_videofile(str(temp_out), codec=codec_v, audio_codec=codec_a)

    print("Burning dual subtitles...")
    tmp_orig = project_dir / "_orig_sub.srt"
    tmp_narr = project_dir / "_narr_sub.srt"
    shutil.copy2(srt_original, tmp_orig)
    shutil.copy2(srt_narration, tmp_narr)

    style_o = _style(style_original)
    style_n = _style(style_narration)
    vf = (
        f"subtitles={tmp_orig.name}:force_style='{style_o}',"
        f"subtitles={tmp_narr.name}:force_style='{style_n}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_out.name,
        "-vf", vf,
        "-c:v", codec_v,
        "-c:a", codec_a,
        "-crf", str(crf),
        output_path.name,
    ]
    subprocess.run(cmd, check=True, cwd=str(project_dir))

    temp_out.unlink(missing_ok=True)
    tmp_orig.unlink(missing_ok=True)
    tmp_narr.unlink(missing_ok=True)
    print(f"Done! {output_path}")
