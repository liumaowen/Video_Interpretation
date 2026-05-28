"""A/B compare CosyVoice 2 (ModelScope) vs edge-tts on the same Chinese text.

Run from repo root:
    python experiments/cosyvoice_demo.py
    python experiments/cosyvoice_demo.py --voice 中文女
    python experiments/cosyvoice_demo.py --text "自定义文本"
    python experiments/cosyvoice_demo.py --from-project my_first       # load first 3 segs

First run downloads iic/CosyVoice2-0.5B (~2GB) to .cache/models/CosyVoice2-0.5B/.

Setup (one-time):
    pip install modelscope torch torchaudio hyperpyyaml onnxruntime soundfile
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
    # If you already cloned without --recursive:
    #   cd third_party/CosyVoice && git submodule update --init --recursive
    pip install -r third_party/CosyVoice/requirements.txt
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Real B站 narration sample, concatenated from projects/my_first/narration_aligned.txt
DEFAULT_TEXT = (
    "今天给大家带来一部末日生存大片。"
    "看这个男人，怀里抱着他的狗，门口站着他的妻子，画面很温馨。"
    "但这温馨背后，藏着巨大的悲伤——"
    "世界变了，只剩满地的废墟，损坏的车辆，还有空荡荡的直升机。"
    "他们的命运将何去何从？"
)


def setup_cosyvoice_path() -> Path | None:
    """Add the local CosyVoice clone *and* its Matcha-TTS submodule to sys.path.

    CosyVoice imports `matcha.*` directly from its bundled submodule at
    `third_party/Matcha-TTS`, so both paths must be importable.
    """
    cv_root = None
    for candidate in (REPO_ROOT / "third_party" / "CosyVoice", REPO_ROOT / "CosyVoice"):
        if candidate.exists():
            cv_root = candidate
            break
    if cv_root is None:
        return None

    matcha_dir = cv_root / "third_party" / "Matcha-TTS"
    if not matcha_dir.exists() or not (matcha_dir / "matcha").exists():
        raise SystemExit(
            f"Matcha-TTS submodule missing at {matcha_dir}.\n"
            f"Fix:\n"
            f"  cd {cv_root.relative_to(REPO_ROOT)} && git submodule update --init --recursive"
        )

    for p in (cv_root, matcha_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return cv_root


def load_text_from_project(name: str, n_segments: int = 5) -> str:
    """Concatenate the first n segments of a project's narration_aligned.txt."""
    aligned = REPO_ROOT / "projects" / name / "narration_aligned.txt"
    if not aligned.exists():
        raise SystemExit(f"Not found: {aligned}")
    blocks = re.split(r"\n\s*\n", aligned.read_text(encoding="utf-8").strip())
    texts = []
    for block in blocks[:n_segments]:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        texts.append(" ".join(lines[2:]))
    return "".join(texts)


def download_cosyvoice_model(model_dir: Path) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        return
    print(f"Downloading iic/CosyVoice2-0.5B → {model_dir} (one-time, ~2GB)")
    from modelscope import snapshot_download
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download("iic/CosyVoice2-0.5B", local_dir=str(model_dir))


def synth_cosyvoice(text: str, voice: str, model_dir: Path, out_path: Path) -> None:
    cv_path = setup_cosyvoice_path()
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
    except ImportError as e:
        raise SystemExit(
            f"CosyVoice import failed: {e}\n"
            "Setup:\n"
            "  git clone https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice\n"
            "  pip install -r third_party/CosyVoice/requirements.txt"
        )
    import torch
    import torchaudio

    download_cosyvoice_model(model_dir)

    print(f"Loading CosyVoice 2 from {model_dir.name} (cold start ~30s)…")
    model = CosyVoice2(str(model_dir), load_jit=False, load_trt=False, fp16=False)

    available = list(model.list_available_spks()) if hasattr(model, "list_available_spks") else []
    if available and voice not in available:
        print(f"  ⚠ voice '{voice}' not in pretrained set: {available}")
        print(f"  Falling back to: {available[0]}")
        voice = available[0]

    print(f"Synthesizing CosyVoice [voice={voice}, {len(text)} chars]…")
    chunks = []
    for piece in model.inference_sft(text, voice, stream=False):
        chunks.append(piece["tts_speech"])
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    torchaudio.save(str(out_path), audio, model.sample_rate)


async def synth_edge_tts(text: str, voice: str, out_path: Path) -> None:
    import edge_tts
    import subprocess

    print(f"Synthesizing edge-tts [voice={voice}]…")
    mp3_path = out_path.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text, voice)
    with open(mp3_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

    # Re-encode to 24kHz wav so the two outputs have matching sample rate
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "24000", str(out_path)],
        check=True, capture_output=True,
    )
    mp3_path.unlink(missing_ok=True)


def report(label: str, path: Path) -> None:
    if not path.exists():
        print(f"  [skipped] {label}")
        return
    size_kb = path.stat().st_size / 1024
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        dur = info.num_frames / info.sample_rate
        print(f"  {label:18}  {dur:5.1f}s  {size_kb:6.0f} KB  {info.sample_rate} Hz  → {path}")
    except Exception:
        print(f"  {label:18}  {size_kb:6.0f} KB  → {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", help="Text to synthesize (overrides --from-project and default)")
    p.add_argument("--from-project", help="Use first N segments of projects/<name>/narration_aligned.txt")
    p.add_argument("--segments", type=int, default=5,
                   help="With --from-project, how many segments to concatenate (default 5)")
    p.add_argument("--voice", default="中文男",
                   help="CosyVoice SFT preset: 中文男 / 中文女 / 粤语女 / 英文男 / 英文女 / 日语男 / 韩语女")
    p.add_argument("--edge-voice", default="zh-CN-YunjianNeural",
                   help="edge-tts voice for baseline")
    p.add_argument("--skip-cosy", action="store_true", help="Skip CosyVoice synthesis")
    p.add_argument("--skip-edge", action="store_true", help="Skip edge-tts baseline")
    p.add_argument("--model-dir", default=".cache/models/CosyVoice2-0.5B",
                   help="CosyVoice model cache dir (relative to repo root)")
    args = p.parse_args()

    # Resolve text source
    if args.text:
        text = args.text
    elif args.from_project:
        text = load_text_from_project(args.from_project, args.segments)
    else:
        text = DEFAULT_TEXT

    out_dir = REPO_ROOT / "experiments" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = REPO_ROOT / args.model_dir

    preview = text if len(text) <= 80 else text[:77] + "..."
    print(f"\nText ({len(text)} chars): {preview}\n")

    cosy_out = out_dir / f"cosyvoice_{args.voice}.wav"
    edge_out = out_dir / f"edge_{args.edge_voice}.wav"

    if not args.skip_cosy:
        synth_cosyvoice(text, args.voice, model_dir, cosy_out)
    if not args.skip_edge:
        asyncio.run(synth_edge_tts(text, args.edge_voice, edge_out))

    print(f"\n--- Output ---")
    report(f"CosyVoice/{args.voice}", cosy_out)
    report(f"edge-tts/{args.edge_voice}", edge_out)
    print(f"\nListen and compare WAVs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
