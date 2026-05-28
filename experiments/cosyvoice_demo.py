"""A/B compare CosyVoice 2 (ModelScope, zero-shot) vs edge-tts on the same text.

Run from repo root:
    python experiments/cosyvoice_demo.py                          # default: zero-shot w/ bundled prompt
    python experiments/cosyvoice_demo.py --from-project my_first
    python experiments/cosyvoice_demo.py --prompt-audio my.wav --prompt-text "我的样本台词"
    python experiments/cosyvoice_demo.py --sft-voice 中文男       # SFT mode (requires CosyVoice-300M-SFT)

CosyVoice2-0.5B is the *base* model — it only supports zero-shot voice cloning.
For built-in voice presets (中文男/中文女/...), use iic/CosyVoice-300M-SFT instead
and pass --sft-voice + --model-dir.

First run downloads iic/CosyVoice2-0.5B (~2GB) to .cache/models/CosyVoice2-0.5B/.

Setup (one-time):
    pip install modelscope torch torchaudio hyperpyyaml onnxruntime soundfile \\
                conformer lightning diffusers inflect WeTextProcessing openai-whisper
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice
    # If you cloned without --recursive:
    #   cd third_party/CosyVoice && git submodule update --init --recursive
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

# Bundled prompt that ships with the CosyVoice repo's asset/ directory
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
DEFAULT_PROMPT_REL = Path("asset") / "zero_shot_prompt.wav"


def setup_cosyvoice_path() -> Path | None:
    """Add the local CosyVoice clone *and* its Matcha-TTS submodule to sys.path."""
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


def download_cosyvoice_model(model_id: str, model_dir: Path) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        return
    print(f"Downloading {model_id} → {model_dir} (one-time)")
    from modelscope import snapshot_download
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, local_dir=str(model_dir))


def _load_cosyvoice_class(use_v2: bool):
    setup_cosyvoice_path()
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice
    except ImportError as e:
        raise SystemExit(
            f"CosyVoice import failed: {e}\n"
            "Setup:\n"
            "  pip install conformer lightning diffusers inflect WeTextProcessing openai-whisper\n"
            "  cd third_party/CosyVoice && git submodule update --init --recursive"
        )
    return CosyVoice2 if use_v2 else CosyVoice


def _patch_torchaudio_io():
    """torchaudio 2.10+ hard-routes all I/O through torchcodec, which needs FFmpeg
    shared libs at very specific versions (libavutil.so.57–60). Most envs don't
    have a matching ffmpeg, so we replace CosyVoice's load_wav with a soundfile-
    based implementation that bypasses the whole torchcodec stack.
    """
    import soundfile as sf
    import torch
    from cosyvoice.utils import file_utils

    def load_wav(wav, target_sr):
        speech_np, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        # soundfile gives (n_samples, channels); transpose → (channels, n_samples)
        speech = torch.from_numpy(speech_np.T).mean(dim=0, keepdim=True)
        if sr != target_sr:
            assert sr > target_sr, f"wav sample rate {sr} < target {target_sr}"
            import torchaudio
            speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)
        return speech

    file_utils.load_wav = load_wav


def _save_wav(path: Path, audio_tensor, sample_rate: int) -> None:
    """Save (channels, samples) tensor → WAV via soundfile (bypasses torchcodec)."""
    import soundfile as sf
    a = audio_tensor.detach().cpu()
    if a.ndim == 2 and a.shape[0] == 1:
        np_audio = a.squeeze(0).numpy()
    elif a.ndim == 2:
        np_audio = a.T.numpy()  # (samples, channels)
    else:
        np_audio = a.numpy()
    sf.write(str(path), np_audio, sample_rate)


def synth_cosyvoice(
    text: str,
    model_dir: Path,
    out_path: Path,
    *,
    sft_voice: str | None = None,
    prompt_audio: Path | None = None,
    prompt_text: str = DEFAULT_PROMPT_TEXT,
    use_v2: bool = True,
    model_id: str = "iic/CosyVoice2-0.5B",
) -> None:
    Klass = _load_cosyvoice_class(use_v2)
    _patch_torchaudio_io()
    import torch

    download_cosyvoice_model(model_id, model_dir)

    print(f"Loading {Klass.__name__} from {model_dir.name} (cold start ~30s)…")
    model = Klass(str(model_dir), load_jit=False, load_trt=False, fp16=False)

    if sft_voice:
        available = list(model.list_available_spks()) if hasattr(model, "list_available_spks") else []
        if not available:
            raise SystemExit(
                f"Model at {model_dir.name} has no built-in SFT speakers.\n"
                f"CosyVoice 2 base only supports zero-shot. For preset voices use:\n"
                f"  --model-dir .cache/models/CosyVoice-300M-SFT --sft-voice 中文男\n"
                f"  (will auto-download iic/CosyVoice-300M-SFT)"
            )
        if sft_voice not in available:
            raise SystemExit(f"Voice '{sft_voice}' not available. Choose from: {available}")
        print(f"Synthesizing SFT [voice={sft_voice}, {len(text)} chars]…")
        gen = model.inference_sft(text, sft_voice, stream=False)
    else:
        if prompt_audio is None or not prompt_audio.exists():
            raise SystemExit(
                f"Prompt audio not found: {prompt_audio}\n"
                f"Pass --prompt-audio <wav> --prompt-text '<台词>', "
                f"or make sure CosyVoice's bundled asset/zero_shot_prompt.wav exists."
            )
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech = load_wav(str(prompt_audio), 16000)
        print(f"Synthesizing zero-shot [prompt={prompt_audio.name}, {len(text)} chars]…")
        gen = model.inference_zero_shot(text, prompt_text, prompt_speech, stream=False)

    chunks = [piece["tts_speech"] for piece in gen]
    if not chunks:
        raise SystemExit("CosyVoice returned no audio chunks.")
    audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    _save_wav(out_path, audio, model.sample_rate)


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
        import soundfile as sf
        info = sf.info(str(path))
        print(f"  {label:24}  {info.duration:5.1f}s  {size_kb:6.0f} KB  {info.samplerate} Hz  → {path}")
    except Exception:
        print(f"  {label:24}  {size_kb:6.0f} KB  → {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", help="Text to synthesize (overrides --from-project)")
    p.add_argument("--from-project", help="Use first N segments of projects/<name>/narration_aligned.txt")
    p.add_argument("--segments", type=int, default=5,
                   help="With --from-project, segments to concatenate (default 5)")

    p.add_argument("--sft-voice", help="SFT preset (only works with CosyVoice-300M-SFT model)")
    p.add_argument("--prompt-audio",
                   help="Reference WAV for zero-shot cloning "
                        "(default: third_party/CosyVoice/asset/zero_shot_prompt.wav)")
    p.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT,
                   help=f"Text spoken in the prompt audio (default matches bundled prompt)")

    p.add_argument("--model-id", default="iic/CosyVoice2-0.5B",
                   help="ModelScope id (e.g. iic/CosyVoice2-0.5B, iic/CosyVoice-300M-SFT)")
    p.add_argument("--model-dir", default=None,
                   help="Local cache dir (default: .cache/models/<model_id>)")
    p.add_argument("--cosy-v1", action="store_true",
                   help="Use CosyVoice (v1) class instead of CosyVoice2 (needed for *-300M-* models)")

    p.add_argument("--edge-voice", default="zh-CN-YunjianNeural")
    p.add_argument("--skip-cosy", action="store_true")
    p.add_argument("--skip-edge", action="store_true")
    args = p.parse_args()

    # Resolve text
    if args.text:
        text = args.text
    elif args.from_project:
        text = load_text_from_project(args.from_project, args.segments)
    else:
        text = DEFAULT_TEXT

    # Resolve model_dir default from model_id
    if args.model_dir is None:
        args.model_dir = f".cache/models/{Path(args.model_id).name}"
    model_dir = REPO_ROOT / args.model_dir

    # Resolve prompt audio default (only matters for zero-shot)
    if not args.sft_voice and not args.prompt_audio:
        cv_root = setup_cosyvoice_path()
        if cv_root is not None:
            args.prompt_audio = cv_root / DEFAULT_PROMPT_REL
    prompt_audio = Path(args.prompt_audio) if args.prompt_audio else None

    out_dir = REPO_ROOT / "experiments" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    preview = text if len(text) <= 80 else text[:77] + "..."
    print(f"\nText ({len(text)} chars): {preview}\n")

    if args.sft_voice:
        cosy_out = out_dir / f"cosyvoice_sft_{args.sft_voice}.wav"
        cosy_label = f"CosyVoice/SFT/{args.sft_voice}"
    else:
        prompt_name = prompt_audio.stem if prompt_audio else "noprompt"
        cosy_out = out_dir / f"cosyvoice_zs_{prompt_name}.wav"
        cosy_label = f"CosyVoice/ZeroShot/{prompt_name}"
    edge_out = out_dir / f"edge_{args.edge_voice}.wav"

    if not args.skip_cosy:
        synth_cosyvoice(
            text, model_dir, cosy_out,
            sft_voice=args.sft_voice,
            prompt_audio=prompt_audio,
            prompt_text=args.prompt_text,
            use_v2=not args.cosy_v1,
            model_id=args.model_id,
        )
    if not args.skip_edge:
        asyncio.run(synth_edge_tts(text, args.edge_voice, edge_out))

    print("\n--- Output ---")
    report(cosy_label, cosy_out)
    report(f"edge-tts/{args.edge_voice}", edge_out)
    print(f"\nListen and compare WAVs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
