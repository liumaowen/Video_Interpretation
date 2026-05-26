"""whisper-faster.exe wrapper — produces JSON for downstream refine."""
import subprocess
from pathlib import Path


def transcribe(
    exe_path: Path,
    video_path: Path,
    out_dir: Path,
    model: str = "small",
    language: str = "en",
    model_dir: Path | None = None,
) -> Path:
    """Run whisper-faster.exe; returns the JSON output path."""
    cmd = [
        str(exe_path),
        str(video_path),
        "--model", model,
        "--language", language,
        "--output_dir", str(out_dir),
        "--output_format", "json",
        "--word_timestamps", "True",
        "--device", "cpu",
    ]
    if model_dir:
        cmd.extend(["--model_dir", str(model_dir)])
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # whisper-faster.exe names output as <video_stem>.json
    return out_dir / f"{video_path.stem}.json"
