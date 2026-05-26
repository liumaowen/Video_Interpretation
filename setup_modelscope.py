"""Setup script for ModelScope Notebook (Linux CPU environment)."""
import subprocess
import sys
import os

def run(cmd: str):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def setup():
    print("=" * 60)
    print("ModelScope Notebook Setup — Video Interpretation")
    print("=" * 60)

    # 1. Install ffmpeg
    print("\n[1/4] Installing ffmpeg...")
    run("apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1")
    run("ffmpeg -version | head -1")

    # 2. Install Chinese fonts for subtitle rendering
    print("\n[2/4] Installing Chinese fonts...")
    run("apt-get install -y -qq fonts-noto-cjk > /dev/null 2>&1")
    print("  Installed fonts-noto-cjk (Noto Sans CJK)")

    # 3. Install Python dependencies
    print("\n[3/4] Installing Python dependencies...")
    run("pip install -q -r requirements.txt")
    print("  All Python packages installed")

    # 4. Verify setup
    print("\n[4/4] Verifying...")
    run("ffmpeg -version | head -1")
    import faster_whisper
    print(f"  faster-whisper: {faster_whisper.__version__}")
    import edge_tts
    print(f"  edge-tts: {edge_tts.__version__}")
    import moviepy
    print(f"  moviepy: {moviepy.__version__}")
    import anthropic
    print(f"  anthropic: {anthropic.__version__}")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

if __name__ == "__main__":
    setup()
