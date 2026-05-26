import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HF_HOME = ROOT / ".cache"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))

if __name__ == "__main__":
    from vi.cli import main
    sys.exit(main())
