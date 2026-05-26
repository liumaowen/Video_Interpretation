"""Shared SRT utilities — replaces ms_to_ts and parse_segments duplicated across old scripts."""
import re
from pathlib import Path


def ms_to_ts(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_to_ms(ts: str) -> int:
    m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", ts.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {ts}")
    h, mi, s, ms = map(int, m.groups())
    return (h * 3600 + mi * 60 + s) * 1000 + ms


def parse_srt(path: str | Path) -> list[tuple[int, int, int, str]]:
    """Return list of (index, start_ms, end_ms, text)."""
    content = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    out = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0])
        except ValueError:
            continue
        m = re.match(
            r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", lines[1]
        )
        if not m:
            continue
        start = ts_to_ms(m.group(1))
        end = ts_to_ms(m.group(2))
        text = "\n".join(lines[2:]).strip()
        out.append((idx, start, end, text))
    return out


def parse_aligned_blocks(path: str | Path) -> list[tuple[int, int, str]]:
    """For narration_aligned.txt — SRT-shaped but text may span multiple lines joined by space."""
    content = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    segs = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(
            r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[1]
        )
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        text = " ".join(lines[2:]).strip()
        segs.append((start, end, text))
    return segs


def write_srt(path: str | Path, entries: list[tuple[int, int, str]]) -> None:
    """entries: list of (start_ms, end_ms, text)."""
    out = []
    for idx, (s, e, text) in enumerate(entries, 1):
        out.append(str(idx))
        out.append(f"{ms_to_ts(s)} --> {ms_to_ts(e)}")
        out.append(text)
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")
