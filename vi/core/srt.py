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


_TS_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _wrap_subtitle_line(text: str, max_chars: int = 20) -> str:
    """Wrap a long subtitle text into short lines suitable for display.

    Splits at natural breakpoints (periods, commas) and falls back to
    character-count splitting if no good breakpoint exists.
    """
    if len(text) <= max_chars:
        return text
    breakpoints = "，。！？、；,.!?;:："
    result_lines = []
    remaining = text
    while len(remaining) > max_chars:
        # Try to find a natural breakpoint within the first max_chars+10 chars
        search_end = min(len(remaining), max_chars + 10)
        last_bp = -1
        for i in range(max_chars, min(len(remaining), search_end)):
            if remaining[i] in breakpoints:
                last_bp = i
                break
        if last_bp >= 0:
            result_lines.append(remaining[:last_bp + 1])
            remaining = remaining[last_bp + 1:]
        else:
            # No good breakpoint, force-split at max_chars
            result_lines.append(remaining[:max_chars])
            remaining = remaining[max_chars:]
    if remaining:
        result_lines.append(remaining)
    return "\n".join(result_lines)


def _fix_carry_bug(h: int, m: int, s: int, ms: int, max_ms: int) -> int:
    """LLMs sometimes write `01:00:01` when they mean `00:01:01` (carry into hour
    instead of minute). Detect that by checking if the value exceeds the video
    duration, and swap h/m when it does."""
    total = (h * 3600 + m * 60 + s) * 1000 + ms
    if total <= max_ms or h == 0:
        return total
    swapped = (m * 3600 + h * 60 + s) * 1000 + ms
    if swapped <= max_ms:
        return swapped
    return total


def sanitize_srt_text(text: str, max_duration_ms: int) -> str:
    """Reformat LLM-generated SRT, fixing common timestamp bugs.

    - Re-parses each block's two timestamps and writes them back via ms_to_ts
      so the format is canonical (`HH:MM:SS,mmm`).
    - Detects the "hour carry" bug where `00:01:XX` is written as `01:00:XX`
      and rewrites by swapping h/m when the original value would exceed
      max_duration_ms.
    - Wraps long subtitle text into short lines (≤20 chars) for readability.
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    fixed_blocks: list[str] = []
    # Allow ~10s slop so a final block that brushes against the duration still passes.
    cap = max_duration_ms + 10_000
    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        m = re.match(
            r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)\s*",
            lines[1],
        )
        if not m:
            fixed_blocks.append("\n".join(lines))
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start_ms = _fix_carry_bug(h1, m1, s1, ms1, cap)
        end_ms = _fix_carry_bug(h2, m2, s2, ms2, cap)
        lines[1] = f"{ms_to_ts(start_ms)} --> {ms_to_ts(end_ms)}"
        # Concatenate all text lines then re-wrap cleanly
        combined = "".join(l.strip() for l in lines[2:])
        wrapped_lines = _wrap_subtitle_line(combined, max_chars=20).split("\n")
        fixed_blocks.append("\n".join(lines[:2] + wrapped_lines))
    return "\n\n".join(fixed_blocks) + "\n"
