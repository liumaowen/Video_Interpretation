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

# Constants mirroring whisper_py.group_words — proven to produce readable subtitles
NARRATION_MAX_CHARS = 60
NARRATION_MIN_DURATION_MS = 800
NARRATION_END_PUNCT = set("。！？")
NARRATION_SOFT_PUNCT = set("，；、")


def _split_narration_sentences(text: str) -> list[str]:
    """Split narration at sentence terminators (。！？), mirroring whisper_py Step 1."""
    sentences = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in NARRATION_END_PUNCT:
            sentences.append(cur)
            cur = ""
    if cur.strip():
        sentences.append(cur)
    return [s for s in sentences if s]


def _split_long_sentence(sent: str) -> list[str]:
    """Split a single long sentence into <=60-char chunks at soft punctuation.
    Falls back to mid-point splitting if no soft punctuation exists.
    """
    if len(sent) <= NARRATION_MAX_CHARS:
        return [sent]

    chars = list(sent)
    split_indices = []
    for i, ch in enumerate(chars[:-1]):
        if ch in NARRATION_SOFT_PUNCT:
            split_indices.append(i)

    if not split_indices:
        # No soft punctuation at all — split evenly at mid-point
        parts = []
        remaining = sent
        while len(remaining) > NARRATION_MAX_CHARS:
            mid = len(remaining) // 2
            parts.append(remaining[:mid])
            remaining = remaining[mid:]
        if remaining:
            parts.append(remaining)
        return parts

    # Build chunks from split points
    segments = []
    buf_start = 0
    buf = []
    buf_len = 0
    for i, ch in enumerate(chars):
        buf.append(ch)
        buf_len += 1
        if i in split_indices and buf_len >= NARRATION_MAX_CHARS // 2:
            segments.append("".join(buf))
            buf = []
            buf_len = 0

    if buf:
        if segments and buf_len < 10:
            # Short tail — merge into previous
            segments[-1] += "".join(buf)
        else:
            segments.append("".join(buf))

    return segments if segments else [sent[:NARRATION_MAX_CHARS]]


def split_narration_text(text: str) -> list[str]:
    """Sentence-aware splitting of narration text, same logic as whisper_py.group_words.

    Returns a list of text chunks, each ≤60 chars, split at natural breakpoints.
    """
    # Step 1: split into sentences at end punctuation
    sentences = _split_narration_sentences(text)

    # Step 2: split long sentences at soft punctuation
    parts = []
    for sent in sentences:
        parts.extend(_split_long_sentence(sent))

    # Step 3: merge short chunks with previous one (mirrors MIN_DURATION_MS merge)
    merged = []
    for chunk in parts:
        if merged and len(chunk.strip()) <= 5:
            merged[-1] = merged[-1].rstrip() + chunk
        elif merged and len(merged[-1]) + len(chunk.strip()) <= NARRATION_MAX_CHARS and len(chunk.strip()) <= 20:
            merged[-1] = merged[-1].rstrip() + chunk
        else:
            merged.append(chunk)

    # Step 4: re-split any merged result that still exceeds the limit
    # (this catches cases like long sentences with only one soft punctuation)
    result = []
    for item in merged:
        if len(item) > NARRATION_MAX_CHARS:
            result.extend(_split_long_sentence(item))
        else:
            result.append(item)

    return result


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
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    fixed_blocks: list[str] = []
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
        # Merge consecutive text lines into one for parse_aligned_blocks compat
        combined = "".join(l.strip() for l in lines[2:])
        fixed_blocks.append("\n".join(lines[:2] + [combined]))
    return "\n\n".join(fixed_blocks) + "\n"
