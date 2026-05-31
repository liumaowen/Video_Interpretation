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
NARRATION_SOFT_PUNCT = set("，；、：")

# For recording-friendly narration lines (one breath = one line)
RECORDING_LINE_MAX_CHARS = 20


def _split_narration_sentences(text: str) -> list[str]:
    """Split narration at sentence terminators (。！？)."""
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


def _split_long_sentence_at(sent: str, max_chars: int, min_chunk: int | None = None) -> list[str]:
    """Split a long sentence into ≤max_chars chunks at soft punctuation.

    Falls back to mid-point splitting if no soft punctuation exists.

    Args:
        min_chunk: Minimum accumulated characters before a soft-punctuation
            split is accepted. Defaults to max_chars // 2.  Lower values
            split earlier, useful for recording-friendly (≤20 char) output.
    """
    if min_chunk is None:
        min_chunk = max_chars // 2
    if len(sent) <= max_chars:
        return [sent]

    chars = list(sent)
    split_indices = []
    for i, ch in enumerate(chars[:-1]):
        if ch in NARRATION_SOFT_PUNCT:
            split_indices.append(i)

    if not split_indices:
        # No soft punctuation — split evenly at mid-point
        parts = []
        remaining = sent
        while len(remaining) > max_chars:
            mid = len(remaining) // 2
            parts.append(remaining[:mid])
            remaining = remaining[mid:]
        if remaining:
            parts.append(remaining)
        return parts

    # Build chunks from split points
    segments = []
    buf = []
    buf_len = 0
    for i, ch in enumerate(chars):
        buf.append(ch)
        buf_len += 1
        if i in split_indices and buf_len >= min_chunk:
            segments.append("".join(buf))
            buf = []
            buf_len = 0

    if buf:
        if segments and buf_len < 10 and len(segments[-1]) + buf_len <= max_chars:
            # Short tail — merge into previous (only if it won't exceed max_chars)
            segments[-1] += "".join(buf)
        else:
            segments.append("".join(buf))

    return segments if segments else [sent[:max_chars]]


def _split_and_merge(text: str, max_chars: int, min_chunk: int | None = None, short_merge_threshold: int = 5, short_merge_limit: int = 20) -> list[str]:
    """Core algorithm: split text at sentence/soft-punctuation boundaries, merge short tails.

    Args:
        text: Input narration text.
        max_chars: Maximum characters per chunk.
        min_chunk: Minimum accumulated chars before a soft-punctuation split
            is accepted. Passed to _split_long_sentence_at.
        short_merge_threshold: Chunks ≤ this many chars get merged into previous.
        short_merge_limit: Only merge if combined length ≤ this.

    Returns list of text chunks, each ≤ max_chars (after re-split pass).
    """
    # Step 1: split into sentences at end punctuation
    sentences = _split_narration_sentences(text)

    # Step 2: split long sentences at soft punctuation
    parts = []
    for sent in sentences:
        parts.extend(_split_long_sentence_at(sent, max_chars, min_chunk))

    # Step 3: merge short chunks with previous
    merged = []
    for chunk in parts:
        if merged and len(chunk.strip()) <= short_merge_threshold:
            merged[-1] = merged[-1].rstrip() + chunk
        elif merged and len(merged[-1]) + len(chunk.strip()) <= max_chars and len(chunk.strip()) <= short_merge_limit:
            merged[-1] = merged[-1].rstrip() + chunk
        else:
            merged.append(chunk)

    # Step 4: re-split any merged result that still exceeds the limit
    result = []
    for item in merged:
        if len(item) > max_chars:
            result.extend(_split_long_sentence_at(item, max_chars, min_chunk))
        else:
            result.append(item)

    return result


def split_narration_text(text: str) -> list[str]:
    """Sentence-aware splitting of narration text, same logic as whisper_py.group_words.

    Returns a list of text chunks, each ≤60 chars, split at natural breakpoints.
    """
    return _split_and_merge(text, NARRATION_MAX_CHARS)


def split_narration_for_recording(text: str) -> list[str]:
    """Split narration into short lines suitable for voice recording.

    Each line is one natural pause (≤20 chars), split at commas/periods/colons.
    Uses min_chunk=3 so soft punctuation splits even after very few characters.
    Returns a list of text chunks.
    """
    return _split_and_merge(
        text,
        RECORDING_LINE_MAX_CHARS,
        min_chunk=3,
        short_merge_threshold=5,
        short_merge_limit=8,
    )


def format_narration_for_recording(text: str) -> str:
    """Like split_narration_for_recording but returns a string with double-newlines between lines."""
    chunks = split_narration_for_recording(text)
    return "\n\n".join(c.strip() for c in chunks if c.strip())


# Speaking rate: comfortable emotional delivery pace (chars per second)
SPEAKING_RATE_CHARS_PER_SEC = 3.5

# Keywords that indicate a visual description should be skipped for narration
VISUAL_SKIP_KEYWORDS = {"片头Logo", "黑屏", "Logo", "导演信息", "显示导演", "显示文字"}


def parse_visual_description(path: str | Path) -> list[tuple[int, str, bool]]:
    """Parse visual_description.txt into (timestamp_ms, description, should_skip) tuples.

    Returns entries sorted by timestamp. Entries with Logo/black-screen keywords
    are marked with should_skip=True for the caller to handle.
    """
    content = Path(path).read_text(encoding="utf-8")
    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'\[(\d+:\d+:\d+)[,.](\d+)\]\s*(.+)', line)
        if not m:
            continue
        ts = m.group(1)
        ms_str = m.group(2).ljust(3, '0')[:3]
        ms = ts_to_ms(f'{ts},{ms_str}')
        desc = m.group(3).strip()
        should_skip = any(kw in desc for kw in VISUAL_SKIP_KEYWORDS)
        entries.append((ms, desc, should_skip))
    entries.sort(key=lambda x: x[0])
    return entries


def allocate_chunk_times(chunks: list[str], start_ms: int, end_ms: int,
                         prev_end_ms: int | None = None,
                         max_end_ms: int | None = None) -> tuple[list[tuple[int, int, str]], int]:
    """Allocate time to text chunks proportional to character count.

    Each chunk gets time based on its length at SPEAKING_RATE_CHARS_PER_SEC
    (3.5 chars/sec), with a minimum of 800ms. Chunks are sequenced
    sequentially starting from max(start_ms, prev_end_ms) to avoid overlap.

    If max_end_ms is set and total allocation would exceed it, durations are
    compressed so the last entry ends at or before max_end_ms.

    Args:
        chunks: List of text chunks.
        start_ms: Original start time of this block.
        end_ms: Original end time of this block (used for scaling if total fits).
        prev_end_ms: End time of the last entry from the previous block.
        max_end_ms: Hard deadline — final entry must end at or before this.

    Returns:
        (entries, actual_end_ms) where entries is list of (start_ms, end_ms, text).
    """
    if not chunks:
        return [], prev_end_ms or start_ms

    # Start from max of original start and previous block's end (no overlap)
    cur = max(start_ms, prev_end_ms) if prev_end_ms is not None else start_ms

    # If cur already exceeds the video deadline (LLM hallucinated timestamps),
    # clamp start so chunks are placed within the valid range.
    if max_end_ms is not None and cur > max_end_ms:
        # Place remaining chunks in the last available window proportionally
        cur = max_end_ms - NARRATION_MIN_DURATION_MS * len(chunks)
        if cur < 0:
            cur = 0

    # Calculate ideal duration for each chunk based on char count
    ideal_durs = []
    for chunk in chunks:
        char_count = len(chunk.strip())
        if char_count == 0:
            ideal_durs.append(NARRATION_MIN_DURATION_MS)
            continue
        ideal = int(char_count / SPEAKING_RATE_CHARS_PER_SEC * 1000)
        ideal_durs.append(max(ideal, NARRATION_MIN_DURATION_MS))

    total_ideal = sum(ideal_durs)
    total_window = end_ms - cur

    # If total ideal fits within window, scale up to fill it
    if total_ideal <= total_window:
        scale = total_window / total_ideal
        final_durs = [int(d * scale) for d in ideal_durs]
    else:
        # Text is too long — keep ideal durations
        final_durs = ideal_durs

    # Enforce max_end_ms: compress if we'd exceed it
    if max_end_ms is not None:
        projected_end = cur + sum(final_durs)
        if projected_end > max_end_ms:
            # Compress proportionally to fit within max_end_ms
            available = max_end_ms - cur
            if available > 0:
                scale = available / total_ideal if total_ideal > 0 else 1.0
                final_durs = [max(int(d * scale), NARRATION_MIN_DURATION_MS) for d in ideal_durs]
            else:
                # cur already at or past max_end_ms — force minimum durations
                final_durs = [NARRATION_MIN_DURATION_MS] * len(chunks)

    entries = []
    for chunk, dur in zip(chunks, final_durs):
        chunk_end = cur + dur
        entries.append((cur, chunk_end, chunk))
        cur = chunk_end

    # Final clamp: ensure the last entry doesn't exceed max_end_ms
    # (the NARRATION_MIN_DURATION_MS floor can push the total past the deadline)
    if max_end_ms is not None and cur > max_end_ms:
        overflow = cur - max_end_ms
        # Trim the last chunk's end to exactly max_end_ms
        last = list(entries[-1])
        last[1] = max_end_ms
        entries[-1] = tuple(last)
        cur = max_end_ms

    return entries, cur


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
