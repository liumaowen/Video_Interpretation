"""Preview SRT — split each aligned narration segment into sentence-level subtitle
entries, evenly distributed across the segment's time window. No TTS required."""
import re
from pathlib import Path

from .srt import ms_to_ts, parse_aligned_blocks


def split_sentences(text):
    parts = re.split(r"(?<=[。！？])", text)
    parts = [p.strip() for p in parts if p.strip()]
    result = []
    for p in parts:
        if len(p) > 22:
            sub = re.split(r"(?<=[，、])", p)
            sub = [s.strip() for s in sub if s.strip()]
            result.extend(sub)
        else:
            result.append(p)
    return result


def build(aligned_path: Path, out_srt: Path) -> int:
    segments = parse_aligned_blocks(aligned_path)
    out_lines = []
    idx = 1
    for start, end, text in segments:
        sentences = split_sentences(text)
        total_chars = sum(len(s) for s in sentences) or 1
        window = end - start
        cursor = start
        for sent in sentences:
            dur = int(window * len(sent) / total_chars)
            sub_start = cursor
            sub_end = min(cursor + dur, end)
            out_lines.append(str(idx))
            out_lines.append(f"{ms_to_ts(sub_start)} --> {ms_to_ts(sub_end)}")
            out_lines.append(sent)
            out_lines.append("")
            cursor = sub_end
            idx += 1
    out_srt.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_srt} ({idx - 1} entries)")
    return idx - 1
