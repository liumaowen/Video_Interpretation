"""Rebuild SRT from whisper JSON using per-word timestamps with sentence-aware splitting.

Hard rule: a subtitle line never crosses a sentence-ending punctuation (. ! ?)."""
import json
from pathlib import Path

from .srt import ms_to_ts

MAX_WORDS = 13
MAX_CHARS = 60
MAX_GAP_S = 0.7
END_PUNCT = (".", "!", "?")
SOFT_PUNCT = (",", ";", ":")


def split_sentence(sentence_words):
    text = "".join(w["word"] for w in sentence_words).strip()
    if len(sentence_words) <= MAX_WORDS and len(text) <= MAX_CHARS:
        return [sentence_words]

    soft_break_indices = [
        i for i, w in enumerate(sentence_words[:-1])
        if w["word"].strip().endswith(SOFT_PUNCT)
    ]
    for i in range(len(sentence_words) - 1):
        gap = sentence_words[i + 1]["start"] - sentence_words[i]["end"]
        if gap > MAX_GAP_S and i not in soft_break_indices:
            soft_break_indices.append(i)
    soft_break_indices.sort()

    if soft_break_indices:
        lines = []
        buf = []
        for i, w in enumerate(sentence_words):
            buf.append(w)
            buf_text = "".join(x["word"] for x in buf).strip()
            if i in soft_break_indices and (
                len(buf) >= MAX_WORDS or len(buf_text) >= MAX_CHARS
            ):
                lines.append(buf)
                buf = []
        if buf:
            if lines and len(buf) <= 2:
                lines[-1].extend(buf)
            else:
                lines.append(buf)
        return lines

    n_parts = (len(sentence_words) + MAX_WORDS - 1) // MAX_WORDS
    n_parts = max(n_parts, 2)
    chunk_size = (len(sentence_words) + n_parts - 1) // n_parts
    return [
        sentence_words[i:i + chunk_size]
        for i in range(0, len(sentence_words), chunk_size)
    ]


def refine(json_path: Path, out_srt: Path) -> int:
    data = json.load(open(json_path, encoding="utf-8"))
    words = []
    for seg in data["segments"]:
        for w in seg.get("words", []):
            words.append(w)

    sentences = []
    cur = []
    for w in words:
        cur.append(w)
        if w["word"].strip().endswith(END_PUNCT):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    lines = []
    for sent in sentences:
        lines.extend(split_sentence(sent))

    out = []
    for idx, grp in enumerate(lines, 1):
        start_ms = int(grp[0]["start"] * 1000)
        end_ms = int(grp[-1]["end"] * 1000)
        text = "".join(x["word"] for x in grp).strip()
        if not text:
            continue
        out.append(str(idx))
        out.append(f"{ms_to_ts(start_ms)} --> {ms_to_ts(end_ms)}")
        out.append(text)
        out.append("")
    out_srt.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_srt} ({len(lines)} entries)")
    return len(lines)
