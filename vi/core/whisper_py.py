"""faster-whisper Python ASR. Produces word-level grouped SRT."""
from pathlib import Path

from .srt import ms_to_ts


def group_words(words, max_words=8, max_gap_s=0.8, max_chars=42):
    lines = []
    buf = []
    for w in words:
        if not buf:
            buf.append(w)
            continue
        gap = w.start - buf[-1].end
        cur_text = "".join(x.word for x in buf)
        too_long = (
            len(buf) >= max_words
            or len(cur_text) + len(w.word) > max_chars
            or gap > max_gap_s
            or buf[-1].word.rstrip().endswith((".", "!", "?"))
        )
        if too_long:
            lines.append(buf)
            buf = [w]
        else:
            buf.append(w)
    if buf:
        lines.append(buf)
    return lines


def transcribe(
    video_path: Path,
    out_srt: Path,
    model_size: str = "tiny",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> int:
    from faster_whisper import WhisperModel
    print(f"Loading whisper model: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"Transcribing {video_path.name}...")
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=False,
    )
    print(f"Detected language: {info.language} (prob {info.language_probability:.2f})")

    out_lines = []
    idx = 1
    for seg in segments:
        if not seg.words:
            continue
        for grp in group_words(seg.words):
            start_ms = int(grp[0].start * 1000)
            end_ms = int(grp[-1].end * 1000)
            text = "".join(w.word for w in grp).strip()
            if not text:
                continue
            out_lines.append(str(idx))
            out_lines.append(f"{ms_to_ts(start_ms)} --> {ms_to_ts(end_ms)}")
            out_lines.append(text)
            out_lines.append("")
            idx += 1
    out_srt.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_srt} ({idx - 1} entries)")
    return idx - 1
