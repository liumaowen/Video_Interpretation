"""faster-whisper Python ASR. Produces sentence-aware SRT with better grouping."""
from pathlib import Path

from .srt import ms_to_ts

MAX_WORDS = 13
MAX_CHARS = 60
MAX_GAP_S = 1.2
MIN_DURATION_MS = 800
END_PUNCT = (".", "!", "?")
SOFT_PUNCT = (",", ";", ":")


def group_words(words):
    """Sentence-aware grouping: split on sentence boundaries first,
    then split long sentences at commas/gaps, finally merge short tails."""
    # Step 1: split into sentences at end punctuation
    sentences = []
    cur = []
    for w in words:
        cur.append(w)
        if w.word.rstrip().endswith(END_PUNCT):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    # Step 2: split long sentences at soft punctuation / long gaps
    lines = []
    for sent in sentences:
        text = "".join(w.word for w in sent).strip()
        if len(sent) <= MAX_WORDS and len(text) <= MAX_CHARS:
            lines.append(sent)
            continue

        # Find split points at soft punctuation or long gaps
        split_indices = set()
        for i, w in enumerate(sent[:-1]):
            if w.word.rstrip().endswith(SOFT_PUNCT):
                split_indices.add(i)
            gap = sent[i + 1].start - w.end
            if gap > MAX_GAP_S:
                split_indices.add(i)

        if not split_indices:
            # No natural breaks — split evenly
            mid = len(sent) // 2
            split_indices = {mid} if mid > 0 else set()

        # Build sub-groups from split points
        buf = []
        for i, w in enumerate(sent):
            buf.append(w)
            buf_text = "".join(x.word for x in buf).strip()
            if i in split_indices and (
                len(buf) >= 3 and (len(buf) >= MAX_WORDS or len(buf_text) >= MAX_CHARS)
            ):
                lines.append(buf)
                buf = []
        if buf:
            # Merge short tail into previous line if possible
            if lines and len(buf) <= 2:
                lines[-1].extend(buf)
            else:
                lines.append(buf)

    # Step 3: merge lines shorter than MIN_DURATION_MS with next line
    merged = []
    for grp in lines:
        if not grp:
            continue
        duration_ms = (grp[-1].end - grp[0].start) * 1000
        if (
            merged
            and duration_ms < MIN_DURATION_MS
            and not grp[-1].word.rstrip().endswith(END_PUNCT)
        ):
            merged[-1].extend(grp)
        else:
            merged.append(grp)

    return merged


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
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
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
