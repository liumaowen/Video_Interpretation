"""FunASR ASR engine (SenseVoiceSmall + FSMN-VAD).

Pipeline:
- FSMN-VAD splits the audio into speech chunks
- SenseVoiceSmall transcribes each chunk (multilingual: en/zh/yue/ja/ko)
- SenseVoice's built-in ITN (`use_itn=True`) adds punctuation

ct-punc is intentionally NOT used by default: when chained after SenseVoice it
merges all VAD chunks into a single text and drops per-chunk timestamps, which
leaves us nothing to build an SRT from. SenseVoice's built-in punctuation is
sufficient for our purposes.

Output is a sentence-level SRT, so the `refine` step is unnecessary when this
engine is used.

Models are downloaded automatically by FunASR from ModelScope on first use.
On the ModelScope Notebook environment they come from the local cache.
"""
import re
from pathlib import Path

from .srt import write_srt


# Tolerate whitespace inside tags (ct-punc tokenizes "<|en|>" as "< | en | >").
_TAG_RE = re.compile(r"<\s*\|[^|<>]*\|\s*>")
# Sentence-ending vs soft-pause punctuation for splitting long SRT lines.
_END_PUNCT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_SOFT_PUNCT_RE = re.compile(r"(?<=[,;:，；：、])\s+")
_MAX_LINE_CHARS = 60
# Each SenseVoice chunk starts with a language tag. Split on lookahead so the
# tag stays attached to the chunk it introduces.
_LANG_TAG_LOOKAHEAD = re.compile(
    r"(?=<\s*\|\s*(?:en|zh|zn|yue|ja|ko|auto|nospeech)\s*\|\s*>)",
    re.IGNORECASE,
)


def _strip_tags(text: str) -> str:
    """Drop SenseVoice rich tokens like <|en|><|HAPPY|><|Speech|><|withitn|>."""
    cleaned = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _split_chunks_by_lang_tag(text: str) -> list[str]:
    """Split SenseVoice's concatenated text into per-VAD-chunk transcriptions."""
    parts = _LANG_TAG_LOOKAHEAD.split(text)
    out: list[str] = []
    for p in parts:
        clean = _strip_tags(p)
        if clean:
            out.append(clean)
    return out


def _vad_only_timestamps(
    video_path: Path, vad_model: str, device: str
) -> list[tuple[int, int]]:
    """Run VAD alone to get speech-chunk [(start_ms, end_ms), ...] boundaries."""
    from funasr import AutoModel
    vm = AutoModel(model=vad_model, device=device, disable_update=True)
    res = vm.generate(input=str(video_path))
    if not res:
        return []
    value = res[0].get("value") or []
    return [(int(s), int(e)) for s, e in value if int(e) > int(s)]


def _hard_split_words(text: str, max_chars: int) -> list[str]:
    """Greedy word-boundary split when no punctuation is available."""
    words = text.split()
    if not words:
        return []
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars and cur:
            out.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        out.append(" ".join(cur))
    return out


def _split_text_cascade(text: str, max_chars: int) -> list[str]:
    """Split overlong text: sentence-end → soft-pause → word-boundary hard split."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    out: list[str] = []
    for part in _END_PUNCT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            out.append(part)
            continue
        # Sentence still too long → try soft pauses.
        for sub in _SOFT_PUNCT_RE.split(part):
            sub = sub.strip()
            if not sub:
                continue
            if len(sub) <= max_chars:
                out.append(sub)
            else:
                out.extend(_hard_split_words(sub, max_chars))
    return out or [text]


def _split_long_segment(
    text: str,
    start_ms: int,
    end_ms: int,
    max_chars: int = _MAX_LINE_CHARS,
) -> list[tuple[int, int, str]]:
    """Split an over-long segment, distributing time proportionally to character count."""
    pieces = _split_text_cascade(text, max_chars)
    if len(pieces) <= 1:
        return [(start_ms, end_ms, pieces[0] if pieces else text.strip())]

    total_chars = sum(len(p) for p in pieces)
    if total_chars == 0:
        return [(start_ms, end_ms, text.strip())]

    out: list[tuple[int, int, str]] = []
    cursor = start_ms
    duration = end_ms - start_ms
    for i, p in enumerate(pieces):
        if i == len(pieces) - 1:
            out.append((cursor, end_ms, p))
        else:
            dur = int(duration * len(p) / total_chars)
            out.append((cursor, cursor + dur, p))
            cursor += dur
    return out


def _entries_from_result(res: list[dict]) -> list[tuple[int, int, str]]:
    """Normalize FunASR result into [(start_ms, end_ms, text), ...].

    FunASR's output shape varies with version and config. We handle:
    1. Item has `sentence_info`: [{text, start, end}, ...]   (preferred, merge_vad=True)
    2. Item has top-level `start`/`end` + `text`             (merge_vad=False, per-chunk)
    3. Item has `timestamp` [[s_ms, e_ms], ...] + `text`     (span-level)
    4. Multiple items, each one VAD chunk
    """
    entries: list[tuple[int, int, str]] = []
    for item in res:
        # Case 1: per-sentence timestamps (best)
        sent_info = item.get("sentence_info")
        if sent_info:
            for s in sent_info:
                txt = _strip_tags(s.get("text", ""))
                if not txt:
                    continue
                start = int(s.get("start", 0))
                end = int(s.get("end", start))
                if end <= start:
                    continue
                entries.append((start, end, txt))
            continue

        text = _strip_tags(item.get("text", ""))
        if not text:
            continue

        # Case 2: per-item start/end (FunASR returns one item per VAD chunk)
        if "start" in item and "end" in item:
            try:
                start = int(item["start"])
                end = int(item["end"])
            except (TypeError, ValueError):
                start = end = 0
            if end > start:
                entries.extend(_split_long_segment(text, start, end))
                continue

        # Case 3: span-level timestamp
        ts = item.get("timestamp")
        if ts and isinstance(ts, list) and ts:
            try:
                start = int(ts[0][0])
                end = int(ts[-1][1])
            except (TypeError, IndexError):
                start = end = 0
            if end > start:
                entries.extend(_split_long_segment(text, start, end))
                continue

        # Case 4: no timestamps — skip silently (VAD should always provide them)

    return entries


def transcribe(
    video_path: Path,
    out_srt: Path,
    model_size: str = "iic/SenseVoiceSmall",
    language: str = "en",
    device: str = "cpu",
    compute_type: str = "fp32",  # accepted for signature parity; unused
    vad_model: str = "fsmn-vad",
    punc_model: str = "",
) -> int:
    """Transcribe *video_path* into a sentence-level SRT at *out_srt*.

    *language* accepts SenseVoice codes: "auto" / "en" / "zh" / "yue" / "ja" / "ko".
    *model_size* can be a ModelScope id (e.g. "iic/SenseVoiceSmall") or alias.
    *punc_model* is OFF by default — passing "ct-punc" here causes FunASR to
    merge VAD chunks and drop their timestamps, which breaks SRT generation.
    """
    try:
        from funasr import AutoModel
    except ImportError:
        raise SystemExit(
            "funasr not installed. Run: pip install funasr modelscope"
        )

    # Friendly aliases → ModelScope ids
    aliases = {
        "tiny": "iic/SenseVoiceSmall",
        "small": "iic/SenseVoiceSmall",
        "base": "iic/SenseVoiceSmall",
        "SenseVoiceSmall": "iic/SenseVoiceSmall",
    }
    model_id = aliases.get(model_size, model_size)

    # SenseVoice expects "zn" for Chinese (not "zh"); normalize common synonyms.
    lang = {"zh": "zn", "cn": "zn"}.get(language, language)

    pipeline_desc = f"{model_id} + {vad_model}"
    auto_kwargs = dict(
        model=model_id,
        trust_remote_code=True,
        vad_model=vad_model,
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        disable_update=True,
    )
    if punc_model and punc_model.lower() != "none":
        auto_kwargs["punc_model"] = punc_model
        pipeline_desc += f" + {punc_model}"

    print(f"Loading FunASR pipeline: {pipeline_desc} (device={device})")
    model = AutoModel(**auto_kwargs)

    print(f"Transcribing {video_path.name}...")
    res = model.generate(
        input=str(video_path),
        cache={},
        language=lang,
        use_itn=True,
        batch_size_s=60,
        # Keep False so the number of <|en|> chunks in `text` matches a
        # standalone VAD pass — the fallback below relies on that alignment.
        merge_vad=False,
    )

    # Surface the raw shape so users can report it if parsing still misses.
    if res:
        keys = sorted(set(k for item in res if isinstance(item, dict) for k in item.keys()))
        print(f"FunASR returned {len(res)} item(s); keys per item: {keys}")

    entries = _entries_from_result(res)
    if not entries:
        # Fallback: funasr 1.3.1's chained pipeline returns merged text without
        # timestamps. Run VAD alone to get chunk boundaries, then split the
        # SenseVoice text on language-tag markers and pair the two.
        text = res[0].get("text", "") if res else ""
        parts = _split_chunks_by_lang_tag(text)
        if parts:
            print(f"Pairing {len(parts)} text segments with a separate VAD pass...")
            chunks = _vad_only_timestamps(video_path, vad_model, device)
            if chunks:
                if len(parts) != len(chunks):
                    print(
                        f"Note: {len(parts)} text segments vs {len(chunks)} VAD "
                        f"chunks — pairing the first {min(len(parts), len(chunks))}."
                    )
                n = min(len(parts), len(chunks))
                for i in range(n):
                    if parts[i]:
                        entries.extend(
                            _split_long_segment(parts[i], chunks[i][0], chunks[i][1])
                        )

    if not entries:
        print("DEBUG raw res:", res)
        raise SystemExit(
            f"FunASR produced no usable segments for {video_path.name}. "
            "The raw result shape is printed above — please report it."
        )

    write_srt(out_srt, entries)
    print(f"Wrote {out_srt} ({len(entries)} entries)")
    return len(entries)
