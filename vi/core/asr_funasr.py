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
_SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _strip_tags(text: str) -> str:
    """Drop SenseVoice rich tokens like <|en|><|HAPPY|><|Speech|><|withitn|>."""
    cleaned = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _split_long_segment(
    text: str,
    start_ms: int,
    end_ms: int,
    max_chars: int = 80,
) -> list[tuple[int, int, str]]:
    """Split an over-long segment at sentence punctuation, distributing time
    proportionally to character count. Keeps short segments intact."""
    text = text.strip()
    if len(text) <= max_chars:
        return [(start_ms, end_ms, text)]

    parts = [p.strip() for p in _SENTENCE_END_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return [(start_ms, end_ms, text)]

    total_chars = sum(len(p) for p in parts)
    out: list[tuple[int, int, str]] = []
    cursor = start_ms
    duration = end_ms - start_ms
    for i, p in enumerate(parts):
        if i == len(parts) - 1:
            out.append((cursor, end_ms, p))
        else:
            dur = int(duration * len(p) / total_chars)
            out.append((cursor, cursor + dur, p))
            cursor += dur
    return out


def _entries_from_result(res: list[dict]) -> list[tuple[int, int, str]]:
    """Normalize FunASR result into [(start_ms, end_ms, text), ...].

    Handles three possible shapes:
    1. Each item has `sentence_info`: [{text, start, end}, ...]   (preferred)
    2. Each item has a `timestamp` field [[s_ms, e_ms], ...] for the whole text
    3. Item has only `text` — falls back to a single span if no timestamps
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

        # Case 2: span-level timestamp
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

        # Case 3: no timestamps — skip silently (VAD should always provide them)

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
        merge_vad=False,
    )

    entries = _entries_from_result(res)
    if not entries:
        raise SystemExit(
            f"FunASR produced no usable segments for {video_path.name}. "
            "Verify the source contains audible speech, and make sure "
            "punc_model is empty (ct-punc strips VAD timestamps)."
        )

    write_srt(out_srt, entries)
    print(f"Wrote {out_srt} ({len(entries)} entries)")
    return len(entries)
