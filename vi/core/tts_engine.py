"""edge-tts engine. Preserves the original tts.py algorithm:
  - per-segment SentenceBoundary capture
  - auto re-synth at higher rate when duration overflows window
  - ffmpeg adelay+apad+amix to assemble final voice.wav
  - subtitle line splitting from SentenceBoundary events"""
import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import edge_tts

from .srt import ms_to_ts, parse_aligned_blocks


async def _synthesize_with_timing(text, out_mp3, voice, volume, rate):
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    sentences = []
    with open(out_mp3, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                sentences.append((chunk["offset"], chunk["duration"], chunk["text"]))
    return sentences


def _probe_duration_ms(path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(float(json.loads(result.stdout)["format"]["duration"]) * 1000)


def _split_into_subtitle_lines(sentences, seg_start_ms, max_len=24):
    inner_punct = re.compile(r"(?<=[，、；：])")
    lines = []
    for offset, dur, sent_text in sentences:
        s_start = seg_start_ms + offset // 10000
        s_end = seg_start_ms + (offset + dur) // 10000
        sent_text = sent_text.strip()
        if not sent_text:
            continue
        if len(sent_text) <= max_len:
            lines.append((s_start, s_end, sent_text))
            continue
        parts = [p.strip() for p in inner_punct.split(sent_text) if p.strip()]
        chunks = []
        buf = ""
        for p in parts:
            if not buf:
                buf = p
            elif len(buf) + len(p) <= max_len:
                buf += p
            else:
                chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        total_chars = sum(len(p) for p in chunks)
        window = s_end - s_start
        cursor = s_start
        for i, p in enumerate(chunks):
            if i == len(chunks) - 1:
                p_end = s_end
            else:
                p_end = cursor + int(window * len(p) / total_chars)
            lines.append((cursor, p_end, p))
            cursor = p_end
    return lines


async def synthesize(
    aligned_path: Path,
    out_wav: Path,
    out_srt: Path,
    tmp_dir: Path,
    voice: str = "zh-CN-YunjianNeural",
    volume: str = "+0%",
    max_rate_boost: int = 40,
) -> None:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    segments = parse_aligned_blocks(aligned_path)
    print(f"Parsed {len(segments)} segments")
    if not segments:
        raise SystemExit(f"No segments parsed from {aligned_path}")

    total_end = max(end for _, end, _ in segments) + 1000

    seg_data = []
    for idx, (start, end, text) in enumerate(segments, 1):
        window_ms = end - start
        seg_path = tmp_dir / f"seg_{idx}.mp3"

        sentences = await _synthesize_with_timing(text, seg_path, voice, volume, "+0%")
        dur = _probe_duration_ms(seg_path)

        if dur > window_ms:
            overflow = dur / window_ms
            rate_pct = min(int((overflow - 1) * 100) + 5, max_rate_boost)
            rate_str = f"+{rate_pct}%"
            print(f"  Seg {idx}: {dur}ms > window {window_ms}ms, re-synth at {rate_str}")
            sentences = await _synthesize_with_timing(text, seg_path, voice, volume, rate_str)
            dur = _probe_duration_ms(seg_path)

        lines = _split_into_subtitle_lines(sentences, start)
        print(f"  Seg {idx}: start={start}ms, dur={dur}ms, {len(lines)} subtitle lines")
        seg_data.append((start, seg_path, lines))

    inputs = []
    filters = []
    mix_inputs = []
    for i, (start, path, _) in enumerate(seg_data):
        inputs += ["-i", str(path)]
        filters.append(f"[{i}:a]adelay={start}|{start},apad=whole_dur={total_end}ms[a{i}]")
        mix_inputs.append(f"[a{i}]")

    n = len(seg_data)
    filter_complex = (
        "; ".join(filters)
        + f"; {''.join(mix_inputs)}amix=inputs={n}:normalize=0,"
          f"atrim=end={total_end/1000}[out]"
    )
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
           "-map", "[out]", str(out_wav)]
    print("Merging audio...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Wrote {out_wav}")

    srt_entries = []
    for _, _, lines in seg_data:
        for s, e, text in lines:
            srt_entries.append((s, e, text))
    out = []
    for idx, (s, e, text) in enumerate(srt_entries, 1):
        out.append(str(idx))
        out.append(f"{ms_to_ts(s)} --> {ms_to_ts(e)}")
        out.append(text)
        out.append("")
    out_srt.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_srt} ({len(srt_entries)} entries)")


def run_sync(*args, **kwargs):
    asyncio.run(synthesize(*args, **kwargs))
