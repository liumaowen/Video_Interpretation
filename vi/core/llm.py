"""LLM client wrapper. Supports Anthropic Claude and OpenAI-compatible APIs.

Providers:
- anthropic: Claude models (requires ANTHROPIC_API_KEY)
- openai_compatible: Any OpenAI-compatible API (Ollama, SiliconFlow, DeepSeek, GLM, etc.)

Configuration via config.toml [llm]:
  provider = "openai_compatible"  # or "anthropic"
  base_url = "http://localhost:11434/v1"  # Ollama default
  model = "qwen2.5:7b"
  api_key_env = "OPENAI_API_KEY"  # or any env var; Ollama can use "dummy"
"""
import os
import re
import sys
from pathlib import Path

from .srt import parse_srt, ms_to_ts, write_srt
from . import llm_prompts


# ---------------------------------------------------------------------------
# Provider: Anthropic Claude
# ---------------------------------------------------------------------------

def _client_anthropic(api_key: str):
    try:
        import anthropic
    except ImportError:
        raise SystemExit("anthropic SDK not installed. Run: pip install anthropic")
    if not api_key:
        raise SystemExit(
            "Anthropic API key not set. Either set [llm].api_key in config.toml, "
            "or set the ANTHROPIC_API_KEY environment variable"
        )
    return anthropic.Anthropic(api_key=api_key)


def _call_anthropic(client, model, system_prompt, user_text):
    """Stream an Anthropic chat completion. Yields text chunks."""
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Provider: OpenAI-compatible (Ollama, SiliconFlow, DeepSeek, GLM, etc.)
# ---------------------------------------------------------------------------

def _client_openai(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai SDK not installed. Run: pip install openai")
    return OpenAI(base_url=base_url, api_key=api_key)


def _call_openai(client, model, system_prompt, user_text):
    """Stream an OpenAI-compatible chat completion. Yields text chunks."""
    stream = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def _resolve_api_key(api_key: str, api_key_env: str) -> str:
    """Resolve API key: direct value takes priority, then env var lookup."""
    if api_key:
        return api_key
    key = os.environ.get(api_key_env, "")
    if not key:
        raise SystemExit(
            f"API key not found. Either set [llm].api_key in config.toml, "
            f"or set environment variable {api_key_env}"
        )
    return key


def _get_backend(provider, base_url, api_key, api_key_env):
    """Return (client, call_fn) based on provider config."""
    # Strip trailing slash from base_url to avoid double slashes
    if base_url:
        base_url = base_url.rstrip("/")
    if provider == "anthropic":
        resolved_key = _resolve_api_key(api_key, api_key_env)
        client = _client_anthropic(resolved_key)
        return client, _call_anthropic
    else:
        # openai_compatible — default for free models
        url = base_url or "http://localhost:11434/v1"
        resolved_key = _resolve_api_key(api_key, api_key_env or "OPENAI_API_KEY")
        client = _client_openai(url, resolved_key)
        return client, _call_openai


def _srt_to_prompt_text(srt_path: Path) -> tuple[str, int]:
    """Return (formatted text, total duration in seconds)."""
    entries = parse_srt(srt_path)
    if not entries:
        raise SystemExit(f"No entries parsed from {srt_path}")
    lines = []
    for _, start, end, text in entries:
        lines.append(f"[{ms_to_ts(start)}] {text}")
    total_ms = entries[-1][2]
    return "\n".join(lines), total_ms // 1000


def _format_visual_section(visual_description: str) -> str:
    """Format visual description for inclusion in prompt. Empty string if no description.

    Filters out uninformative entries (Logo, black screen, credits) and
    condenses the list to keep the prompt compact.
    """
    if not visual_description:
        return ""
    # Filter out entries that are just logos, black screens, or credits
    skip_keywords = ["片头Logo", "黑屏", "Logo", "导演信息", "显示导演", "显示文字"]
    useful_lines = []
    for line in visual_description.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Extract description part after timestamp
        desc = line.split("] ", 1)[-1] if "] " in line else line
        if any(kw in desc for kw in skip_keywords):
            continue
        useful_lines.append(line)
    if not useful_lines:
        return ""
    condensed = "\n".join(useful_lines)
    return f"【视频画面描述】（视觉模型分析的关键帧，仅保留有剧情意义的画面）：\n\n{condensed}\n\n"


def _stream_and_collect(call_fn, client, model, system_prompt, user_text,
                        max_retries=3):
    """Call LLM with streaming, print dots to stderr, return full text.

    Retries on rate-limit (429) errors with exponential backoff.
    """
    import time
    from openai import RateLimitError

    for attempt in range(max_retries):
        try:
            parts = []
            for chunk in call_fn(client, model, system_prompt, user_text):
                parts.append(chunk)
                sys.stderr.write(".")
                sys.stderr.flush()
            sys.stderr.write("\n")
            result = "".join(parts).strip()
            if not result:
                print(f"\n  ⚠ LLM returned empty response (attempt {attempt+1}/{max_retries})",
                      file=sys.stderr)
                if attempt < max_retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"  Retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
            return result
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"\n  ⚠ Rate limited (429), retrying in {wait}s... "
                      f"(attempt {attempt+1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


def generate_narration(
    subtitle_path: Path,
    out_path: Path,
    style: str,
    length: int,
    video_title: str = "",
    visual_description: str = "",
    model: str = "qwen2.5:7b",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    provider: str = "openai_compatible",
    base_url: str = "",
) -> None:
    """Generate narration.txt from English subtitle.srt via LLM."""
    client, call_fn = _get_backend(provider, base_url, api_key, api_key_env)
    subtitle_text, duration_sec = _srt_to_prompt_text(subtitle_path)

    # Calculate max chars: video duration × comfortable speaking rate (3.5 chars/sec)
    from ..core.srt import SPEAKING_RATE_CHARS_PER_SEC
    max_chars = int(duration_sec * SPEAKING_RATE_CHARS_PER_SEC)

    visual_section = _format_visual_section(visual_description)
    user_text = llm_prompts.NARRATE_USER_TEMPLATE.format(
        video_title=video_title or "未知",
        visual_section=visual_section,
        subtitle_text=subtitle_text,
        duration_sec=duration_sec,
        max_chars=max_chars,
        style=style,
        length=min(length, max_chars),  # Don't ask for more than the video can hold
    )

    print(f"Calling {model} (style={style!r}, length={length})...", file=sys.stderr)
    result = _stream_and_collect(
        call_fn, client, model, llm_prompts.NARRATE_SYSTEM, user_text
    )

    # Post-process: split into recording-friendly short lines (≤20 chars each)
    from ..core.srt import format_narration_for_recording
    result = format_narration_for_recording(result)

    out_path.write_text(result + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def generate_narration_srt(
    subtitle_path: Path,
    out_path: Path,
    style: str,
    video_title: str = "",
    visual_description: str = "",
    model: str = "qwen2.5:7b",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    provider: str = "openai_compatible",
    base_url: str = "",
) -> None:
    """One-step: generate narration_aligned.txt (SRT) directly from subtitle.srt."""
    client, call_fn = _get_backend(provider, base_url, api_key, api_key_env)
    subtitle_text, duration_sec = _srt_to_prompt_text(subtitle_path)

    # Calculate max chars: video duration × comfortable speaking rate (3.5 chars/sec)
    from ..core.srt import SPEAKING_RATE_CHARS_PER_SEC
    max_chars = int(duration_sec * SPEAKING_RATE_CHARS_PER_SEC)

    visual_section = _format_visual_section(visual_description)
    user_text = llm_prompts.NARRATE_SRT_USER_TEMPLATE.format(
        video_title=video_title or "未知",
        visual_section=visual_section,
        subtitle_text=subtitle_text,
        duration_sec=duration_sec,
        max_chars=max_chars,
        style=style,
    )

    print(f"Calling {model} for one-step SRT narration (style={style!r})...", file=sys.stderr)
    result = _stream_and_collect(
        call_fn, client, model, llm_prompts.NARRATE_SRT_SYSTEM, user_text
    )

    # Re-process: split each block into recording-friendly chunks (≤20 chars)
    # so the voice actor can read each line in one breath.
    from ..core.srt import split_narration_for_recording, allocate_chunk_times, write_srt
    _blocks = re.split(r"\n\s*\n", result.strip())
    final_entries = []
    prev_end = None
    for block in _blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(
            r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)\s*",
            lines[1],
        )
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start_ms = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end_ms = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        text = "".join(l.strip() for l in lines[2:])

        chunks = split_narration_for_recording(text)
        if not chunks:
            continue

        # Allocate time proportional to character count, no overlap with previous
        entries, prev_end = allocate_chunk_times(chunks, start_ms, end_ms, prev_end)
        final_entries.extend(entries)

    write_srt(out_path, final_entries)
    print(f"Wrote {out_path} ({len(final_entries)} entries)")
    print("  ⚠ LLM 生成为草稿，请人工 review 时间与文本后再跑 tts。", file=sys.stderr)


def generate_narration_visual(
    subtitle_path: Path,
    visual_description_path: Path,
    out_path: Path,
    video_duration_ms: int,
    style: str = "",
    video_title: str = "",
    model: str = "qwen2.5:7b",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    provider: str = "openai_compatible",
    base_url: str = "",
) -> None:
    """Generate narration SRT anchored to visual event timestamps.

    Parses visual_description.txt, groups nearby events into larger windows
    (≥3s each), then asks the LLM to write one narration line per window.
    SRT is built using the exact window timestamps — no drift, no pile-up.
    """
    from ..core.srt import (
        parse_visual_description, split_narration_for_recording,
        allocate_chunk_times, SPEAKING_RATE_CHARS_PER_SEC, write_srt,
    )

    # Parse visual description
    visual_events = parse_visual_description(visual_description_path)
    if not visual_events:
        raise SystemExit(f"No visual events parsed from {visual_description_path}")

    # Filter out skip events (Logo, black screen) for narration
    active_events = [(ms, desc) for ms, desc, skip in visual_events if not skip]
    if not active_events:
        raise SystemExit("No active visual events after filtering.")

    # Group consecutive events into windows with a target duration.
    # Events are typically ~1s apart (max_frame_interval=1), so 87 events in 87s
    # = 1s/event — too short for narration. We aim for ~3s per window.
    _TARGET_WINDOW_MS = 3000

    windows: list[tuple[int, int, str, list[int]]] = []  # (start, end, desc, event_indices)
    win_start = active_events[0][0]
    win_descs: list[str] = [active_events[0][1]]
    win_indices: list[int] = [0]

    for i in range(1, len(active_events)):
        cur_ts, cur_desc = active_events[i]
        prev_ts = active_events[i - 1][0]
        elapsed = cur_ts - win_start

        if elapsed >= _TARGET_WINDOW_MS:
            # Finalize current window
            win_end = cur_ts
            desc = " → ".join(win_descs) if len(win_descs) > 1 else win_descs[0]
            windows.append((win_start, win_end, desc, win_indices))
            win_start = cur_ts
            win_descs = [cur_desc]
            win_indices = [i]
        else:
            win_descs.append(cur_desc)
            win_indices.append(i)

    # Last window: extend past the last event (use max of video_duration_ms
    # and last_event_ts + 3s to handle subtitle.srt that ends before video)
    last_event_ts = win_start
    effective_video_dur = max(video_duration_ms, last_event_ts + 3000)
    desc = " → ".join(win_descs) if len(win_descs) > 1 else win_descs[0]
    windows.append((win_start, effective_video_dur, desc, win_indices))

    # Calculate max chars
    duration_sec = video_duration_ms // 1000
    max_chars = int(duration_sec * SPEAKING_RATE_CHARS_PER_SEC)

    # Build visual events text for prompt — numbered list with timestamps
    visual_events_lines = []
    for i, (win_start_ms, win_end_ms, win_desc, _) in enumerate(windows):
        ts = ms_to_ts(win_start_ms)
        window_ms = win_end_ms - win_start_ms
        window_sec = max(1, window_ms // 1000)
        window_chars = int(window_sec * SPEAKING_RATE_CHARS_PER_SEC)
        visual_events_lines.append(
            f"  {i+1}. [{ts}] {win_desc}  （窗口 {window_sec} 秒，最多 {window_chars} 字）"
        )

    # Build optional subtitle section
    subtitle_section = ""
    if subtitle_path.exists():
        subtitle_text, _ = _srt_to_prompt_text(subtitle_path)
        subtitle_section = f"【原视频英文字幕】（带时间戳，仅供参考剧情）：\n\n{subtitle_text}\n\n"

    client, call_fn = _get_backend(provider, base_url, api_key, api_key_env)

    user_text = llm_prompts.VISUAL_ANCHOR_USER_TEMPLATE.format(
        video_title=video_title or "未知",
        duration_sec=duration_sec,
        max_chars=max_chars,
        subtitle_section=subtitle_section,
        visual_events="\n".join(visual_events_lines),
    )

    print(f"Calling {model} for visual-anchored narration (style={style!r}, {len(windows)} windows from {len(active_events)} events)...",
          file=sys.stderr)
    result = _stream_and_collect(
        call_fn, client, model, llm_prompts.VISUAL_ANCHOR_SYSTEM, user_text
    )

    # Post-process: parse LLM output as "N. text" lines, build SRT with
    # exact window timestamps — no time drift.
    _line_re = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
    narration_map: dict[int, str] = {}
    for line in result.splitlines():
        m = _line_re.match(line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            if text:
                narration_map[idx] = text

    # Build SRT entries: each window gets one narration with its exact timestamp
    final_entries = []
    prev_end = None

    for i, (win_start_ms, win_end_ms, win_desc, _) in enumerate(windows):
        idx = i + 1
        if idx not in narration_map:
            continue

        text = narration_map[idx]

        # Split text into recording-friendly chunks
        chunks = split_narration_for_recording(text)
        if not chunks:
            continue

        # Allocate time proportional to character count, anchored to window timestamp
        entries, prev_end = allocate_chunk_times(
            chunks, win_start_ms, win_end_ms, prev_end, max_end_ms=video_duration_ms,
        )
        final_entries.extend(entries)

    write_srt(out_path, final_entries)
    total_chars = sum(len(t) for _, _, t in final_entries)
    print(f"Wrote {out_path} ({len(final_entries)} entries, {total_chars} chars)")
    print("  ⚠ LLM 生成为草稿，请人工 review 时间与文本后再跑 tts。", file=sys.stderr)


def _segment_subtitle_windows(subtitle_path: Path, n: int) -> list[tuple[int, int]]:
    """Split the subtitle timeline into n approximately equal windows."""
    entries = parse_srt(subtitle_path)
    if not entries:
        raise SystemExit(f"No entries in {subtitle_path}")
    total_ms = entries[-1][2]
    window = total_ms // n
    windows = []
    for i in range(n):
        start = i * window
        end = (i + 1) * window if i < n - 1 else total_ms
        windows.append((start, end))
    return windows


def _visual_description_windows(
    visual_description_path: Path,
    video_duration_ms: int,
    min_window_sec: int = 3,
) -> list[tuple[int, int, str]]:
    """Create time windows from visual description events.

    Consecutive events are grouped into windows of approximately
    *min_window_sec* seconds so each has enough duration for a meaningful
    narration segment. Without consolidation, 87 events in 87 seconds = 1s
    each, which is too short for narration.

    Returns list of (start_ms, end_ms, description) where each window spans
    from the first event in the group to the next group's first event.
    Skip events (Logo/black-screen) are filtered out but their time gaps are
    preserved.
    """
    from ..core.srt import parse_visual_description
    events = parse_visual_description(visual_description_path)
    if not events:
        return []

    # Get non-skip events as anchors
    anchors = [(ms, desc) for ms, desc, skip in events if not skip]
    if not anchors:
        return []

    # Group consecutive events by target duration (not gap)
    windows = []
    group_start = anchors[0][0]
    group_descs = [anchors[0][1]]

    for i in range(1, len(anchors)):
        cur_ts, cur_desc = anchors[i]
        elapsed = cur_ts - group_start

        group_descs.append(cur_desc)

        if elapsed >= min_window_sec * 1000:
            window_end = cur_ts
            combined = group_descs[0] if len(group_descs) == 1 else f"{group_descs[0]} → {group_descs[-1]}"
            windows.append((group_start, window_end, combined))
            group_start = cur_ts
            group_descs = [cur_desc]

    # Last group: extend to max(video_end, last_event + min_window_sec)
    if group_descs:
        combined = group_descs[0] if len(group_descs) == 1 else f"{group_descs[0]} → {group_descs[-1]}"
        effective_dur = max(video_duration_ms, group_start + min_window_sec * 1000)
        windows.append((group_start, effective_dur, combined))

    return windows


def align_narration(
    narration_path: Path,
    subtitle_path: Path,
    out_path: Path,
    n_segments: int = 3,
    char_limits: list[int] | None = None,
    video_duration_ms: int | None = None,
    visual_description_path: Path | None = None,
    model: str = "qwen2.5:7b",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    provider: str = "openai_compatible",
    base_url: str = "",
) -> None:
    """Split narration.txt into timed blocks → narration_aligned.txt.

    If visual_description_path is provided, time windows are created from
    visual event timestamps (one window per visual event), ensuring each
    narration segment aligns with what the viewer sees on screen.
    Otherwise, falls back to equal subtitle windows.
    """
    client, call_fn = _get_backend(provider, base_url, api_key, api_key_env)
    narration = narration_path.read_text(encoding="utf-8").strip()

    # Prefer visual-based windows if available
    visual_windows = []
    if visual_description_path and visual_description_path.exists() and video_duration_ms:
        visual_windows = _visual_description_windows(visual_description_path, video_duration_ms)

    if visual_windows:
        from ..core.srt import SPEAKING_RATE_CHARS_PER_SEC
        windows_text_lines = []
        for i, (s, e, desc) in enumerate(visual_windows):
            dur = (e - s) // 1000
            max_c = int(dur * SPEAKING_RATE_CHARS_PER_SEC)
            windows_text_lines.append(
                f"  第 {i+1} 段：{ms_to_ts(s)} --> {ms_to_ts(e)}  时长 {dur} 秒（最多 {max_c} 字）\n"
                f"    画面内容：{desc}"
            )
        windows_text = "\n".join(windows_text_lines)
        n_actual = len(visual_windows)
    else:
        windows = _segment_subtitle_windows(subtitle_path, n_segments)
        windows_text_lines = []
        for i, (s, e) in enumerate(windows):
            dur = (e - s) // 1000
            char_info = f"（最多 {char_limits[i]} 字）" if char_limits else ""
            windows_text_lines.append(f"  第 {i+1} 段：{ms_to_ts(s)} --> {ms_to_ts(e)}  时长 {dur} 秒 {char_info}")
        windows_text = "\n".join(windows_text_lines)
        n_actual = n_segments

    user_text = llm_prompts.ALIGN_USER_TEMPLATE.format(
        narration=narration,
        n_segments=n_actual,
        windows_text=windows_text,
    )

    print(f"Calling {model} for alignment ({n_actual} segments)...", file=sys.stderr)
    result = _stream_and_collect(
        call_fn, client, model, llm_prompts.ALIGN_SYSTEM, user_text
    )

    # Re-process: split each block's text into recording-friendly chunks (≤20 chars)
    # so the voice actor can read each line in one breath.
    from ..core.srt import split_narration_for_recording, allocate_chunk_times, write_srt
    _blocks = re.split(r"\n\s*\n", result.strip())
    final_entries = []
    prev_end = None
    for block in _blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(
            r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)\s*",
            lines[1],
        )
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start_ms = (h1 * 3600 + m1 * 60 + s1) * 1000 + ms1
        end_ms = (h2 * 3600 + m2 * 60 + s2) * 1000 + ms2
        text = "".join(l.strip() for l in lines[2:])

        # Clamp LLM timestamps to video duration (LLMs often hallucinate times
        # beyond the actual video length, especially with many segments)
        if video_duration_ms is not None:
            start_ms = min(start_ms, max(0, video_duration_ms - 800))
            end_ms = min(end_ms, video_duration_ms)

        # Split text into ≤20 char chunks at natural breakpoints
        chunks = split_narration_for_recording(text)
        if not chunks:
            continue

        # Allocate time proportional to character count, constrained to video duration
        max_end = video_duration_ms or end_ms
        entries, prev_end = allocate_chunk_times(
            chunks, start_ms, end_ms, prev_end, max_end_ms=max_end,
        )
        final_entries.extend(entries)

    write_srt(out_path, final_entries)
    print(f"Wrote {out_path} ({len(final_entries)} entries)")
    print("  ⚠ LLM 对齐为草稿，请人工 review 时间与文本后再跑 tts。", file=sys.stderr)


_TRANSLATE_LINE_RE = re.compile(r"^\s*(\d+)\s*[.\．、:：)]\s*(.+?)\s*$")


def translate_srt(
    src_path: Path,
    out_path: Path,
    model: str = "qwen2.5:7b",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    provider: str = "openai_compatible",
    base_url: str = "",
) -> int:
    """Translate an English SRT into Chinese, preserving every timestamp.

    Missing or unparseable lines fall back to the original English text so the
    output SRT is never shorter than the input.
    """
    entries = parse_srt(src_path)
    if not entries:
        raise SystemExit(f"No entries parsed from {src_path}")

    client, call_fn = _get_backend(provider, base_url, api_key, api_key_env)

    numbered = "\n".join(f"{i}. {text}" for i, (_, _, _, text) in enumerate(entries, 1))
    user_text = llm_prompts.TRANSLATE_SRT_USER_TEMPLATE.format(numbered_text=numbered)

    print(f"Calling {model} to translate {len(entries)} entries...", file=sys.stderr)
    result = _stream_and_collect(
        call_fn, client, model, llm_prompts.TRANSLATE_SRT_SYSTEM, user_text
    )

    translations: dict[int, str] = {}
    for line in result.splitlines():
        m = _TRANSLATE_LINE_RE.match(line)
        if m:
            translations[int(m.group(1))] = m.group(2).strip()

    missing = [i for i in range(1, len(entries) + 1) if i not in translations]
    if missing:
        print(
            f"  ⚠ Missing {len(missing)} translation(s); falling back to English for: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}",
            file=sys.stderr,
        )

    out_entries = [
        (s, e, translations.get(i, text))
        for i, (_, s, e, text) in enumerate(entries, 1)
    ]
    write_srt(out_path, out_entries)
    print(f"Wrote {out_path} ({len(out_entries)} entries)")
    return len(out_entries)
