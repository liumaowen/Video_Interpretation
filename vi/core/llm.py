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


def align_narration(
    narration_path: Path,
    subtitle_path: Path,
    out_path: Path,
    n_segments: int = 3,
    char_limits: list[int] | None = None,
    model: str = "qwen2.5:7b",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    provider: str = "openai_compatible",
    base_url: str = "",
) -> None:
    """Split narration.txt into n_segments timed blocks → narration_aligned.txt.

    Uses LLM to draft the alignment, then re-splits each block using the same
    sentence-aware algorithm as whisper_py.group_words (≤60 chars per chunk,
    split at punctuation, ≥800ms per entry).
    """
    client, call_fn = _get_backend(provider, base_url, api_key, api_key_env)
    narration = narration_path.read_text(encoding="utf-8").strip()
    windows = _segment_subtitle_windows(subtitle_path, n_segments)
    windows_text_lines = []
    for i, (s, e) in enumerate(windows):
        dur = (e - s) // 1000
        char_info = f"（最多 {char_limits[i]} 字）" if char_limits else ""
        windows_text_lines.append(f"  第 {i+1} 段：{ms_to_ts(s)} --> {ms_to_ts(e)}  时长 {dur} 秒 {char_info}")
    windows_text = "\n".join(windows_text_lines)

    user_text = llm_prompts.ALIGN_USER_TEMPLATE.format(
        narration=narration,
        n_segments=n_segments,
        windows_text=windows_text,
    )

    print(f"Calling {model} for alignment ({n_segments} segments)...", file=sys.stderr)
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

        # Split text into ≤20 char chunks at natural breakpoints
        chunks = split_narration_for_recording(text)
        if not chunks:
            continue

        # Allocate time proportional to character count, no overlap with previous
        entries, prev_end = allocate_chunk_times(chunks, start_ms, end_ms, prev_end)
        final_entries.extend(entries)

    total_ms = windows[-1][1] if windows else 0
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
