"""Claude API client wrapper. Uses claude-opus-4-7 + adaptive thinking + prompt caching.

Caching strategy: system prompt (stable) is cached. User content (varies per request)
follows. Re-running with the same style/system but different lengths still hits cache.
"""
import os
import sys
from pathlib import Path

from .srt import parse_srt, ms_to_ts
from . import llm_prompts


def _client(api_key_env: str = "ANTHROPIC_API_KEY"):
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "anthropic SDK not installed. Run: pip install anthropic"
        )
    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(
            f"Environment variable {api_key_env} not set.\n"
            f"Get an API key from https://console.anthropic.com/ and set it:\n"
            f"  export {api_key_env}=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


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


def generate_narration(
    subtitle_path: Path,
    out_path: Path,
    style: str,
    length: int,
    model: str = "claude-opus-4-7",
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> None:
    """Generate narration.txt from English subtitle.srt via Claude."""
    client = _client(api_key_env)
    subtitle_text, duration_sec = _srt_to_prompt_text(subtitle_path)

    user_text = llm_prompts.NARRATE_USER_TEMPLATE.format(
        subtitle_text=subtitle_text,
        duration_sec=duration_sec,
        style=style,
        length=length,
    )

    print(f"Calling {model} (style={style!r}, length={length})...", file=sys.stderr)
    parts = []
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": llm_prompts.NARRATE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
            sys.stderr.write(".")
            sys.stderr.flush()
        final = stream.get_final_message()

    sys.stderr.write("\n")
    print(
        f"Tokens — input: {final.usage.input_tokens}, "
        f"cache_write: {final.usage.cache_creation_input_tokens}, "
        f"cache_read: {final.usage.cache_read_input_tokens}, "
        f"output: {final.usage.output_tokens}",
        file=sys.stderr,
    )

    out_path.write_text("".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


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
    model: str = "claude-opus-4-7",
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> None:
    """Split narration.txt into n_segments timed blocks → narration_aligned.txt."""
    client = _client(api_key_env)
    narration = narration_path.read_text(encoding="utf-8").strip()
    windows = _segment_subtitle_windows(subtitle_path, n_segments)
    windows_text = "\n".join(
        f"  第 {i+1} 段：{ms_to_ts(s)} --> {ms_to_ts(e)}"
        for i, (s, e) in enumerate(windows)
    )

    user_text = llm_prompts.ALIGN_USER_TEMPLATE.format(
        narration=narration,
        n_segments=n_segments,
        windows_text=windows_text,
    )

    print(f"Calling {model} for alignment ({n_segments} segments)...", file=sys.stderr)
    parts = []
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": llm_prompts.ALIGN_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
            sys.stderr.write(".")
            sys.stderr.flush()
        final = stream.get_final_message()

    sys.stderr.write("\n")
    print(
        f"Tokens — input: {final.usage.input_tokens}, "
        f"cache_write: {final.usage.cache_creation_input_tokens}, "
        f"cache_read: {final.usage.cache_read_input_tokens}, "
        f"output: {final.usage.output_tokens}",
        file=sys.stderr,
    )

    out_path.write_text("".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("  ⚠ LLM 对齐为草稿，请人工 review 时间与文本后再跑 tts。", file=sys.stderr)
