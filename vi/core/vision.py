"""Video frame extraction + vision model analysis.

Extract key frames from the source video via ffmpeg, then describe each
frame using a multimodal LLM (e.g. GLM-4V-Flash) via the OpenAI-compatible
chat completions API with image inputs.

The result is a timeline of visual descriptions that can be fed into the
narration prompt so the LLM knows what's actually happening on screen.
"""
import base64
import subprocess
import sys
import tempfile
from pathlib import Path

from .srt import ms_to_ts


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_key_frames(
    video_path: Path,
    interval_sec: int = 3,
    tmp_dir: Path | None = None,
) -> list[tuple[int, Path]]:
    """Extract one frame every *interval_sec* seconds from *video_path*.

    Returns a list of (timestamp_ms, frame_path) sorted by time.
    Frames are saved as JPEG in *tmp_dir* (or a temp dir if None).
    """
    out_dir = tmp_dir or Path(tempfile.mkdtemp(prefix="vi_frames_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg -i source.mp4 -vf fps=1/3 frame_%04d.jpg
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_sec}",
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found. Install it: apt-get install ffmpeg")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg frame extraction failed: {e.stderr.decode()}")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"No frames extracted from {video_path}")

    result = []
    for i, fp in enumerate(frames):
        ts_ms = (i + 1) * interval_sec * 1000  # 1-based: frame_0001 = 3s
        # Clamp first frame to 0 for accuracy
        if i == 0:
            ts_ms = 0
        result.append((ts_ms, fp))
    return result


# ---------------------------------------------------------------------------
# Vision model analysis
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """你是一位视频画面描述专家。你会看到视频的多个关键帧截图，请用简洁的中文描述每一帧的画面内容。

要求：
- 每帧用一句话描述（不超过20个字）
- 重点描述：人物、动作、场景、氛围、特效
- 如果是片头Logo或黑屏，直接说"片头Logo"或"黑屏过渡"
- 按时间顺序输出，格式：[时间] 描述"""

VISION_USER_TEMPLATE = """以下是视频【{video_title}】的关键帧截图，请依次描述每帧画面内容：

{frame_list}"""


def _encode_image_base64(path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _analyze_batch(
    client,
    vision_model: str,
    batch: list[tuple[int, Path]],
    video_title: str,
) -> str:
    """Analyze a single batch of frames via vision model. Returns description text."""
    # Build message content: interleaved text descriptions and images
    content_parts = []
    for ts_ms, fp in batch:
        ts_str = ms_to_ts(ts_ms)
        content_parts.append({
            "type": "text",
            "text": f"[{ts_str}]",
        })
        b64 = _encode_image_base64(fp)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        })

    user_text = VISION_USER_TEMPLATE.format(
        video_title=video_title,
        frame_list=f"（共 {len(batch)} 帧，见下方图片）",
    )

    stream = client.chat.completions.create(
        model=vision_model,
        max_tokens=4096,
        stream=True,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ],
    )

    parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            parts.append(delta.content)
            sys.stderr.write(".")
            sys.stderr.flush()
    sys.stderr.write("\n")

    return "".join(parts).strip()


BATCH_SIZE = 15  # Max frames per API call to avoid truncation


def analyze_frames(
    frames: list[tuple[int, Path]],
    video_title: str,
    base_url: str,
    api_key: str,
    vision_model: str = "glm-4v-flash",
) -> str:
    """Send frames to a vision model and return a combined description string.

    Frames are processed in batches to avoid token limit truncation.
    Returns a multi-line string like:
        [00:00:00,000] 片头Logo
        [00:00:03,000] 荒野中一个男人带着狗行走
        ...
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai SDK not installed. Run: pip install openai")

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)

    if len(frames) <= BATCH_SIZE:
        # Single batch
        print(f"Calling {vision_model} for frame analysis ({len(frames)} frames)...", file=sys.stderr)
        return _analyze_batch(client, vision_model, frames, video_title)

    # Multiple batches
    all_descriptions = []
    n_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(frames))
        batch = frames[start:end]
        print(
            f"Calling {vision_model} for frame analysis "
            f"(batch {i+1}/{n_batches}, frames {start+1}-{end})...",
            file=sys.stderr,
        )
        desc = _analyze_batch(client, vision_model, batch, video_title)
        all_descriptions.append(desc)

    return "\n".join(all_descriptions)


# ---------------------------------------------------------------------------
# High-level: extract + analyze + cleanup
# ---------------------------------------------------------------------------

def describe_video(
    video_path: Path,
    video_title: str,
    base_url: str,
    api_key: str,
    vision_model: str = "glm-4v-flash",
    interval_sec: int = 3,
) -> str:
    """Extract key frames from video, analyze with vision model, return description.

    Also saves description to <project_dir>/visual_description.txt for caching.
    Temporary frame images are cleaned up automatically.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vi_frames_") as tmp:
        tmp_dir = Path(tmp)
        frames = extract_key_frames(video_path, interval_sec, tmp_dir)
        description = analyze_frames(
            frames, video_title, base_url, api_key, vision_model
        )

    # Cache to project dir
    cache_path = video_path.parent / "visual_description.txt"
    cache_path.write_text(description + "\n", encoding="utf-8")
    print(f"Cached visual description to {cache_path}", file=sys.stderr)

    return description
