"""Preview utilities — frame extraction, video probing, and sample caption rendering."""

import re
import subprocess
import numpy as np
from PIL import Image

from utils.gpu import get_ffmpeg_path


def probe_video(video_path: str) -> dict | None:
    """Probe a video for duration and resolution.

    Returns dict with 'duration' (seconds), 'width', 'height', or None on failure.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return None

    cmd = [ffmpeg, "-i", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None

    info = {}

    # Parse resolution
    m = re.search(r"(\d{2,5})x(\d{2,5})", result.stderr)
    if m:
        info["width"] = int(m.group(1))
        info["height"] = int(m.group(2))

    # Parse duration  (format: HH:MM:SS.ff)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if m:
        hours = int(m.group(1))
        mins = int(m.group(2))
        secs = int(m.group(3))
        frac = int(m.group(4)) / (10 ** len(m.group(4)))
        info["duration"] = hours * 3600 + mins * 60 + secs + frac

    return info if info else None


def extract_frame(video_path: str, time_sec: float = 2.0) -> np.ndarray | None:
    """Extract a single frame from a video at the given timestamp.

    Returns an RGB numpy array (H, W, 3) or None on failure.
    Uses FFmpeg directly for speed (no need to open the whole clip).
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return None

    cmd = [
        ffmpeg,
        "-ss", str(time_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "error",
        "-"
    ]

    # We need to know the resolution first
    probe_cmd = [ffmpeg, "-i", video_path]

    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    m = re.search(r"(\d{2,5})x(\d{2,5})", probe_result.stderr)
    if not m:
        return None
    width, height = int(m.group(1)), int(m.group(2))

    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or len(result.stdout) == 0:
        return None

    expected_size = width * height * 3
    raw = result.stdout[:expected_size]
    if len(raw) < expected_size:
        return None

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    return frame


def render_preview_image(
    frame: np.ndarray,
    font_size: int,
    y_position_pct: float,
    drop_shadow: bool,
    max_chars_per_line: int = 25,
) -> Image.Image:
    """Render a sample caption on a video frame and return composited PIL Image."""
    from core.transcribe import Word, WordGroup
    from core.captions import render_caption

    h, w = frame.shape[:2]

    # Show W's at exactly max_chars_per_line length — W is the widest
    # letter, so this shows the true worst-case line width
    n = max_chars_per_line
    third = n // 3
    remainder = n - third * 2 - 2  # subtract 2 for spaces between words
    chunks = ["W" * third, "W" * third, "W" * max(1, remainder)]
    colors = ["#FFFFFF", "#00FF6A", "#FFD700"]
    sample_words = [
        Word(text=chunk, start=i * 0.3, end=(i + 1) * 0.3, probability=1.0, color=colors[i])
        for i, chunk in enumerate(chunks)
    ]

    group = WordGroup(words=sample_words)

    overlay = render_caption(
        group=group,
        frame_width=w,
        frame_height=h,
        font_size=font_size,
        y_position_pct=y_position_pct,
        scale=1.0,
        all_caps=True,
        drop_shadow=drop_shadow,
    )

    # Alpha composite using PIL (no dependency on compositor module)
    bg = Image.fromarray(frame).convert("RGBA")
    composited = Image.alpha_composite(bg, overlay)
    return composited.convert("RGB")
