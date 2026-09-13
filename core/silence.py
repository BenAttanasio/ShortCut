"""Silence detection and removal using FFmpeg."""

import logging
import re
import subprocess
import tempfile
import os
from dataclasses import dataclass

from utils.gpu import get_ffmpeg_path, get_ffprobe_path

log = logging.getLogger(__name__)


@dataclass
class SilenceSegment:
    start: float
    end: float


@dataclass
class SilenceResult:
    """Result of silence detection and removal."""
    original_duration: float
    trimmed_duration: float
    silence_segments: list[SilenceSegment]
    output_path: str


def _get_duration(video_path: str, ffmpeg: str) -> float:
    """Get video duration using ffprobe, or ffmpeg as fallback."""
    ffprobe = get_ffprobe_path()
    if ffprobe:
        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass

    # Fallback: use ffmpeg to get duration from stderr (header only)
    cmd = [ffmpeg, "-i", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if m:
        h, mi, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return h * 3600 + mi * 60 + s + cs / 100.0
    raise RuntimeError("Could not determine video duration.")


def detect_silences(
    video_path: str,
    threshold_db: float = -35.0,
    min_duration: float = 0.4,
) -> tuple[list[SilenceSegment], float]:
    """Detect silent segments in a video using FFmpeg silencedetect.

    Returns (silence_segments, total_duration).
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and add it to PATH.\n"
            "On Windows: winget install Gyan.FFmpeg"
        )

    total_duration = _get_duration(video_path, ffmpeg)

    # Run silence detection
    cmd = [
        ffmpeg, "-i", video_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            f"Silence detection failed: {(result.stderr or '')[-500:]}"
        )
    stderr = result.stderr

    # Parse silence_start and silence_end from FFmpeg output
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", stderr)]

    segments = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else total_duration
        segments.append(SilenceSegment(start=start, end=end))

    return segments, total_duration


def get_non_silent_segments(
    silences: list[SilenceSegment],
    total_duration: float,
    padding: float = 0.05,
) -> list[tuple[float, float]]:
    """Compute non-silent time ranges from silence segments.

    Returns list of (start, end) tuples for segments to keep.
    """
    if not silences:
        return [(0.0, total_duration)]

    segments = []
    cursor = 0.0

    for silence in silences:
        seg_start = cursor
        seg_end = silence.start + padding  # keep a tiny bit into the silence

        if seg_end > seg_start + 0.01:  # skip tiny segments
            segments.append((
                max(0.0, seg_start),
                min(total_duration, seg_end),
            ))

        cursor = max(cursor, silence.end - padding)  # resume slightly before speech

    # Trailing segment after last silence
    if cursor < total_duration - 0.01:
        segments.append((cursor, total_duration))

    return segments


def remove_silences(
    video_path: str,
    threshold_db: float = -35.0,
    min_duration: float = 0.4,
    padding: float = 0.05,
    progress_callback=None,
) -> SilenceResult:
    """Full pipeline: detect silences and produce a trimmed video.

    Uses FFmpeg concat demuxer for reliable audio-video sync.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg and add it to PATH.\n"
            "On Windows: winget install Gyan.FFmpeg"
        )

    if progress_callback:
        progress_callback("Detecting silences...")

    silences, total_duration = detect_silences(video_path, threshold_db, min_duration)

    if not silences:
        if progress_callback:
            progress_callback("No silences detected.")
        return SilenceResult(
            original_duration=total_duration,
            trimmed_duration=total_duration,
            silence_segments=[],
            output_path=video_path,
        )

    segments = get_non_silent_segments(silences, total_duration, padding)

    if progress_callback:
        progress_callback(f"Found {len(silences)} silences. Cutting {len(segments)} segments...")

    # Create temp directory for segment files
    tmp_dir = tempfile.mkdtemp(prefix="reel_silence_")
    segment_files = []

    # Cut each non-silent segment with FFmpeg
    # Re-encode for frame-accurate cuts (-c copy only cuts at keyframes,
    # causing repeated words at segment boundaries)
    for i, (start, end) in enumerate(segments):
        seg_path = os.path.join(tmp_dir, f"seg_{i:04d}.ts")
        cmd = [
            ffmpeg, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            seg_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            log.error("Segment %d cut failed: %s", i, (proc.stderr or "")[-300:])
            continue
        segment_files.append(seg_path)

        if progress_callback:
            pct = (i + 1) / len(segments)
            progress_callback(f"Cutting segment {i + 1}/{len(segments)}", pct)

    # Write concat list
    concat_list_path = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list_path, "w") as f:
        for seg_path in segment_files:
            safe_path = seg_path.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    # Concatenate all segments
    output_path = os.path.join(tmp_dir, "trimmed.mp4")
    concat_cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg concat failed during silence removal: {(proc.stderr or '')[-500:]}"
        )

    # Calculate trimmed duration
    try:
        trimmed_duration = _get_duration(output_path, ffmpeg)
    except Exception:
        trimmed_duration = sum(end - start for start, end in segments)

    # Clean up segment files (keep output)
    for seg_path in segment_files:
        try:
            os.remove(seg_path)
        except OSError:
            pass
    try:
        os.remove(concat_list_path)
    except OSError:
        pass
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise RuntimeError("Silence removal produced an invalid output file")

    return SilenceResult(
        original_duration=total_duration,
        trimmed_duration=trimmed_duration,
        silence_segments=silences,
        output_path=output_path,
    )
