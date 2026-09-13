"""Video export — per-frame caption compositing and simple re-encode modes.

Caption mode:  Decode (FFmpeg) → composite captions (Pillow) → encode (FFmpeg).
Simple mode:   Single FFmpeg re-encode command (no captions, much faster).
"""

import logging
import os
import re
import subprocess

import numpy as np
from PIL import Image

from core.captions import render_caption
from core.transcribe import WordGroup
from utils.gpu import get_encoder, get_ffmpeg_path, has_nvenc

log = logging.getLogger(__name__)

# Quality levels — plain-English labels mapped to encoder settings
QUALITY_LEVELS = ["Tiny file", "Small file", "Medium", "High quality", "Best quality"]

QUALITY_PRESETS = {
    "Tiny file":     {"libx264": {"crf": "32", "preset": "fast"},   "h264_nvenc": {"qp": "32", "preset": "fast"}},
    "Small file":    {"libx264": {"crf": "28", "preset": "fast"},   "h264_nvenc": {"qp": "28", "preset": "fast"}},
    "Medium":        {"libx264": {"crf": "24", "preset": "medium"}, "h264_nvenc": {"qp": "24", "preset": "medium"}},
    "High quality":  {"libx264": {"crf": "21", "preset": "medium"}, "h264_nvenc": {"qp": "21", "preset": "medium"}},
    "Best quality":  {"libx264": {"crf": "18", "preset": "slow"},   "h264_nvenc": {"qp": "18", "preset": "slow"}},
}

# Estimated video bitrate (Mbps) at 1080x1920 30fps
_EST_BITRATES = {
    "Tiny file": 0.8,
    "Small file": 2.0,
    "Medium": 4.5,
    "High quality": 7.5,
    "Best quality": 12.0,
}
_AUDIO_BITRATE_MBPS = 0.192  # 192 kbps AAC

# Map old preset names -> new names (for saved-preset migration)
_LEGACY_MAP = {
    "Fast": "Small file",
    "Balanced": "Medium",
    "High Quality": "High quality",
}


def migrate_quality_preset(name: str) -> str:
    """Convert a legacy quality preset name to the current label."""
    return _LEGACY_MAP.get(name, name)


def estimate_file_size_mb(
    quality_level: str,
    duration_seconds: float,
    width: int = 1080,
    height: int = 1920,
) -> float:
    """Estimate output file size in MB."""
    video_bitrate = _EST_BITRATES.get(quality_level, 4.5)
    pixel_ratio = (width * height) / (1080 * 1920)
    total_mbps = video_bitrate * pixel_ratio + _AUDIO_BITRATE_MBPS
    return total_mbps * duration_seconds / 8


# ──────────────────────────────────────────────
# Probing helpers
# ──────────────────────────────────────────────

def _probe_video(ffmpeg: str, video_path: str) -> dict:
    """Probe a video for fps, duration, and resolution."""
    cmd = [ffmpeg, "-i", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return {}

    info = {}
    stderr = result.stderr

    # Resolution
    m = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
    if m:
        info["width"] = int(m.group(1))
        info["height"] = int(m.group(2))

    # Duration
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
    if m:
        h, mm, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        frac = int(m.group(4)) / (10 ** len(m.group(4)))
        info["duration"] = h * 3600 + mm * 60 + s + frac

    # FPS (try "fps" first, then "tbr")
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)\s*tbr", stderr)
    if m:
        info["fps"] = float(m.group(1))

    return info


# ──────────────────────────────────────────────
# Per-frame render loop
# ──────────────────────────────────────────────

def _render_loop(
    ffmpeg: str,
    video_path: str,
    output_path: str,
    groups: list[WordGroup],
    frame_width: int,
    frame_height: int,
    fps: float,
    encoder: str,
    enc_config: dict,
    font_size: int,
    y_position_pct: float,
    bounce_intensity: float,
    bounce_duration_ms: float,
    all_caps: bool,
    drop_shadow: bool,
    stroke_width: int,
    total_frames: int,
    progress_callback,
) -> str:
    """Decode → Pillow composite → encode pipeline."""
    output_path = os.path.normpath(output_path)

    # Precompute caption overlays at scale=1.0 (reused for most frames)
    static_overlays: dict[int, Image.Image] = {}
    for i, group in enumerate(groups):
        static_overlays[i] = render_caption(
            group, frame_width, frame_height,
            font_size, y_position_pct, 1.0, all_caps, drop_shadow, stroke_width,
        )
    log.info("Precomputed %d caption overlays", len(static_overlays))

    # Decode: raw RGB frames from input video
    # Force constant frame rate output so the frame count matches
    # the audio duration exactly. Without this, VFR sources (phone
    # recordings, post-concat videos) produce more/fewer frames than
    # duration*fps, causing progressive A/V desync.
    decode_cmd = [
        ffmpeg,
        "-i", video_path,
        "-vsync", "cfr",
        "-r", str(fps),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error",
        "-",
    ]

    # Encode: raw RGB frames from pipe + audio from original file
    encode_cmd = [
        ffmpeg, "-y",
        "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{frame_width}x{frame_height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", video_path,
        "-map", "0:v", "-map", "1:a",
        "-pix_fmt", "yuv420p",
        "-c:v", encoder,
    ]

    if encoder == "h264_nvenc":
        encode_cmd.extend([
            "-preset", enc_config["preset"],
            "-qp", enc_config["qp"],
            "-rc", "constqp",
        ])
    else:
        encode_cmd.extend([
            "-preset", enc_config["preset"],
            "-crf", enc_config["crf"],
        ])

    encode_cmd.extend([
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ])

    log.info("Decode cmd: %s", " ".join(decode_cmd))
    log.info("Encode cmd: %s", " ".join(encode_cmd))

    decode_proc = subprocess.Popen(
        decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    encode_proc = subprocess.Popen(
        encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    frame_size = frame_width * frame_height * 3
    frame_idx = 0
    group_idx = 0

    try:
        while True:
            raw = decode_proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            t = frame_idx / fps

            # Advance past expired groups
            # Hold caption 150ms past word end, but not into the next group
            while group_idx < len(groups):
                grp_end = groups[group_idx].end
                next_start = groups[group_idx + 1].start if group_idx + 1 < len(groups) else float('inf')
                hold_end = min(grp_end + 0.15, next_start)
                if hold_end <= t:
                    group_idx += 1
                else:
                    break

            # Check if a caption group is active
            # Whisper timestamps tend to be ~50ms early; nudge forward to sync
            CAPTION_DELAY = 0.05
            if group_idx < len(groups) and groups[group_idx].start + CAPTION_DELAY <= t:
                active_group = groups[group_idx]

                # Bounce scale for the first few frames of each group
                elapsed_ms = (t - active_group.start) * 1000
                needs_bounce = bounce_intensity > 0 and elapsed_ms < bounce_duration_ms

                if needs_bounce:
                    scale = 1.0 + bounce_intensity * (1.0 - elapsed_ms / bounce_duration_ms)
                    overlay = render_caption(
                        active_group, frame_width, frame_height,
                        font_size, y_position_pct, scale, all_caps, drop_shadow, stroke_width,
                    )
                else:
                    overlay = static_overlays[group_idx]

                # Alpha composite caption onto frame
                frame_arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (frame_height, frame_width, 3),
                )
                bg = Image.fromarray(frame_arr).convert("RGBA")
                composited = Image.alpha_composite(bg, overlay)
                raw = np.array(composited.convert("RGB")).tobytes()

            encode_proc.stdin.write(raw)
            frame_idx += 1

            # Progress updates every 30 frames
            if progress_callback and frame_idx % 30 == 0:
                if total_frames > 0:
                    progress_callback(
                        f"Rendering frame {frame_idx}/{total_frames}...",
                    )
                else:
                    progress_callback(f"Rendering frame {frame_idx}...")

    except BrokenPipeError:
        log.warning("Encode pipe closed early at frame %d", frame_idx)

    finally:
        try:
            encode_proc.stdin.close()
        except OSError:
            pass
        decode_proc.stdout.close()

    # Wait for processes to finish and capture stderr
    try:
        encode_ret = encode_proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        encode_proc.kill()
        encode_ret = -1
    encode_stderr = encode_proc.stderr.read().decode(errors="replace") if encode_proc.stderr else ""

    try:
        decode_ret = decode_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        decode_proc.kill()
        decode_ret = -1
    decode_stderr = decode_proc.stderr.read().decode(errors="replace") if decode_proc.stderr else ""

    if encode_ret != 0:
        log.error("FFmpeg encode failed (exit %d): %s", encode_ret, encode_stderr[-1000:])
        raise RuntimeError(
            f"FFmpeg export failed (exit {encode_ret}): {encode_stderr[-500:] if encode_stderr else 'unknown error'}",
        )

    if decode_ret != 0:
        log.warning("FFmpeg decode exited with code %d: %s", decode_ret, decode_stderr[-300:])

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise RuntimeError("Export produced an invalid file")

    log.info("Export complete: %s (%d frames rendered)", output_path, frame_idx)
    return output_path


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def export_video(
    video_path: str,
    output_path: str,
    groups: list[WordGroup],
    frame_width: int,
    frame_height: int,
    font_dir: str = "",
    font_name: str = "",
    font_size: int = 80,
    y_position_pct: float = 38.0,
    bounce_intensity: float = 0.1,
    bounce_duration_ms: float = 150.0,
    all_caps: bool = True,
    drop_shadow: bool = True,
    stroke_width: int = 3,
    quality_preset: str = "High quality",
    use_gpu: bool = True,
    progress_callback=None,
) -> str:
    """Export video with burned-in captions via Pillow per-frame compositing.

    Returns path to the exported file.
    (font_dir and font_name are accepted for API compat but unused —
     fonts are resolved by core.captions internally.)
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found.")

    # Probe video metadata
    info = _probe_video(ffmpeg, video_path)
    fps = info.get("fps", 30.0)
    duration = info.get("duration", 60.0)
    total_frames = int(duration * fps)

    if progress_callback:
        progress_callback("Starting export...")

    log.info(
        "Export: %dx%d @ %.1f fps, ~%d frames, %d caption groups",
        frame_width, frame_height, fps, total_frames, len(groups),
    )

    # Build encoder fallback list
    encoders: list[str] = []
    if use_gpu:
        enc = get_encoder()
        if enc != "libx264":
            encoders.append(enc)
    encoders.append("libx264")

    for encoder in encoders:
        preset_config = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["High quality"])
        enc_config = preset_config.get(encoder, preset_config["libx264"])

        if progress_callback:
            progress_callback(f"Exporting with {encoder}...")
        log.info("Trying encoder: %s (%s)", encoder, quality_preset)

        try:
            result = _render_loop(
                ffmpeg, video_path, output_path, groups,
                frame_width, frame_height, fps, encoder, enc_config,
                font_size, y_position_pct, bounce_intensity, bounce_duration_ms,
                all_caps, drop_shadow, stroke_width, total_frames, progress_callback,
            )
            if progress_callback:
                progress_callback("Export complete!")
            return result

        except RuntimeError:
            if encoder != encoders[-1]:
                log.warning("%s failed, falling back to next encoder...", encoder)
                if progress_callback:
                    progress_callback(f"{encoder} failed, trying libx264...")
                continue
            raise


def export_video_simple(
    video_path: str,
    output_path: str,
    quality_preset: str = "High quality",
    use_gpu: bool = True,
    progress_callback=None,
) -> str:
    """Export video without captions — fast single-pass FFmpeg re-encode.

    Used for long-form videos where no per-frame caption compositing is needed.
    Returns path to the exported file.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found.")

    output_path = os.path.normpath(output_path)

    if progress_callback:
        progress_callback("Starting export (no captions)...")

    info = _probe_video(ffmpeg, video_path)
    log.info(
        "Simple export: %dx%d, quality=%s",
        info.get("width", "?"), info.get("height", "?"), quality_preset,
    )

    # Build encoder fallback list
    encoders: list[str] = []
    if use_gpu:
        enc = get_encoder()
        if enc != "libx264":
            encoders.append(enc)
    encoders.append("libx264")

    for encoder in encoders:
        preset_config = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["High quality"])
        enc_config = preset_config.get(encoder, preset_config["libx264"])

        if progress_callback:
            progress_callback(f"Encoding with {encoder}...")
        log.info("Simple export: trying %s (%s)", encoder, quality_preset)

        cmd = [
            ffmpeg, "-y",
            "-i", video_path,
            "-pix_fmt", "yuv420p",
            "-c:v", encoder,
        ]

        if encoder == "h264_nvenc":
            cmd.extend([
                "-preset", enc_config["preset"],
                "-qp", enc_config["qp"],
                "-rc", "constqp",
            ])
        else:
            cmd.extend([
                "-preset", enc_config["preset"],
                "-crf", enc_config["crf"],
            ])

        cmd.extend([
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ])

        log.info("Simple export cmd: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed: {result.stderr[-500:] if result.stderr else 'unknown error'}"
                )

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise RuntimeError("Export produced an invalid file")

            log.info("Simple export complete: %s", output_path)
            if progress_callback:
                progress_callback("Export complete!")
            return output_path

        except RuntimeError:
            if encoder != encoders[-1]:
                log.warning("%s failed, falling back to next encoder...", encoder)
                if progress_callback:
                    progress_callback(f"{encoder} failed, trying libx264...")
                continue
            raise
