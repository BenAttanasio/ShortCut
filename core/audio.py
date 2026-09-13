"""Audio loudness helpers — normalize or boost volume with FFmpeg."""

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Callable, Optional

from utils.gpu import get_ffmpeg_path, get_ffprobe_path

log = logging.getLogger(__name__)

# ── Natural method targets ───────────────────────
TARGET_I = -16.0    # Integrated loudness (LUFS)
TARGET_TP = -1.5    # True peak (dBTP) — headroom to prevent clipping
TARGET_LRA = 11.0   # Loudness range — preserves natural dynamics

_LOUDNORM_BASE = f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"

# Safety limiter (shared by both methods)
_ALIMITER_LIMIT = 0.95      # ~-0.45 dBFS ceiling
_ALIMITER_ATTACK = 5         # ms — fast enough for transients, gentle on voice
_ALIMITER_RELEASE = 50       # ms — short release avoids pumping
_ALIMITER_ASC = True         # Smooth softclip

_MAX_GAIN_DB = 20.0          # Cap to prevent noise floor amplification
_SKIP_THRESHOLD_DB = 1.0     # Skip if already within 1 dB of target

# ── Output verification (closed loop) ────────────
_VERIFY_TOLERANCE_DB = 1.5   # Acceptable undershoot before a correction pass
_VERIFY_CLIP_TP = -0.1       # Output true peak at/above this (dBTP) = real clipping
                             # (alimiter holds peaks ~-0.45 dBFS, so a healthy
                             #  output sits near -0.5; only flag genuine 0 dBFS hits)
_CORRECTION_MAX_DB = 12.0    # Cap on the single correction pass

_ALIMITER_CHAIN = (
    f"alimiter=limit={_ALIMITER_LIMIT}"
    f":attack={_ALIMITER_ATTACK}"
    f":release={_ALIMITER_RELEASE}"
    f":asc={'1' if _ALIMITER_ASC else '0'}"
)


# ── Multi-track audio handling ───────────────────
def count_audio_streams(path: str) -> int:
    """Return the number of audio streams in a media file.

    Uses ffprobe; returns 0 if ffprobe is unavailable or errors, so callers
    degrade gracefully to legacy single-track behavior instead of crashing.
    """
    ffprobe = get_ffprobe_path()
    if not ffprobe:
        log.warning("ffprobe not found — assuming single audio track")
        return 0
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("ffprobe failed (%s) — assuming single audio track", e)
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def combine_audio_tracks(
    input_path: str,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Mix all audio streams of a file down to one normalized track.

    The video stream is copied (no re-encode). Returns the path to a temp
    working copy that has a single combined audio track. Intended to be called
    only when ``count_audio_streams(input_path) > 1``.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Install it (Windows: winget install Gyan.FFmpeg)"
        )

    n = count_audio_streams(input_path)
    suffix = os.path.splitext(input_path)[1] or ".mkv"
    out_path = tempfile.mktemp(suffix=suffix, prefix="reel_combined_")

    # amix with normalize=1 sums all streams then auto-scales by ~1/N so the mix
    # can't clip. If desktop audio comes out too quiet on real recordings, swap
    # the filter for a summing pan that keeps full level, e.g. (2-track stereo):
    #   amerge=inputs=2,pan=stereo|c0<c0+c2|c1<c1+c3
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-filter_complex", f"amix=inputs={n}:duration=longest:normalize=1",
        "-map", "0:v:0",
        "-c:v", "copy",
        out_path,
    ]
    log.info("Combining %d audio tracks: %s", n, os.path.basename(input_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if not os.path.exists(out_path):
        raise RuntimeError(
            f"Failed to combine audio tracks. "
            f"FFmpeg: {(proc.stderr or '')[-300:]}"
        )
    return out_path


def normalize_loudness(
    input_path: str,
    method: str = "natural",
    boost_db: float = 6.0,
    target_lufs: float = TARGET_I,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Normalize or boost audio loudness.

    Methods:
      'natural'  — Measure LUFS, apply linear gain to reach the target, then
                    re-measure the output and correct if it fell short.
                    Preserves dynamics and verifies the result is loud enough
                    with no clipping.
      'boost'    — Apply a flat dB boost (no measurement, no analysis).
                    Just makes it louder, like turning up the volume knob.

    Returns path to the output file (or input_path if skipped).
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Install it (Windows: winget install Gyan.FFmpeg)"
        )

    if method == "boost":
        return _boost_pass(input_path, ffmpeg, boost_db, progress_callback)

    # method == "natural" — measure, apply, then verify
    return _natural_normalize(input_path, ffmpeg, target_lufs, progress_callback)


def _measure_loudness(input_path: str, ffmpeg: str) -> Optional[dict]:
    """Run loudnorm analysis and return its JSON stats, or None if it failed.

    Keys of interest: input_i (integrated LUFS), input_tp (true peak dBTP).
    """
    measure_cmd = [
        ffmpeg, "-hide_banner",
        "-i", input_path,
        "-af", f"{_LOUDNORM_BASE}:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        measure_cmd, capture_output=True, text=True, timeout=600,
    )
    json_match = re.search(
        r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.DOTALL,
    )
    if not json_match:
        return None
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return None


def _apply_gain(
    input_path: str, ffmpeg: str, gain_db: float, prefix: str,
) -> str:
    """Apply a linear gain + safety limiter, returning a new temp file path."""
    out_path = tempfile.mktemp(suffix=".mp4", prefix=prefix)
    af_filter = f"volume={gain_db:.2f}dB,{_ALIMITER_CHAIN}"
    apply_cmd = [
        ffmpeg, "-hide_banner",
        "-i", input_path,
        "-c:v", "copy",
        "-af", af_filter,
        "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000",
        "-y", out_path,
    ]
    try:
        subprocess.run(apply_cmd, capture_output=True, text=True, timeout=600, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Audio normalization failed: {(e.stderr or '')[-500:]}"
        ) from e
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("Audio normalization produced an invalid file")
    return out_path


def _natural_normalize(
    input_path: str,
    ffmpeg: str,
    target_lufs: float = TARGET_I,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Measure loudness, apply gain to hit the target, then verify the output.

    Closed loop: after applying gain we re-measure the rendered file. If it
    undershot the target (the limiter eating peaks, non-linear behaviour), we
    apply a single correction pass. We also check the output true peak so we
    know for sure it isn't clipping.
    """
    # ── Pass 1: Measure input ────────────────────
    if progress_callback:
        progress_callback("Measuring audio loudness...", 0.0)

    stats = _measure_loudness(input_path, ffmpeg)
    if stats is None:
        if progress_callback:
            progress_callback("Measurement failed — using single-pass fallback", 0.3)
        return _single_pass(input_path, ffmpeg, progress_callback)

    measured_i = float(stats.get("input_i", 0))

    if progress_callback:
        progress_callback(
            f"Measured {measured_i:.1f} LUFS — targeting {target_lufs:.0f} LUFS", 0.4,
        )

    # ── Guard: near-silent input ─────────────────
    if measured_i <= -70.0:
        if progress_callback:
            progress_callback(
                f"Audio is near-silent ({measured_i:.1f} LUFS) — skipping", 1.0,
            )
        return input_path

    # ── Guard: already close to target ───────────
    delta_db = target_lufs - measured_i
    if abs(delta_db) <= _SKIP_THRESHOLD_DB:
        if progress_callback:
            progress_callback(
                f"Already at {measured_i:.1f} LUFS — skipping normalization", 1.0,
            )
        return input_path

    # ── Pass 2: Apply linear gain ────────────────
    if delta_db > _MAX_GAIN_DB:
        if progress_callback:
            progress_callback(
                f"Gain capped at +{_MAX_GAIN_DB:.0f} dB (was +{delta_db:.1f} dB)", 0.5,
            )
        delta_db = _MAX_GAIN_DB
    elif delta_db < -_MAX_GAIN_DB:
        delta_db = -_MAX_GAIN_DB

    if progress_callback:
        direction = "Boosting" if delta_db > 0 else "Reducing"
        progress_callback(f"{direction} volume by {abs(delta_db):.1f} dB...", 0.6)

    out_path = _apply_gain(input_path, ffmpeg, delta_db, "reel_norm_")

    # ── Pass 3: Verify the rendered output ───────
    if progress_callback:
        progress_callback("Verifying levels...", 0.8)

    verify = _measure_loudness(out_path, ffmpeg)
    if verify is None:
        # Can't verify — trust the gain math rather than fail the whole job.
        if progress_callback:
            progress_callback(
                f"Leveled (+{delta_db:.1f} dB, unverified)", 1.0,
            )
        return out_path

    out_i = float(verify.get("input_i", 0))
    out_tp = float(verify.get("input_tp", 0))
    shortfall = target_lufs - out_i

    # Undershot the target → one correction pass.
    if shortfall > _VERIFY_TOLERANCE_DB:
        correction = min(shortfall, _CORRECTION_MAX_DB)
        if progress_callback:
            progress_callback(
                f"Output landed at {out_i:.1f} LUFS — correcting +{correction:.1f} dB", 0.85,
            )
        corrected = _apply_gain(out_path, ffmpeg, correction, "reel_norm2_")
        try:
            os.remove(out_path)
        except OSError:
            pass
        out_path = corrected
        verify = _measure_loudness(out_path, ffmpeg) or verify
        out_i = float(verify.get("input_i", out_i))
        out_tp = float(verify.get("input_tp", out_tp))

    # ── Final report: loud enough? clipping? ─────
    clip_note = "" if out_tp <= _VERIFY_CLIP_TP else f" — WARNING peak {out_tp:.1f} dBTP near clipping"
    if progress_callback:
        progress_callback(
            f"Verified {out_i:.1f} LUFS, peak {out_tp:.1f} dBTP{clip_note}", 1.0,
        )

    return out_path


def _boost_pass(
    input_path: str,
    ffmpeg: str,
    boost_db: float,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Apply a flat dB boost with a safety limiter to prevent clipping."""
    if abs(boost_db) < 0.1:
        if progress_callback:
            progress_callback("Boost is ~0 dB — skipping", 1.0)
        return input_path

    out_path = tempfile.mktemp(suffix=".mp4", prefix="reel_boost_")

    af_filter = f"volume={boost_db:.1f}dB,{_ALIMITER_CHAIN}"

    apply_cmd = [
        ffmpeg, "-hide_banner",
        "-i", input_path,
        "-c:v", "copy",
        "-af", af_filter,
        "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000",
        "-y", out_path,
    ]

    if progress_callback:
        sign = "+" if boost_db > 0 else ""
        progress_callback(f"Applying {sign}{boost_db:.1f} dB volume boost...", 0.5)

    try:
        subprocess.run(apply_cmd, capture_output=True, text=True, timeout=600, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Audio boost failed: {(e.stderr or '')[-500:]}"
        ) from e

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("Audio boost produced an invalid file")

    if progress_callback:
        progress_callback("Volume boosted", 1.0)

    return out_path


def _single_pass(
    input_path: str,
    ffmpeg: str,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Fallback single-pass normalization when measurement fails."""
    out_path = tempfile.mktemp(suffix=".mp4", prefix="reel_norm_")
    cmd = [
        ffmpeg, "-hide_banner",
        "-i", input_path,
        "-c:v", "copy",
        "-af", _LOUDNORM_BASE,
        "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000",
        "-y", out_path,
    ]

    if progress_callback:
        progress_callback("Normalizing audio (single-pass)...", 0.5)

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Audio normalization (single-pass) failed: {(e.stderr or '')[-500:]}"
        ) from e

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError("Audio normalization produced an invalid file")

    return out_path
