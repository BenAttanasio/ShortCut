"""GPU detection utilities for CUDA, NVENC, and FFmpeg support."""

import os
import sys
import subprocess
import shutil
import functools
import glob


def _setup_nvidia_dll_paths():
    """Add nvidia pip package bin dirs to DLL search path on Windows.

    CTranslate2/faster-whisper need cublas64_12.dll, cudnn*.dll etc.
    These are installed by nvidia-cublas-cu12, nvidia-cudnn-cu12 pip packages
    but their bin/ dirs aren't on PATH by default.
    """
    if sys.platform != "win32":
        return

    try:
        import nvidia
        nvidia_root = os.path.dirname(nvidia.__path__[0])
    except (ImportError, AttributeError, IndexError):
        return

    # Find all bin/ directories under nvidia packages
    bin_dirs = glob.glob(os.path.join(nvidia_root, "nvidia", "*", "bin"))
    for bin_dir in bin_dirs:
        if os.path.isdir(bin_dir):
            os.add_dll_directory(bin_dir)
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


# Run on import so DLLs are findable before ctranslate2 loads
_setup_nvidia_dll_paths()


@functools.lru_cache(maxsize=1)
def get_ffmpeg_path() -> str | None:
    """Find FFmpeg binary — system PATH first, then imageio_ffmpeg bundled copy."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def get_ffprobe_path() -> str | None:
    """Find ffprobe binary — system PATH first, then alongside imageio ffmpeg."""
    path = shutil.which("ffprobe")
    if path:
        return path
    # imageio_ffmpeg only bundles ffmpeg, not ffprobe — try same directory
    ffmpeg = get_ffmpeg_path()
    if ffmpeg:
        ffprobe = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
        if os.path.exists(ffprobe) or os.path.exists(ffprobe + ".exe"):
            return ffprobe
    return None


@functools.lru_cache(maxsize=1)
def has_cuda() -> bool:
    """Check if CUDA is available via faster-whisper/CTranslate2."""
    try:
        import ctranslate2
        types = ctranslate2.get_supported_compute_types("cuda")
        return len(types) > 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def has_nvenc() -> bool:
    """Check if FFmpeg h264_nvenc encoder is available."""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def has_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    return get_ffmpeg_path() is not None


def get_whisper_device() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper."""
    if has_cuda():
        return "cuda", "float16"
    return "cpu", "int8"


def get_gpu_memory_info() -> dict | None:
    """Query NVIDIA GPU memory usage via nvidia-smi.

    Returns dict with 'total_mb', 'used_mb', 'free_mb', or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                return {
                    "total_mb": int(parts[0].strip()),
                    "used_mb": int(parts[1].strip()),
                    "free_mb": int(parts[2].strip()),
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_encoder() -> str:
    """Return the best available H.264 encoder."""
    if has_nvenc():
        return "h264_nvenc"
    return "libx264"
