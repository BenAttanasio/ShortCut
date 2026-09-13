"""Reel/Short Auto-Editor — Streamlit UI."""

import logging
import subprocess as _subprocess
import sys
import os
import json
import tempfile
import time
import glob as glob_mod
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

# Load .env for API keys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── File logging with auto-cleanup ──
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Clean up logs older than 30 days on startup
for old_log in glob_mod.glob(os.path.join(LOG_DIR, "*.log")):
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(old_log))
        if datetime.now() - mtime > timedelta(days=30):
            os.remove(old_log)
    except OSError:
        pass

_log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_filename, encoding="utf-8"),
    ],
)
log = logging.getLogger("app")
log.info("Session started — log file: %s", _log_filename)

import streamlit as st

import numpy as np

from utils.gpu import has_cuda, has_nvenc, has_ffmpeg, get_gpu_memory_info
from utils.preview import extract_frame, render_preview_image, probe_video
from core.audio import normalize_loudness
from core.silence import remove_silences
from core.transcribe import transcribe, group_words, unload_model, remove_filler_words
from core.captions import get_font_dir, get_font_name
from core.exporter import (
    export_video,
    export_video_simple,
    estimate_file_size_mb,
    migrate_quality_preset,
    QUALITY_LEVELS,
)
from core.duplicates import remove_duplicates
from core.keywords import assign_colors, fix_group_contrast


def _build_queue(folder_vids, upload_files) -> list[dict]:
    """Return list of {'path': str, 'name': str} for all videos to process."""
    # Source folder takes priority
    if folder_vids:
        return [{"path": p, "name": os.path.basename(p)} for p in folder_vids]

    # Uploaded files from the widget
    if upload_files:
        queue = []
        for uf in upload_files:
            file_id = f"{uf.name}_{uf.size}"
            cache_key = f"_tmp_upload_{file_id}"
            if cache_key not in st.session_state:
                tmp = tempfile.mktemp(
                    suffix=os.path.splitext(uf.name)[1], prefix="reel_",
                )
                with open(tmp, "wb") as f:
                    f.write(uf.getbuffer())
                st.session_state[cache_key] = tmp
            queue.append({"path": st.session_state[cache_key], "name": uf.name})
        # Cache queue info so it survives st.rerun() from browse dialogs
        st.session_state["_cached_upload_queue"] = queue
        return queue

    # Fallback: cached uploads (survive st.rerun after browse dialog)
    cached = st.session_state.get("_cached_upload_queue", [])
    return [q for q in cached if os.path.exists(q["path"])]

st.set_page_config(
    page_title="Reel Auto-Editor",
    page_icon="🎬",
    layout="wide",
)

# ──────────────────────────────────────────────
# Constants & preset system
# ──────────────────────────────────────────────
PRESETS_FILE = os.path.join(os.path.dirname(__file__), "presets.json")
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}

DEFAULTS = {
    "video_mode": "shorts",  # "shorts" or "longform"
    "silence_threshold": -35.0,
    "min_silence_dur": 0.4,
    "silence_padding": 0.05,
    "font_size": 80,
    "caption_y": 38,
    "bounce_intensity": 0.1,
    "bounce_duration": 150,
    "max_words_per_group": 3,
    "max_chars_per_line": 25,
    "drop_shadow": True,
    "remove_duplicates": False,
    "audio_leveling": False,
    "leveling_method": "natural",
    "boost_db": 6.0,
    "target_lufs": -16.0,
    "quality_preset": "High quality",
    "use_gpu_encoding": has_nvenc(),
    "auto_unload_model": True,
    "source_folder": "",
    "export_folder": os.path.join(
        os.path.expanduser("~"),
        "Documents", "Adobe", "Premiere Pro", "24.0", "Exports", "2026",
        "Shorts Processor Exports",
    ),
}

# Keys that belong in presets (exclude folder paths — those stay independent)
_PRESET_KEYS = sorted(k for k in DEFAULTS if k not in ("source_folder", "export_folder"))


def load_presets() -> dict[str, dict]:
    """Load all presets from JSON file."""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_presets(presets: dict[str, dict]):
    """Save all presets to JSON file."""
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)


def get_current_settings() -> dict:
    """Gather current processing settings (no folder paths)."""
    return {k: st.session_state.get(k, DEFAULTS[k]) for k in _PRESET_KEYS}


def apply_preset(preset: dict):
    """Write a preset's values into session state so widgets pick them up."""
    preset = dict(preset)  # don't mutate the original
    if "quality_preset" in preset:
        preset["quality_preset"] = migrate_quality_preset(preset["quality_preset"])
    _excluded = {
        "enable_highlighting", "green_color", "yellow_color",
        "white_color", "all_caps", "source_folder", "export_folder",
    }
    for key, value in preset.items():
        if key not in _excluded:
            st.session_state[key] = value


def discover_videos(folder: str) -> list[str]:
    """Find all video files in a folder, sorted by name."""
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )


def _browse_folder(title: str = "Select folder", initial_dir: str = "") -> str | None:
    """Open a native OS folder picker dialog and return the selected path."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    kwargs = {"title": title}
    if initial_dir and os.path.isdir(initial_dir):
        kwargs["initialdir"] = initial_dir
    folder = filedialog.askdirectory(**kwargs)
    root.destroy()
    return folder if folder else None


# Initialize session state from defaults on first run
if "initialized" not in st.session_state:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["initialized"] = True
    # Load default preset if available
    _init_presets = load_presets()
    if "Ben Main Shorts" in _init_presets:
        apply_preset(_init_presets["Ben Main Shorts"])
        st.session_state["_selected_preset"] = "Ben Main Shorts"
    else:
        st.session_state["_selected_preset"] = "(none)"


# ──────────────────────────────────────────────
# Callbacks (run before widgets render on next rerun)
# ──────────────────────────────────────────────
def _on_preset_change():
    """Auto-apply preset values when the selectbox changes."""
    if st.session_state.pop("_skip_preset_apply", False):
        return
    name = st.session_state.get("_selected_preset", "(none)")
    if name != "(none)":
        presets = load_presets()
        if name in presets:
            apply_preset(presets[name])


# ──────────────────────────────────────────────
# Sidebar: Settings (compressed layout)
# ──────────────────────────────────────────────
with st.sidebar:
    # Compact header + inline system status
    st.markdown("#### Reel Auto-Editor")
    _ff = "✓" if has_ffmpeg() else "✗"
    _cu = "✓" if has_cuda() else "—"
    _nv = "✓" if has_nvenc() else "—"
    st.caption(f"FFmpeg {_ff}  ·  CUDA {_cu}  ·  NVENC {_nv}")

    # ── Video Mode ──
    video_mode = st.radio(
        "Video Mode",
        options=["shorts", "longform"],
        format_func=lambda x: {"shorts": "Shorts", "longform": "Long-form"}.get(x, x),
        key="video_mode",
        horizontal=True,
        help="Long-form mode skips transcription and caption rendering for faster processing — ideal when you don't need captions.",
    )
    is_longform = video_mode == "longform"

    # ── Presets (auto-apply on selection) ──
    presets = load_presets()
    preset_names = ["(none)"] + sorted(presets.keys())

    # Apply any pending preset selection BEFORE the widget is instantiated
    if "_pending_preset_select" in st.session_state:
        st.session_state["_selected_preset"] = st.session_state.pop("_pending_preset_select")

    # Validate stored selection still exists (e.g. preset was deleted externally)
    if st.session_state.get("_selected_preset", "(none)") not in preset_names:
        st.session_state["_selected_preset"] = "(none)"

    selected_preset = st.selectbox(
        "Preset", preset_names,
        key="_selected_preset",
        label_visibility="collapsed",
        on_change=_on_preset_change,
    )
    if selected_preset != "(none)":
        if st.button("Delete preset", key="_del_preset", use_container_width=True):
            del presets[selected_preset]
            save_presets(presets)
            st.session_state["_pending_preset_select"] = "(none)"
            st.rerun()

    with st.expander("Save preset"):
        # Default to currently selected preset name for easy overwrite
        _default_name = selected_preset if selected_preset != "(none)" else ""
        _pn = st.text_input("Name", value=_default_name, placeholder="My Style", label_visibility="collapsed")
        _pn_stripped = (_pn or "").strip()
        _is_overwrite = _pn_stripped in presets
        _btn_label = f"Overwrite \"{_pn_stripped}\"" if _is_overwrite else "Save new preset"
        if st.button(_btn_label, disabled=not _pn_stripped, use_container_width=True):
            presets[_pn_stripped] = get_current_settings()
            save_presets(presets)
            if not _is_overwrite:
                # New preset — switch selectbox to it on next rerun
                st.session_state["_pending_preset_select"] = _pn_stripped
                st.session_state["_skip_preset_apply"] = True
            # Overwrite: selectbox already shows this preset, just save & rerun
            st.rerun()

    # ── Input / Output ──
    # Apply any pending folder picks BEFORE the widgets are instantiated
    if "_pending_source_folder" in st.session_state:
        st.session_state["source_folder"] = st.session_state.pop("_pending_source_folder")
    if "_pending_export_folder" in st.session_state:
        st.session_state["export_folder"] = st.session_state.pop("_pending_export_folder")

    with st.expander("Input / Output", expanded=True):
        col_sf, col_sfbtns = st.columns([5, 1])
        source_folder = col_sf.text_input(
            "Source folder", key="source_folder",
            placeholder=r"D:\Screen Capture\OBS\...",
            help="Folder with raw videos. All .mp4/.mov/.webm/.m4v files will be queued.",
        )
        if col_sfbtns.button("📂", key="_browse_source", help="Browse for source folder", use_container_width=True):
            picked = _browse_folder("Select source folder", initial_dir=r"D:\Screen Capture\OBS")
            if picked:
                st.session_state["_pending_source_folder"] = picked
                st.rerun()
        if source_folder and col_sfbtns.button("✕", key="_clear_source", help="Clear source folder", use_container_width=True):
            st.session_state["_pending_source_folder"] = ""
            st.rerun()

        col_ef, col_efb = st.columns([5, 1])
        export_folder = col_ef.text_input(
            "Export folder", key="export_folder",
            placeholder=r"C:\Videos\Exports",
            help="Where processed videos are saved. Defaults to temp if empty.",
        )
        if col_efb.button("📂", key="_browse_export", help="Browse for export folder", use_container_width=True):
            picked = _browse_folder("Select export folder")
            if picked:
                st.session_state["_pending_export_folder"] = picked
                st.rerun()

        # Show discovered file count
        folder_videos = discover_videos(source_folder)
        _sf = (source_folder or "").strip()
        if _sf:
            if os.path.isdir(_sf):
                st.caption(f"{len(folder_videos)} video(s) found")
            else:
                st.caption("Folder not found")

        # Or drag files directly
        uploaded_files = st.file_uploader(
            "Or drag files",
            type=["mp4", "mov", "webm", "m4v"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")

    with st.expander("Silence Removal", expanded=True):
        silence_threshold = st.slider(
            "Threshold (dB)", -60.0, -20.0, step=1.0,
            key="silence_threshold",
            help="Audio below this = silence. Recommended: **-28 dB** (tight) to **-35 dB** (natural).",
        )
        min_silence_dur = st.slider(
            "Min duration (s)", 0.05, 2.0, step=0.05,
            key="min_silence_dur",
            help="Only cut silences longer than this. Recommended: **0.15 s** (tight) to **0.40 s** (natural).",
        )
        silence_padding = st.slider(
            "Padding (s)", 0.0, 0.2, step=0.01,
            key="silence_padding",
            help="Buffer at cut edges. Recommended: **0.02 s** (tight) to **0.05 s** (natural).",
        )

    if not is_longform:
        with st.expander("Captions", expanded=True):
            font_size = st.slider("Font size", 30, 120, key="font_size")
            caption_y = st.slider("Y position (% from top)", 10, 90, key="caption_y")
            bounce_intensity = st.slider(
                "Bounce intensity", 0.0, 0.3, step=0.01,
                key="bounce_intensity",
                help="How much captions 'pop' when appearing.",
            )
            bounce_duration = st.slider(
                "Bounce duration (ms)", 50, 300, step=10, key="bounce_duration",
            )
            max_words_per_group = st.number_input(
                "Max words/group", min_value=1, max_value=5, key="max_words_per_group",
            )
            max_chars_per_line = st.number_input(
                "Max chars/line", min_value=10, max_value=50, key="max_chars_per_line",
                help="Maximum characters (with spaces) before wrapping to next group. Preview shows wide text for calibration.",
            )
            drop_shadow = st.toggle("Drop shadow", key="drop_shadow")
            st.caption("Auto-removes filler words (um, uh, ahh, etc.) and colors keywords by importance")
    else:
        # Pull values from session state even when hidden (for presets)
        font_size = st.session_state.get("font_size", DEFAULTS["font_size"])
        caption_y = st.session_state.get("caption_y", DEFAULTS["caption_y"])
        bounce_intensity = st.session_state.get("bounce_intensity", DEFAULTS["bounce_intensity"])
        bounce_duration = st.session_state.get("bounce_duration", DEFAULTS["bounce_duration"])
        max_words_per_group = st.session_state.get("max_words_per_group", DEFAULTS["max_words_per_group"])
        max_chars_per_line = st.session_state.get("max_chars_per_line", DEFAULTS["max_chars_per_line"])
        drop_shadow = st.session_state.get("drop_shadow", DEFAULTS["drop_shadow"])
        st.caption("No captions in long-form mode")

    with st.expander("Duplicate Removal"):
        remove_dupes = st.toggle(
            "Remove duplicate takes", key="remove_duplicates",
            help="Uses AI to detect when you retry the same content. "
                 "Keeps only the last take.",
        )
        if remove_dupes:
            _has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            if _has_key:
                st.caption("Using Anthropic API (Claude Haiku)")
            else:
                st.warning("Set ANTHROPIC_API_KEY in .env to enable")

    with st.expander("Audio", expanded=True):
        audio_leveling = st.toggle(
            "Audio Leveling", key="audio_leveling",
            help="Adjust audio volume before export.",
        )
        if audio_leveling:
            st.selectbox(
                "Method",
                options=["natural", "boost"],
                format_func=lambda x: {
                    "natural": "Natural (Volume Match)",
                    "boost": "Flat Boost (dB)",
                }.get(x, x),
                key="leveling_method",
                help=(
                    "**Natural**: Measures loudness, adjusts to the target, "
                    "then re-measures the output to confirm it landed there "
                    "with no clipping. Preserves dynamics.\n\n"
                    "**Flat Boost**: Adds a fixed dB amount. "
                    "No analysis — just turns the volume up."
                ),
            )
            if st.session_state.get("leveling_method") == "boost":
                st.slider(
                    "Boost (dB)", -10.0, 20.0, step=0.5,
                    key="boost_db",
                    help="How much louder (or quieter) to make the audio. "
                         "6 dB ≈ twice as loud.",
                )
            else:
                st.slider(
                    "Target loudness (LUFS)", -24.0, -12.0, step=0.5,
                    key="target_lufs",
                    help="Target integrated loudness. −16 is the YouTube/platform "
                         "norm. Lower = quieter, higher (toward −12) = louder. "
                         "The output is verified against this.",
                )

    with st.expander("Export", expanded=True):
        quality_preset = st.select_slider(
            "Output quality", options=QUALITY_LEVELS, key="quality_preset",
            help="Left = smaller files, right = better quality.",
        )

        # Show estimated file size
        video_info = st.session_state.get("_video_info")
        if video_info and "duration" in video_info:
            est_mb = estimate_file_size_mb(
                quality_preset,
                video_info["duration"],
                video_info.get("width", 1080),
                video_info.get("height", 1920),
            )
            if est_mb >= 1000:
                st.caption(f"Estimated: **~{est_mb / 1000:.1f} GB** per video")
            else:
                st.caption(f"Estimated: **~{est_mb:.0f} MB** per video")

        use_gpu_encoding = st.toggle(
            "GPU encoding (NVENC)", key="use_gpu_encoding",
            disabled=not has_nvenc(),
        )
        # Only show model unload toggle when transcription will run
        _needs_transcription = not is_longform or remove_dupes
        if _needs_transcription:
            st.toggle(
                "Free GPU after transcription", key="auto_unload_model",
                help="Unloads the Whisper model from VRAM before export. "
                     "Frees ~3-4 GB. Disable for faster batch processing "
                     "(model stays cached between videos).",
            )

    # ── Process button ──
    video_queue = _build_queue(folder_videos, uploaded_files)
    queue_count = len(video_queue)
    has_queue = queue_count > 0
    btn_label = f"Process {queue_count} Video{'s' if queue_count != 1 else ''}" if queue_count else "Process Video"
    process_btn = st.button(
        btn_label, type="primary", use_container_width=True, disabled=not has_queue,
    )


# ──────────────────────────────────────────────
# Build video queue & update preview
# ──────────────────────────────────────────────
# video_queue already built in sidebar for the process button


def _update_preview():
    """Extract preview frame and probe metadata from the first video in the queue."""
    if not video_queue:
        return  # keep cached preview — queue may be temporarily empty during rerun

    first_path = video_queue[0]["path"]
    if st.session_state.get("_preview_id") == first_path:
        return  # already cached

    frame = extract_frame(first_path, time_sec=2.0)
    info = probe_video(first_path)
    st.session_state["_preview_frame"] = frame
    st.session_state["_video_info"] = info
    st.session_state["_preview_id"] = first_path


_update_preview()


# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────
def _output_path_for(video_name: str) -> str:
    """Determine the output path for a processed video."""
    base = os.path.splitext(video_name)[0]
    _mode = st.session_state.get("video_mode", "shorts")
    suffix = "_long_edited" if _mode == "longform" else "_edited"
    out_name = f"{base}{suffix}.mp4"
    ef = (st.session_state.get("export_folder") or "").strip()
    if ef and os.path.isdir(ef):
        return os.path.normpath(os.path.join(ef, out_name))
    safe_base = base.replace(" ", "_")
    return os.path.normpath(os.path.join(tempfile.gettempdir(), f"{safe_base}{suffix}.mp4"))


def _process_single(input_path: str, output_path: str, file_label: str, container, settings: dict):
    """Run the full pipeline on a single video. Returns stats dict."""
    progress_bar = container.progress(0, text=f"{file_label}: Starting...")
    status_text = container.empty()
    stage_start = time.time()
    _is_longform = settings["video_mode"] == "longform"
    _do_dedup = settings["remove_duplicates"]
    _api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    stats = {
        "duplicates_found": 0,
        "dedup_time_saved": 0.0,
        "words": 0,
    }

    def update_status(msg, pct=None):
        status_text.text(msg)
        if pct is not None:
            progress_bar.progress(pct, text=msg)

    # Track intermediate temp files for cleanup
    _temp_files = []

    # ── Stage 1: Silence Removal ──
    update_status(f"{file_label} — Removing silences...", 0.0)

    def silence_progress(msg, pct=None):
        if pct is not None:
            update_status(f"{file_label} — {msg}", pct * 0.15)
        else:
            update_status(f"{file_label} — {msg}")

    silence_result = remove_silences(
        input_path,
        threshold_db=settings["silence_threshold"],
        min_duration=settings["min_silence_dur"],
        padding=settings["silence_padding"],
        progress_callback=silence_progress,
    )
    trimmed_path = silence_result.output_path
    if trimmed_path != input_path:
        _temp_files.append(trimmed_path)

    # ── Stage 1.5: Audio Leveling (optional) ──
    if settings["audio_leveling"]:
        update_status(f"{file_label} — Leveling audio...", 0.10)

        def audio_progress(msg, pct=None):
            if pct is not None:
                update_status(f"{file_label} — {msg}", 0.10 + pct * 0.05)
            else:
                update_status(f"{file_label} — {msg}")

        prev_path = trimmed_path
        trimmed_path = normalize_loudness(
            trimmed_path,
            method=settings["leveling_method"],
            boost_db=settings["boost_db"],
            target_lufs=settings.get("target_lufs", -16.0),
            progress_callback=audio_progress,
        )
        if trimmed_path != prev_path:
            _temp_files.append(trimmed_path)

    # ── Branching: Long-form vs Shorts ──
    if _is_longform:
        # Long-form pipeline: optional dedup, then simple export (no captions)
        if _do_dedup:
            # Transcribe for duplicate detection only
            update_status(f"{file_label} — Transcribing for duplicate detection...", 0.15)
            log.info("[%s] Transcribing for dedup...", file_label)

            def transcribe_progress(msg, pct=None):
                update_status(f"{file_label} — {msg}", 0.15)

            words = transcribe(trimmed_path, progress_callback=transcribe_progress)
            stats["words"] = len(words)
            log.info("[%s] Transcribed %d words", file_label, len(words))

            if not words:
                container.warning(f"{file_label}: No speech detected — skipping duplicate detection.")

            # Free GPU before dedup API call
            if settings["auto_unload_model"]:
                unload_model()

            # Duplicate removal
            update_status(f"{file_label} — Detecting duplicate takes...", 0.40)

            def dedup_progress(msg, pct=None):
                if pct is not None:
                    update_status(f"{file_label} — {msg}", 0.40 + pct * 0.20)
                else:
                    update_status(f"{file_label} — {msg}")

            dedup_result = remove_duplicates(
                trimmed_path, words, _api_key, progress_callback=dedup_progress,
            )
            if dedup_result.output_path != trimmed_path:
                _temp_files.append(dedup_result.output_path)
            trimmed_path = dedup_result.output_path
            stats["duplicates_found"] = dedup_result.duplicates_found
            stats["dedup_time_saved"] = dedup_result.time_saved
            log.info("[%s] Dedup: removed %d takes (%.1fs)",
                     file_label, dedup_result.duplicates_found, dedup_result.time_saved)

        # Simple export (no captions)
        update_status(f"{file_label} — Exporting (no captions)...", 0.65)
        log.info("[%s] Simple export (long-form)...", file_label)

        def export_progress(msg, pct=None):
            update_status(f"{file_label} — {msg}", 0.65)

        export_video_simple(
            video_path=trimmed_path,
            output_path=output_path,
            quality_preset=settings["quality_preset"],
            use_gpu=settings["use_gpu_encoding"],
            progress_callback=export_progress,
        )

    else:
        # Shorts pipeline: transcription, optional dedup, captions, export
        # ── Stage 2: Transcription ──
        update_status(f"{file_label} — Transcribing...", 0.15)
        log.info("[%s] Transcribing...", file_label)

        def transcribe_progress(msg, pct=None):
            update_status(f"{file_label} — {msg}", 0.15)

        words = transcribe(trimmed_path, progress_callback=transcribe_progress)
        progress_bar.progress(0.40, text=f"{file_label} — {len(words)} words")
        log.info("[%s] Transcribed %d words", file_label, len(words))

        if not words:
            container.warning(f"{file_label}: No speech detected — captions will be empty.")

        # ── Stage 2.5: Duplicate Removal (optional) ──
        if _do_dedup:
            update_status(f"{file_label} — Detecting duplicate takes...", 0.40)

            def dedup_progress(msg, pct=None):
                if pct is not None:
                    update_status(f"{file_label} — {msg}", 0.40 + pct * 0.10)
                else:
                    update_status(f"{file_label} — {msg}")

            dedup_result = remove_duplicates(
                trimmed_path, words, _api_key, progress_callback=dedup_progress,
            )
            stats["duplicates_found"] = dedup_result.duplicates_found
            stats["dedup_time_saved"] = dedup_result.time_saved

            if dedup_result.duplicates_found > 0:
                if dedup_result.output_path != trimmed_path:
                    _temp_files.append(dedup_result.output_path)
                trimmed_path = dedup_result.output_path
                log.info("[%s] Dedup: removed %d takes, re-transcribing...",
                         file_label, dedup_result.duplicates_found)
                # Re-transcribe the cleaned video for accurate captions
                update_status(f"{file_label} — Re-transcribing after dedup...", 0.45)
                words = transcribe(trimmed_path, progress_callback=transcribe_progress)
                log.info("[%s] Re-transcribed %d words", file_label, len(words))

        # ── Filler word removal ──
        pre_filler = len(words)
        words = remove_filler_words(words)
        if pre_filler != len(words):
            log.info("[%s] Removed %d filler words", file_label, pre_filler - len(words))

        stats["words"] = len(words)

        # Free GPU VRAM before export
        if settings["auto_unload_model"]:
            mem = get_gpu_memory_info()
            if mem:
                log.info("[%s] GPU VRAM before unload: %d/%d MB", file_label, mem["used_mb"], mem["total_mb"])
            unload_model()
            mem = get_gpu_memory_info()
            if mem:
                log.info("[%s] GPU VRAM after unload: %d/%d MB", file_label, mem["used_mb"], mem["total_mb"])

        # ── Stage 3: Caption Grouping ──
        update_status(f"{file_label} — Grouping captions...", 0.50)
        words = assign_colors(words)
        groups = group_words(words, max_per_group=int(settings["max_words_per_group"]), max_chars=int(settings.get("max_chars_per_line", 25)))
        groups = fix_group_contrast(groups)
        log.info("[%s] %d caption groups", file_label, len(groups))

        # ── Stage 4: Export with captions ──
        update_status(f"{file_label} — Exporting with captions...", 0.55)
        log.info("[%s] Exporting with captions...", file_label)

        video_info = probe_video(trimmed_path)
        fw = video_info.get("width", 1080) if video_info else 1080
        fh = video_info.get("height", 1920) if video_info else 1920

        def export_progress(msg, pct=None):
            update_status(f"{file_label} — {msg}", 0.55)

        export_video(
            video_path=trimmed_path,
            output_path=output_path,
            groups=groups,
            frame_width=fw,
            frame_height=fh,
            font_dir=get_font_dir(),
            font_name=get_font_name(),
            font_size=settings["font_size"],
            y_position_pct=float(settings["caption_y"]),
            bounce_intensity=settings["bounce_intensity"],
            bounce_duration_ms=float(settings["bounce_duration"]),
            all_caps=True,
            drop_shadow=settings["drop_shadow"],
            quality_preset=settings["quality_preset"],
            use_gpu=settings["use_gpu_encoding"],
            progress_callback=export_progress,
        )

    progress_bar.progress(1.0, text=f"{file_label} — Done!")
    log.info("[%s] Done!", file_label)

    # Clean up intermediate temp files (keep final output)
    for _tf in _temp_files:
        if _tf != output_path:
            try:
                os.remove(_tf)
            except OSError:
                pass

    elapsed = time.time() - stage_start
    return {
        "elapsed": elapsed,
        "original_duration": silence_result.original_duration,
        "trimmed_duration": silence_result.trimmed_duration,
        "words": stats["words"],
        "duplicates_found": stats["duplicates_found"],
        "dedup_time_saved": stats["dedup_time_saved"],
        "output_path": output_path,
        "output_size_mb": os.path.getsize(output_path) / (1024 * 1024),
    }


if process_btn and video_queue:
    # Check FFmpeg
    if not has_ffmpeg():
        st.error(
            "FFmpeg is not installed or not on PATH. "
            "Please install FFmpeg before processing. "
            "On Windows: `winget install Gyan.FFmpeg`"
        )
        st.stop()

    # Create export folder if needed
    ef = (st.session_state.get("export_folder") or "").strip()
    if ef and not os.path.exists(ef):
        os.makedirs(ef, exist_ok=True)

    # Verify export folder is writable
    if ef and os.path.isdir(ef):
        _test_file = os.path.join(ef, ".write_test")
        try:
            with open(_test_file, "w") as _f:
                _f.write("test")
            os.remove(_test_file)
        except OSError:
            st.warning(f"Cannot write to export folder: {ef}. Files will be saved to a temp directory.")
            ef = ""

    # Snapshot all settings so they can't change mid-batch
    _settings = {
        "video_mode": st.session_state.get("video_mode", "shorts"),
        "silence_threshold": st.session_state.get("silence_threshold", DEFAULTS["silence_threshold"]),
        "min_silence_dur": st.session_state.get("min_silence_dur", DEFAULTS["min_silence_dur"]),
        "silence_padding": st.session_state.get("silence_padding", DEFAULTS["silence_padding"]),
        "remove_duplicates": st.session_state.get("remove_duplicates", False),
        "audio_leveling": st.session_state.get("audio_leveling", False),
        "leveling_method": st.session_state.get("leveling_method", "natural"),
        "boost_db": st.session_state.get("boost_db", 6.0),
        "target_lufs": st.session_state.get("target_lufs", -16.0),
        "auto_unload_model": st.session_state.get("auto_unload_model", True),
        "quality_preset": st.session_state.get("quality_preset", DEFAULTS["quality_preset"]),
        "use_gpu_encoding": st.session_state.get("use_gpu_encoding", False),
        "font_size": st.session_state.get("font_size", DEFAULTS["font_size"]),
        "caption_y": st.session_state.get("caption_y", DEFAULTS["caption_y"]),
        "bounce_intensity": st.session_state.get("bounce_intensity", DEFAULTS["bounce_intensity"]),
        "bounce_duration": st.session_state.get("bounce_duration", DEFAULTS["bounce_duration"]),
        "max_words_per_group": st.session_state.get("max_words_per_group", DEFAULTS["max_words_per_group"]),
        "max_chars_per_line": st.session_state.get("max_chars_per_line", DEFAULTS["max_chars_per_line"]),
        "drop_shadow": st.session_state.get("drop_shadow", DEFAULTS["drop_shadow"]),
    }

    total = len(video_queue)
    st.markdown(f"### Processing {total} video{'s' if total != 1 else ''}...")
    overall_bar = st.progress(0, text="Starting batch...")

    results = []
    for i, video in enumerate(video_queue):
        overall_bar.progress(i / total, text=f"File {i + 1}/{total}: {video['name']}")
        out_path = _output_path_for(video["name"])
        container = st.container()
        container.markdown(f"**{video['name']}**")

        try:
            stats = _process_single(video["path"], out_path, video["name"], container, _settings)
            results.append({"name": video["name"], "status": "ok", **stats})

            _show_dedup = stats.get("duplicates_found", 0) > 0
            _ncols = 6 if _show_dedup else 5
            cols = container.columns(_ncols)
            cols[0].metric("Original", f"{stats['original_duration']:.1f}s")
            cols[1].metric("Trimmed", f"{stats['trimmed_duration']:.1f}s")
            sr = stats["original_duration"] - stats["trimmed_duration"]
            cols[2].metric("Silence Cut", f"{sr:.1f}s")
            if _show_dedup:
                cols[3].metric("Dupes Cut", f"{stats['duplicates_found']} ({stats['dedup_time_saved']:.1f}s)")
                col_offset = 4
            else:
                col_offset = 3
            if stats.get("words", 0) > 0:
                cols[col_offset].metric("Words", str(stats["words"]))
                col_offset += 1
            cols[min(col_offset, _ncols - 1)].metric("Size", f"{stats['output_size_mb']:.1f} MB")

            if stats["output_size_mb"] >= 5000:
                container.warning(
                    f"**{stats['output_size_mb'] / 1000:.1f} GB** — large file. "
                    "Consider lowering output quality."
                )

            # Offer download if no export folder was set
            if not (ef and os.path.isdir(ef)):
                with open(out_path, "rb") as f:
                    container.download_button(
                        f"Download {video['name']}",
                        data=f,
                        file_name=os.path.basename(out_path),
                        mime="video/mp4",
                        use_container_width=True,
                    )

        except Exception as e:
            results.append({"name": video["name"], "status": "failed", "error": str(e)})
            container.error(f"Failed: {e}")

    overall_bar.progress(1.0, text="Batch complete!")

    # Batch summary
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    if ok:
        total_size = sum(r["output_size_mb"] for r in ok)
        total_time = sum(r["elapsed"] for r in ok)
        st.success(
            f"**{len(ok)}/{total}** video(s) processed in {total_time:.0f}s "
            f"— {total_size:.0f} MB total"
        )
        if ef and os.path.isdir(ef):
            _out_path_display = os.path.normpath(ok[0]["output_path"]) if len(ok) == 1 and ok[0].get("output_path") else os.path.normpath(ef)
            st.code(_out_path_display, language=None)
            if st.button("📂 Open in Explorer", use_container_width=True):
                if len(ok) == 1 and ok[0].get("output_path"):
                    os.startfile(os.path.dirname(os.path.normpath(ok[0]["output_path"])))
                else:
                    os.startfile(os.path.normpath(ef))
    if failed:
        st.error(f"**{len(failed)}** video(s) failed")

else:
    # ── Live output preview ──
    preview_frame = st.session_state.get("_preview_frame")

    if not video_queue:
        st.markdown(
            """
            ## How to use

            1. Set a **source folder** or drag files in the sidebar
            2. Set an **export folder** for outputs
            3. **Adjust settings** — the preview updates live
            4. **Click Process** to run the full pipeline
            """
        )
        placeholder = np.full((1920, 1080, 3), 30, dtype=np.uint8)
        preview_frame = placeholder
    else:
        # Show queued files
        st.markdown(f"**{len(video_queue)} video(s) queued:**")
        for v in video_queue[:20]:
            st.caption(f"· {v['name']}")
        if len(video_queue) > 20:
            st.caption(f"… and {len(video_queue) - 20} more")

    if preview_frame is not None:
        st.subheader("Output Preview")
        if is_longform:
            # No caption overlay for long-form
            preview_img = preview_frame
        else:
            preview_img = render_preview_image(
                frame=preview_frame,
                font_size=font_size,
                y_position_pct=float(caption_y),
                drop_shadow=drop_shadow,
                max_chars_per_line=int(max_chars_per_line),
            )

        frame_h, frame_w = preview_frame.shape[:2]
        aspect = frame_w / frame_h
        display_h = min(600, frame_h)
        display_w = int(display_h * aspect)

        st.image(preview_img, width=display_w)

        if video_queue:
            if is_longform:
                st.caption(
                    "Long-form mode — no captions. "
                    "Click **Process** to export."
                )
            else:
                st.caption(
                    "Live preview — adjust sliders to see changes. "
                    "Click **Process** to export."
                )
