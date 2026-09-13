"""Headless batch processor for ShortCut — no Streamlit required.

Usage:
    python cli.py --input <folder> --output <folder> --preset <preset_name>

Processes every video in the input folder through the full shorts pipeline:
silence removal → transcription → keyword coloring → caption rendering → export.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from core.silence import remove_silences
from core.audio import normalize_loudness, count_audio_streams, combine_audio_tracks
from core.transcribe import transcribe, group_words, unload_model, remove_filler_words
from core.exporter import export_video, export_video_simple
from core.duplicates import remove_duplicates
from core.captions import get_font_dir, get_font_name

try:
    from core.keywords import assign_colors, fix_group_contrast
except OSError as e:
    print("ERROR: spaCy model not found. Run:  python -m spacy download en_core_web_sm")
    sys.exit(1)

# Video file extensions to process
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_preset(preset_name: str) -> dict:
    """Load a named preset from presets.json."""
    presets_path = Path(__file__).parent / "presets.json"
    if not presets_path.exists():
        print(f"ERROR: presets.json not found at {presets_path}")
        sys.exit(1)

    with open(presets_path, "r") as f:
        presets = json.load(f)

    if preset_name not in presets:
        available = ", ".join(f'"{k}"' for k in presets.keys())
        print(f'ERROR: Preset "{preset_name}" not found. Available: {available}')
        sys.exit(1)

    return presets[preset_name]


def find_videos(input_path: str) -> list[Path]:
    """Find all video files in the input path (file or folder)."""
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            return [p]
        print(f"ERROR: Not a supported video file: {input_path}")
        print(f"  (Supported: {', '.join(sorted(VIDEO_EXTENSIONS))})")
        sys.exit(1)
    if not p.is_dir():
        print(f"ERROR: Input path does not exist: {input_path}")
        sys.exit(1)
    return sorted(
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )


def cli_progress(file_label: str):
    """Create a progress callback that prints to stdout."""
    def callback(message: str, percentage: float = None):
        if percentage is not None:
            print(f"  [{file_label}] [{percentage * 100:5.1f}%] {message}")
        else:
            print(f"  [{file_label}] {message}")
    return callback


def probe_video_info(video_path: str) -> dict | None:
    """Probe video for width/height using FFmpeg."""
    import subprocess
    import re
    from utils.gpu import get_ffmpeg_path

    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        return None

    cmd = [ffmpeg, "-i", video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None

    info = {}
    m = re.search(r"(\d{2,5})x(\d{2,5})", result.stderr)
    if m:
        info["width"] = int(m.group(1))
        info["height"] = int(m.group(2))

    return info if info else None


def process_single(input_path: Path, output_path: Path, settings: dict, save_transcript: bool = False) -> dict:
    """Run the full pipeline on a single video. Returns stats dict."""
    file_label = input_path.name
    progress = cli_progress(file_label)
    start_time = time.time()
    _temp_files = []

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    do_dedup = settings.get("remove_duplicates", False)

    # ── Stage 0: Combine audio tracks (auto-detect) ──
    # OBS recordings often have separate mic + desktop audio streams; mix them
    # down to one track up front so the rest of the pipeline (which re-encodes
    # only the first audio stream) doesn't silently drop the others.
    working_input = str(input_path)
    combine_mode = settings.get("combine_audio_tracks", "auto")  # auto | first_only | off
    if combine_mode == "auto":
        n_tracks = count_audio_streams(working_input)
        if n_tracks > 1:
            print(f"  [{file_label}] Detected {n_tracks} audio tracks — combining...")
            combined = combine_audio_tracks(working_input, progress_callback=progress)
            _temp_files.append(combined)
            working_input = combined
        else:
            print(f"  [{file_label}] {n_tracks} audio track — no mix needed")
    # "first_only"/"off" -> leave working_input as the original (legacy behavior)

    # ── Stage 1: Silence Removal ──
    print(f"\n[{file_label}] Stage 1: Removing silences...")
    silence_result = remove_silences(
        working_input,
        threshold_db=settings.get("silence_threshold", -35.0),
        min_duration=settings.get("min_silence_dur", 0.4),
        padding=settings.get("silence_padding", 0.05),
        progress_callback=progress,
    )
    trimmed_path = silence_result.output_path
    if trimmed_path != working_input:
        _temp_files.append(trimmed_path)

    saved = silence_result.original_duration - silence_result.trimmed_duration
    print(f"  [{file_label}] Silence removed: {saved:.1f}s cut "
          f"({silence_result.original_duration:.1f}s -> {silence_result.trimmed_duration:.1f}s)")

    # ── Stage 1.5: Audio Leveling (optional) ──
    if settings.get("audio_leveling", False):
        print(f"  [{file_label}] Leveling audio...")
        prev_path = trimmed_path
        trimmed_path = normalize_loudness(
            trimmed_path,
            method=settings.get("leveling_method", "natural"),
            boost_db=settings.get("boost_db", 6.0),
            target_lufs=settings.get("target_lufs", -16.0),
            progress_callback=progress,
        )
        if trimmed_path != prev_path:
            _temp_files.append(trimmed_path)

    # ── Long-form branch: no captions ──
    # Mirrors the Streamlit app's longform path: silence -> audio -> optional
    # dedup -> simple (captionless) export. Transcription only runs when needed.
    if settings.get("video_mode") == "longform":
        words = []
        duplicates_found = 0
        dedup_time_saved = 0.0

        # Transcribe only if dedup or a saved transcript needs it
        if do_dedup or save_transcript:
            print(f"  [{file_label}] Transcribing...")
            words = transcribe(trimmed_path, progress_callback=progress)
            print(f"  [{file_label}] Transcribed {len(words)} words")

        # Optional duplicate removal
        if do_dedup and words:
            print(f"  [{file_label}] Detecting duplicate takes...")
            dedup_result = remove_duplicates(
                trimmed_path, words, api_key, progress_callback=progress,
            )
            if dedup_result.duplicates_found > 0:
                duplicates_found = dedup_result.duplicates_found
                dedup_time_saved = dedup_result.time_saved
                if dedup_result.output_path != trimmed_path:
                    _temp_files.append(dedup_result.output_path)
                    trimmed_path = dedup_result.output_path
                if save_transcript:
                    print(f"  [{file_label}] Re-transcribing after duplicate removal...")
                    words = transcribe(trimmed_path, progress_callback=progress)
            print(f"  [{file_label}] Duplicates removed: {duplicates_found} "
                  f"({dedup_time_saved:.1f}s saved)")

        # Optional transcript save
        if save_transcript:
            transcript_path = output_path.with_name(f"{output_path.stem}_transcript.json")
            transcript_data = {
                "text": " ".join(w.text for w in words),
                "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
                "word_count": len(words),
            }
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2)
            print(f"  [{file_label}] Transcript saved: {transcript_path.name}")

        # Free GPU before export
        if settings.get("auto_unload_model", False):
            unload_model()

        # Simple export — no captions burned in
        print(f"  [{file_label}] Exporting (no captions)...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_video_simple(
            video_path=trimmed_path,
            output_path=str(output_path),
            quality_preset=settings.get("quality_preset", "High quality"),
            use_gpu=settings.get("use_gpu_encoding", True),
            progress_callback=progress,
        )

        # Clean up intermediate temp files
        for tf in _temp_files:
            try:
                os.remove(tf)
            except OSError:
                pass

        elapsed = time.time() - start_time
        output_size = os.path.getsize(str(output_path)) / (1024 * 1024)
        print(f"  [{file_label}] Done in {elapsed:.1f}s — output: {output_size:.1f} MB")
        return {
            "input": str(input_path),
            "output": str(output_path),
            "status": "success",
            "stats": {
                "elapsed_seconds": round(elapsed, 1),
                "original_duration": round(silence_result.original_duration, 1),
                "trimmed_duration": round(silence_result.trimmed_duration, 1),
                "words_transcribed": len(words),
                "duplicates_found": duplicates_found,
                "dedup_time_saved": round(dedup_time_saved, 1),
                "output_size_mb": round(output_size, 1),
            },
        }

    # ── Stage 2: Transcription ──
    print(f"  [{file_label}] Stage 2: Transcribing...")
    words = transcribe(trimmed_path, progress_callback=progress)
    print(f"  [{file_label}] Transcribed {len(words)} words")

    # ── Filler word removal ──
    pre_filler = len(words)
    words = remove_filler_words(words)
    if pre_filler != len(words):
        print(f"  [{file_label}] Removed {pre_filler - len(words)} filler words")

    # ── Stage 2.5: Duplicate Removal (optional) ──
    duplicates_found = 0
    dedup_time_saved = 0.0
    if do_dedup:
        print(f"  [{file_label}] Detecting duplicate takes...")
        dedup_result = remove_duplicates(
            trimmed_path, words, api_key, progress_callback=progress,
        )
        if dedup_result.duplicates_found > 0:
            duplicates_found = dedup_result.duplicates_found
            dedup_time_saved = dedup_result.time_saved
            if dedup_result.output_path != trimmed_path:
                _temp_files.append(dedup_result.output_path)
                trimmed_path = dedup_result.output_path
            # Re-transcribe after cuts
            print(f"  [{file_label}] Re-transcribing after duplicate removal...")
            words = transcribe(trimmed_path, progress_callback=progress)
        print(f"  [{file_label}] Duplicates removed: {duplicates_found} "
              f"({dedup_time_saved:.1f}s saved)")

    # ── Save transcript if requested ──
    if save_transcript:
        transcript_path = output_path.with_name(f"{output_path.stem}_transcript.json")
        transcript_data = {
            "text": " ".join(w.text for w in words),
            "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
            "word_count": len(words),
        }
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2)
        print(f"  [{file_label}] Transcript saved: {transcript_path.name}")

    # ── Stage 3: Keyword Coloring & Caption Grouping ──
    print(f"  [{file_label}] Stage 3: Coloring keywords & grouping captions...")
    words = assign_colors(words)
    groups = group_words(words, max_per_group=int(settings.get("max_words_per_group", 3)), max_chars=int(settings.get("max_chars_per_line", 25)))
    groups = fix_group_contrast(groups)
    print(f"  [{file_label}] {len(groups)} caption groups created")

    # ── Stage 4: Export with Captions ──
    print(f"  [{file_label}] Stage 4: Exporting with captions...")
    video_info = probe_video_info(trimmed_path)
    fw = video_info.get("width", 1080) if video_info else 1080
    fh = video_info.get("height", 1920) if video_info else 1920

    # Unload whisper model before export to free GPU VRAM
    if settings.get("auto_unload_model", False):
        unload_model()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_video(
        video_path=trimmed_path,
        output_path=str(output_path),
        groups=groups,
        frame_width=fw,
        frame_height=fh,
        font_dir=get_font_dir(),
        font_name=get_font_name(),
        font_size=settings.get("font_size", 60),
        y_position_pct=float(settings.get("caption_y", 38)),
        bounce_intensity=settings.get("bounce_intensity", 0.1),
        bounce_duration_ms=float(settings.get("bounce_duration", 150)),
        all_caps=True,
        drop_shadow=settings.get("drop_shadow", True),
        quality_preset=settings.get("quality_preset", "High quality"),
        use_gpu=settings.get("use_gpu_encoding", True),
        progress_callback=progress,
    )

    # Clean up intermediate temp files
    for tf in _temp_files:
        try:
            os.remove(tf)
        except OSError:
            pass

    elapsed = time.time() - start_time
    output_size = os.path.getsize(str(output_path)) / (1024 * 1024)
    print(f"  [{file_label}] Done in {elapsed:.1f}s — output: {output_size:.1f} MB")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "status": "success",
        "stats": {
            "elapsed_seconds": round(elapsed, 1),
            "original_duration": round(silence_result.original_duration, 1),
            "trimmed_duration": round(silence_result.trimmed_duration, 1),
            "words_transcribed": len(words),
            "duplicates_found": duplicates_found,
            "dedup_time_saved": round(dedup_time_saved, 1),
            "output_size_mb": round(output_size, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="ShortCut — Headless batch video processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python cli.py --input ./videos --output ./exports --preset \"Ben Main Shorts\"\n  python cli.py --input ./clip.mp4 --output ./exports --preset \"Ben Main Shorts\"",
    )
    parser.add_argument("--input", required=True, help="Input folder or video file")
    parser.add_argument("--output", required=True, help="Output folder for processed videos")
    parser.add_argument("--preset", required=True, help="Preset name from presets.json")
    parser.add_argument("--save-transcript", action="store_true", help="Save transcript JSON alongside each output video")
    parser.add_argument("--dedup", action="store_true", help="Force duplicate-take removal on, overriding the preset (opt-in per run)")
    args = parser.parse_args()

    # Load preset
    settings = load_preset(args.preset)
    if args.dedup:
        settings["remove_duplicates"] = True
    print(f'Loaded preset: "{args.preset}"')
    print(f"  Video mode: {settings.get('video_mode', 'shorts')}")
    print(f"  Quality: {settings.get('quality_preset', 'High quality')}")
    print(f"  GPU encoding: {settings.get('use_gpu_encoding', True)}")

    # Find videos
    videos = find_videos(args.input)
    if not videos:
        print(f"No video files found in: {args.input}")
        print(f"  (Looking for: {', '.join(sorted(VIDEO_EXTENSIONS))})")
        sys.exit(0)

    print(f"\nFound {len(videos)} video(s) to process:")
    for v in videos:
        print(f"  - {v.name}")

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each video
    results = []
    failures = 0
    total_start = time.time()

    for i, video_path in enumerate(videos, 1):
        print(f"\n{'='*60}")
        print(f"Processing [{i}/{len(videos)}]: {video_path.name}")
        print(f"{'='*60}")

        suffix = "_long_edited" if settings.get("video_mode") == "longform" else "_short"
        output_name = f"{video_path.stem}{suffix}{video_path.suffix}"
        output_path = output_folder / output_name

        try:
            result = process_single(video_path, output_path, settings, save_transcript=args.save_transcript)
            results.append(result)
        except Exception as e:
            failures += 1
            print(f"\n  ERROR processing {video_path.name}: {e}")
            results.append({
                "input": str(video_path),
                "output": str(output_path),
                "status": "failed",
                "error": str(e),
            })

    # Print summary
    total_elapsed = time.time() - total_start
    successful = len(videos) - failures

    summary = {
        "files_processed": successful,
        "failures": failures,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": results,
    }

    print(f"\n{'='*60}")
    print("--- BATCH SUMMARY ---")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))

    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
