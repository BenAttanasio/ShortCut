"""Transcription using faster-whisper with word-level timestamps."""

import gc
import logging
import subprocess
import tempfile
import os
from dataclasses import dataclass, field

from utils.gpu import get_whisper_device, get_ffmpeg_path
from core.audio import count_audio_streams

log = logging.getLogger(__name__)

# ── Cached Whisper model (avoid reloading per video) ──
_model = None
_model_size = None


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float
    color: str = "#FFFFFF"  # set later by keywords module


@dataclass
class WordGroup:
    """A group of 2-4 words displayed together on screen."""
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def extract_audio(video_path: str) -> str:
    """Extract audio from video as WAV for transcription."""
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    audio_path = tempfile.mktemp(suffix=".wav", prefix="reel_audio_")
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vn",
    ]
    # If the source still has multiple audio streams (e.g. raw OBS mic + desktop
    # passed straight to the GUI), mix them so Whisper hears everything. In the
    # CLI flow the working copy is already single-track, so this is a no-op there.
    n_tracks = count_audio_streams(video_path)
    if n_tracks > 1:
        cmd += ["-filter_complex", f"amix=inputs={n_tracks}:duration=longest:normalize=1"]
    cmd += [
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    log.info("Extracting audio: %s", os.path.basename(video_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if not os.path.exists(audio_path):
        raise RuntimeError(
            f"Failed to extract audio from video. "
            f"FFmpeg: {(proc.stderr or '')[-300:]}"
        )

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    log.info("Audio extracted: %.1f MB", size_mb)
    return audio_path


def _get_model(model_size: str):
    """Get or create cached Whisper model."""
    global _model, _model_size
    if _model is not None and _model_size == model_size:
        log.info("Reusing cached Whisper model (%s)", model_size)
        return _model

    from faster_whisper import WhisperModel

    device, compute_type = get_whisper_device()
    log.info("Loading Whisper model %s on %s (%s)...", model_size, device, compute_type)
    _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_size = model_size
    log.info("Whisper model loaded")
    return _model


def unload_model():
    """Unload the cached Whisper model to free GPU VRAM."""
    global _model, _model_size
    if _model is None:
        return

    log.info("Unloading Whisper model to free GPU memory...")
    try:
        _model.model.unload_model(to_cpu=False)
    except (AttributeError, Exception) as e:
        log.debug("CTranslate2 unload_model not available: %s", e)
    _model = None
    _model_size = None
    gc.collect()
    log.info("Whisper model unloaded")


def transcribe(
    video_path: str,
    model_size: str = "large-v3",
    progress_callback=None,
) -> list[Word]:
    """Transcribe video audio and return word-level timestamps."""
    if progress_callback:
        progress_callback("Extracting audio...")

    audio_path = extract_audio(video_path)

    try:
        if progress_callback:
            progress_callback(f"Loading Whisper model ({model_size})...")

        model = _get_model(model_size)

        if progress_callback:
            progress_callback("Transcribing...")

        log.info("Starting transcription...")
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt="Claude, Anthropic, AI, API, n8n",
            hotwords="Claude n8n",
        )

        words = []
        segment_count = 0
        try:
            for segment in segments:
                segment_count += 1
                if segment.words:
                    for w in segment.words:
                        words.append(Word(
                            text=w.word.strip(),
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        ))
                log.info(
                    "Segment %d: %.1fs-%.1fs | %d words so far",
                    segment_count, segment.start, segment.end, len(words),
                )
        except Exception as e:
            log.error("Transcription failed at segment %d (%d words collected): %s", segment_count, len(words), e)
            if not words:
                raise RuntimeError(f"Transcription failed: {e}") from e
            log.warning("Returning %d partial words from %d segments", len(words), segment_count)

        log.info("Transcription complete: %d words from %d segments", len(words), segment_count)

        words = fix_claude_transcription(words)

        if progress_callback:
            progress_callback(f"Transcribed {len(words)} words.")

        return words

    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


_CLOUD_CONTEXT = {"computing", "storage", "service", "services", "platform", "infrastructure", "server", "servers", "hosted", "based"}


def fix_claude_transcription(words: list[Word]) -> list[Word]:
    """Fix 'Cloud' → 'Claude' and split fused 'claudecode' into two words."""
    result = []
    for i, w in enumerate(words):
        stripped = w.text.lower().strip(".,!?;:'\"")

        # Split fused "claudecode" into "Claude" + "Code"
        if "claudecode" in stripped or "cloudcode" in stripped:
            mid = (w.start + w.end) / 2
            result.append(Word(text="Claude", start=w.start, end=mid, probability=w.probability))
            result.append(Word(text="Code", start=mid, end=w.end, probability=w.probability))
            continue

        # Fix standalone "Cloud" → "Claude"
        if stripped == "cloud":
            next_word = words[i + 1].text.lower().strip(".,!?;:'\"") if i + 1 < len(words) else ""
            if next_word not in _CLOUD_CONTEXT:
                w.text = "Claude" if w.text[0].isupper() else "claude"

        # Fix "nnn" / "n-n-n" / "nate n" etc. → "n8n"
        elif stripped in {"nnn", "n.n.n", "nen", "nan"}:
            w.text = "n8n"

        result.append(w)
    return result


FILLER_WORDS = {
    "um", "uh", "umm", "uhh", " uhm", "hmm", "hm", "mm",
    "ah", "ahh", "er", "err", "eh", "like,", "like",
    "you know", "i mean", "so,", "right,", "okay so",
}

# Single-word fillers for fast matching
_FILLER_SINGLE = {
    "um", "uh", "umm", "uhh", "uhm", "hmm", "hm", "mm",
    "ah", "ahh", "er", "err", "eh", "mhm", "erm",
}


def remove_filler_words(words: list[Word]) -> list[Word]:
    """Remove filler/hesitation words (um, uh, ahh, etc.) from transcript.

    Returns filtered word list with fillers removed.
    """
    filtered = []
    for w in words:
        cleaned = w.text.lower().strip(".,!?;:'\"")
        if cleaned in _FILLER_SINGLE:
            log.info("Filler removed: '%s' at %.2fs", w.text, w.start)
            continue
        filtered.append(w)
    log.info("Filler removal: %d → %d words (%d removed)",
             len(words), len(filtered), len(words) - len(filtered))
    return filtered


def group_words(
    words: list[Word],
    max_per_group: int = 3,
    pause_threshold: float = 0.3,
    max_chars: int = 25,
) -> list[WordGroup]:
    """Group words into display groups of 2-4 words.

    Groups break on:
    - Punctuation at end of word
    - Pause > pause_threshold between words
    - Reaching max_per_group words
    - Exceeding max_chars total characters (including spaces)
    """
    if not words:
        return []

    groups: list[WordGroup] = []
    current = WordGroup()

    for i, word in enumerate(words):
        # Check if adding this word would exceed the character limit
        # (but never break a single word — show it in full even if over the limit)
        prospective = " ".join(w.text for w in current.words + [word])
        if current.words and len(prospective) > max_chars:
            groups.append(current)
            current = WordGroup()

        current.words.append(word)

        should_break = False

        # Break on reaching max group size
        if len(current.words) >= max_per_group:
            should_break = True

        # Break on punctuation
        elif word.text and word.text[-1] in ".!?,;:":
            should_break = True

        # Break on pause before next word
        elif i + 1 < len(words):
            gap = words[i + 1].start - word.end
            if gap > pause_threshold:
                should_break = True

        if should_break:
            groups.append(current)
            current = WordGroup()

    # Don't forget the last group
    if current.words:
        groups.append(current)

    return groups
