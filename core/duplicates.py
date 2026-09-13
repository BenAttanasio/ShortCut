"""Duplicate take detection and removal using Claude Haiku."""

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher

from utils.gpu import get_ffmpeg_path

log = logging.getLogger(__name__)


@dataclass
class Take:
    """A contiguous speech segment (one 'take')."""
    index: int
    text: str
    start: float
    end: float
    words: list = None  # original word objects for sub-splitting


@dataclass
class DuplicateResult:
    """Result of duplicate detection and removal."""
    output_path: str
    duplicates_found: int
    time_saved: float
    removed_ranges: list[tuple[float, float]]


def segment_into_takes(
    words: list,
    pause_threshold: float = 0.5,
) -> list[Take]:
    """Segment transcribed words into takes based on pauses.

    A 'take' is a contiguous group of words separated by pauses
    longer than pause_threshold seconds.
    """
    if not words:
        return []

    takes: list[Take] = []
    current_words = [words[0]]

    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap > pause_threshold:
            # End current take
            text = " ".join(w.text for w in current_words)
            takes.append(Take(
                index=len(takes),
                text=text,
                start=current_words[0].start,
                end=current_words[-1].end,
                words=list(current_words),
            ))
            current_words = [words[i]]
        else:
            current_words.append(words[i])

    # Last take
    if current_words:
        text = " ".join(w.text for w in current_words)
        takes.append(Take(
            index=len(takes),
            text=text,
            start=current_words[0].start,
            end=current_words[-1].end,
            words=list(current_words),
        ))

    return takes


def _split_long_takes(
    takes: list[Take],
    min_words_to_split: int = 12,
    min_gap: float = 0.3,
    min_words_per_side: int = 3,
) -> list[Take]:
    """Split long takes at the largest internal word gap.

    When a speaker does two retakes back-to-back without a long pause,
    they get merged into one take. This function detects that by finding
    the biggest gap between consecutive words inside a long take and
    splitting there.
    """
    result: list[Take] = []

    for take in takes:
        word_count = len(take.words) if take.words else 0
        if word_count < min_words_to_split or not take.words:
            result.append(take)
            continue

        # Find the largest internal gap
        best_gap = 0.0
        best_idx = -1
        for i in range(1, len(take.words)):
            gap = take.words[i].start - take.words[i - 1].end
            if gap > best_gap:
                best_gap = gap
                best_idx = i

        # Only split if the gap is meaningful and both sides have enough words
        if (
            best_gap >= min_gap
            and best_idx >= min_words_per_side
            and (len(take.words) - best_idx) >= min_words_per_side
        ):
            left_words = take.words[:best_idx]
            right_words = take.words[best_idx:]
            result.append(Take(
                index=0,  # re-indexed below
                text=" ".join(w.text for w in left_words),
                start=left_words[0].start,
                end=left_words[-1].end,
                words=left_words,
            ))
            result.append(Take(
                index=0,
                text=" ".join(w.text for w in right_words),
                start=right_words[0].start,
                end=right_words[-1].end,
                words=right_words,
            ))
            log.info("Split long take (%d words, gap=%.2fs) into two sub-takes",
                     word_count, best_gap)
        else:
            result.append(take)

    # Re-index all takes sequentially
    for i, take in enumerate(result):
        take.index = i

    return result


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _detect_local_duplicates(
    takes: list[Take],
    threshold: float = 0.7,
) -> list[int]:
    """Detect duplicate takes using text similarity (no API needed).

    Returns list of take indices to remove (keeps last in each group).
    """
    if len(takes) < 2:
        return []

    remove = set()
    for i in range(len(takes)):
        if i in remove:
            continue
        text_i = _normalize_text(takes[i].text)
        for j in range(i + 1, len(takes)):
            if j in remove:
                continue
            text_j = _normalize_text(takes[j].text)
            ratio = SequenceMatcher(None, text_i, text_j).ratio()
            if ratio >= threshold:
                remove.add(takes[i].index)
                log.info("Local match: take [%d] ~ take [%d] (%.0f%% similar)",
                         takes[i].index, takes[j].index, ratio * 100)
                break

    return sorted(remove)


def identify_duplicates(
    takes: list[Take],
    api_key: str,
    progress_callback=None,
) -> list[tuple[float, float]]:
    """Use Claude Haiku to identify duplicate takes.

    Returns list of (start, end) time ranges to REMOVE from the video.
    Always keeps the LAST take in a duplicate group.
    """
    if len(takes) < 2:
        return []

    import anthropic

    # Build transcript for the prompt
    transcript_lines = []
    for t in takes:
        mins = int(t.start // 60)
        secs = t.start % 60
        transcript_lines.append(f"[{t.index}] ({mins}:{secs:05.2f}-{int(t.end // 60)}:{t.end % 60:05.2f}) \"{t.text}\"")

    transcript_text = "\n".join(transcript_lines)

    prompt = f"""Analyze this transcript from a video recording session. The speaker sometimes does multiple "takes" — repeating roughly the same content to get it right.

Your job: identify groups of takes where the speaker is saying the same thing multiple times. For each group, ALL BUT THE LAST take should be marked for removal (the last attempt is always the keeper).

IMPORTANT distinctions:
- A "retake" = speaker tries to say roughly the same thing again (may use different words, rephrase, or slightly expand). Mark earlier attempts for removal.
- A "continuation" = speaker finishes a thought and moves to a NEW topic. These are NOT duplicates.
- Brief self-corrections WITHIN a single take (stutters, restarts mid-sentence) are NOT separate duplicates.
- If in doubt, do NOT mark as duplicate. Only flag clear retakes.

Transcript (format: [index] (start-end) "text"):
{transcript_text}

Return ONLY a JSON object with this exact structure, no other text:
{{"remove_indices": [list of take indices to remove]}}

If there are no duplicates, return: {{"remove_indices": []}}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        log.info("Sending %d takes to Haiku for duplicate detection...", len(takes))
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        if not response.content:
            log.error("Anthropic returned empty response content")
            return []

        # Parse the response
        response_text = response.content[0].text.strip()
        log.info("Haiku response: %s", response_text[:200])
    except Exception as e:
        log.error("Anthropic API call failed: %s", e)
        if progress_callback:
            progress_callback(f"AI detection failed: {e}")
        return []

    # Extract JSON from response (handle markdown code blocks)
    if "```" in response_text:
        # Strip markdown code fences
        lines = response_text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            if in_block or (not json_lines and not line.strip().startswith("```")):
                json_lines.append(line)
        response_text = "\n".join(json_lines).strip()

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response_text[start:end])
        else:
            log.error("Could not parse Haiku response as JSON: %s", response_text)
            return []

    remove_indices = result.get("remove_indices", [])
    log.info("Haiku identified %d takes to remove: %s", len(remove_indices), remove_indices)

    # Convert indices to time ranges
    takes_by_index = {t.index: t for t in takes}
    remove_ranges = []
    for idx in remove_indices:
        if idx in takes_by_index:
            t = takes_by_index[idx]
            remove_ranges.append((t.start, t.end))

    return sorted(remove_ranges)


def _get_keep_ranges(
    remove_ranges: list[tuple[float, float]],
    total_duration: float,
    padding: float = 0.05,
) -> list[tuple[float, float]]:
    """Convert remove ranges to keep ranges (inverse), with padding."""
    if not remove_ranges:
        return [(0.0, total_duration)]

    keep = []
    cursor = 0.0

    for rm_start, rm_end in sorted(remove_ranges):
        seg_end = rm_start + padding
        if seg_end > cursor + 0.01:
            keep.append((max(0.0, cursor), min(total_duration, seg_end)))
        cursor = max(cursor, rm_end - padding)

    if cursor < total_duration - 0.01:
        keep.append((cursor, total_duration))

    return keep


def _get_duration(video_path: str, ffmpeg: str) -> float:
    """Get video duration using ffprobe or ffmpeg."""
    from utils.gpu import get_ffprobe_path

    ffprobe = get_ffprobe_path()
    if ffprobe:
        cmd = [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass

    import re
    cmd = [ffmpeg, "-i", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if m:
        h, mi, s, cs = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        return h * 3600 + mi * 60 + s + cs / 100.0
    raise RuntimeError("Could not determine video duration.")


def remove_duplicates(
    video_path: str,
    words: list,
    api_key: str,
    progress_callback=None,
) -> DuplicateResult:
    """Full pipeline: segment takes, identify duplicates via Haiku, cut video.

    Returns DuplicateResult with the cleaned video path and stats.
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found.")

    # Step 1: Segment into takes
    if progress_callback:
        progress_callback("Segmenting into takes...")
    takes = segment_into_takes(words)
    log.info("Segmented transcript into %d takes", len(takes))

    # Split long takes that may contain back-to-back retakes
    takes = _split_long_takes(takes)
    log.info("After sub-splitting: %d takes", len(takes))
    for t in takes:
        log.info("  Take [%d] (%.1f-%.1fs): %s", t.index, t.start, t.end, t.text[:120])

    if len(takes) < 2:
        log.info("Only %d take(s) — nothing to deduplicate", len(takes))
        return DuplicateResult(
            output_path=video_path,
            duplicates_found=0,
            time_saved=0.0,
            removed_ranges=[],
        )

    # Step 2a: Local text-similarity detection (fast, no API needed)
    if progress_callback:
        progress_callback("Detecting duplicate takes...")
    local_remove_indices = _detect_local_duplicates(takes)

    if local_remove_indices:
        log.info("Local detection found %d duplicate takes: %s",
                 len(local_remove_indices), local_remove_indices)
        takes_by_index = {t.index: t for t in takes}
        remove_ranges = sorted(
            (takes_by_index[idx].start, takes_by_index[idx].end)
            for idx in local_remove_indices
            if idx in takes_by_index
        )
    elif api_key:
        # Step 2b: Fall through to API for semantic detection
        if progress_callback:
            progress_callback("Detecting duplicate takes (AI)...")
        remove_ranges = identify_duplicates(
            takes, api_key, progress_callback=progress_callback,
        )
    else:
        log.info("No local duplicates found and no API key available")
        remove_ranges = []

    if not remove_ranges:
        log.info("No duplicates found")
        return DuplicateResult(
            output_path=video_path,
            duplicates_found=0,
            time_saved=0.0,
            removed_ranges=[],
        )

    time_saved = sum(end - start for start, end in remove_ranges)
    log.info("Found %d duplicate ranges totaling %.1fs", len(remove_ranges), time_saved)

    # Step 3: Cut video — keep non-duplicate segments
    if progress_callback:
        progress_callback("Cutting duplicate takes...")

    total_duration = _get_duration(video_path, ffmpeg)
    keep_ranges = _get_keep_ranges(remove_ranges, total_duration)

    if not keep_ranges:
        log.warning("No segments to keep after dedup — returning original")
        return DuplicateResult(
            output_path=video_path,
            duplicates_found=len(remove_ranges),
            time_saved=0.0,
            removed_ranges=remove_ranges,
        )

    # Cut each keep segment and concat (same approach as silence removal)
    tmp_dir = tempfile.mkdtemp(prefix="reel_dedup_")
    segment_files = []

    for i, (start, end) in enumerate(keep_ranges):
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
            pct = (i + 1) / len(keep_ranges)
            progress_callback(f"Cutting segment {i + 1}/{len(keep_ranges)}", pct)

    # Concat all segments
    concat_list_path = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list_path, "w") as f:
        for seg_path in segment_files:
            safe_path = seg_path.replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    output_path = os.path.join(tmp_dir, "deduped.mp4")
    concat_cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    proc = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=1200)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg concat failed during dedup: {(proc.stderr or '')[-500:]}"
        )

    # Cleanup segment files
    for seg_path in segment_files:
        try:
            os.remove(seg_path)
        except OSError:
            pass
    try:
        os.remove(concat_list_path)
    except OSError:
        pass

    # Clean up tmp directory
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        raise RuntimeError("Dedup produced an invalid or missing output file")

    log.info("Dedup complete: removed %d takes (%.1fs), output: %s",
             len(remove_ranges), time_saved, output_path)

    return DuplicateResult(
        output_path=output_path,
        duplicates_found=len(remove_ranges),
        time_saved=time_saved,
        removed_ranges=remove_ranges,
    )
