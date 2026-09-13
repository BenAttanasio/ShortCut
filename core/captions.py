"""Caption rendering with Pillow — used for both live preview and video export.

The exporter calls render_caption() per-frame via FFmpeg pipe compositing.
"""

import os
from PIL import Image, ImageDraw, ImageFont
from core.transcribe import WordGroup

# Font search paths
_FONT_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "Montserrat-ExtraBold.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def get_font_path() -> str | None:
    """Return the file system path to the first available caption font."""
    for path in _FONT_PATHS:
        if os.path.exists(path):
            return path
    return None


def get_font_dir() -> str:
    """Return the directory containing the caption font (for FFmpeg fontsdir)."""
    path = get_font_path()
    if path:
        return os.path.dirname(path)
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts")


def get_font_name() -> str:
    """Return the font family name for ASS subtitle styles."""
    path = get_font_path()
    if path and "Montserrat" in os.path.basename(path):
        return "Montserrat ExtraBold"
    if path and "impact" in os.path.basename(path).lower():
        return "Impact"
    if path and "arialbd" in os.path.basename(path).lower():
        return "Arial"
    return "Sans"


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load and cache the caption font at the given size."""
    if size in _font_cache:
        return _font_cache[size]

    for path in _FONT_PATHS:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font

    # Last resort: default font scaled
    font = ImageFont.load_default(size=size)
    _font_cache[size] = font
    return font


def render_caption(
    group: WordGroup,
    frame_width: int,
    frame_height: int,
    font_size: int = 60,
    y_position_pct: float = 38.0,
    scale: float = 1.0,
    all_caps: bool = True,
    drop_shadow: bool = True,
    stroke_width: int = 3,
) -> Image.Image:
    """Render a word group as a transparent RGBA image.

    Each word is colored individually. The whole group is centered
    and optionally scaled (for bounce animation).

    Returns an RGBA PIL Image the same size as the video frame.
    """
    overlay = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    effective_size = max(8, int(font_size * scale))
    font = _get_font(effective_size)

    # Build word texts and colors
    word_texts = []
    word_colors = []
    for w in group.words:
        text = w.text.upper() if all_caps else w.text
        word_texts.append(text)
        word_colors.append(w.color)

    # Measure total width (words + spaces)
    space_width = draw.textlength(" ", font=font)
    word_widths = [draw.textlength(t, font=font) for t in word_texts]
    total_width = sum(word_widths) + space_width * (len(word_texts) - 1)

    # Position: centered horizontally, at y_position_pct vertically
    x_start = (frame_width - total_width) / 2
    y_pos = int(frame_height * y_position_pct / 100.0)

    # Adjust Y for text height (center the text vertically around y_pos)
    bbox = font.getbbox("Ay")  # representative characters
    text_height = bbox[3] - bbox[1]
    y_pos -= text_height // 2

    # Draw each word
    x_cursor = x_start
    for text, color, width in zip(word_texts, word_colors, word_widths):
        # Drop shadow
        if drop_shadow:
            shadow_offset = max(1, int(2 * scale))
            draw.text(
                (x_cursor + shadow_offset, y_pos + shadow_offset),
                text, font=font,
                fill=(0, 0, 0, 100),
            )

        # Stroke/outline
        draw.text(
            (x_cursor, y_pos),
            text, font=font,
            fill=color,
            stroke_width=max(1, int(stroke_width * scale)),
            stroke_fill="#000000",
        )

        x_cursor += width + space_width

    return overlay
