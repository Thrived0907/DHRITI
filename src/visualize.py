"""Draw the two-stage results on the image (PIL only)."""

from __future__ import annotations

from typing import Dict

from PIL import Image, ImageDraw, ImageFont

ACTION_COLORS: Dict[str, str] = {
    "Recycle": "#2e9e4f",
    "Compost": "#8d6e2f",
    "Reuse / Drop-off": "#2b7bd6",
    "Landfill": "#d64541",
    "Manual review": "#f08c00",
}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


def draw_results(image: Image.Image, result) -> Image.Image:
    """Return a copy of `image` with boxes and 'Stage 1 -> Stage 2' labels drawn on it."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    thickness = max(2, round(min(w, h) / 200))
    font = _font(max(12, round(min(w, h) / 40)))

    for item in result.items:
        rec = item.recommendation
        color = ACTION_COLORS.get(rec.action, "#555555")
        x1, y1, x2, y2 = item.crop_box
        if item.stage1 is not None:  # draw the raw detector box, not the padded crop
            x1, y1, x2, y2 = (int(round(v)) for v in item.stage1.box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        parts = [f"#{item.index}"]
        if item.stage1:
            parts.append(f"{item.stage1.class_name} {item.stage1.confidence:.0%}")
        if item.stage2:
            parts.append(f"> {item.stage2.label} {item.stage2.confidence:.0%}")
        parts.append(f"= {rec.action}")
        label = " ".join(parts)

        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        ty = y1 - th - 6 if y1 - th - 6 >= 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill="white", font=font)
    return canvas
