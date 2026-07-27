#!/usr/bin/env python3
"""Compose compact per-frame montages that compare Stage 1, Stage 3, and Stage 5 outputs.

For each rendered frame/camera entry in decision_visualization.json, this script
places a compact Stage 1 summary card, the Stage 3 reprojection overlay, and
the Stage 5 decision overlay side by side with short labels. It keeps the text
minimal and is intended as a quick same-moment summary of how the episode-level
candidate generation, fused scene, and decision overlay relate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-meta-json", required=True, help="Path to decision_visualization.json from stage 5.")
    parser.add_argument("--stage1-candidates-json", default=None, help="Optional explicit episode_candidates.json path (default: inferred next to the episode output dir).")
    parser.add_argument("--output-dir", default=None, help="Default: viz_compare next to decision_visualization.json.")
    parser.add_argument("--panel-gap", type=int, default=8, help="Gap in pixels between panels.")
    parser.add_argument("--label-height", type=int, default=26, help="Height of the small stage label strip.")
    parser.add_argument("--summary-width", type=int, default=360, help="Width in pixels for the Stage 1 summary card.")
    parser.add_argument("--background", default="white", help="Background color for the montage canvas.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_image(path: str | None) -> Image.Image | None:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.is_file():
        return None
    return Image.open(image_path).convert("RGBA")


def _fit_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    if image.height <= 0:
        return image
    target_width = max(1, int(round(image.width * (target_height / image.height))))
    return ImageOps.contain(image, (target_width, target_height))


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        return ImageFont.load_default()


def _small_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        return ImageFont.load_default()


def _draw_panel_label(image: Image.Image, label: str, label_height: int) -> Image.Image:
    canvas = Image.new("RGBA", (image.width, image.height + label_height), (255, 255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, image.width, label_height], fill=(24, 24, 24, 255))
    draw.text((8, 4), label, fill=(255, 255, 255, 255), font=_font())
    return canvas


def _summary_lines(stage1: Mapping[str, Any]) -> list[str]:
    instruction = str(stage1.get("instruction") or "")
    target = stage1.get("role_spec", {}).get("target", {}) if isinstance(stage1.get("role_spec"), Mapping) else {}
    target_name = str(target.get("name") or "")
    cameras = stage1.get("camera_names", [])
    frames = stage1.get("frames", [])
    first_frame = frames[0] if frames else {}
    views = first_frame.get("views", {}) if isinstance(first_frame, Mapping) else {}
    counts = []
    for camera in cameras:
        view = views.get(camera, {}) if isinstance(views, Mapping) else {}
        counts.append(str(view.get("num_candidates", 0)))
    return [
        f"I: {instruction}",
        f"T: {target_name}",
        f"cams: {','.join(cameras)}",
        f"frames: {len(frames)}",
        f"cand1: {'/'.join(counts)}" if counts else "cand1: -",
    ]


def _stage1_card(stage1: Mapping[str, Any], width: int, label_height: int) -> Image.Image:
    lines = _summary_lines(stage1)
    body_font = _small_font()
    # Create a small card with concise lines; the height adapts to the text but
    # remains visually lighter than the image panels.
    body_lines = [textwrap.shorten(line, width=52, placeholder="…") for line in lines]
    metrics = ImageDraw.Draw(Image.new("RGBA", (1, 1))).multiline_textbbox((0, 0), "\n".join(body_lines), font=body_font, spacing=3)
    text_w = metrics[2] - metrics[0]
    text_h = metrics[3] - metrics[1]
    card_w = max(width, text_w + 24)
    card_h = text_h + label_height + 20
    card = Image.new("RGBA", (card_w, card_h), (248, 248, 248, 255))
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, card_w, label_height], fill=(24, 24, 24, 255))
    draw.text((8, 4), "S1", fill=(255, 255, 255, 255), font=_font())
    draw.multiline_text((10, label_height + 4), "\n".join(body_lines), fill=(30, 30, 30, 255), font=body_font, spacing=3)
    return card


def _compose_triplet(summary: Image.Image, left: Image.Image, right: Image.Image, gap: int, label_height: int) -> Image.Image:
    target_height = max(summary.height, left.height, right.height)
    summary = _fit_height(summary, target_height)
    left = _fit_height(left, target_height)
    right = _fit_height(right, target_height)
    summary = _draw_panel_label(summary, "S1", label_height)
    left = _draw_panel_label(left, "S3", label_height)
    right = _draw_panel_label(right, "S5", label_height)
    height = max(summary.height, left.height, right.height)
    width = summary.width + gap + left.width + gap + right.width
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    x = 0
    canvas.paste(summary, (x, 0))
    x += summary.width + gap
    canvas.paste(left, (x, 0))
    x += left.width + gap
    canvas.paste(right, (x, 0))
    return canvas
    target_height = max(left.height, right.height)
    left = _fit_height(left, target_height)
    right = _fit_height(right, target_height)
    left = _draw_panel_label(left, "S3", label_height)
    right = _draw_panel_label(right, "S5", label_height)
    height = max(left.height, right.height)
    width = left.width + gap + right.width
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def main() -> None:
    args = build_parser().parse_args()
    meta_path = Path(args.decision_meta_json).expanduser().resolve()
    meta = _load_json(meta_path)
    episode_root = meta_path.parent.parent
    stage1_path = Path(args.stage1_candidates_json).expanduser().resolve() if args.stage1_candidates_json else episode_root / "episode_candidates.json"
    stage1 = _load_json(stage1_path) if stage1_path.is_file() else {}

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else meta_path.with_name("viz_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for item in meta.get("rendered", []):
        if not isinstance(item, Mapping):
            continue
        stage5_path = _load_image(str(item.get("output_path")))
        stage3_path = _load_image(str(item.get("background_path")))
        if stage3_path is None or stage5_path is None:
            continue

        summary_card = _stage1_card(stage1, args.summary_width, args.label_height)
        composed = _compose_triplet(summary_card, stage3_path, stage5_path, args.panel_gap, args.label_height)
        frame_id = str(item.get("frame_id"))
        camera = str(item.get("camera"))
        out_path = output_dir / f"{frame_id}_{camera}_stage_compare.png"
        composed.convert("RGB").save(out_path)
        rendered.append(
            {
                "frame_id": frame_id,
                "frame_index": item.get("frame_index"),
                "camera": camera,
                "output_path": str(out_path),
                "stage3_path": str(item.get("background_path") or ""),
                "stage5_path": str(item.get("output_path") or ""),
            }
        )

    summary = {
        "decision_meta_json": str(meta_path),
        "stage1_candidates_json": str(stage1_path),
        "output_dir": str(output_dir),
        "rendered": rendered,
    }
    summary_path = output_dir / "stage_compare.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rendered_images": len(rendered),
                "summary_json": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()