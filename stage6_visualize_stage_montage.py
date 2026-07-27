#!/usr/bin/env python3
"""Compose compact per-frame montages that compare Stage 3 and Stage 5 outputs.

For each rendered frame/camera entry in decision_visualization.json, this script
places the Stage 3 reprojection overlay and Stage 5 decision overlay side by
side with short labels. It keeps the text minimal and is intended as a quick
same-moment summary of how the fused scene and the decision overlay relate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-meta-json", required=True, help="Path to decision_visualization.json from stage 5.")
    parser.add_argument("--output-dir", default=None, help="Default: viz_compare next to decision_visualization.json.")
    parser.add_argument("--panel-gap", type=int, default=8, help="Gap in pixels between panels.")
    parser.add_argument("--label-height", type=int, default=26, help="Height of the small stage label strip.")
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


def _draw_panel_label(image: Image.Image, label: str, label_height: int) -> Image.Image:
    canvas = Image.new("RGBA", (image.width, image.height + label_height), (255, 255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, image.width, label_height], fill=(24, 24, 24, 255))
    draw.text((8, 4), label, fill=(255, 255, 255, 255), font=_font())
    return canvas


def _compose_pair(left: Image.Image, right: Image.Image, gap: int, label_height: int) -> Image.Image:
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

        composed = _compose_pair(stage3_path, stage5_path, args.panel_gap, args.label_height)
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