#!/usr/bin/env python3
"""
Refine Qwen3-VL RLBench target/reference grounding results with SAM 3.

Input
-----
A grounding output directory produced by
qwen3vl_rlbench_episode_grounding_verified_updated.py, for example:

    role_grounding_output/place_cups/
    ├── role_spec.json
    ├── episode_grounding.json
    └── frames/
        ├── 000000_0.json
        ├── 000001_5.json
        └── ...

For every selected frame and camera, this script reads Qwen's target/reference
bbox, uses that bbox as an interactive SAM 3 box prompt, and saves precise
binary masks and overlays.

Important
---------
- Qwen3-VL determines semantic roles and the object instance bbox.
- SAM 3 only refines the selected bbox into a pixel-level instance mask.
- No additional Qwen call is made by this script.
- The SAM model is loaded once for the whole episode.

Example
-------
python qwen3_bbox_guided_sam3_demo.py \
  --grounding-dir role_grounding_output/place_cups \
  --sam-model-dir /common-data-32t/.cache/facebook/sam3 \
  --sam-checkpoint /common-data-32t/.cache/facebook/sam3/sam3.pt \
  --output-dir sam3_role_masks/place_cups \
  --roles target,reference \
  --frame-interval 1 \
  --device cuda \
  --no-bf16 \
  --make-video
"""

from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
except ImportError:
    Sam3Processor = None
    build_sam3_image_model = None


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_ROLES = ("target", "reference")
ROLE_STYLE: dict[str, dict[str, Any]] = {
    "target": {"label": "Target", "color": (40, 220, 40)},
    "reference": {"label": "Reference", "color": (255, 170, 30)},
    "interaction_part": {"label": "Interaction", "color": (80, 170, 255)},
}


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--grounding-dir",
        required=True,
        help="Qwen episode grounding output, e.g. role_grounding_output/place_cups.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <grounding-dir>/sam3_segmentation.",
    )
    parser.add_argument("--sam-model-dir", required=True)
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Explicit .pt/.pth checkpoint; otherwise searched below --sam-model-dir.",
    )
    parser.add_argument(
        "--episode-dir",
        default=None,
        help=(
            "Optional original RLBench episode directory. Use this when source_images "
            "stored in frame JSON are unavailable or moved."
        ),
    )
    parser.add_argument(
        "--roles",
        type=parse_csv,
        default=DEFAULT_ROLES,
        help="Comma-separated roles. Default: target,reference.",
    )
    parser.add_argument(
        "--cameras",
        type=parse_csv,
        default=None,
        help="Optional camera subset. Default: cameras recorded in episode_grounding.json.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive index in the naturally sorted grounding frame records.",
    )
    parser.add_argument(
        "--frame-interval",
        "--stride",
        dest="frame_interval",
        type=int,
        default=1,
        help="Process one grounding result every N records.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--bbox-padding-ratio",
        type=float,
        default=0.05,
        help="Expand each Qwen bbox before SAM prompting. Default: 0.05.",
    )
    parser.add_argument(
        "--skip-uncertain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip Qwen localizations marked uncertain.",
    )
    parser.add_argument(
        "--min-box-side",
        type=int,
        default=2,
        help="Skip boxes whose width or height is smaller than this value.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-camera result JSON files already written.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--save-multiview-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--make-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--visualization-cell-width", type=int, default=384)
    parser.add_argument(
        "--mask-alpha",
        type=int,
        default=105,
        help="Mask overlay alpha in [0,255]. Default: 105.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the processing plan without loading SAM 3.",
    )
    return parser


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def natural_sort_key(value: str | Path) -> list[Any]:
    text = Path(value).stem if isinstance(value, Path) else str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def find_checkpoint(model_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {path}")
        return path

    preferred_names = (
        "sam3.pt",
        "sam3.pth",
        "sam3.1_multiplex.pt",
        "sam3.1_multiplex.pth",
    )
    for name in preferred_names:
        path = model_dir / name
        if path.is_file():
            return path.resolve()

    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth"):
        candidates.extend(model_dir.rglob(pattern))
    candidates = sorted(
        (path.resolve() for path in candidates),
        key=lambda path: (
            "sam3.1" not in path.name.lower(),
            "multiplex" not in path.name.lower(),
            len(str(path)),
        ),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No SAM 3 .pt/.pth checkpoint found below {model_dir}."
        )
    return candidates[0]


def autocast_context(device: str, no_bf16: bool):
    if (
        device == "cuda"
        and not no_bf16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def normalize_masks(value: Any) -> np.ndarray:
    masks = tensor_to_numpy(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected SAM masks as NxHxW, got {masks.shape}")
    return masks > 0.5


def normalize_scores(value: Any, count: int) -> np.ndarray:
    if value is None:
        return np.ones((count,), dtype=np.float32)
    scores = tensor_to_numpy(value).reshape(-1)
    if len(scores) != count:
        raise ValueError(
            f"SAM mask/score mismatch: {count} masks and {len(scores)} scores"
        )
    return scores.astype(np.float32)


def run_box_prompt(
    model: Any,
    inference_state: Any,
    image: Image.Image,
    box_xyxy: Sequence[float],
    device: str,
    no_bf16: bool,
) -> tuple[np.ndarray, float]:
    """Use Qwen's pixel-space XYXY bbox as an interactive SAM 3 prompt.

    ``inference_state`` is computed once per source image and reused for all
    roles, avoiding duplicate SAM image-backbone execution.
    """
    box = np.asarray(box_xyxy, dtype=np.float32)
    with autocast_context(device, no_bf16):
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=box,
            mask_input=None,
            multimask_output=False,
            return_logits=False,
        )

    masks_np = normalize_masks(masks)
    scores_np = normalize_scores(scores, len(masks_np))
    best_index = int(np.argmax(scores_np))
    mask = masks_np[best_index]
    if mask.shape != (image.height, image.width):
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image) > 127
    return mask, float(scores_np[best_index])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_frame_result_path(
    grounding_dir: Path,
    item: Mapping[str, Any],
) -> Path:
    value = item.get("result_json")
    if value:
        path = Path(str(value))
        if not path.is_absolute():
            path = grounding_dir / path
        return path
    frame_index = int(item["frame_index"])
    frame_id = str(item["frame_id"])
    return grounding_dir / "frames" / f"{frame_index:06d}_{frame_id}.json"


def discover_grounding_records(
    grounding_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the episode summary and all per-frame Qwen grounding records."""
    summary_path = grounding_dir / "episode_grounding.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}

    records: list[dict[str, Any]] = []
    frame_items = summary.get("frames", [])
    if isinstance(frame_items, list) and frame_items:
        for item in frame_items:
            if not isinstance(item, Mapping):
                continue
            path = resolve_frame_result_path(grounding_dir, item)
            if not path.is_file():
                raise FileNotFoundError(f"Grounding frame JSON not found: {path}")
            records.append(load_json(path))
    else:
        frame_dir = grounding_dir / "frames"
        paths = sorted(frame_dir.glob("*.json"), key=natural_sort_key)
        records = [load_json(path) for path in paths]

    if not records:
        # Also support the original single-frame role_grounding.json layout.
        single_path = grounding_dir / "role_grounding.json"
        if single_path.is_file():
            single = load_json(single_path)
            records = [
                {
                    "frame_index": 0,
                    "frame_id": "0",
                    "source_images": single.get("source_images", {}),
                    "views": single.get("views", {}),
                    "errors": [],
                }
            ]
            summary = {
                **summary,
                "instruction": single.get("instruction"),
                "role_spec": single.get("role_spec", {}),
                "camera_names": list(single.get("views", {}).keys()),
            }

    if not records:
        raise FileNotFoundError(
            f"No frame grounding results found below {grounding_dir}. "
            "Expected episode_grounding.json + frames/*.json."
        )

    records.sort(key=lambda item: int(item.get("frame_index", 0)))
    return summary, records


def select_records(
    records: Sequence[dict[str, Any]],
    start: int,
    end: int | None,
    interval: int,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    if interval <= 0:
        raise ValueError("--frame-interval must be positive.")
    count = len(records)
    if start < 0:
        start += count
    if not 0 <= start < count:
        raise IndexError(f"--start={start} is outside 0..{count - 1}")

    stop = count if end is None else end
    if stop < 0:
        stop += count
    stop = min(stop, count)
    if stop <= start:
        raise ValueError(f"Empty record range: start={start}, end={end}")

    selected = list(records[start:stop:interval])
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive.")
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No grounding records selected.")
    return selected


def discover_cameras(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested:
        return tuple(requested)

    summary_cameras = summary.get("camera_names")
    if isinstance(summary_cameras, list) and summary_cameras:
        return tuple(str(item) for item in summary_cameras)

    first_views = records[0].get("views", {})
    if isinstance(first_views, Mapping) and first_views:
        return tuple(str(name) for name in first_views.keys())
    raise ValueError("Could not determine camera names from grounding output.")


def resolve_source_image(
    grounding_dir: Path,
    episode_dir: Path | None,
    record: Mapping[str, Any],
    camera: str,
) -> Path:
    source_images = record.get("source_images", {})
    if isinstance(source_images, Mapping) and source_images.get(camera):
        raw = Path(str(source_images[camera])).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend(
                [
                    grounding_dir / raw,
                    grounding_dir.parent / raw,
                    Path.cwd() / raw,
                ]
            )
        for path in candidates:
            if path.is_file():
                return path.resolve()

    if episode_dir is not None:
        frame_id = str(record.get("frame_id"))
        camera_dir = episode_dir / f"{camera}_rgb"
        for extension in IMAGE_EXTENSIONS:
            path = camera_dir / f"{frame_id}{extension}"
            if path.is_file():
                return path.resolve()
        matches = list(camera_dir.glob(f"{frame_id}.*"))
        for path in matches:
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                return path.resolve()

    raise FileNotFoundError(
        f"Could not resolve source image for camera={camera}, "
        f"frame_id={record.get('frame_id')}. Supply --episode-dir if files moved."
    )


def bbox_from_grounding(
    role_result: Mapping[str, Any],
    image_width: int,
    image_height: int,
) -> list[int] | None:
    raw = role_result.get("bbox_xyxy")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 4:
        values = [int(round(float(item))) for item in raw]
    else:
        normalized = role_result.get("bbox_2d")
        if not (
            isinstance(normalized, Sequence)
            and not isinstance(normalized, (str, bytes))
            and len(normalized) == 4
        ):
            return None
        x1, y1, x2, y2 = [float(item) for item in normalized]
        values = [
            round(x1 * image_width / 1000),
            round(y1 * image_height / 1000),
            round(x2 * image_width / 1000),
            round(y2 * image_height / 1000),
        ]

    x1, y1, x2, y2 = values
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return [x1, y1, x2, y2]


def expand_bbox(
    box: Sequence[int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> list[int]:
    x1, y1, x2, y2 = [int(value) for value in box]
    width = x2 - x1
    height = y2 - y1
    pad_x = round(width * padding_ratio)
    pad_y = round(height * padding_ratio)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    ]


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_transparent_label(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    image_rgba = image.convert("RGBA")
    probe = ImageDraw.Draw(image_rgba)
    left, top, right, bottom = probe.textbbox(xy, text, font=font)
    padding = 2
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        fill=(0, 0, 0, 90),
    )
    draw.text(xy, text, font=font, fill=(255, 255, 255, 255))
    composed = Image.alpha_composite(image_rgba, overlay).convert("RGB")
    image.paste(composed)


def add_mask_overlay(
    rgba_pixels: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: int,
) -> None:
    layer = np.zeros_like(rgba_pixels)
    layer[mask, :3] = color
    layer[mask, 3] = alpha
    # Alpha-composite is handled after all masks. Since semantic masks may overlap,
    # keeping the larger alpha per pixel produces a stable visualization.
    replace = layer[..., 3] > rgba_pixels[..., 3]
    rgba_pixels[replace] = layer[replace]


def draw_mask_boundary(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    # Lightweight four-neighbour erosion without OpenCV/scipy.
    eroded = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        eroded[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    boundary = mask & ~eroded
    ys, xs = np.nonzero(boundary)
    # Drawing every point is fine for small RLBench images; subsample on large images.
    step = max(1, len(xs) // 12000)
    for x, y in zip(xs[::step], ys[::step]):
        draw.point((int(x), int(y)), fill=(*color, 255))


def render_camera_overlay(
    image: Image.Image,
    role_outputs: Mapping[str, Mapping[str, Any]],
    masks: Mapping[str, np.ndarray],
    mask_alpha: int,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay_pixels = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for role, mask in masks.items():
        style = ROLE_STYLE.get(role, {"color": (255, 220, 40)})
        add_mask_overlay(
            overlay_pixels,
            mask.astype(bool),
            tuple(style["color"]),
            mask_alpha,
        )
    rendered = Image.alpha_composite(
        base,
        Image.fromarray(overlay_pixels, mode="RGBA"),
    )
    draw = ImageDraw.Draw(rendered)
    font = load_font(max(9, round(min(image.size) * 0.035)))
    line_width = max(1, min(image.size) // 128)

    for role, output in role_outputs.items():
        if not output.get("segmented", False):
            continue
        style = ROLE_STYLE.get(
            role,
            {"label": role.replace("_", " ").title(), "color": (255, 220, 40)},
        )
        color = tuple(style["color"])
        prompt_box = output.get("sam_prompt_bbox_xyxy")
        if prompt_box:
            draw.rectangle(tuple(prompt_box), outline=(*color, 255), width=line_width)
        mask_box = output.get("sam_mask_bbox_xyxy")
        if mask_box:
            label_x = int(mask_box[0]) + line_width + 1
            label_y = max(1, int(mask_box[1]) - 14)
        elif prompt_box:
            label_x = int(prompt_box[0]) + line_width + 1
            label_y = max(1, int(prompt_box[1]) - 14)
        else:
            continue
        suffix = " ?" if output.get("qwen_uncertain", False) else ""
        rendered_rgb = rendered.convert("RGB")
        draw_transparent_label(
            rendered_rgb,
            (label_x, label_y),
            f"{style['label']}{suffix}",
            font,
        )
        rendered = rendered_rgb.convert("RGBA")
        draw = ImageDraw.Draw(rendered)

    for role, mask in masks.items():
        style = ROLE_STYLE.get(role, {"color": (255, 220, 40)})
        draw_mask_boundary(draw, mask, tuple(style["color"]))
    return rendered.convert("RGB")


def fit_to_cell(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    canvas.paste(
        resized,
        ((width - resized.width) // 2, (height - resized.height) // 2),
    )
    return canvas


def compose_multiview(
    images: Mapping[str, Path],
    cameras: Sequence[str],
    frame_index: int,
    frame_id: str,
    cell_width: int,
) -> Image.Image:
    loaded = [Image.open(images[camera]).convert("RGB") for camera in cameras]
    aspect = max(image.height / image.width for image in loaded)
    cell_height = max(128, round(cell_width * aspect))
    header_height = 35
    canvas = Image.new(
        "RGB",
        (cell_width * len(cameras), header_height + cell_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(11, cell_width // 30))
    draw.text(
        (10, 8),
        f"Frame {frame_index} · source id {frame_id}",
        font=font,
        fill=(235, 235, 235),
    )
    for index, (camera, image) in enumerate(zip(cameras, loaded)):
        x = index * cell_width
        canvas.paste(fit_to_cell(image, cell_width, cell_height), (x, header_height))
        draw_transparent_label(
            canvas,
            (x + 7, header_height + 7),
            camera.replace("_", " ").title(),
            font,
        )
    return canvas


def write_mp4(
    frames: Iterable[Image.Image],
    output_path: Path,
    fps: float,
) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        return False, "no frames"

    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        try:
            writer.append_data(np.asarray(first.convert("RGB")))
            for frame in frames:
                writer.append_data(np.asarray(frame.convert("RGB")))
        finally:
            writer.close()
        return True, "imageio"
    except Exception as imageio_error:
        try:
            import cv2

            first_array = np.asarray(first.convert("RGB"))
            height, width = first_array.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter could not open output file")
            try:
                writer.write(cv2.cvtColor(first_array, cv2.COLOR_RGB2BGR))
                for frame in frames:
                    array = np.asarray(frame.convert("RGB"))
                    if array.shape[:2] != (height, width):
                        array = cv2.resize(array, (width, height))
                    writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return True, "opencv"
        except Exception as cv2_error:
            output_path.unlink(missing_ok=True)
            return False, f"imageio failed: {imageio_error}; OpenCV failed: {cv2_error}"


def role_name_from_spec(
    role_spec: Mapping[str, Any],
    role: str,
) -> str | None:
    value = role_spec.get(role)
    if not isinstance(value, Mapping):
        return None
    name = value.get("name")
    return str(name) if name is not None else None


def process_episode(args: argparse.Namespace) -> dict[str, Any]:
    grounding_dir = Path(args.grounding_dir).expanduser().resolve()
    if not grounding_dir.is_dir():
        raise NotADirectoryError(f"Grounding directory not found: {grounding_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else grounding_dir / "sam3_segmentation"
    )
    episode_dir = (
        Path(args.episode_dir).expanduser().resolve()
        if args.episode_dir
        else None
    )

    summary, all_records = discover_grounding_records(grounding_dir)
    records = select_records(
        all_records,
        start=args.start,
        end=args.end,
        interval=args.frame_interval,
        max_frames=args.max_frames,
    )
    cameras = discover_cameras(summary, records, args.cameras)
    roles = tuple(args.roles)
    invalid_roles = set(roles) - set(ROLE_STYLE)
    if invalid_roles:
        raise ValueError(
            f"Unknown role(s): {sorted(invalid_roles)}; "
            f"allowed: {sorted(ROLE_STYLE)}"
        )

    role_spec = summary.get("role_spec", {})
    if not isinstance(role_spec, Mapping):
        role_spec = {}
    instruction = summary.get("instruction")

    # A partially completed grounding directory may have role_spec.json even
    # when episode_grounding.json has not yet been written.
    role_spec_path = grounding_dir / "role_spec.json"
    if role_spec_path.is_file() and (not role_spec or not instruction):
        role_document = load_json(role_spec_path)
        if not role_spec:
            loaded_role_spec = role_document.get("role_spec", role_document)
            if isinstance(loaded_role_spec, Mapping):
                role_spec = loaded_role_spec
        if not instruction:
            instruction = role_document.get("instruction")

    plan: list[dict[str, Any]] = []
    for record in records:
        images = {
            camera: str(
                resolve_source_image(
                    grounding_dir,
                    episode_dir,
                    record,
                    camera,
                )
            )
            for camera in cameras
        }
        plan.append(
            {
                "frame_index": int(record.get("frame_index", 0)),
                "frame_id": str(record.get("frame_id", "0")),
                "images": images,
            }
        )

    print(f"Grounding directory: {grounding_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames: {len(records)} selected from {len(all_records)}")
    print(f"Cameras: {', '.join(cameras)}")
    print(f"Roles: {', '.join(roles)}")
    print(f"Instruction: {instruction}")

    if args.dry_run:
        result = {
            "dry_run": True,
            "grounding_dir": str(grounding_dir),
            "output_dir": str(output_dir),
            "cameras": list(cameras),
            "roles": list(roles),
            "frames": plan,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    if Sam3Processor is None or build_sam3_image_model is None:
        raise ImportError(
            "The sam3 package is required. Run this script in the environment "
            "where your ModelScope SAM3 demo already works."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    model_dir = Path(args.sam_model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(f"SAM model directory not found: {model_dir}")
    checkpoint = find_checkpoint(model_dir, args.sam_checkpoint)
    print(f"Loading SAM 3 once: {checkpoint}")

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        compile=args.compile,
    )
    processor = Sam3Processor(
        model=model,
        device=args.device,
        confidence_threshold=0.5,
    )
    if model.inst_interactive_predictor is None:
        raise RuntimeError("SAM 3 interactive predictor was not created.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_output_root = output_dir / "frames"
    multiview_root = output_dir / "multiview"
    frame_output_root.mkdir(parents=True, exist_ok=True)
    if args.save_multiview_images:
        multiview_root.mkdir(parents=True, exist_ok=True)

    final_frames: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for selected_index, record in enumerate(records):
        frame_index = int(record.get("frame_index", selected_index))
        frame_id = str(record.get("frame_id", selected_index))
        frame_key = f"{frame_index:06d}_{frame_id}"
        print(
            f"[{selected_index + 1}/{len(records)}] "
            f"SAM refine frame_index={frame_index}, frame_id={frame_id}"
        )
        frame_dir = frame_output_root / frame_key
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_result: dict[str, Any] = {
            "frame_index": frame_index,
            "frame_id": frame_id,
            "views": {},
            "errors": [],
        }
        overlay_paths: dict[str, Path] = {}

        views = record.get("views", {})
        if not isinstance(views, Mapping):
            views = {}

        for camera in cameras:
            camera_dir = frame_dir / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            camera_result_path = camera_dir / "result.json"
            overlay_path = camera_dir / "overlay.png"

            if args.resume and camera_result_path.is_file() and overlay_path.is_file():
                camera_result = load_json(camera_result_path)
                frame_result["views"][camera] = camera_result
                overlay_paths[camera] = overlay_path
                print(f"  {camera}: resume")
                continue

            image_path = resolve_source_image(
                grounding_dir,
                episode_dir,
                record,
                camera,
            )
            image = Image.open(image_path).convert("RGB")
            view_result = views.get(camera, {})
            if not isinstance(view_result, Mapping):
                view_result = {}

            camera_output: dict[str, Any] = {
                "source_image": str(image_path),
                "image_size": [image.width, image.height],
                "roles": {},
            }
            masks: dict[str, np.ndarray] = {}

            # Compute the image backbone exactly once, then reuse the state for
            # target/reference box prompts.
            with autocast_context(args.device, args.no_bf16):
                inference_state = processor.set_image(image)

            for role in roles:
                qwen_result = view_result.get(role, {})
                if not isinstance(qwen_result, Mapping):
                    qwen_result = {}

                role_output: dict[str, Any] = {
                    "role": role,
                    "role_name": role_name_from_spec(role_spec, role),
                    "qwen_visible": bool(qwen_result.get("visible", False)),
                    "qwen_uncertain": bool(qwen_result.get("uncertain", False)),
                    "qwen_uncertain_reason": qwen_result.get("uncertain_reason"),
                    "qwen_evidence": qwen_result.get("evidence"),
                    "qwen_bbox_2d": qwen_result.get("bbox_2d"),
                    "qwen_bbox_xyxy": qwen_result.get("bbox_xyxy"),
                    "segmented": False,
                    "skip_reason": None,
                }

                if not role_output["qwen_visible"]:
                    role_output["skip_reason"] = "qwen_visible_false"
                    camera_output["roles"][role] = role_output
                    continue
                if args.skip_uncertain and role_output["qwen_uncertain"]:
                    role_output["skip_reason"] = "qwen_uncertain"
                    camera_output["roles"][role] = role_output
                    continue

                qwen_box = bbox_from_grounding(
                    qwen_result,
                    image.width,
                    image.height,
                )
                if qwen_box is None:
                    role_output["skip_reason"] = "missing_qwen_bbox"
                    camera_output["roles"][role] = role_output
                    continue
                if (
                    qwen_box[2] - qwen_box[0] < args.min_box_side
                    or qwen_box[3] - qwen_box[1] < args.min_box_side
                ):
                    role_output["skip_reason"] = "qwen_bbox_too_small"
                    camera_output["roles"][role] = role_output
                    continue

                sam_box = expand_bbox(
                    qwen_box,
                    image.width,
                    image.height,
                    args.bbox_padding_ratio,
                )
                try:
                    mask, sam_score = run_box_prompt(
                        model=model,
                        inference_state=inference_state,
                        image=image,
                        box_xyxy=sam_box,
                        device=args.device,
                        no_bf16=args.no_bf16,
                    )
                    mask_path = camera_dir / f"{role}_mask.png"
                    Image.fromarray(
                        (mask.astype(np.uint8) * 255),
                        mode="L",
                    ).save(mask_path)
                    masked_rgba = np.zeros(
                        (image.height, image.width, 4), dtype=np.uint8
                    )
                    image_array = np.asarray(image)
                    masked_rgba[mask, :3] = image_array[mask]
                    masked_rgba[mask, 3] = 255
                    masked_path = camera_dir / f"{role}_masked.png"
                    Image.fromarray(masked_rgba, mode="RGBA").save(masked_path)

                    role_output.update(
                        {
                            "segmented": True,
                            "sam_score": sam_score,
                            "sam_prompt_bbox_xyxy": sam_box,
                            "sam_mask_bbox_xyxy": mask_bbox(mask),
                            "mask_area_pixels": int(mask.sum()),
                            "mask_area_ratio": float(mask.mean()),
                            "mask_file": mask_path.name,
                            "masked_file": masked_path.name,
                        }
                    )
                    masks[role] = mask
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    role_output["skip_reason"] = "sam_error"
                    role_output["error"] = error_text
                    frame_result["errors"].append(
                        {
                            "camera": camera,
                            "role": role,
                            "error": error_text,
                        }
                    )
                    failures.append(
                        {
                            "frame_index": frame_index,
                            "frame_id": frame_id,
                            "camera": camera,
                            "role": role,
                            "error": error_text,
                        }
                    )
                    traceback.print_exc()
                    if args.fail_fast:
                        raise

                camera_output["roles"][role] = role_output

            overlay = render_camera_overlay(
                image=image,
                role_outputs=camera_output["roles"],
                masks=masks,
                mask_alpha=max(0, min(255, args.mask_alpha)),
            )
            overlay.save(overlay_path)
            camera_output["overlay"] = overlay_path.name
            atomic_json_dump(camera_output, camera_result_path)
            frame_result["views"][camera] = camera_output
            overlay_paths[camera] = overlay_path

        frame_result_path = frame_dir / "frame_result.json"
        atomic_json_dump(frame_result, frame_result_path)
        frame_result["result_json"] = str(frame_result_path.relative_to(output_dir))

        if args.save_multiview_images:
            multiview_path = multiview_root / f"{frame_key}.png"
            compose_multiview(
                images=overlay_paths,
                cameras=cameras,
                frame_index=frame_index,
                frame_id=frame_id,
                cell_width=args.visualization_cell_width,
            ).save(multiview_path)
            frame_result["multiview"] = str(multiview_path.relative_to(output_dir))

        final_frames.append(frame_result)

    video_info: dict[str, Any] = {"created": False}
    if args.make_video and args.save_multiview_images:
        video_path = output_dir / "sam3_multiview.mp4"

        def video_frames() -> Iterable[Image.Image]:
            for frame in final_frames:
                path = output_dir / str(frame["multiview"])
                yield Image.open(path).convert("RGB")

        created, backend = write_mp4(
            video_frames(),
            video_path,
            args.video_fps,
        )
        video_info = {
            "created": created,
            "path": str(video_path) if created else None,
            "backend": backend,
        }
        if not created:
            print(f"Warning: video not created: {backend}")

    final_summary = {
        "grounding_dir": str(grounding_dir),
        "output_dir": str(output_dir),
        "sam_model_dir": str(model_dir),
        "sam_checkpoint": str(checkpoint),
        "instruction": instruction,
        "role_spec": role_spec,
        "roles": list(roles),
        "cameras": list(cameras),
        "frame_interval": args.frame_interval,
        "bbox_padding_ratio": args.bbox_padding_ratio,
        "num_frames": len(final_frames),
        "num_failures": len(failures),
        "failures": failures,
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "frame_id": frame["frame_id"],
                "result_json": frame["result_json"],
                "multiview": frame.get("multiview"),
            }
            for frame in final_frames
        ],
        "video": video_info,
    }
    atomic_json_dump(final_summary, output_dir / "sam3_segmentation.json")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return final_summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process_episode(args)


if __name__ == "__main__":
    main()
