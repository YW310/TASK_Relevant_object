#!/usr/bin/env python3
"""
Run SAM 3 image segmentation using a local ModelScope checkpoint.

Supported input modes
---------------------
Concept segmentation:
  text          : open-vocabulary text prompt
  exemplar_box  : positive/negative example boxes
  text_box      : text + example boxes

Interactive instance segmentation:
  point         : one or more positive/negative points
  box           : an XYXY box
  point_box     : points + XYXY box
  mask          : an external binary/grayscale mask prompt
  mask_refine   : first point prediction, then refinement with previous mask logits
  all           : run every applicable mode from the supplied arguments

Coordinate conventions
----------------------
--point X Y LABEL
    Pixel coordinates. LABEL=1 is foreground; LABEL=0 is background.

--box X1 Y1 X2 Y2
    Pixel-space XYXY box. For concept modes it is converted internally to
    normalized [center_x, center_y, width, height]. For interactive modes it
    remains pixel-space XYXY.

--negative-box X1 Y1 X2 Y2
    Negative concept example box. It is only used by exemplar_box/text_box.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


MODE_CHOICES = (
    "text",
    "exemplar_box",
    "text_box",
    "point",
    "box",
    "point_box",
    "mask",
    "mask_refine",
    "all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--model-dir", required=True, help="ModelScope snapshot directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path. Otherwise searched under --model-dir.",
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", default="./sam31_outputs")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="all")
    parser.add_argument("--text", default=None, help='Text prompt, e.g. "white mug".')
    parser.add_argument(
        "--point",
        nargs=3,
        action="append",
        metavar=("X", "Y", "LABEL"),
        help="Pixel point. Repeatable. LABEL: 1 foreground, 0 background.",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Positive pixel XYXY box. Repeatable for concept prompting; "
            "the first box is used by interactive modes."
        ),
    )
    parser.add_argument(
        "--negative-box",
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Negative pixel XYXY concept example box. Repeatable.",
    )
    parser.add_argument(
        "--mask-input",
        default=None,
        help="Binary/grayscale mask image used by mode=mask.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Concept segmentation confidence threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum masks kept for concept modes.",
    )
    parser.add_argument(
        "--interactive-top-k",
        type=int,
        default=3,
        help="Maximum interactive candidate masks kept.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Use CUDA in practice; CPU inference is extremely demanding.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable CUDA bfloat16 autocast.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile in the SAM 3 builder.",
    )
    return parser.parse_args()


def find_checkpoint(model_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
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
            return path

    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth"):
        candidates.extend(model_dir.rglob(pattern))

    # The official builder uses torch.load, so a safetensors-only file is not
    # directly usable without conversion.
    candidates = sorted(
        (p.resolve() for p in candidates),
        key=lambda p: (
            "sam3.1" not in p.name.lower(),
            "multiplex" not in p.name.lower(),
            len(str(p)),
        ),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .pt or .pth checkpoint found below {model_dir}. "
            "Expected a file such as sam3.1_multiplex.pt."
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


def to_float_box(raw: Sequence[str]) -> np.ndarray:
    values = np.asarray([float(v) for v in raw], dtype=np.float32)
    x1, y1, x2, y2 = values.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid XYXY box: {values.tolist()}")
    return values


def parse_boxes(raw_boxes: Sequence[Sequence[str]] | None) -> list[np.ndarray]:
    return [to_float_box(raw) for raw in (raw_boxes or [])]


def parse_points(
    raw_points: Sequence[Sequence[str]] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not raw_points:
        return None, None

    coords: list[list[float]] = []
    labels: list[int] = []
    for x, y, label in raw_points:
        int_label = int(label)
        if int_label not in (0, 1):
            raise ValueError(f"Point label must be 0 or 1, got {label}")
        coords.append([float(x), float(y)])
        labels.append(int_label)

    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def xyxy_to_normalized_cxcywh(
    box: np.ndarray, width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = box.astype(float).tolist()
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    values = np.clip([cx, cy, bw, bh], 0.0, 1.0)
    return values.astype(float).tolist()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def normalize_mask_array(masks: Any) -> np.ndarray:
    masks_np = tensor_to_numpy(masks)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None]
    if masks_np.ndim != 3:
        raise ValueError(f"Expected masks as NxHxW, got shape {masks_np.shape}")
    return masks_np > 0.5


def normalize_scores(scores: Any, count: int) -> np.ndarray:
    if scores is None:
        return np.ones((count,), dtype=np.float32)
    scores_np = tensor_to_numpy(scores).reshape(-1)
    if len(scores_np) != count:
        raise ValueError(f"Mask/score count mismatch: {count} masks, {len(scores_np)} scores")
    return scores_np


def sort_and_limit(
    masks: Any,
    scores: Any,
    top_k: int,
    boxes: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    masks_np = normalize_mask_array(masks)
    scores_np = normalize_scores(scores, len(masks_np))
    order = np.argsort(-scores_np)
    if top_k > 0:
        order = order[:top_k]

    boxes_np = None
    if boxes is not None:
        boxes_np = tensor_to_numpy(boxes).reshape(-1, 4)[order]
    return masks_np[order], scores_np[order], boxes_np


def read_external_mask(path: str, target_size: int = 256) -> np.ndarray:
    mask = Image.open(path).convert("L").resize(
        (target_size, target_size), Image.Resampling.NEAREST
    )
    binary = np.asarray(mask, dtype=np.float32) > 127
    # SAM mask_input expects logits rather than a 0/1 bitmap.
    logits = np.where(binary, 10.0, -10.0).astype(np.float32)
    return logits[None, :, :]


def ensure_inside_image(
    points: np.ndarray | None,
    boxes: Iterable[np.ndarray],
    width: int,
    height: int,
) -> None:
    if points is not None:
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] >= width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] >= height)
        ):
            raise ValueError(f"At least one point lies outside image size {width}x{height}")

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(
                f"Box {box.tolist()} lies outside image size {width}x{height}"
            )


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = (
        (255, 80, 80),
        (80, 220, 120),
        (80, 150, 255),
        (255, 190, 60),
        (190, 90, 255),
        (40, 220, 220),
        (255, 100, 190),
        (160, 210, 60),
    )
    return palette[index % len(palette)]


def save_visualization(
    image: Image.Image,
    output_dir: Path,
    name: str,
    masks: np.ndarray,
    scores: np.ndarray,
    predicted_boxes: np.ndarray | None,
    input_points: np.ndarray | None,
    input_labels: np.ndarray | None,
    input_box: np.ndarray | None,
) -> dict[str, Any]:
    mode_dir = output_dir / name
    mode_dir.mkdir(parents=True, exist_ok=True)

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_pixels = np.asarray(overlay).copy()

    mask_files: list[str] = []
    for index, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        color = color_for_index(index)
        layer = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
        layer[mask_bool, :3] = color
        layer[mask_bool, 3] = 105
        overlay_pixels = np.maximum(overlay_pixels, layer)

        mask_path = mode_dir / f"mask_{index:02d}.png"
        Image.fromarray((mask_bool * 255).astype(np.uint8)).save(mask_path)
        mask_files.append(str(mask_path))

    overlay = Image.fromarray(overlay_pixels, mode="RGBA")
    rendered = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(rendered)

    if predicted_boxes is not None:
        for index, box in enumerate(predicted_boxes):
            x1, y1, x2, y2 = [float(v) for v in box]
            color = color_for_index(index)
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
            draw.text(
                (x1 + 3, y1 + 3),
                f"{index}: {scores[index]:.3f}",
                fill=color + (255,),
            )

    if input_box is not None:
        x1, y1, x2, y2 = [float(v) for v in input_box]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255, 255), width=4)

    if input_points is not None and input_labels is not None:
        radius = max(4, round(min(image.size) * 0.007))
        for (x, y), label in zip(input_points, input_labels):
            color = (20, 255, 20, 255) if int(label) == 1 else (255, 30, 30, 255)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
                outline=(255, 255, 255, 255),
                width=2,
            )

    overlay_path = mode_dir / "overlay.png"
    rendered.convert("RGB").save(overlay_path, quality=95)

    result = {
        "mode": name,
        "num_masks": int(len(masks)),
        "scores": [float(v) for v in scores],
        "overlay": str(overlay_path),
        "mask_files": mask_files,
    }
    if predicted_boxes is not None:
        result["predicted_boxes_xyxy"] = predicted_boxes.astype(float).tolist()

    with (mode_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def run_concept_prompt(
    processor: Sam3Processor,
    image: Image.Image,
    text: str | None,
    positive_boxes: list[np.ndarray],
    negative_boxes: list[np.ndarray],
    threshold: float,
    top_k: int,
    device: str,
    no_bf16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    processor.set_confidence_threshold(threshold)
    with autocast_context(device, no_bf16):
        state = processor.set_image(image)

        output = None
        if text:
            output = processor.set_text_prompt(prompt=text, state=state)

        for box in positive_boxes:
            normalized = xyxy_to_normalized_cxcywh(box, image.width, image.height)
            output = processor.add_geometric_prompt(
                box=normalized, label=True, state=state
            )

        for box in negative_boxes:
            normalized = xyxy_to_normalized_cxcywh(box, image.width, image.height)
            output = processor.add_geometric_prompt(
                box=normalized, label=False, state=state
            )

    if output is None:
        raise ValueError("Concept prompting requires text or at least one example box.")

    return sort_and_limit(
        output["masks"],
        output["scores"],
        top_k=top_k,
        boxes=output.get("boxes"),
    )


def run_interactive_prompt(
    model: Any,
    processor: Sam3Processor,
    image: Image.Image,
    points: np.ndarray | None,
    labels: np.ndarray | None,
    box: np.ndarray | None,
    mask_input: np.ndarray | None,
    multimask_output: bool,
    top_k: int,
    device: str,
    no_bf16: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAM1-style point/box/mask prompting through SAM 3.

    The interactive tracker created by the image builder intentionally has no
    standalone visual backbone. Compute image features with the main SAM 3
    image model, then invoke ``model.predict_inst``.
    """
    with autocast_context(device, no_bf16):
        inference_state = processor.set_image(image)
        masks, scores, low_res_logits = model.predict_inst(
            inference_state,
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=False,
        )

    masks_np, scores_np, _ = sort_and_limit(
        masks, scores, top_k=top_k, boxes=None
    )
    return masks_np, scores_np, np.asarray(low_res_logits)


def requested_modes(
    mode: str,
    text: str | None,
    points: np.ndarray | None,
    positive_boxes: list[np.ndarray],
    mask_input_path: str | None,
) -> list[str]:
    if mode != "all":
        return [mode]

    modes: list[str] = []
    if text:
        modes.append("text")
    if positive_boxes:
        modes.append("exemplar_box")
    if text and positive_boxes:
        modes.append("text_box")
    if points is not None:
        modes.append("point")
    if positive_boxes:
        modes.append("box")
    if points is not None and positive_boxes:
        modes.append("point_box")
    if mask_input_path:
        modes.append("mask")
    if points is not None and len(points) >= 2:
        modes.append("mask_refine")

    if not modes:
        raise ValueError(
            "mode=all needs at least one of --text, --point, --box, or --mask-input."
        )
    return modes


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory not found: {model_dir}")

    image_path = Path(args.image).expanduser().resolve()
    image = Image.open(image_path).convert("RGB")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    positive_boxes = parse_boxes(args.box)
    negative_boxes = parse_boxes(args.negative_box)
    points, point_labels = parse_points(args.point)
    ensure_inside_image(
        points,
        [*positive_boxes, *negative_boxes],
        image.width,
        image.height,
    )

    checkpoint = find_checkpoint(model_dir, args.checkpoint)
    print(f"[SAM 3] checkpoint: {checkpoint}")
    print(f"[SAM 3] image: {image_path} ({image.width}x{image.height})")

    # enable_inst_interactivity=True loads the tracker branch needed for point,
    # interactive box, and mask prompts. The same model also serves text and
    # concept-example-box prompting.
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
        confidence_threshold=args.threshold,
    )
    predictor = model.inst_interactive_predictor
    if predictor is None:
        raise RuntimeError("Interactive predictor was not created.")
    # Do not call predictor.set_image() directly. The image builder shares the
    # main model's backbone through model.predict_inst(inference_state, ...).

    modes = requested_modes(
        args.mode,
        args.text,
        points,
        positive_boxes,
        args.mask_input,
    )

    results: list[dict[str, Any]] = []
    first_box = positive_boxes[0] if positive_boxes else None

    for mode in modes:
        print(f"[RUN] {mode}")

        if mode == "text":
            if not args.text:
                raise ValueError("mode=text requires --text.")
            masks, scores, boxes = run_concept_prompt(
                processor,
                image,
                text=args.text,
                positive_boxes=[],
                negative_boxes=[],
                threshold=args.threshold,
                top_k=args.top_k,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, boxes,
                    None, None, None
                )
            )

        elif mode == "exemplar_box":
            if not positive_boxes:
                raise ValueError("mode=exemplar_box requires at least one --box.")
            masks, scores, boxes = run_concept_prompt(
                processor,
                image,
                text=None,
                positive_boxes=positive_boxes,
                negative_boxes=negative_boxes,
                threshold=args.threshold,
                top_k=args.top_k,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, boxes,
                    None, None, first_box
                )
            )

        elif mode == "text_box":
            if not args.text or not positive_boxes:
                raise ValueError("mode=text_box requires --text and at least one --box.")
            masks, scores, boxes = run_concept_prompt(
                processor,
                image,
                text=args.text,
                positive_boxes=positive_boxes,
                negative_boxes=negative_boxes,
                threshold=args.threshold,
                top_k=args.top_k,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, boxes,
                    None, None, first_box
                )
            )

        elif mode == "point":
            if points is None:
                raise ValueError("mode=point requires one or more --point arguments.")
            masks, scores, _ = run_interactive_prompt(
                model,
                processor,
                image,
                points=points,
                labels=point_labels,
                box=None,
                mask_input=None,
                multimask_output=(len(points) == 1),
                top_k=args.interactive_top_k,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, None,
                    points, point_labels, None
                )
            )

        elif mode == "box":
            if first_box is None:
                raise ValueError("mode=box requires --box.")
            masks, scores, _ = run_interactive_prompt(
                model,
                processor,
                image,
                points=None,
                labels=None,
                box=first_box,
                mask_input=None,
                multimask_output=False,
                top_k=1,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, None,
                    None, None, first_box
                )
            )

        elif mode == "point_box":
            if points is None or first_box is None:
                raise ValueError("mode=point_box requires --point and --box.")
            masks, scores, _ = run_interactive_prompt(
                model,
                processor,
                image,
                points=points,
                labels=point_labels,
                box=first_box,
                mask_input=None,
                multimask_output=False,
                top_k=1,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, None,
                    points, point_labels, first_box
                )
            )

        elif mode == "mask":
            if not args.mask_input:
                raise ValueError("mode=mask requires --mask-input.")
            mask_input = read_external_mask(args.mask_input)
            masks, scores, _ = run_interactive_prompt(
                model,
                processor,
                image,
                points=points,
                labels=point_labels,
                box=first_box,
                mask_input=mask_input,
                multimask_output=False,
                top_k=1,
                device=args.device,
                no_bf16=args.no_bf16,
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, None,
                    points, point_labels, first_box
                )
            )

        elif mode == "mask_refine":
            if points is None or len(points) < 2:
                raise ValueError(
                    "mode=mask_refine requires at least two --point arguments."
                )

            # First click: compute main-model image features, then decode candidates.
            with autocast_context(args.device, args.no_bf16):
                inference_state = processor.set_image(image)
                initial_masks, initial_scores, initial_low_res = model.predict_inst(
                    inference_state,
                    point_coords=points[:1],
                    point_labels=point_labels[:1],
                    multimask_output=True,
                    return_logits=False,
                )
                best_index = int(np.argmax(np.asarray(initial_scores)))
                previous_logits = np.asarray(initial_low_res)[best_index][None, :, :]

                # Second pass reuses the cached main-model backbone output.
                refined_masks, refined_scores, _ = model.predict_inst(
                    inference_state,
                    point_coords=points,
                    point_labels=point_labels,
                    mask_input=previous_logits,
                    multimask_output=False,
                    return_logits=False,
                )

            masks, scores, _ = sort_and_limit(
                refined_masks, refined_scores, top_k=1, boxes=None
            )
            results.append(
                save_visualization(
                    image, output_dir, mode, masks, scores, None,
                    points, point_labels, None
                )
            )

        else:
            raise AssertionError(f"Unhandled mode: {mode}")

    summary = {
        "checkpoint": str(checkpoint),
        "image": str(image_path),
        "image_size": [image.width, image.height],
        "text": args.text,
        "points_xy_label": args.point or [],
        "positive_boxes_xyxy": [box.tolist() for box in positive_boxes],
        "negative_boxes_xyxy": [box.tolist() for box in negative_boxes],
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[DONE] Results written to: {output_dir}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
