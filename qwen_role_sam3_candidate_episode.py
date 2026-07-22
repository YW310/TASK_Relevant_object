#!/usr/bin/env python3
"""
Episode-level Qwen role specification + SAM 3 text-prompt candidate generation.

Stage 1 runs Qwen3-VL once on a synchronized multiview frame to produce a
bbox-free role_spec.json. Stage 2 runs SAM 3 text prompts for target/reference/
interaction_part on every selected frame and camera, saving numbered candidate
masks, crops, masked crops, and visual summaries under:

    outputs/<episode>/frames/<frame_key>/<camera>/qwen_candidates/
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from qwen3_bbox_guided_sam3_demo import (
    autocast_context,
    find_checkpoint,
    load_font,
    normalize_masks,
    normalize_scores,
)
from qwen3vl_rlbench_episode_grounding import (
    DEFAULT_CAMERAS,
    Qwen3VLRLBenchGrounder,
    ROLE_PROMPT,
    atomic_json_dump,
    collect_camera_frames,
    discover_instruction,
    extract_json,
    parse_csv,
    role_identity_cues,
    role_display_text,
    select_frame_ids,
    resolve_role_frame,
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

ROLE_PREFIX = {"target": "T", "reference": "R", "interaction_part": "P"}
ROLE_ORDER = ("target", "reference", "interaction_part")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root used to create outputs/<episode>/... unless --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit episode output directory. Default: <output-root>/<episode name>.",
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument(
        "--cameras",
        type=parse_csv,
        default=DEFAULT_CAMERAS,
        help="Comma-separated camera names; directories must be NAME_rgb.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--frame-interval", "--stride", dest="frame_interval", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--role-frame", default="first")
    parser.add_argument("--role-spec-json", default=None, help="Reuse an existing role_spec.json.")
    parser.add_argument("--grounding-min-side", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--sam-model-dir", required=True)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k-per-role", type=int, default=8)
    parser.add_argument("--mask-alpha", type=int, default=105)
    parser.add_argument(
        "--save-frame-contact-sheet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one per-frame PNG that juxtaposes all camera numbered candidate overlays.",
    )
    parser.add_argument(
        "--visualization-cell-width",
        type=int,
        default=384,
        help="Cell width for per-frame visualization contact sheets.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def strip_bbox_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): strip_bbox_fields(child)
            for key, child in value.items()
            if "bbox" not in str(key).lower() and "box" not in str(key).lower()
        }
    if isinstance(value, list):
        return [strip_bbox_fields(item) for item in value]
    return value


def role_spec_document(instruction: str, raw_role_spec: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = strip_bbox_fields(raw_role_spec)
    return {
        "instruction": instruction,
        "target": cleaned.get("target"),
        "reference": cleaned.get("reference"),
        "interaction_part": cleaned.get("interaction_part"),
        "relation": cleaned.get("relation"),
    }


def load_or_identify_role_spec(
    args: argparse.Namespace,
    instruction: str,
    role_views: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    if args.role_spec_json:
        loaded = json.loads(Path(args.role_spec_json).read_text(encoding="utf-8"))
        doc = role_spec_document(instruction, loaded.get("role_spec", loaded))
    else:
        grounder = Qwen3VLRLBenchGrounder(
            model_path=args.model_path,
            grounding_min_side=args.grounding_min_side,
            max_retries=args.max_retries,
        )
        # Reuses qwen3vl_rlbench_episode_grounding.py's ROLE_PROMPT,
        # extract_json(), model loading, and multiview image message logic.
        raw_role_spec, raw_text = grounder.identify_roles(instruction, role_views)
        doc = role_spec_document(instruction, raw_role_spec)
        atomic_json_dump({"raw_text": raw_text, "prompt": ROLE_PROMPT}, output_dir / "raw_role_spec_output.json")
    atomic_json_dump(doc, output_dir / "role_spec.json")
    return doc


def text_prompt_for_role(role_spec: Mapping[str, Any], role: str) -> str | None:
    spec = role_spec.get(role)
    if not isinstance(spec, Mapping):
        return None
    text = role_display_text(role_spec, role)
    cues = role_identity_cues(role_spec, role)
    if cues:
        return f"{text}. Visual cues: {'; '.join(cues)}"
    return text


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_crop_sets(image: Image.Image, mask: np.ndarray, bbox: Sequence[int], stem: str, out_dir: Path) -> tuple[str, str, str]:
    masks_dir = out_dir / "masks"
    crops_dir = out_dir / "crops"
    masked_dir = out_dir / "masked_crops"
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    mask_path = masks_dir / f"{stem}.png"
    crop_path = crops_dir / f"{stem}.png"
    masked_path = masked_dir / f"{stem}.png"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    crop = image.crop(tuple(bbox))
    crop.save(crop_path)
    mask_crop = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").crop(tuple(bbox))
    rgba = crop.convert("RGBA")
    rgba.putalpha(mask_crop)
    rgba.save(masked_path)
    return str(mask_path), str(crop_path), str(masked_path)


def run_text_prompt(processor: Any, image: Image.Image, prompt: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    processor.set_confidence_threshold(args.threshold)
    with autocast_context(args.device, args.no_bf16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(prompt=prompt, state=state)
    masks = normalize_masks(output["masks"])
    scores = normalize_scores(output.get("scores"), len(masks))
    order = np.argsort(-scores)[: args.top_k_per_role]
    boxes = output.get("boxes")
    boxes_np = None
    if boxes is not None:
        raw_boxes = np.asarray(boxes).reshape((-1, 4))
        if len(raw_boxes) == len(masks):
            boxes_np = raw_boxes[order]
    return masks[order], scores[order], boxes_np


def save_visuals(
    image: Image.Image,
    candidates: Sequence[Mapping[str, Any]],
    out_dir: Path,
    mask_alpha: int,
) -> None:
    rendered = image.convert("RGBA")
    overlay_pixels = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    for idx, cand in enumerate(candidates):
        mask = np.asarray(Image.open(cand["mask_path"]).convert("L")) > 127
        color = color_for_index(idx)
        layer = np.zeros_like(overlay_pixels)
        layer[mask, :3] = color
        layer[mask, 3] = mask_alpha
        overlay_pixels = np.maximum(overlay_pixels, layer)
    rendered = Image.alpha_composite(rendered, Image.fromarray(overlay_pixels, mode="RGBA"))
    draw = ImageDraw.Draw(rendered)
    font = load_font(max(10, min(image.size) // 18))
    for idx, cand in enumerate(candidates):
        color = color_for_index(idx)
        box = cand.get("mask_bbox_xyxy")
        if box:
            draw.rectangle(tuple(box), outline=(*color, 255), width=max(1, min(image.size) // 96))
            draw.text((box[0] + 2, box[1] + 2), cand["id"], fill=(*color, 255), font=font)
    rendered.convert("RGB").save(out_dir / "numbered_candidates.png")

    thumbs = [Image.open(c["masked_crop_path"]).convert("RGBA") for c in candidates]
    cell = 128
    if not thumbs:
        grid = Image.new("RGB", (cell * 2, cell), (25, 25, 25))
        ImageDraw.Draw(grid).text((10, 10), "No SAM3 candidates", fill=(235, 235, 235))
        grid.save(out_dir / "candidate_grid.png")
        return
    cols = max(1, min(4, len(thumbs)))
    rows = max(1, (len(thumbs) + cols - 1) // cols)
    grid = Image.new("RGB", (cols * cell, rows * (cell + 20)), (25, 25, 25))
    gd = ImageDraw.Draw(grid)
    for i, (thumb, cand) in enumerate(zip(thumbs, candidates)):
        scale = min(cell / max(1, thumb.width), cell / max(1, thumb.height))
        resized = thumb.resize((max(1, round(thumb.width * scale)), max(1, round(thumb.height * scale))), Image.Resampling.BICUBIC)
        x = (i % cols) * cell + (cell - resized.width) // 2
        y = (i // cols) * (cell + 20)
        grid.paste(resized.convert("RGB"), (x, y), resized.getchannel("A"))
        gd.text(((i % cols) * cell + 4, y + cell + 2), f"{cand['id']} {cand['score']:.3f}", fill=(235, 235, 235), font=font)
    grid.save(out_dir / "candidate_grid.png")


def process_camera(
    processor: Any,
    image_path: Path,
    role_doc: Mapping[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    resume_files = (
        out_dir / "candidates.json",
        out_dir / "numbered_candidates.png",
        out_dir / "candidate_grid.png",
    )
    if args.resume and all(path.is_file() for path in resume_files):
        return json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    candidates: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        prompt = text_prompt_for_role(role_doc, role)
        if not prompt:
            continue
        masks, scores, boxes = run_text_prompt(processor, image, prompt, args)
        prefix = ROLE_PREFIX[role]
        for index, (mask, score) in enumerate(zip(masks, scores), start=1):
            bbox = mask_bbox(mask)
            if bbox is None:
                continue
            cid = f"{prefix}{index}"
            mask_path, crop_path, masked_path = save_crop_sets(image, mask, bbox, cid, out_dir)
            item: dict[str, Any] = {
                "id": cid,
                "role": role,
                "text_prompt": prompt,
                "score": float(score),
                "mask_bbox_xyxy": bbox,
                "mask_path": mask_path,
                "crop_path": crop_path,
                "masked_crop_path": masked_path,
            }
            if boxes is not None and index - 1 < len(boxes):
                item["sam_box_xyxy"] = [float(v) for v in boxes[index - 1]]
            candidates.append(item)
    result = {"image_path": str(image_path), "candidates": candidates}
    atomic_json_dump(result, out_dir / "candidates.json")
    save_visuals(image, candidates, out_dir, args.mask_alpha)
    return result


def load_sam3_components() -> tuple[Any, Any]:
    processor_module = importlib.import_module("sam3.model.sam3_image_processor")
    builder_module = importlib.import_module("sam3.model_builder")
    return processor_module.Sam3Processor, builder_module.build_sam3_image_model


def fit_to_cell(image: Image.Image, cell_width: int, cell_height: int) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), (cell_width, cell_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (cell_width, cell_height), (18, 18, 18))
    canvas.paste(fitted, ((cell_width - fitted.width) // 2, (cell_height - fitted.height) // 2))
    return canvas


def save_frame_contact_sheet(
    camera_overlays: Mapping[str, Path],
    frame_key: str,
    output_path: Path,
    cell_width: int,
) -> None:
    if not camera_overlays:
        return
    loaded = [(camera, Image.open(path).convert("RGB")) for camera, path in camera_overlays.items()]
    aspect = max(image.height / image.width for _, image in loaded)
    cell_height = max(128, round(cell_width * aspect))
    header_height = 36
    canvas = Image.new("RGB", (cell_width * len(loaded), header_height + cell_height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(11, cell_width // 30))
    draw.text((10, 8), f"SAM3 text candidates · {frame_key}", fill=(235, 235, 235), font=font)
    for index, (camera, image) in enumerate(loaded):
        x = index * cell_width
        canvas.paste(fit_to_cell(image, cell_width, cell_height), (x, header_height))
        draw.rectangle((x, header_height, x + cell_width - 1, header_height + cell_height - 1), outline=(80, 80, 80))
        draw.text((x + 8, header_height + 8), camera, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = build_parser().parse_args()

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(args.output_root).expanduser().resolve() / episode_dir.name
    instruction, instruction_source = discover_instruction(episode_dir, args.instruction, args.instruction_file, args.instruction_index)
    all_frame_ids, camera_frames = collect_camera_frames(episode_dir, args.cameras)
    selected_frame_ids = select_frame_ids(all_frame_ids, args.start, args.end, args.frame_interval, args.max_frames)
    role_frame_id = resolve_role_frame(args.role_frame, selected_frame_ids)
    role_views = {camera: camera_frames[camera][role_frame_id] for camera in args.cameras}

    print(f"Episode: {episode_dir}")
    print(f"Output: {output_dir}")
    print(f"Instruction ({instruction_source}): {instruction}")
    print(f"Frames: {len(selected_frame_ids)} selected from {len(all_frame_ids)}")
    print(f"Cameras: {', '.join(args.cameras)}")

    if args.dry_run:
        role_doc = None
        if args.role_spec_json:
            loaded = json.loads(Path(args.role_spec_json).read_text(encoding="utf-8"))
            role_doc = role_spec_document(instruction, loaded.get("role_spec", loaded))
        plan = {
            "dry_run": True,
            "episode_dir": str(episode_dir),
            "output_dir": str(output_dir),
            "instruction": instruction,
            "camera_names": list(args.cameras),
            "selected_frame_ids": list(selected_frame_ids),
            "role_frame_id": role_frame_id,
            "role_spec": role_doc,
            "would_load_qwen": role_doc is None,
            "would_load_sam3": True,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    role_doc = load_or_identify_role_spec(args, instruction, role_views, output_dir)

    Sam3Processor, build_sam3_image_model = load_sam3_components()
    model_dir = Path(args.sam_model_dir).expanduser().resolve()
    checkpoint = find_checkpoint(model_dir, args.sam_checkpoint)
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
        compile=args.compile,
    )
    processor = Sam3Processor(model=model, device=args.device, confidence_threshold=args.threshold)

    frames_summary: list[dict[str, Any]] = []
    for frame_index, frame_id in enumerate(selected_frame_ids):
        frame_key = f"{frame_index:06d}_{frame_id}"
        frame_entry = {"frame_index": frame_index, "frame_id": frame_id, "views": {}}
        camera_overlays: dict[str, Path] = {}
        for camera in args.cameras:
            out = output_dir / "frames" / frame_key / camera / "qwen_candidates"
            result = process_camera(processor, camera_frames[camera][frame_id], role_doc, out, args)
            camera_overlays[camera] = out / "numbered_candidates.png"
            frame_entry["views"][camera] = {
                "output_dir": str(out),
                "candidates_json": str(out / "candidates.json"),
                "numbered_candidates": str(out / "numbered_candidates.png"),
                "candidate_grid": str(out / "candidate_grid.png"),
                "num_candidates": len(result["candidates"]),
            }
        if args.save_frame_contact_sheet:
            contact_sheet = output_dir / "frames" / frame_key / "qwen_candidates_contact_sheet.png"
            save_frame_contact_sheet(
                camera_overlays,
                frame_key,
                contact_sheet,
                args.visualization_cell_width,
            )
            frame_entry["contact_sheet"] = str(contact_sheet)
        frames_summary.append(frame_entry)
    summary = {
        "episode_dir": str(episode_dir),
        "instruction": instruction,
        "role_spec": role_doc,
        "camera_names": list(args.cameras),
        "frames": frames_summary,
    }
    atomic_json_dump(summary, output_dir / "episode_candidates.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
