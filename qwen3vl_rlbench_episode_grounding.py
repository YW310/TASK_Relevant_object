#!/usr/bin/env python3
"""
Qwen3-VL target/reference grounding for RLBench, revised version.

Key changes:
1. Preserve the user's proven Hugging Face inference path:
   processor.apply_chat_template(..., tokenize=True, return_dict=True).
2. Do not let the role-identification stage invent view-specific locations.
3. Localize one role at a time in one image.
4. Resize low-resolution RLBench images before inference while keeping
   normalized [0, 1000] coordinates.
5. Save annotated images for immediate visual inspection.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
try:
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
    )
except ImportError:
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None
    Qwen3VLMoeForConditionalGeneration = None


# Visualization-only settings. They never affect prompts or model outputs.
ROLE_VISUAL_STYLE: dict[str, dict[str, Any]] = {
    "target": {
        "color": (40, 220, 40),
        "label": "Target",
    },
    "reference": {
        "color": (255, 170, 30),
        "label": "Reference",
    },
    "interaction_part": {
        "color": (80, 170, 255),
        "label": "Interaction",
    },
}

LABEL_BACKGROUND_RGBA = (0, 0, 0, 90)  # 90 / 255 ~= 0.35 opacity.


ROLE_PROMPT = """You are resolving semantic object roles for an RLBench robot task.

Instruction:
{instruction}

The supplied images are synchronized views of the same scene.

Definitions:
- target: the specific object whose pose, location, configuration, or state must change.
- reference: the specific object, support, container, slot, surface, or region that defines the goal relation. It may be null.
- interaction_part: a local part directly contacted by the robot, such as a handle, button, lid edge, or opening. It may be null.

Rules:
1. Select specific instances, not only categories.
2. Use color, size, relative position, and instruction relations to distinguish identical categories.
3. Do not select the robot arm or gripper.
4. Do not invent unsupported view-specific statements such as "on the floor",
   "behind the robot", or "under another object".
5. Describe only stable identity cues needed to find the same instance in another view.
6. For every non-null role, include sam_prompts: 3 to 5 short text prompts that SAM3 can use to segment the object or part. Prefer concise noun phrases, aliases, and visually grounded variants; do not include long instructions.
7. For every non-null role, include negative_cues: objects or regions that must not be selected.
8. Do not output bounding boxes in this stage.
9. Return one valid JSON object only.

Schema:
{{
  "task_type": "short task type",
  "relation": "short target-reference relation",
  "target": {{
    "name": "category or object name",
    "sam_prompts": ["short SAM3 prompt", "alias", "visual variant"],
    "identity_cues": ["stable visible cue", "instruction-based cue"],
    "negative_cues": ["nearby distractor", "robot gripper"]
  }},
  "reference": null or {{
    "name": "category or object name",
    "sam_prompts": ["short SAM3 prompt", "alias", "visual variant"],
    "identity_cues": ["stable visible cue", "instruction-based cue"],
    "negative_cues": ["nearby distractor", "robot gripper"]
  }},
  "interaction_part": null or {{
    "name": "part name",
    "sam_prompts": ["short SAM3 prompt", "alias", "visual variant"],
    "identity_cues": ["stable visible cue"],
    "negative_cues": ["nearby distractor", "robot gripper"]
  }},
  "uncertain": false,
  "uncertain_reason": null
}}
"""


LOCALIZE_ONE_ROLE_PROMPT = """Perform visual grounding in this single RLBench image.

Instruction:
{instruction}

Role:
{role_name}

Object specification:
{role_spec}

Requirements:
1. Locate the exact physical instance corresponding to this role.
2. The object may be small or partially occluded.
3. If any recognizable part of the object is visible, set visible=true and
   return the best tight bounding box around its complete visible extent.
4. Set visible=false only when no part of the specified object appears.
5. Do not select the robot arm or gripper.
6. bbox_2d format is [left, top, right, bottom].
7. Coordinates must be integers normalized to [0, 1000].
8. Return one valid JSON object only.

Schema:
{{
  "visible": true,
  "bbox_2d": [x1, y1, x2, y2],
  "evidence": "brief visual evidence",
  "uncertain": false,
  "uncertain_reason": null
}}
"""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found: {text[:300]!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("JSON top level is not an object.")
                return value

    raise ValueError(f"Unbalanced JSON: {text[:300]!r}")


def resize_for_grounding(image: Image.Image, min_side: int) -> Image.Image:
    """Upscale small RLBench frames; normalized boxes remain size-independent."""
    image = image.convert("RGB")
    width, height = image.size
    if min(width, height) >= min_side:
        return image

    scale = min_side / min(width, height)
    new_width = max(32, round(width * scale / 32) * 32)
    new_height = max(32, round(height * scale / 32) * 32)
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def validate_bbox(raw_box: Any) -> list[int] | None:
    if raw_box is None:
        return None
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError(f"Invalid bbox_2d: {raw_box!r}")

    box = [max(0, min(1000, int(round(float(v))))) for v in raw_box]
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox order: {box}")
    return box


def normalized_to_pixels(
    box: list[int] | None,
    width: int,
    height: int,
) -> list[int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    result = [
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    ]
    result[0] = max(0, min(result[0], width - 1))
    result[1] = max(0, min(result[1], height - 1))
    result[2] = max(result[0] + 1, min(result[2], width))
    result[3] = max(result[1] + 1, min(result[3], height))
    return result


class Qwen3VLRLBenchGrounder:
    def __init__(
        self,
        model_path: str,
        grounding_min_side: int = 512,
        max_retries: int = 1,
    ) -> None:
        if (
            AutoProcessor is None
            or Qwen3VLForConditionalGeneration is None
            or Qwen3VLMoeForConditionalGeneration is None
        ):
            raise ImportError(
                "transformers with Qwen3-VL support is required. "
                "Install the same transformers environment used by the verified single-frame script."
            )

        model_cls = (
            Qwen3VLMoeForConditionalGeneration
            if ("A3B" in model_path or "MoE" in model_path)
            else Qwen3VLForConditionalGeneration
        )

        self.model = model_cls.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.grounding_min_side = grounding_min_side
        self.max_retries = max_retries

    @property
    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> str:
        # This intentionally follows the inference path already verified by the user.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.input_device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def generate_json(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
    ) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        raw_text = ""

        for _ in range(self.max_retries + 1):
            raw_text = self.generate_text(messages, max_new_tokens)
            try:
                return extract_json(raw_text), raw_text
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                messages = [
                    *messages,
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": raw_text}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Return the same answer as one valid JSON object only.",
                            }
                        ],
                    },
                ]

        raise RuntimeError(f"Failed to parse JSON: {last_error}; output={raw_text!r}")

    def load_view(self, path: str | Path) -> tuple[Image.Image, Image.Image]:
        original = Image.open(path).convert("RGB")
        model_image = resize_for_grounding(original, self.grounding_min_side)
        return original, model_image

    def identify_roles(
        self,
        instruction: str,
        views: Mapping[str, str | Path],
    ) -> tuple[dict[str, Any], str]:
        content: list[dict[str, Any]] = []

        for view_name, path in views.items():
            _, model_image = self.load_view(path)
            content.extend(
                [
                    {"type": "text", "text": f"VIEW_{view_name.upper()}"},
                    {"type": "image", "image": model_image},
                ]
            )

        content.append(
            {
                "type": "text",
                "text": ROLE_PROMPT.format(instruction=instruction),
            }
        )
        return self.generate_json(
            [{"role": "user", "content": content}],
            max_new_tokens=320,
        )

    def localize_role(
        self,
        instruction: str,
        role_name: str,
        role_spec: dict[str, Any],
        image_path: str | Path,
    ) -> tuple[dict[str, Any], str]:
        original, model_image = self.load_view(image_path)

        prompt = LOCALIZE_ONE_ROLE_PROMPT.format(
            instruction=instruction,
            role_name=role_name,
            role_spec=json.dumps(role_spec, ensure_ascii=False),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": model_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        result, raw = self.generate_json(messages, max_new_tokens=192)
        visible = bool(result.get("visible", False))
        bbox = validate_bbox(result.get("bbox_2d"))

        if visible and bbox is None:
            raise ValueError(f"{role_name}: visible=true but bbox_2d is null")
        if not visible:
            bbox = None

        result["visible"] = visible
        result["bbox_2d"] = bbox
        result["bbox_xyxy"] = normalized_to_pixels(
            bbox,
            original.width,
            original.height,
        )
        result.setdefault("evidence", None)
        result["uncertain"] = bool(result.get("uncertain", False))
        result.setdefault("uncertain_reason", None)
        return result, raw

    def ground(
        self,
        instruction: str,
        views: Mapping[str, str | Path],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        role_spec, raw_role = self.identify_roles(instruction, views)

        results: dict[str, Any] = {}
        raw_localization: dict[str, Any] = {}

        for view_name, image_path in views.items():
            per_view: dict[str, Any] = {}
            per_view_raw: dict[str, str] = {}

            for role_name in ("target", "reference", "interaction_part"):
                spec = role_spec.get(role_name)
                if spec is None:
                    per_view[role_name] = {
                        "visible": False,
                        "bbox_2d": None,
                        "bbox_xyxy": None,
                        "evidence": None,
                        "uncertain": False,
                        "uncertain_reason": None,
                    }
                    continue

                localized, raw = self.localize_role(
                    instruction=instruction,
                    role_name=role_name,
                    role_spec=spec,
                    image_path=image_path,
                )
                per_view[role_name] = localized
                per_view_raw[role_name] = raw

            results[view_name] = per_view
            raw_localization[view_name] = per_view_raw
            self.save_annotation(image_path, per_view, output_dir / f"{view_name}_boxes.png")

        final = {
            "instruction": instruction,
            "role_spec": role_spec,
            "views": results,
            "raw_outputs": {
                "role_identification": raw_role,
                "localization": raw_localization,
            },
        }

        with (output_dir / "role_grounding.json").open("w", encoding="utf-8") as file:
            json.dump(final, file, ensure_ascii=False, indent=2)
        return final

    @staticmethod
    def save_annotation(
        image_path: str | Path,
        view_result: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        save_detailed_annotation(
            image_path=image_path,
            view_result=view_result,
            output_path=output_path,
            roles=("target", "reference", "interaction_part"),
        )

    def ground_rlbench_observation(
        self,
        instruction: str,
        observation: Any,
        output_dir: str | Path,
        camera_names: tuple[str, ...] = (
            "front",
            "left_shoulder",
            "right_shoulder",
        ),
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        views: dict[str, Path] = {}
        for camera_name in camera_names:
            rgb = getattr(observation, f"{camera_name}_rgb", None)
            if rgb is None:
                continue

            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8:
                if rgb.max() <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            path = image_dir / f"{camera_name}.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            views[camera_name] = path

        if not views:
            raise ValueError("No RGB views found in observation.")

        return self.ground(instruction, views, output_dir)



IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_CAMERAS = ("front", "left_shoulder", "right_shoulder")
DEFAULT_ROLES = ("target", "reference")


def natural_sort_key(value: str | Path) -> list[Any]:
    """Sort frame names numerically when possible: 2.png before 10.png."""
    text = Path(value).stem if isinstance(value, Path) else str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def atomic_json_dump(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty list.")
    return values


def normalize_instruction_candidates(value: Any) -> list[str]:
    """Extract instruction strings from common RLBench pickle/JSON layouts."""
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, bytes):
        try:
            return normalize_instruction_candidates(value.decode("utf-8"))
        except UnicodeDecodeError:
            return []

    if isinstance(value, Mapping):
        preferred_keys = (
            "instruction",
            "instructions",
            "description",
            "descriptions",
            "variation_description",
            "variation_descriptions",
            "language",
            "task_description",
        )
        collected: list[str] = []
        for key in preferred_keys:
            if key in value:
                collected.extend(normalize_instruction_candidates(value[key]))
        if collected:
            return collected

        for nested in value.values():
            collected.extend(normalize_instruction_candidates(nested))
        return collected

    if isinstance(value, (list, tuple, set)):
        collected: list[str] = []
        for item in value:
            collected.extend(normalize_instruction_candidates(item))
        return collected

    return []


def load_instruction_file(path: str | Path) -> list[str]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as file:
            value = pickle.load(file)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    else:
        text = path.read_text(encoding="utf-8")
        # A plain-text file can contain one instruction per line.
        value = [line.strip() for line in text.splitlines() if line.strip()]
        if not value and text.strip():
            value = text.strip()

    candidates = normalize_instruction_candidates(value)
    # Stable de-duplication.
    return list(dict.fromkeys(candidate for candidate in candidates if candidate.strip()))


def discover_instruction(
    episode_dir: str | Path,
    explicit_instruction: str | None,
    instruction_file: str | Path | None,
    instruction_index: int,
) -> tuple[str, str]:
    if explicit_instruction:
        return explicit_instruction.strip(), "command line"

    candidate_files: list[Path] = []
    if instruction_file is not None:
        candidate_files.append(Path(instruction_file))
    else:
        episode_dir = Path(episode_dir)
        names = (
            "variation_descriptions.pkl",
            "variation_description.pkl",
            "descriptions.pkl",
            "description.pkl",
            "instructions.pkl",
            "instruction.pkl",
            "variation_descriptions.json",
            "descriptions.json",
            "instruction.json",
            "instruction.txt",
            "descriptions.txt",
        )
        # Prefer metadata stored directly inside the episode.
        for name in names:
            path = episode_dir / name
            if path.is_file():
                candidate_files.append(path)

        # Some converted datasets put the description one directory above episodes.
        if not candidate_files:
            for parent in list(episode_dir.parents)[:3]:
                for name in names:
                    path = parent / name
                    if path.is_file():
                        candidate_files.append(path)

    errors: list[str] = []
    for path in candidate_files:
        try:
            instructions = load_instruction_file(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue

        if not instructions:
            errors.append(f"{path}: no instruction string found")
            continue

        index = instruction_index
        if index < 0:
            index += len(instructions)
        if not 0 <= index < len(instructions):
            raise IndexError(
                f"--instruction-index={instruction_index} is invalid for {path}; "
                f"the file contains {len(instructions)} instruction(s)."
            )
        return instructions[index], str(path)

    details = "\n".join(errors)
    raise FileNotFoundError(
        "Could not determine the RLBench instruction. Supply --instruction or "
        "--instruction-file. Checked common variation_descriptions/instruction files."
        + (f"\nDetails:\n{details}" if details else "")
    )


def collect_camera_frames(
    episode_dir: str | Path,
    camera_names: Sequence[str],
) -> tuple[list[str], dict[str, dict[str, Path]]]:
    episode_dir = Path(episode_dir)
    camera_frames: dict[str, dict[str, Path]] = {}

    for camera_name in camera_names:
        directory = episode_dir / f"{camera_name}_rgb"
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing camera directory: {directory}")

        frame_map: dict[str, Path] = {}
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                frame_map[path.stem] = path

        if not frame_map:
            raise FileNotFoundError(f"No RGB frames found in {directory}")
        camera_frames[camera_name] = frame_map

    common_ids = set.intersection(*(set(frames) for frames in camera_frames.values()))
    if not common_ids:
        counts = {name: len(frames) for name, frames in camera_frames.items()}
        raise ValueError(
            "The selected camera folders do not share any frame IDs. "
            f"Per-camera counts: {counts}"
        )

    ordered_ids = sorted(common_ids, key=natural_sort_key)
    return ordered_ids, camera_frames


def select_frame_ids(
    all_frame_ids: Sequence[str],
    start: int,
    end: int | None,
    stride: int,
    max_frames: int | None,
) -> list[str]:
    if stride <= 0:
        raise ValueError("--stride must be positive.")
    if start < 0:
        start += len(all_frame_ids)
    if start < 0 or start >= len(all_frame_ids):
        raise IndexError(f"--start={start} is outside 0..{len(all_frame_ids) - 1}")

    stop = len(all_frame_ids) if end is None else end
    if stop < 0:
        stop += len(all_frame_ids)
    stop = min(stop, len(all_frame_ids))
    if stop <= start:
        raise ValueError(
            f"Empty frame interval: start={start}, end={end}, resolved stop={stop}."
        )

    selected = list(all_frame_ids[start:stop:stride])
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive.")
        selected = selected[:max_frames]

    if not selected:
        raise ValueError("Frame selection is empty.")
    return selected


def resolve_role_frame(role_frame: str, selected_ids: Sequence[str]) -> str:
    if role_frame.lower() == "first":
        return selected_ids[0]
    if role_frame in selected_ids:
        return role_frame

    try:
        index = int(role_frame)
    except ValueError as exc:
        raise ValueError(
            f"--role-frame must be 'first', a selected frame ID, or a selected-frame index; "
            f"got {role_frame!r}."
        ) from exc

    if index < 0:
        index += len(selected_ids)
    if not 0 <= index < len(selected_ids):
        raise IndexError(
            f"--role-frame index {role_frame} is outside 0..{len(selected_ids) - 1}"
        )
    return selected_ids[index]


def empty_role_result(error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "visible": False,
        "bbox_2d": None,
        "bbox_xyxy": None,
        "evidence": None,
        "uncertain": error is not None,
        "uncertain_reason": error,
    }
    if error is not None:
        result["error"] = error
    return result


def short_role_label(role_spec: Mapping[str, Any], role_name: str) -> str:
    value = role_spec.get(role_name)
    if not isinstance(value, Mapping):
        return "null"
    return str(value.get("name") or role_name)


def role_identity_cues(role_spec: Mapping[str, Any], role_name: str) -> list[str]:
    value = role_spec.get(role_name)
    if not isinstance(value, Mapping):
        return []
    cues = value.get("identity_cues", [])
    if isinstance(cues, str):
        return [cues]
    if not isinstance(cues, Sequence):
        return []
    return [str(item) for item in cues if str(item).strip()]


def role_display_text(role_spec: Mapping[str, Any], role_name: str) -> str:
    """Human-readable role identity for HTML and composed visualizations."""
    name = short_role_label(role_spec, role_name)
    cues = role_identity_cues(role_spec, role_name)
    if name == "null":
        return name
    if not cues:
        return name
    return f"{name} — {'; '.join(cues)}"


def load_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_with_background(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: Any,
    fill: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int, int] = LABEL_BACKGROUND_RGBA,
    padding: int = 2,
) -> None:
    """Draw a compact text label on an RGBA overlay with transparent background."""
    if image.mode != "RGB":
        raise ValueError("draw_text_with_background expects an RGB image.")

    x, y = xy
    probe = ImageDraw.Draw(image)
    try:
        left, top, right, bottom = probe.textbbox((x, y), text, font=font)
    except AttributeError:
        width, height = probe.textsize(text, font=font)
        left, top, right, bottom = x, y, x + width, y + height

    # Clamp the label background to the image bounds.
    shift_x = max(0, -(left - padding))
    shift_y = max(0, -(top - padding))
    shift_x -= max(0, right + padding + shift_x - image.width)
    shift_y -= max(0, bottom + padding + shift_y - image.height)
    x += shift_x
    y += shift_y
    left += shift_x
    right += shift_x
    top += shift_y
    bottom += shift_y

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (left - padding, top - padding, right + padding, bottom + padding),
        fill=background,
    )
    overlay_draw.text((x, y), text, font=font, fill=(*fill, 255))
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.paste(composed)


def save_detailed_annotation(
    image_path: str | Path,
    view_result: Mapping[str, Any],
    output_path: str | Path,
    roles: Sequence[str],
) -> None:
    """Visualization only; model outputs are not altered."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(1, min(image.size) // 128)
    # RLBench images are often 128-256 px. Keep labels intentionally compact.
    font = load_font(max(9, round(min(image.size) * 0.035)))

    for role in roles:
        result = view_result.get(role, {})
        box = result.get("bbox_xyxy") if isinstance(result, Mapping) else None
        if box is None:
            continue

        style = ROLE_VISUAL_STYLE.get(
            role,
            {"color": (255, 220, 40), "label": role.replace("_", " ").title()},
        )
        color = tuple(style["color"])
        draw.rectangle(box, outline=color, width=line_width)
        suffix = " ?" if result.get("uncertain", False) else ""
        label = f"{style['label']}{suffix}"

        # Prefer a label just above the box. Fall back to the box interior when
        # the object touches the top border.
        try:
            _, _, _, text_bottom = draw.textbbox((0, 0), label, font=font)
            text_height = text_bottom
        except AttributeError:
            _, text_height = draw.textsize(label, font=font)
        label_y = box[1] - text_height - 5
        if label_y < 0:
            label_y = box[1] + line_width + 1

        draw_text_with_background(
            image,
            (box[0] + line_width + 1, label_y),
            label,
            font=font,
            fill=(255, 255, 255),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def fit_image_to_cell(image: Image.Image, cell_width: int, cell_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(cell_width / image.width, cell_height / image.height)
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (cell_width, cell_height), (18, 18, 18))
    x = (cell_width - resized.width) // 2
    y = (cell_height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def compose_multiview_frame(
    image_paths: Mapping[str, Path],
    camera_names: Sequence[str],
    frame_id: str,
    frame_index: int,
    instruction: str,
    role_spec: Mapping[str, Any],
    cell_width: int,
    legend: str | None = None,
    paper_vis: bool = False,
) -> Image.Image:
    """Compose synchronized views into one image for saving or video output."""
    images = [Image.open(image_paths[name]).convert("RGB") for name in camera_names]
    aspect = max(image.height / image.width for image in images)
    cell_height = max(128, round(cell_width * aspect))
    header_height = 0 if paper_vis else max(92, cell_width // 4)
    canvas = Image.new(
        "RGB",
        (cell_width * len(camera_names), header_height + cell_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(13, cell_width // 26))
    small_font = load_font(max(10, cell_width // 34))

    if not paper_vis:
        draw.text(
            (12, 8),
            f"Frame {frame_index} · source id {frame_id}",
            font=font,
            fill=(255, 255, 255),
        )

        wrapped = textwrap.wrap(instruction, width=max(45, 42 * len(camera_names)))
        line_height = small_font.size + 3 if hasattr(small_font, "size") else 15
        for line_index, line in enumerate(wrapped[:2]):
            draw.text(
                (12, 32 + line_index * line_height),
                line,
                font=small_font,
                fill=(220, 220, 220),
            )

        role_x = 12
        role_y = header_height - 27
        for role_name in ("target", "reference", "interaction_part"):
            if not isinstance(role_spec.get(role_name), Mapping):
                continue
            style = ROLE_VISUAL_STYLE[role_name]
            role_text = f"{style['label']}: {short_role_label(role_spec, role_name)}"
            try:
                _, _, text_right, _ = draw.textbbox((role_x, role_y), role_text, font=small_font)
                text_width = text_right - role_x
            except AttributeError:
                text_width, _ = draw.textsize(role_text, font=small_font)
            draw_text_with_background(
                canvas,
                (role_x, role_y),
                role_text,
                font=small_font,
                fill=tuple(style["color"]),
            )
            role_x += text_width + 14

        if legend and role_x < canvas.width - 160:
            draw.text(
                (role_x, role_y),
                legend,
                font=small_font,
                fill=(180, 180, 180),
            )

    for camera_index, camera_name in enumerate(camera_names):
        cell = fit_image_to_cell(images[camera_index], cell_width, cell_height)
        x = camera_index * cell_width
        canvas.paste(cell, (x, header_height))
        draw_text_with_background(
            canvas,
            (x + 8, header_height + 8),
            camera_name.replace("_", " ").title(),
            font=small_font,
        )
    return canvas


def write_mp4(
    frame_factory: Callable[[], Iterable[Image.Image]],
    output_path: str | Path,
    fps: float,
) -> tuple[bool, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer imageio because it handles RGB directly.
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        count = 0
        try:
            for frame in frame_factory():
                writer.append_data(np.asarray(frame.convert("RGB")))
                count += 1
        finally:
            writer.close()
        if count == 0:
            output_path.unlink(missing_ok=True)
            return False, "no frames"
        return True, "imageio"
    except Exception as imageio_error:
        # Fall back to OpenCV, which is commonly present in RLBench environments.
        try:
            import cv2

            iterator = iter(frame_factory())
            first = next(iterator, None)
            if first is None:
                return False, "no frames"

            first_array = np.asarray(first.convert("RGB"))
            height, width = first_array.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter could not open the output file.")

            try:
                writer.write(cv2.cvtColor(first_array, cv2.COLOR_RGB2BGR))
                for frame in iterator:
                    array = np.asarray(frame.convert("RGB"))
                    if array.shape[:2] != (height, width):
                        array = cv2.resize(array, (width, height))
                    writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return True, "opencv"
        except Exception as cv2_error:
            output_path.unlink(missing_ok=True)
            return (
                False,
                f"imageio failed: {imageio_error}; OpenCV failed: {cv2_error}",
            )


def create_episode_html(
    output_path: str | Path,
    instruction: str,
    role_spec: Mapping[str, Any],
    frame_records: Sequence[Mapping[str, Any]],
    camera_names: Sequence[str],
) -> None:
    """Create a local synchronized multiview viewer with role/debug information."""
    output_path = Path(output_path)
    frames_for_html = []
    for record in frame_records:
        frame_id = str(record["frame_id"])
        frame_index = int(record["frame_index"])
        images = {
            camera: f"annotations/{camera}/{frame_index:06d}_{frame_id}.png"
            for camera in camera_names
        }
        frames_for_html.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "images": images,
                "views": record.get("views", {}),
            }
        )

    data_json = json.dumps(frames_for_html, ensure_ascii=False).replace("</", "<\\/")
    role_spec_js = json.dumps(role_spec, ensure_ascii=False).replace("</", "<\\/")
    role_json = html.escape(json.dumps(role_spec, ensure_ascii=False, indent=2))
    instruction_html = html.escape(instruction)
    camera_options = "\n".join(
        f'<option value="{html.escape(camera)}">{html.escape(camera.replace("_", " ").title())}</option>'
        for camera in camera_names
    )

    present_roles = [
        role_name
        for role_name in ("target", "reference", "interaction_part")
        if isinstance(role_spec.get(role_name), Mapping)
    ]
    if not present_roles:
        present_roles = ["target", "reference"]
    role_options = "\n".join(
        f'<option value="{role_name}">{ROLE_VISUAL_STYLE.get(role_name, {}).get("label", role_name.title())}</option>'
        for role_name in present_roles
    )

    def role_card(role_name: str) -> str:
        style = ROLE_VISUAL_STYLE.get(
            role_name,
            {"label": role_name.replace("_", " ").title()},
        )
        value = role_spec.get(role_name)
        title = html.escape(str(style["label"]))
        if not isinstance(value, Mapping):
            body = '<div class="role-name muted">None</div>'
        else:
            name = html.escape(str(value.get("name") or role_name))
            cues = role_identity_cues(role_spec, role_name)
            cue_html = "".join(f"<li>{html.escape(cue)}</li>" for cue in cues)
            body = f'<div class="role-name">{name}</div>'
            if cue_html:
                body += f'<ul class="role-cues">{cue_html}</ul>'
        return f'<section class="role-card {role_name}"><div class="role-title">{title}</div>{body}</section>'

    role_cards = "\n".join(
        role_card(role_name)
        for role_name in ("target", "reference", "interaction_part")
        if role_name != "interaction_part" or isinstance(role_spec.get(role_name), Mapping)
    )
    task_type = html.escape(str(role_spec.get("task_type") or "—"))
    relation = html.escape(str(role_spec.get("relation") or "—"))

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RLBench episode grounding</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #131313;
  --panel: rgba(255, 255, 255, 0.045);
  --border: rgba(255, 255, 255, 0.14);
  --muted: #a7a7a7;
  --target: rgb(40, 220, 40);
  --reference: rgb(255, 170, 30);
  --interaction: rgb(80, 170, 255);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: #eee; }}
main {{ max-width: 1440px; margin: auto; padding: 16px; }}
h1, h2, h3, p {{ margin-top: 0; }}
h1 {{ font-size: 20px; font-weight: 600; margin-bottom: 14px; }}
h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 8px; }}
.layout {{ display: grid; grid-template-columns: minmax(220px, 290px) minmax(0, 1fr); gap: 16px; align-items: start; }}
.sidebar {{ position: sticky; top: 12px; display: grid; gap: 10px; }}
.card, .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
.instruction {{ white-space: pre-wrap; line-height: 1.45; font-size: 14px; }}
.task-meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; }}
.task-meta strong {{ display: block; color: var(--muted); font-weight: 500; margin-bottom: 3px; }}
.role-card {{ border: 1px solid var(--border); border-left-width: 3px; border-radius: 8px; padding: 9px 10px; background: rgba(255, 255, 255, 0.025); }}
.role-card.target {{ border-left-color: var(--target); background: rgba(40, 220, 40, 0.045); }}
.role-card.reference {{ border-left-color: var(--reference); background: rgba(255, 170, 30, 0.045); }}
.role-card.interaction_part {{ border-left-color: var(--interaction); background: rgba(80, 170, 255, 0.045); }}
.role-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 4px; }}
.role-name {{ font-size: 13px; font-weight: 600; line-height: 1.3; }}
.role-cues {{ margin: 5px 0 0 16px; padding: 0; color: #c8c8c8; font-size: 11px; line-height: 1.35; }}
.muted {{ color: var(--muted); }}
.content {{ min-width: 0; }}
.controls {{ display: grid; grid-template-columns: minmax(200px, 1fr) 180px 150px; gap: 12px; align-items: end; margin-bottom: 12px; }}
label {{ display: grid; gap: 5px; font-size: 12px; color: var(--muted); }}
input[type=range], select {{ width: 100%; }}
select {{ border: 1px solid var(--border); border-radius: 7px; background: #252525; color: #eee; padding: 7px 8px; }}
.grid {{ display: grid; grid-template-columns: repeat({len(camera_names)}, minmax(0, 1fr)); gap: 10px; }}
.view-panel {{ min-width: 0; background: var(--panel); border: 1px solid var(--border); border-radius: 9px; padding: 6px; }}
.view-title {{ font-size: 11px; color: var(--muted); margin: 0 0 5px 2px; text-transform: capitalize; }}
.view-panel img {{ width: 100%; image-rendering: auto; display: block; border-radius: 5px; }}
.debug-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }}
.meta {{ white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 11px; line-height: 1.4; max-height: 330px; overflow: auto; }}
details {{ margin-top: 10px; }}
summary {{ cursor: pointer; color: var(--muted); font-size: 12px; }}
@media (max-width: 960px) {{
  .layout {{ grid-template-columns: 1fr; }}
  .sidebar {{ position: static; grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .sidebar > .instruction-card {{ grid-column: 1 / -1; }}
  .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 640px) {{
  .sidebar, .grid, .debug-grid, .controls {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<h1>RLBench episode target/reference grounding</h1>
<div class="layout">
  <aside class="sidebar">
    <section class="card instruction-card">
      <h2>Instruction</h2>
      <div class="instruction">{instruction_html}</div>
    </section>
    <section class="card">
      <div class="task-meta">
        <div><strong>Task type</strong>{task_type}</div>
        <div><strong>Relation</strong>{relation}</div>
      </div>
    </section>
    {role_cards}
  </aside>

  <section class="content">
    <div class="controls card">
      <label>Frame <span id="frameLabel"></span>
        <input id="frameSlider" type="range" min="0" max="{max(0, len(frames_for_html) - 1)}" value="0" step="1">
      </label>
      <label>Inspect view
        <select id="viewSelect">{camera_options}</select>
      </label>
      <label>Inspect role
        <select id="roleSelect">{role_options}</select>
      </label>
    </div>

    <div id="grid" class="grid"></div>

    <div class="debug-grid">
      <section class="panel">
        <h2>Selected-view result</h2>
        <pre id="json" class="meta"></pre>
      </section>
      <section class="panel">
        <h2>Grounding role specification</h2>
        <pre id="roleDebug" class="meta"></pre>
      </section>
    </div>

    <details>
      <summary>Complete role specification</summary>
      <pre class="panel meta">{role_json}</pre>
    </details>
  </section>
</div>
</main>
<script>
const frames = {data_json};
const cameras = {json.dumps(list(camera_names))};
const roleSpec = {role_spec_js};
const slider = document.getElementById("frameSlider");
const frameLabel = document.getElementById("frameLabel");
const grid = document.getElementById("grid");
const jsonBox = document.getElementById("json");
const roleDebug = document.getElementById("roleDebug");
const viewSelect = document.getElementById("viewSelect");
const roleSelect = document.getElementById("roleSelect");

function render() {{
  const item = frames[Number(slider.value)];
  if (!item) return;
  frameLabel.textContent = `${{item.frame_index}} (source id ${{item.frame_id}})`;
  grid.innerHTML = "";
  for (const camera of cameras) {{
    const panel = document.createElement("section");
    panel.className = "view-panel";
    const title = document.createElement("div");
    title.className = "view-title";
    title.textContent = camera.replaceAll("_", " ");
    const image = document.createElement("img");
    image.src = item.images[camera];
    image.alt = `${{camera}} frame ${{item.frame_id}}`;
    panel.append(title, image);
    grid.append(panel);
  }}

  const selectedView = viewSelect.value;
  const selectedRole = roleSelect.value;
  const viewResult = item.views[selectedView] || {{}};
  jsonBox.textContent = JSON.stringify(viewResult[selectedRole] || {{}}, null, 2);
  roleDebug.textContent = JSON.stringify({{
    role: selectedRole,
    instruction: {json.dumps(instruction, ensure_ascii=False)},
    object_specification: roleSpec[selectedRole] || null
  }}, null, 2);
}}
slider.addEventListener("input", render);
viewSelect.addEventListener("change", render);
roleSelect.addEventListener("change", render);
render();
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")

def load_frame_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid frame result: {path}")
    return value


def ground_episode(
    grounder: Qwen3VLRLBenchGrounder,
    episode_dir: str | Path,
    output_dir: str | Path,
    instruction: str,
    instruction_source: str,
    camera_names: Sequence[str],
    roles: Sequence[str],
    selected_frame_ids: Sequence[str],
    camera_frames: Mapping[str, Mapping[str, Path]],
    role_frame_id: str,
    role_spec_path: str | Path | None,
    resume: bool,
    fail_fast: bool,
    save_raw_outputs: bool,
    save_multiview_images: bool,
    multiview_image_mode: str,
    make_video: bool,
    video_fps: float,
    visualization_cell_width: int,
    frame_interval: int,
    paper_vis: bool,
) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_json_dir = output_dir / "frames"
    annotation_dir = output_dir / "annotations"
    raw_dir = output_dir / "raw_outputs"
    multiview_dir = output_dir / "multiview_images"
    multiview_annotated_dir = multiview_dir / "annotated"
    multiview_original_dir = multiview_dir / "original"
    frame_json_dir.mkdir(exist_ok=True)
    annotation_dir.mkdir(exist_ok=True)
    if save_raw_outputs:
        raw_dir.mkdir(exist_ok=True)
    if save_multiview_images:
        if multiview_image_mode in {"annotated", "both"}:
            multiview_annotated_dir.mkdir(parents=True, exist_ok=True)
        if multiview_image_mode in {"original", "both"}:
            multiview_original_dir.mkdir(parents=True, exist_ok=True)

    if role_spec_path is None and resume:
        resume_role_path = output_dir / "role_spec.json"
        if resume_role_path.is_file():
            role_spec_path = resume_role_path

    if role_spec_path is not None:
        with Path(role_spec_path).open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        role_spec = loaded.get("role_spec", loaded) if isinstance(loaded, dict) else loaded
        if not isinstance(role_spec, dict):
            raise ValueError("--role-spec-json must contain a JSON object.")
        raw_role = None
        role_source = str(role_spec_path)
        atomic_json_dump(
            {
                "instruction": instruction,
                "instruction_source": instruction_source,
                "role_frame_id": role_frame_id,
                "role_spec": role_spec,
                "raw_output": None,
                "reused_from": str(role_spec_path),
            },
            output_dir / "role_spec.json",
        )
    else:
        role_views = {
            camera: camera_frames[camera][role_frame_id]
            for camera in camera_names
        }
        role_spec, raw_role = grounder.identify_roles(instruction, role_views)
        role_source = f"Qwen3-VL role identification at frame {role_frame_id}"
        atomic_json_dump(
            {
                "instruction": instruction,
                "instruction_source": instruction_source,
                "role_frame_id": role_frame_id,
                "role_spec": role_spec,
                "raw_output": raw_role if save_raw_outputs else None,
            },
            output_dir / "role_spec.json",
        )
        if save_raw_outputs:
            (raw_dir / "role_identification.txt").write_text(raw_role, encoding="utf-8")

    frame_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for frame_index, frame_id in enumerate(selected_frame_ids):
        frame_path = frame_json_dir / f"{frame_index:06d}_{frame_id}.json"
        if resume and frame_path.is_file():
            try:
                record = load_frame_record(frame_path)
                frame_records.append(record)
                print(
                    f"[{frame_index + 1}/{len(selected_frame_ids)}] "
                    f"resume frame id={frame_id}"
                )
                # Rebuild a missing annotation without re-running the model.
                for camera in camera_names:
                    annotation_path = (
                        annotation_dir / camera / f"{frame_index:06d}_{frame_id}.png"
                    )
                    if not annotation_path.is_file():
                        save_detailed_annotation(
                            camera_frames[camera][frame_id],
                            record["views"][camera],
                            annotation_path,
                            roles,
                        )
                continue
            except Exception as exc:
                print(f"Warning: could not resume {frame_path}: {exc}; recomputing.")

        print(
            f"[{frame_index + 1}/{len(selected_frame_ids)}] "
            f"ground frame id={frame_id}"
        )
        per_view_results: dict[str, Any] = {}
        per_view_raw: dict[str, dict[str, str]] = {}
        frame_errors: list[dict[str, str]] = []

        for camera in camera_names:
            image_path = camera_frames[camera][frame_id]
            camera_result: dict[str, Any] = {}
            camera_raw: dict[str, str] = {}

            for role_name in roles:
                spec = role_spec.get(role_name)
                if spec is None:
                    camera_result[role_name] = empty_role_result()
                    continue

                try:
                    localized, raw = grounder.localize_role(
                        instruction=instruction,
                        role_name=role_name,
                        role_spec=spec,
                        image_path=image_path,
                    )
                    camera_result[role_name] = localized
                    if save_raw_outputs:
                        camera_raw[role_name] = raw
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    camera_result[role_name] = empty_role_result(error_text)
                    frame_errors.append(
                        {
                            "camera": camera,
                            "role": role_name,
                            "error": error_text,
                        }
                    )
                    if fail_fast:
                        raise
                    traceback.print_exc()

            # Keep absent roles explicit for a stable result schema.
            for role_name in ("target", "reference", "interaction_part"):
                camera_result.setdefault(role_name, empty_role_result())

            per_view_results[camera] = camera_result
            if save_raw_outputs:
                per_view_raw[camera] = camera_raw

            annotation_path = (
                annotation_dir / camera / f"{frame_index:06d}_{frame_id}.png"
            )
            save_detailed_annotation(
                image_path=image_path,
                view_result=camera_result,
                output_path=annotation_path,
                roles=roles,
            )

        record = {
            "frame_index": frame_index,
            "frame_id": frame_id,
            "source_images": {
                camera: str(camera_frames[camera][frame_id])
                for camera in camera_names
            },
            "views": per_view_results,
            "errors": frame_errors,
        }
        if save_raw_outputs:
            record["raw_outputs"] = per_view_raw

        atomic_json_dump(record, frame_path)
        frame_records.append(record)
        if frame_errors:
            failures.append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "errors": frame_errors,
                }
            )

    multiview_records: list[dict[str, Any]] = []
    if save_multiview_images:
        print(
            f"Saving synchronized multiview images: mode={multiview_image_mode}"
        )
        for record in frame_records:
            frame_index = int(record["frame_index"])
            frame_id = str(record["frame_id"])
            filename = f"{frame_index:06d}_{frame_id}.png"
            item: dict[str, Any] = {
                "frame_index": frame_index,
                "frame_id": frame_id,
            }

            if multiview_image_mode in {"annotated", "both"}:
                annotated_paths = {
                    camera: (
                        annotation_dir
                        / camera
                        / f"{frame_index:06d}_{frame_id}.png"
                    )
                    for camera in camera_names
                }
                annotated_output = multiview_annotated_dir / filename
                compose_multiview_frame(
                    image_paths=annotated_paths,
                    camera_names=camera_names,
                    frame_id=frame_id,
                    frame_index=frame_index,
                    instruction=instruction,
                    role_spec=role_spec,
                    cell_width=visualization_cell_width,
                    paper_vis=paper_vis,
                ).save(annotated_output)
                item["annotated"] = str(annotated_output)

            if multiview_image_mode in {"original", "both"}:
                original_paths = {
                    camera: camera_frames[camera][frame_id]
                    for camera in camera_names
                }
                original_output = multiview_original_dir / filename
                compose_multiview_frame(
                    image_paths=original_paths,
                    camera_names=camera_names,
                    frame_id=frame_id,
                    frame_index=frame_index,
                    instruction=instruction,
                    role_spec=role_spec,
                    cell_width=visualization_cell_width,
                    legend="original synchronized RGB views",
                    paper_vis=paper_vis,
                ).save(original_output)
                item["original"] = str(original_output)

            multiview_records.append(item)

    html_path = output_dir / "episode_viewer.html"
    create_episode_html(
        output_path=html_path,
        instruction=instruction,
        role_spec=role_spec,
        frame_records=frame_records,
        camera_names=camera_names,
    )

    video_info: dict[str, Any] = {"created": False}
    if make_video:
        video_path = output_dir / "episode_multiview.mp4"

        def video_frames() -> Iterable[Image.Image]:
            for record in frame_records:
                frame_index = int(record["frame_index"])
                frame_id = str(record["frame_id"])
                paths = {
                    camera: (
                        annotation_dir
                        / camera
                        / f"{frame_index:06d}_{frame_id}.png"
                    )
                    for camera in camera_names
                }
                yield compose_multiview_frame(
                    image_paths=paths,
                    camera_names=camera_names,
                    frame_id=frame_id,
                    frame_index=frame_index,
                    instruction=instruction,
                    role_spec=role_spec,
                    cell_width=visualization_cell_width,
                    paper_vis=paper_vis,
                )

        ok, backend = write_mp4(video_frames, video_path, fps=video_fps)
        video_info = {
            "created": ok,
            "path": str(video_path) if ok else None,
            "backend": backend,
        }
        if not ok:
            print(f"Warning: MP4 was not created: {backend}")

    summary = {
        "episode_dir": str(episode_dir),
        "output_dir": str(output_dir),
        "instruction": instruction,
        "instruction_source": instruction_source,
        "camera_names": list(camera_names),
        "roles": list(roles),
        "num_frames": len(frame_records),
        "frame_interval": frame_interval,
        "paper_vis": paper_vis,
        "role_frame_id": role_frame_id,
        "role_source": role_source,
        "role_spec": role_spec,
        "frames": [
            {
                "frame_index": record["frame_index"],
                "frame_id": record["frame_id"],
                "result_json": (
                    f"frames/{int(record['frame_index']):06d}_{record['frame_id']}.json"
                ),
            }
            for record in frame_records
        ],
        "num_failed_localizations": sum(
            len(item["errors"]) for item in failures
        ),
        "failures": failures,
        "visualization": {
            "html": str(html_path),
            "video": video_info,
            "annotations": str(annotation_dir),
            "multiview_images": {
                "enabled": save_multiview_images,
                "mode": multiview_image_mode if save_multiview_images else None,
                "root": str(multiview_dir) if save_multiview_images else None,
                "frames": multiview_records,
            },
        },
    }
    atomic_json_dump(summary, output_dir / "episode_grounding.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the verified Qwen3-VL RLBench target/reference grounding logic "
            "over every selected frame in one episode folder."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/new-common-data/new-common-data/huggingface/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", default="./episode_role_grounding")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Explicit instruction. Overrides episode metadata discovery.",
    )
    parser.add_argument(
        "--instruction-file",
        default=None,
        help="Optional .pkl/.json/.txt file containing one or more instructions.",
    )
    parser.add_argument(
        "--instruction-index",
        type=int,
        default=0,
        help="Instruction index when the metadata contains multiple descriptions.",
    )
    parser.add_argument(
        "--cameras",
        type=parse_csv,
        default=DEFAULT_CAMERAS,
        help="Comma-separated camera names; directories must be NAME_rgb.",
    )
    parser.add_argument(
        "--roles",
        type=parse_csv,
        default=DEFAULT_ROLES,
        help="Comma-separated roles. Default: target,reference.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive end index in the naturally sorted common frame list.",
    )
    parser.add_argument(
        "--frame-interval",
        "--stride",
        dest="frame_interval",
        type=int,
        default=10,
        help=(
            "Process one frame every N source frames. For example, 5 selects "
            "source-frame indices 0, 5, 10, ... within the start/end range. "
            "--stride is retained as a backward-compatible alias."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--role-frame",
        default="first",
        help=(
            "'first', a selected frame ID, or an index into selected frames. "
            "Role identity is resolved once at this synchronized multiview frame."
        ),
    )
    parser.add_argument(
        "--role-spec-json",
        default=None,
        help=(
            "Reuse a prior role_spec.json or a JSON object containing role_spec; "
            "skips semantic role identification."
        ),
    )
    parser.add_argument("--grounding-min-side", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--save-raw-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-multiview-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one horizontally composed synchronized multiview PNG per selected frame.",
    )
    parser.add_argument(
        "--multiview-image-mode",
        choices=("annotated", "original", "both"),
        default="both",
        help=(
            "Which composed multiview PNGs to save: annotated boxes, original "
            "RGB views, or both."
        ),
    )
    parser.add_argument(
        "--make-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--visualization-cell-width", type=int, default=384)
    parser.add_argument(
        "--paper-vis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Create clean multiview PNG/video frames without the metadata header. "
            "Bounding boxes and compact transparent role/view labels are retained."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    allowed_roles = {"target", "reference", "interaction_part"}
    invalid_roles = set(args.roles) - allowed_roles
    if invalid_roles:
        parser.error(
            f"Unknown role(s): {sorted(invalid_roles)}; "
            f"allowed roles are {sorted(allowed_roles)}"
        )

    instruction, instruction_source = discover_instruction(
        episode_dir=args.episode_dir,
        explicit_instruction=args.instruction,
        instruction_file=args.instruction_file,
        instruction_index=args.instruction_index,
    )
    all_frame_ids, camera_frames = collect_camera_frames(
        args.episode_dir,
        args.cameras,
    )
    selected_frame_ids = select_frame_ids(
        all_frame_ids=all_frame_ids,
        start=args.start,
        end=args.end,
        stride=args.frame_interval,
        max_frames=args.max_frames,
    )
    role_frame_id = resolve_role_frame(args.role_frame, selected_frame_ids)

    print(f"Episode: {args.episode_dir}")
    print(f"Instruction ({instruction_source}): {instruction}")
    print(f"Cameras: {', '.join(args.cameras)}")
    print(
        f"Frames: {len(selected_frame_ids)} selected from {len(all_frame_ids)}; "
        f"frame interval={args.frame_interval}; role frame id={role_frame_id}"
    )

    grounder = Qwen3VLRLBenchGrounder(
        model_path=args.model_path,
        grounding_min_side=args.grounding_min_side,
        max_retries=args.max_retries,
    )
    summary = ground_episode(
        grounder=grounder,
        episode_dir=args.episode_dir,
        output_dir=args.output_dir,
        instruction=instruction,
        instruction_source=instruction_source,
        camera_names=args.cameras,
        roles=args.roles,
        selected_frame_ids=selected_frame_ids,
        camera_frames=camera_frames,
        role_frame_id=role_frame_id,
        role_spec_path=args.role_spec_json,
        resume=args.resume,
        fail_fast=args.fail_fast,
        save_raw_outputs=args.save_raw_outputs,
        save_multiview_images=args.save_multiview_images,
        multiview_image_mode=args.multiview_image_mode,
        make_video=args.make_video,
        video_fps=args.video_fps,
        visualization_cell_width=args.visualization_cell_width,
        frame_interval=args.frame_interval,
        paper_vis=args.paper_vis,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
