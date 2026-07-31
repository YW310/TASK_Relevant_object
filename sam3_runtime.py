"""Reusable SAM3 checkpoint, autocast, and tensor normalization helpers."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch


def find_checkpoint(model_dir: Path, explicit: str | None) -> Path:
    """Resolve a torch-loadable SAM3 checkpoint deterministically."""
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
    """Enable CUDA BF16 only when both hardware and caller allow it."""
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
