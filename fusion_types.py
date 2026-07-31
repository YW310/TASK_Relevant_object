"""Shared data types for multi-view candidate fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


ROLE_NAMES = ("target", "reference", "interaction_part")


@dataclass
class Observation3D:
    observation_id: str
    role_evidence: Mapping[str, float]
    provenance: Mapping[str, Any]
    camera: str
    candidate: Mapping[str, Any]
    points_world: np.ndarray
    centroid_world: np.ndarray
    bbox3d_world: np.ndarray
