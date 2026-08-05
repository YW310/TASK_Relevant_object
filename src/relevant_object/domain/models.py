"""Domain models shared across the relevant-object pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class Observation3D:
    observation_id: str
    semantic_evidence: Mapping[str, Mapping[str, Any]]
    provenance: Mapping[str, Any]
    camera: str
    candidate: Mapping[str, Any]
    points_world: np.ndarray
    centroid_world: np.ndarray
    bbox3d_world: np.ndarray
