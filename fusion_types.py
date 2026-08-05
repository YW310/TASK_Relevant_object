"""Compatibility exports for legacy imports; use relevant_object.domain."""

from _legacy_bootstrap import ensure_src_path


ensure_src_path()

from relevant_object.domain.models import Observation3D


__all__ = ["Observation3D"]
