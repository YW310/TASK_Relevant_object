"""Validated, reproducible offline thresholds; no model activation settings."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path
from typing import Any

from slot_geometry_fusion import FusionConfig
from slot_guided_sam3 import SelectionConfig
from slot_prompt_adapter import AdapterConfig


@dataclass(frozen=True)
class SlotGuidedConfig:
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def __post_init__(self):
        if not isinstance(self.adapter, AdapterConfig) or not isinstance(self.selection, SelectionConfig) or not isinstance(self.fusion, FusionConfig):
            raise ValueError("configuration sections must be their typed offline configs")
        self.adapter.validate()
        # Validate exact numeric types even for Python callers, before NumPy can
        # silently accept booleans or coercible strings as metric thresholds.
        for section in (self.adapter, self.selection, self.fusion):
            for descriptor in fields(section):
                value, default = getattr(section, descriptor.name), descriptor.default
                valid = type(value) is int if type(default) is int else type(value) in (int, float)
                if not valid:
                    raise ValueError(f"{type(section).__name__}.{descriptor.name} has the wrong numeric type")
        self.selection.__post_init__()
        self.fusion.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_slot_guided_config(path: str | Path | None = None) -> SlotGuidedConfig:
    """Missing sections use unchanged defaults; unknown/mistyped fields fail."""
    if path is None:
        return SlotGuidedConfig()
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate config key {key}")
            out[key] = value
        return out
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1 or set(value) - {"schema_version", "adapter", "selection", "fusion"}:
        raise ValueError("offline config requires schema_version=1 and only adapter/selection/fusion sections")
    sections = {}
    for name, cls in (("adapter", AdapterConfig), ("selection", SelectionConfig), ("fusion", FusionConfig)):
        supplied = value.get(name, {})
        defaults = asdict(cls())
        if not isinstance(supplied, dict) or set(supplied) - defaults.keys():
            raise ValueError(f"unknown or malformed {name} config fields")
        for key, setting in supplied.items():
            expected = defaults[key]
            valid = type(setting) is int if type(expected) is int else type(setting) in (int, float)
            if not valid:
                raise ValueError(f"{name}.{key} has the wrong numeric type")
        sections[name] = cls(**supplied)
    return SlotGuidedConfig(**sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_slot_guided_config(args.config)
    print(json.dumps({"effective_config": config.to_dict(), "config_sha256": config.digest()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
