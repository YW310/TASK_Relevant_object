"""Task-agnostic manipulation schema compiled from an RLBench instruction.

The schema intentionally describes actions and goal predicates instead of
enumerating RLBench task names.  It is a lightweight prior for the deterministic
temporal reasoner; Qwen remains responsible for resolving visual identity.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSchema:
    action_family: str
    goal_predicate: str
    manipulated_role: str = "target"
    goal_anchor_role: str | None = None
    interaction_part_role: str | None = None
    repeat_policy: str = "single"
    reference_required: bool = False
    compiler: str = "instruction_predicate_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(child) for child in value)
    return str(value)


def _contains(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def compile_task_schema(
    instruction: Any,
    role_spec: Mapping[str, Any] | None = None,
) -> TaskSchema:
    """Compile common manipulation predicates without branching on task names."""
    text = f"{_flatten_text(instruction)} {_flatten_text(role_spec)}".lower()

    if _contains(
        text,
        (
            r"\bpress\b",
            r"\bpush\b[^.]*\bbutton\b",
            r"\bactivate\b[^.]*\bbutton\b",
        ),
    ):
        action_family, predicate = "press", "PRESSED"
    elif _contains(text, (r"\bopen\b", r"\bpull open\b")):
        action_family, predicate = "articulate", "OPEN"
    elif _contains(text, (r"\bclose\b", r"\bshut\b")):
        action_family, predicate = "articulate", "CLOSED"
    elif _contains(text, (r"\binsert\b", r"\bplug\b", r"\bput .* into .*slot\b")):
        action_family, predicate = "insert", "INSERTED_IN"
    elif _contains(text, (r"\bput\b.*\b(?:in|inside|into)\b", r"\bplace\b.*\b(?:in|inside|into)\b")):
        action_family, predicate = "pick_place", "IN"
    elif _contains(text, (r"\bstack\b", r"\bon top of\b", r"\bplace\b.*\bon\b", r"\bput\b.*\bon\b")):
        action_family, predicate = "pick_place", "ON"
    elif _contains(text, (r"\bpour\b",)):
        action_family, predicate = "pour", "TRANSFERRED_IN"
    elif _contains(text, (r"\bsweep\b", r"\buse\b.*\bto\b")):
        action_family, predicate = "tool_use", "APPLIED_TO"
    elif _contains(text, (r"\bpush\b", r"\bslide\b", r"\bmove\b")):
        action_family, predicate = "move", "AT_GOAL"
    elif _contains(text, (r"\bpick\b", r"\blift\b", r"\bgrasp\b", r"\btake\b")):
        action_family, predicate = "pick", "GRASPED"
    else:
        action_family, predicate = "interact", "TASK_STATE_CHANGED"

    reference_required = predicate in {
        "ON",
        "IN",
        "INSERTED_IN",
        "TRANSFERRED_IN",
        "APPLIED_TO",
        "AT_GOAL",
    }
    interaction_part_role = (
        "interaction_part" if action_family in {"press", "articulate"} else None
    )
    repeat_policy = (
        "repeat_until_satisfied"
        if _contains(
            text,
            (
                r"\bstack\b",
                r"\ball\b",
                r"\beach\b",
                r"\bevery\b",
                r"\bremaining\b",
            ),
        )
        else "single"
    )
    return TaskSchema(
        action_family=action_family,
        goal_predicate=predicate,
        goal_anchor_role="reference" if reference_required else None,
        interaction_part_role=interaction_part_role,
        repeat_policy=repeat_policy,
        reference_required=reference_required,
    )
