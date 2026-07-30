"""Observable-only public contracts for reactive visual DoorNav."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    FrozenSet,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

from ...core import NavigationObservation


# The existing action set lives in the policy-prompt layer. Importing it here
# would make this platform-independent contract depend on model prompting.
# Keep this small vocabulary local until a shared action contract moves to core.
DOORNAV_ACTIONS: FrozenSet[str] = frozenset(
    {"move_forward", "turn_left", "turn_right", "stop"}
)

PRIVILEGED_FIELD_DENYLIST: FrozenSet[str] = frozenset(
    {
        "goal",
        "goal_position",
        "target_position",
        "goal_distance",
        "goal_distance_m",
        "goal_angle",
        "goal_angle_deg",
        "success_distance",
        "success_distance_m",
        "success_radius",
        "shortest_path",
        "geodesic_distance",
        "oracle_waypoint",
        "global_x",
        "global_y",
        "waypoint_position",
    }
)


class DoorNavState(str, Enum):
    """States planned for the reactive DoorNav control loop."""

    SEARCH = "search"
    TRACK = "track"
    APPROACH = "approach"
    AVOID = "avoid"
    REACQUIRE = "reacquire"
    VERIFY = "verify"
    STOP = "stop"
    FAILED = "failed"


class DoorNavTerminationReason(str, Enum):
    """Auditable reasons for ending a DoorNav episode."""

    REACHED_DOOR = "reached_door"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_LOST = "target_lost"
    INVALID_VISUAL_TRACK = "invalid_visual_track"
    LOCAL_PATH_BLOCKED = "local_path_blocked"
    NO_PROGRESS = "no_progress"
    SEARCH_STEP_LIMIT = "search_step_limit"
    EPISODE_STEP_LIMIT = "episode_step_limit"
    INVALID_OBSERVATION = "invalid_observation"
    POLICY_ERROR = "policy_error"


def _validate_bbox_xyxy(bbox_xyxy: Tuple[int, int, int, int]) -> None:
    """Validate one image-space bounding box."""
    if (
        not isinstance(bbox_xyxy, tuple)
        or len(bbox_xyxy) != 4
        or any(type(value) is not int for value in bbox_xyxy)
    ):
        raise ValueError("bbox_xyxy must contain exactly four integers")

    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must satisfy x2 > x1 and y2 > y1")


@dataclass(frozen=True)
class DoorCandidate:
    """One door detection derived from the current observable RGB input."""

    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    label: str
    source: str

    def __post_init__(self):
        """Reject malformed detections before they enter visual tracking."""
        _validate_bbox_xyxy(self.bbox_xyxy)
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class VisualTargetTrack:
    """A current or recently visible doorway tracked in RGB image space.

    Normalized centers use ``[-1, 1]``: horizontal values run from left to
    right, while vertical values run from top to bottom. ``area_ratio`` is an
    observable image fraction, not a physical distance estimate.
    """

    bbox_xyxy: Tuple[int, int, int, int]
    center_x_norm: float
    center_y_norm: float
    area_ratio: float
    confidence: float
    visible: bool
    missing_steps: int
    source: str

    def __post_init__(self):
        """Reject malformed image-space tracks and inconsistent visibility."""
        _validate_bbox_xyxy(self.bbox_xyxy)

        for name, value in {
            "center_x_norm": self.center_x_norm,
            "center_y_norm": self.center_y_norm,
            "area_ratio": self.area_ratio,
            "confidence": self.confidence,
        }.items():
            try:
                is_finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not is_finite:
                raise ValueError(f"{name} must be finite")

        if not -1.0 <= self.center_x_norm <= 1.0:
            raise ValueError("center_x_norm must be between -1 and 1")
        if not -1.0 <= self.center_y_norm <= 1.0:
            raise ValueError("center_y_norm must be between -1 and 1")
        if not 0.0 < self.area_ratio <= 1.0:
            raise ValueError("area_ratio must be greater than 0 and at most 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if type(self.visible) is not bool:
            raise ValueError("visible must be a boolean")
        if type(self.missing_steps) is not int or self.missing_steps < 0:
            raise ValueError("missing_steps must be a non-negative integer")
        if self.visible and self.missing_steps != 0:
            raise ValueError("a visible track must have zero missing_steps")
        if not self.visible and self.missing_steps == 0:
            raise ValueError("a missing track must have at least one missing step")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class LocalExecutionDecision:
    """One auditable decision produced by observable-only local execution."""

    action: str
    state: DoorNavState
    reason: str
    target_track: Optional[VisualTargetTrack]
    obstacle_avoidance_active: bool

    def __post_init__(self):
        """Keep execution actions within the active project vocabulary."""
        if self.action not in DOORNAV_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(DOORNAV_ACTIONS)}"
            )


@runtime_checkable
class DoorGrounder(Protocol):
    """Detect door candidates using only RGB evidence and local instruction."""

    def detect(self, rgb: Any, instruction: str) -> Sequence[DoorCandidate]:
        ...


@runtime_checkable
class VisualTargetTracker(Protocol):
    """Track doorway candidates using RGB image-space evidence only."""

    def reset(self) -> None:
        ...

    def update(
        self,
        rgb: Any,
        candidates: Sequence[DoorCandidate],
        step: int,
    ) -> Optional[VisualTargetTrack]:
        ...


@runtime_checkable
class ReactiveVisualExecutor(Protocol):
    """Choose atomic actions from observable visual and target-free context."""

    def reset(self) -> None:
        ...

    def step(
        self,
        observation: NavigationObservation,
        target: Optional[VisualTargetTrack],
    ) -> LocalExecutionDecision:
        ...
