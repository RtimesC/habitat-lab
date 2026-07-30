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
    INVALID_LOCAL_SUBGOAL = "invalid_local_subgoal"
    LOCAL_PATH_BLOCKED = "local_path_blocked"
    NO_PROGRESS = "no_progress"
    SEARCH_STEP_LIMIT = "search_step_limit"
    EPISODE_STEP_LIMIT = "episode_step_limit"
    INVALID_OBSERVATION = "invalid_observation"
    POLICY_ERROR = "policy_error"


@dataclass(frozen=True)
class DoorCandidate:
    """One door detection derived from the current observable RGB input."""

    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    label: str
    source: str

    def __post_init__(self):
        """Reject malformed detections before they enter local estimation."""
        if (
            not isinstance(self.bbox_xyxy, tuple)
            or len(self.bbox_xyxy) != 4
            or any(type(value) is not int for value in self.bbox_xyxy)
        ):
            raise ValueError("bbox_xyxy must contain exactly four integers")

        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must satisfy x2 > x1 and y2 > y1")
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class LocalSubgoal:
    """An observable or model-produced target in the robot-local frame.

    It may come from RGB visual target derivation, a future navigation model's
    local waypoint, or other observable target-free local perception. The
    contract requires no particular depth sensor and contains neither global
    coordinates nor a hidden Habitat target.
    """

    target_type: str
    relative_x_m: float
    relative_y_m: float
    desired_heading_rad: Optional[float]
    stop_distance_m: float
    confidence: float
    source: str

    def __post_init__(self):
        """Reject invalid local geometry and untraceable target sources."""
        finite_values = {
            "relative_x_m": self.relative_x_m,
            "relative_y_m": self.relative_y_m,
            "stop_distance_m": self.stop_distance_m,
            "confidence": self.confidence,
        }
        if self.desired_heading_rad is not None:
            finite_values["desired_heading_rad"] = self.desired_heading_rad
        for name, value in finite_values.items():
            try:
                is_finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not is_finite:
                raise ValueError(f"{name} must be finite")

        if self.stop_distance_m <= 0.0:
            raise ValueError("stop_distance_m must be greater than zero")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if (
            not isinstance(self.target_type, str)
            or not self.target_type.strip()
        ):
            raise ValueError("target_type must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class LocalExecutionDecision:
    """One auditable decision produced by observable-only local execution."""

    action: str
    state: DoorNavState
    reason: str
    subgoal: Optional[LocalSubgoal]
    target_visible: bool
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
class LocalNavigationExecutor(Protocol):
    """Execute observable robot-local subgoals without privileged target geometry."""

    def reset(self) -> None:
        ...

    def step(
        self,
        observation: NavigationObservation,
        subgoal: LocalSubgoal,
    ) -> LocalExecutionDecision:
        ...
