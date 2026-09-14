"""Executable observable-only components for the reactive DoorNav baseline."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

try:
    from ...core import (
        ALLOWED_BUILDING_PRIOR_FIELDS,
        NavigationObservation,
        PolicyOutput,
    )
except ImportError:
    from core import (
        ALLOWED_BUILDING_PRIOR_FIELDS,
        NavigationObservation,
        PolicyOutput,
    )

from .contracts import (
    PRIVILEGED_FIELD_DENYLIST,
    DoorCandidate,
    DoorNavState,
    DoorNavTerminationReason,
    LocalExecutionDecision,
    VisualTargetTrack,
)

_ALLOWED_NAVIGATION_CONTEXT_FIELDS = frozenset(
    {
        "step",
        "collided",
        "depth_left_m",
        "depth_center_m",
        "depth_right_m",
        "previous_action",
        "previous_action_count",
        "previous_collision",
        "building_prior",
    }
)
_B1_TEXT_BUILDING_PRIOR_FIELDS = frozenset(
    {
        "building_id",
        "start_region",
        "footprint_summary",
        "public_structure_summary",
    }
)
_B1_TEXT_LIST_BUILDING_PRIOR_FIELDS = frozenset(
    {"known_entrances", "floor_labels", "known_regions"}
)


@dataclass(frozen=True)
class ReactiveDoorNavConfig:
    """Tunable image-space thresholds for the executable engineering baseline."""

    max_search_steps: int = 24
    max_episode_steps: int = 200
    arrival_area_ratio_threshold: float = 0.18
    arrival_center_tolerance_norm: float = 0.15
    arrival_confirm_frames: int = 3
    grounding_confidence_threshold: float = 0.55
    target_lost_tolerance_steps: int = 3
    no_progress_tolerance_steps: int = 8
    min_area_progress: float = 0.002
    tracker_match_iou_threshold: float = 0.1
    tracker_confidence_decay: float = 0.8
    grounder_min_area_ratio: float = 0.02
    grounder_max_area_ratio: float = 0.65
    grounder_min_aspect_ratio: float = 1.2
    grounder_min_height_ratio: float = 0.25
    grounder_nms_iou_threshold: float = 0.45
    grounder_max_candidates: int = 5
    use_depth_for_avoidance: bool = False

    def __post_init__(self):
        """Reject inconsistent or unsafe engineering parameter values."""
        integer_fields = {
            "max_search_steps": self.max_search_steps,
            "max_episode_steps": self.max_episode_steps,
            "arrival_confirm_frames": self.arrival_confirm_frames,
            "target_lost_tolerance_steps": self.target_lost_tolerance_steps,
            "no_progress_tolerance_steps": self.no_progress_tolerance_steps,
            "grounder_max_candidates": self.grounder_max_candidates,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        bounded_fields = {
            "arrival_area_ratio_threshold": self.arrival_area_ratio_threshold,
            "arrival_center_tolerance_norm": self.arrival_center_tolerance_norm,
            "grounding_confidence_threshold": self.grounding_confidence_threshold,
            "tracker_match_iou_threshold": self.tracker_match_iou_threshold,
            "tracker_confidence_decay": self.tracker_confidence_decay,
            "grounder_min_area_ratio": self.grounder_min_area_ratio,
            "grounder_max_area_ratio": self.grounder_max_area_ratio,
            "grounder_min_height_ratio": self.grounder_min_height_ratio,
        }
        for name, value in bounded_fields.items():
            if (
                type(value) not in {int, float}
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must be in (0, 1]")

        if self.grounder_min_area_ratio >= self.grounder_max_area_ratio:
            raise ValueError(
                "grounder area bounds must satisfy minimum < maximum"
            )
        if (
            type(self.grounder_min_aspect_ratio) not in {int, float}
            or not math.isfinite(self.grounder_min_aspect_ratio)
            or self.grounder_min_aspect_ratio <= 1.0
        ):
            raise ValueError(
                "grounder_min_aspect_ratio must be greater than 1"
            )
        if (
            type(self.grounder_nms_iou_threshold) not in {int, float}
            or not math.isfinite(self.grounder_nms_iou_threshold)
            or not 0.0 <= self.grounder_nms_iou_threshold <= 1.0
        ):
            raise ValueError("grounder_nms_iou_threshold must be in [0, 1]")
        if (
            type(self.min_area_progress) not in {int, float}
            or not math.isfinite(self.min_area_progress)
            or not 0.0 <= self.min_area_progress <= 1.0
        ):
            raise ValueError(
                "min_area_progress must be finite and non-negative"
            )
        if type(self.use_depth_for_avoidance) is not bool:
            raise ValueError("use_depth_for_avoidance must be a boolean")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        """Build a config from the versioned baseline specification."""
        field_names = set(cls.__dataclass_fields__)
        unknown_fields = sorted(set(values) - field_names)
        if unknown_fields:
            raise ValueError(
                "Reactive DoorNav config has unknown parameters: "
                + ", ".join(unknown_fields)
            )
        return cls(**dict(values))


@dataclass(frozen=True)
class ReactiveDoorNavBaselineSpec:
    """Validated metadata and parameters loaded from ``baseline_spec.yaml``."""

    name: str
    description: str
    status: str
    instruction: str
    output_dir: str
    config: ReactiveDoorNavConfig
    parameter_semantics: Mapping[str, str]
    path: Path


def load_reactive_doornav_spec(path: Any) -> ReactiveDoorNavBaselineSpec:
    """Load the single versioned source of B1 engineering parameters."""
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"DoorNav baseline spec does not exist: {spec_path}"
        )
    with spec_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("DoorNav baseline spec must contain a YAML mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("DoorNav baseline spec requires schema_version 1")

    expected_metadata = {
        "schema_version",
        "name",
        "description",
        "status",
        "instruction",
        "output_dir",
        "parameters",
    }
    unknown_metadata = sorted(set(payload) - expected_metadata)
    if unknown_metadata:
        raise ValueError(
            "DoorNav baseline spec has unknown fields: "
            + ", ".join(unknown_metadata)
        )
    missing_metadata = sorted(expected_metadata - set(payload))
    if missing_metadata:
        raise ValueError(
            "DoorNav baseline spec is missing fields: "
            + ", ".join(missing_metadata)
        )

    text_fields = {
        name: payload[name]
        for name in [
            "name",
            "description",
            "status",
            "instruction",
            "output_dir",
        ]
    }
    for name, value in text_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"DoorNav baseline spec {name!r} must be non-empty"
            )
    if text_fields["name"] != "reactive_doornav_b1":
        raise ValueError(
            "DoorNav baseline spec name must be reactive_doornav_b1"
        )
    if text_fields["status"] != "engineering_baseline":
        raise ValueError(
            "DoorNav baseline status must be engineering_baseline"
        )

    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("DoorNav baseline parameters must be a mapping")
    expected_parameters = set(ReactiveDoorNavConfig.__dataclass_fields__)
    missing_parameters = sorted(expected_parameters - set(parameters))
    unknown_parameters = sorted(set(parameters) - expected_parameters)
    if missing_parameters or unknown_parameters:
        details = []
        if missing_parameters:
            details.append("missing " + ", ".join(missing_parameters))
        if unknown_parameters:
            details.append("unknown " + ", ".join(unknown_parameters))
        raise ValueError(
            "DoorNav baseline parameters are invalid: " + "; ".join(details)
        )

    values: Dict[str, Any] = {}
    semantics: Dict[str, str] = {}
    for name, entry in parameters.items():
        if not isinstance(entry, dict) or set(entry) != {"value", "semantics"}:
            raise ValueError(
                f"DoorNav parameter {name!r} requires value and semantics"
            )
        meaning = entry["semantics"]
        if not isinstance(meaning, str) or not meaning.strip():
            raise ValueError(f"DoorNav parameter {name!r} needs semantics")
        values[name] = entry["value"]
        semantics[name] = meaning.strip()

    return ReactiveDoorNavBaselineSpec(
        name=text_fields["name"].strip(),
        description=text_fields["description"].strip(),
        status=text_fields["status"].strip(),
        instruction=text_fields["instruction"].strip(),
        output_dir=text_fields["output_dir"].strip(),
        config=ReactiveDoorNavConfig.from_mapping(values),
        parameter_semantics=semantics,
        path=spec_path,
    )


class OpenCVDoorGrounder:
    """Detect tall rectangular doorway candidates from the current RGB image.

    This deterministic contour detector is an engineering baseline for synthetic
    and easy-split validation. It is not a learned open-world door recognizer and
    must not be presented as benchmark-grade semantic grounding.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.02,
        max_area_ratio: float = 0.65,
        min_aspect_ratio: float = 1.2,
        min_height_ratio: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_candidates: int = 5,
    ):
        """Configure deterministic contour filtering and candidate NMS."""
        if not 0.0 < min_area_ratio < max_area_ratio <= 1.0:
            raise ValueError(
                "area-ratio bounds must satisfy 0 < min < max <= 1"
            )
        if min_aspect_ratio <= 1.0:
            raise ValueError("min_aspect_ratio must be greater than 1")
        if not 0.0 < min_height_ratio <= 1.0:
            raise ValueError("min_height_ratio must be in (0, 1]")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1]")
        if type(max_candidates) is not int or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.min_height_ratio = min_height_ratio
        self.nms_iou_threshold = nms_iou_threshold
        self.max_candidates = max_candidates

    def detect(self, rgb: Any, instruction: str) -> Sequence[DoorCandidate]:
        """Return image-space candidates using only the supplied RGB frame."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        frame = _as_rgb_uint8(rgb)
        height, width = frame.shape[:2]
        image_area = float(height * width)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), dtype=np.uint8)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue
            area_ratio = (box_width * box_height) / image_area
            aspect_ratio = box_height / float(box_width)
            height_ratio = box_height / float(height)
            bottom_ratio = (y + box_height) / float(height)
            if not self.min_area_ratio <= area_ratio <= self.max_area_ratio:
                continue
            if aspect_ratio < self.min_aspect_ratio:
                continue
            if height_ratio < self.min_height_ratio or bottom_ratio < 0.5:
                continue

            contour_area = max(float(cv2.contourArea(contour)), 0.0)
            rectangularity = min(
                contour_area / float(box_width * box_height),
                1.0,
            )
            aspect_score = min(
                (aspect_ratio - self.min_aspect_ratio) / 1.8,
                1.0,
            )
            height_score = min(
                (height_ratio - self.min_height_ratio) / 0.55,
                1.0,
            )
            confidence = min(
                0.45
                + 0.25 * max(aspect_score, 0.0)
                + 0.20 * max(height_score, 0.0)
                + 0.10 * rectangularity,
                0.99,
            )
            candidates.append(
                DoorCandidate(
                    bbox_xyxy=(x, y, x + box_width, y + box_height),
                    confidence=confidence,
                    label="doorway",
                    source="opencv_vertical_contour",
                )
            )

        return _non_maximum_suppression(
            candidates,
            self.nms_iou_threshold,
            self.max_candidates,
        )


class IoUVisualTargetTracker:
    """Track one doorway by combining image-space overlap and confidence."""

    def __init__(
        self,
        lost_tolerance_steps: int = 3,
        confidence_decay: float = 0.8,
        match_iou_threshold: float = 0.1,
    ):
        """Configure short-term IoU matching and confidence decay."""
        if type(lost_tolerance_steps) is not int or lost_tolerance_steps <= 0:
            raise ValueError("lost_tolerance_steps must be a positive integer")
        if not 0.0 < confidence_decay <= 1.0:
            raise ValueError("confidence_decay must be in (0, 1]")
        if not 0.0 < match_iou_threshold <= 1.0:
            raise ValueError("match_iou_threshold must be in (0, 1]")
        self.lost_tolerance_steps = lost_tolerance_steps
        self.confidence_decay = confidence_decay
        self.match_iou_threshold = match_iou_threshold
        self._track = None
        self._last_step = None

    def reset(self) -> None:
        """Forget the current image-space identity and step counter."""
        self._track = None
        self._last_step = None

    def update(
        self,
        rgb: Any,
        candidates: Sequence[DoorCandidate],
        step: int,
    ) -> Optional[VisualTargetTrack]:
        """Update one target from current RGB candidates and a monotonic step."""
        frame = _as_rgb_uint8(rgb)
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("step must strictly increase")
        self._last_step = step

        candidates = list(candidates)
        for candidate in candidates:
            if not isinstance(candidate, DoorCandidate):
                raise ValueError(
                    "candidates must contain DoorCandidate values"
                )
            _validate_bbox_in_frame(candidate.bbox_xyxy, frame.shape)
        if candidates:
            selected = self._select_candidate(candidates)
            if selected is not None:
                self._track = _candidate_to_track(selected, frame.shape)
                return self._track

        if self._track is None:
            return None
        missing_steps = self._track.missing_steps + 1
        if missing_steps > self.lost_tolerance_steps:
            self._track = None
            return None
        self._track = VisualTargetTrack(
            bbox_xyxy=self._track.bbox_xyxy,
            center_x_norm=self._track.center_x_norm,
            center_y_norm=self._track.center_y_norm,
            area_ratio=self._track.area_ratio,
            confidence=self._track.confidence * self.confidence_decay,
            visible=False,
            missing_steps=missing_steps,
            source=self._track.source,
        )
        return self._track

    def _select_candidate(
        self, candidates: Sequence[DoorCandidate]
    ) -> Optional[DoorCandidate]:
        """Initialize by confidence or preserve identity using IoU matching."""
        if self._track is None:
            return max(
                candidates,
                key=lambda candidate: (
                    candidate.confidence,
                    _bbox_area(candidate.bbox_xyxy),
                ),
            )
        matching_candidates = [
            candidate
            for candidate in candidates
            if _bbox_iou(candidate.bbox_xyxy, self._track.bbox_xyxy)
            >= self.match_iou_threshold
        ]
        if not matching_candidates:
            return None
        return max(
            matching_candidates,
            key=lambda candidate: (
                0.7 * _bbox_iou(candidate.bbox_xyxy, self._track.bbox_xyxy)
                + 0.3 * candidate.confidence,
                candidate.confidence,
            ),
        )


class ReactiveDoorNavExecutor:
    """Observable image-space state machine for local doorway approach."""

    def __init__(self, config: Optional[ReactiveDoorNavConfig] = None):
        """Initialize the observable-only local state machine."""
        self.config = config or ReactiveDoorNavConfig()
        self.reset()

    def reset(self) -> None:
        """Reset all state and image-space history for a new episode."""
        self.state = DoorNavState.SEARCH
        self.termination_reason = None
        self._terminal_decision = None
        self._has_seen_target = False
        self._steps = 0
        self._search_steps = 0
        self._arrival_frames = 0
        self._no_progress_steps = 0
        self._last_area_ratio = None

    def step(
        self,
        observation: NavigationObservation,
        target: Optional[VisualTargetTrack],
    ) -> LocalExecutionDecision:
        """Choose one local action from an observation and optional RGB track."""
        if self._terminal_decision is not None:
            return self._terminal_decision
        observation = NavigationObservation.from_legacy_inputs(observation)
        _validate_navigation_context(observation.navigation_context)
        self._steps += 1
        if self._steps > self.config.max_episode_steps:
            return self._fail(
                DoorNavTerminationReason.EPISODE_STEP_LIMIT,
                "episode step limit reached",
                target,
            )

        if self._collision_observed(observation.navigation_context):
            self._arrival_frames = 0
            self._reset_progress()
            self.state = DoorNavState.AVOID
            return LocalExecutionDecision(
                action=self._collision_recovery_turn(
                    observation.navigation_context
                ),
                state=self.state,
                reason="observable collision feedback triggered local avoidance",
                target_track=target,
                obstacle_avoidance_active=True,
            )

        if target is None:
            self._arrival_frames = 0
            self._reset_progress()
            if self._has_seen_target:
                return self._fail(
                    DoorNavTerminationReason.TARGET_LOST,
                    "tracked doorway exceeded the missing-frame tolerance",
                    None,
                )
            self._search_steps += 1
            if self._search_steps > self.config.max_search_steps:
                return self._fail(
                    DoorNavTerminationReason.SEARCH_STEP_LIMIT,
                    "doorway was not found within the search budget",
                    None,
                )
            self.state = DoorNavState.SEARCH
            return LocalExecutionDecision(
                action="turn_left",
                state=self.state,
                reason="rotate locally to search for a visible doorway",
                target_track=None,
                obstacle_avoidance_active=False,
            )

        self._has_seen_target = True
        self._search_steps = 0
        if not target.visible:
            self._arrival_frames = 0
            self._reset_progress()
            if target.missing_steps > self.config.target_lost_tolerance_steps:
                return self._fail(
                    DoorNavTerminationReason.TARGET_LOST,
                    "tracked doorway exceeded the missing-frame tolerance",
                    target,
                )
            self.state = DoorNavState.REACQUIRE
            return LocalExecutionDecision(
                action=_turn_toward(target.center_x_norm),
                state=self.state,
                reason="turn toward the last observed doorway position",
                target_track=target,
                obstacle_avoidance_active=False,
            )

        if target.confidence < self.config.grounding_confidence_threshold:
            self._arrival_frames = 0
            self._reset_progress()
            self.state = DoorNavState.TRACK
            return LocalExecutionDecision(
                action=_turn_toward(target.center_x_norm),
                state=self.state,
                reason=(
                    "collect another view before approaching a "
                    "low-confidence target"
                ),
                target_track=target,
                obstacle_avoidance_active=False,
            )

        if (
            abs(target.center_x_norm)
            > self.config.arrival_center_tolerance_norm
        ):
            self._arrival_frames = 0
            self._reset_progress()
            self.state = DoorNavState.TRACK
            return LocalExecutionDecision(
                action=_turn_toward(target.center_x_norm),
                state=self.state,
                reason="center the tracked doorway in the current image",
                target_track=target,
                obstacle_avoidance_active=False,
            )

        self._update_progress(target.area_ratio)
        if self._no_progress_steps > self.config.no_progress_tolerance_steps:
            self._reset_progress()
            self._arrival_frames = 0
            self.state = DoorNavState.REACQUIRE
            return LocalExecutionDecision(
                action="turn_left",
                state=self.state,
                reason=(
                    "image-space target size stopped increasing; "
                    "reacquire a better view"
                ),
                target_track=target,
                obstacle_avoidance_active=False,
            )

        if target.area_ratio >= self.config.arrival_area_ratio_threshold:
            self._arrival_frames += 1
            if self._arrival_frames >= self.config.arrival_confirm_frames:
                self.state = DoorNavState.STOP
                self.termination_reason = DoorNavTerminationReason.REACHED_DOOR
                self._terminal_decision = LocalExecutionDecision(
                    action="stop",
                    state=self.state,
                    reason=(
                        "centered doorway satisfied the unvalidated engineering "
                        "default image-space arrival rule"
                    ),
                    target_track=target,
                    obstacle_avoidance_active=False,
                    termination_reason=self.termination_reason,
                )
                return self._terminal_decision
            self.state = DoorNavState.VERIFY
            return LocalExecutionDecision(
                action="move_forward",
                state=self.state,
                reason="arrival cue observed; require consecutive confirmation frames",
                target_track=target,
                obstacle_avoidance_active=False,
            )

        self._arrival_frames = 0
        self.state = DoorNavState.APPROACH
        return LocalExecutionDecision(
            action="move_forward",
            state=self.state,
            reason="doorway is centered but remains below the arrival area threshold",
            target_track=target,
            obstacle_avoidance_active=False,
        )

    def _update_progress(self, area_ratio: float) -> None:
        """Update no-progress counts from consecutive image-area observations."""
        if self._last_area_ratio is None:
            self._no_progress_steps = 0
        elif (
            area_ratio <= self._last_area_ratio + self.config.min_area_progress
        ):
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        self._last_area_ratio = area_ratio

    def _reset_progress(self) -> None:
        """Clear image-area history when centered visible tracking is interrupted."""
        self._no_progress_steps = 0
        self._last_area_ratio = None

    def _fail(
        self,
        reason: DoorNavTerminationReason,
        detail: str,
        target: Optional[VisualTargetTrack],
    ) -> LocalExecutionDecision:
        """Enter a sticky FAILED state with one auditable reason."""
        self.state = DoorNavState.FAILED
        self.termination_reason = reason
        self._terminal_decision = LocalExecutionDecision(
            action="stop",
            state=self.state,
            reason=detail,
            target_track=target,
            obstacle_avoidance_active=False,
            termination_reason=reason,
        )
        return self._terminal_decision

    @staticmethod
    def _collision_observed(context: Mapping[str, Any]) -> bool:
        """Read only observable collision flags from local context."""
        return bool(
            context.get("collided") or context.get("previous_collision")
        )

    def _collision_recovery_turn(self, context: Mapping[str, Any]) -> str:
        """Choose a deterministic target-free collision recovery turn."""
        if not self.config.use_depth_for_avoidance:
            return "turn_left"
        left = _finite_float(context.get("depth_left_m"))
        right = _finite_float(context.get("depth_right_m"))
        if left is not None and right is not None:
            return "turn_left" if left >= right else "turn_right"
        return "turn_left"


class ReactiveDoorNavPolicy:
    """Compose grounding, tracking, and reactive execution as one policy."""

    def __init__(
        self,
        grounder=None,
        tracker=None,
        executor=None,
        config: Optional[ReactiveDoorNavConfig] = None,
    ):
        """Compose replaceable grounder, tracker, and executor components."""
        self.config = config or ReactiveDoorNavConfig()
        self.grounder = (
            grounder
            if grounder is not None
            else OpenCVDoorGrounder(
                min_area_ratio=self.config.grounder_min_area_ratio,
                max_area_ratio=self.config.grounder_max_area_ratio,
                min_aspect_ratio=self.config.grounder_min_aspect_ratio,
                min_height_ratio=self.config.grounder_min_height_ratio,
                nms_iou_threshold=self.config.grounder_nms_iou_threshold,
                max_candidates=self.config.grounder_max_candidates,
            )
        )
        self.tracker = (
            tracker
            if tracker is not None
            else IoUVisualTargetTracker(
                lost_tolerance_steps=self.config.target_lost_tolerance_steps,
                confidence_decay=self.config.tracker_confidence_decay,
                match_iou_threshold=self.config.tracker_match_iou_threshold,
            )
        )
        self.executor = (
            executor
            if executor is not None
            else ReactiveDoorNavExecutor(self.config)
        )
        self._next_step = 0

    def start_episode(self, episode_id: str) -> None:
        """Reset component and implicit-step state for one episode."""
        del episode_id
        self.tracker.reset()
        self.executor.reset()
        self._next_step = 0

    def predict(
        self,
        observation,
        instruction=None,
        step=None,
        navigation_context=None,
    ) -> PolicyOutput:
        """Run grounding, tracking, and execution for one RGB observation."""
        try:
            observation = NavigationObservation.from_legacy_inputs(
                observation,
                instruction,
                step,
                navigation_context,
            )
            effective_step = self._validate_observation(observation)
        except Exception as exc:
            return self._failure_output(
                DoorNavTerminationReason.INVALID_OBSERVATION,
                exc,
            )

        candidates = []
        target = None
        try:
            detected_candidates = list(
                self.grounder.detect(
                    observation.rgb,
                    observation.instruction,
                )
            )
            for candidate in detected_candidates:
                if not isinstance(candidate, DoorCandidate):
                    raise ValueError(
                        "grounder must return DoorCandidate values"
                    )
                if (
                    candidate.confidence
                    >= self.config.grounding_confidence_threshold
                ):
                    candidates.append(candidate)
            target = self.tracker.update(
                observation.rgb,
                candidates,
                effective_step,
            )
            self._next_step = effective_step + 1
            decision = self.executor.step(observation, target)
            if not isinstance(decision, LocalExecutionDecision):
                raise ValueError(
                    "executor must return a LocalExecutionDecision"
                )
        except Exception as exc:
            self._next_step = effective_step + 1
            return self._failure_output(
                DoorNavTerminationReason.POLICY_ERROR,
                exc,
                candidates=candidates,
                target=target,
            )

        termination_reason = (
            decision.termination_reason.value
            if decision.termination_reason is not None
            else None
        )
        failed = decision.state == DoorNavState.FAILED
        metadata = {
            "baseline": "reactive_doornav_b1",
            "state": decision.state.value,
            "reason": decision.reason,
            "candidate_count": len(candidates),
            "candidates": [asdict(candidate) for candidate in candidates],
            "target_track": (
                asdict(decision.target_track)
                if decision.target_track is not None
                else None
            ),
            "obstacle_avoidance_active": decision.obstacle_avoidance_active,
            "termination_reason": termination_reason,
            "arrival_rule": {
                "status": "unvalidated_engineering_default",
                "area_ratio_threshold": (
                    self.config.arrival_area_ratio_threshold
                ),
                "center_tolerance_norm": (
                    self.config.arrival_center_tolerance_norm
                ),
                "confirm_frames": self.config.arrival_confirm_frames,
            },
        }
        if failed:
            metadata["error"] = decision.reason
        raw_text = json.dumps(
            {
                "action": None if failed else decision.action,
                "state": decision.state.value,
                "reason": decision.reason,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return PolicyOutput(
            action=None if failed else decision.action,
            raw_text=raw_text,
            is_valid=not failed,
            termination_reason=termination_reason,
            metadata=metadata,
        )

    def _validate_observation(self, observation: NavigationObservation) -> int:
        """Validate one RGB-first policy input and return its effective step."""
        _as_rgb_uint8(observation.rgb)
        if (
            not isinstance(observation.instruction, str)
            or not observation.instruction.strip()
        ):
            raise ValueError("instruction must be a non-empty string")
        _validate_navigation_context(observation.navigation_context)
        if observation.step is None:
            return self._next_step
        if type(observation.step) is not int or observation.step < 0:
            raise ValueError("step must be a non-negative integer or None")
        return observation.step

    @staticmethod
    def _failure_output(
        reason: DoorNavTerminationReason,
        error: Exception,
        candidates: Optional[Sequence[DoorCandidate]] = None,
        target: Optional[VisualTargetTrack] = None,
    ) -> PolicyOutput:
        """Return one explicit terminal failure without inventing an action."""
        error_text = f"{type(error).__name__}: {error}"
        termination_reason = reason.value
        metadata = {
            "baseline": "reactive_doornav_b1",
            "state": DoorNavState.FAILED.value,
            "reason": error_text,
            "candidate_count": len(candidates or ()),
            "candidates": [
                asdict(candidate) for candidate in (candidates or ())
            ],
            "target_track": asdict(target) if target is not None else None,
            "obstacle_avoidance_active": False,
            "termination_reason": termination_reason,
            "error": error_text,
        }
        return PolicyOutput(
            action=None,
            raw_text=json.dumps(
                {
                    "action": None,
                    "state": DoorNavState.FAILED.value,
                    "reason": error_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            is_valid=False,
            termination_reason=termination_reason,
            metadata=metadata,
        )


def _as_rgb_uint8(rgb: Any) -> np.ndarray:
    """Validate RGB shape and range, then return contiguous uint8 RGB."""
    if not isinstance(rgb, np.ndarray):
        raise ValueError("rgb must be a numpy.ndarray")
    frame = rgb
    if (
        frame.ndim != 3
        or frame.shape[0] <= 0
        or frame.shape[1] <= 0
        or frame.shape[2] not in {3, 4}
    ):
        raise ValueError(
            "rgb must have non-empty shape H x W x 3 or H x W x 4"
        )
    frame = frame[:, :, :3]
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        if not np.all(np.isfinite(frame)):
            raise ValueError("floating-point rgb values must all be finite")
        if float(np.min(frame)) < 0.0 or float(np.max(frame)) > 1.0:
            raise ValueError("floating-point rgb values must be in [0, 1]")
        frame = np.rint(frame * 255.0).astype(np.uint8)
    elif np.issubdtype(frame.dtype, np.integer) and frame.dtype != np.bool_:
        if int(np.min(frame)) < 0 or int(np.max(frame)) > 255:
            raise ValueError("integer rgb values must be in [0, 255]")
        frame = frame.astype(np.uint8)
    else:
        raise ValueError("rgb dtype must be uint8 or a real numeric type")
    return np.ascontiguousarray(frame)


def _candidate_to_track(
    candidate: DoorCandidate,
    frame_shape: Tuple[int, ...],
) -> VisualTargetTrack:
    """Normalize one image-space candidate into a visible target track."""
    height, width = frame_shape[:2]
    _validate_bbox_in_frame(candidate.bbox_xyxy, frame_shape)
    x1, y1, x2, y2 = candidate.bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return VisualTargetTrack(
        bbox_xyxy=candidate.bbox_xyxy,
        center_x_norm=2.0 * center_x / float(width) - 1.0,
        center_y_norm=2.0 * center_y / float(height) - 1.0,
        area_ratio=_bbox_area(candidate.bbox_xyxy) / float(width * height),
        confidence=candidate.confidence,
        visible=True,
        missing_steps=0,
        source=candidate.source,
    )


def _validate_bbox_in_frame(
    bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, ...]
) -> None:
    """Reject image-space candidates that extend outside the current RGB frame."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            "candidate bbox must stay within current image bounds"
        )


def _non_maximum_suppression(
    candidates: Sequence[DoorCandidate],
    iou_threshold: float,
    max_candidates: int,
) -> Sequence[DoorCandidate]:
    """Keep high-confidence candidates whose IoU stays below the threshold."""
    remaining = sorted(
        candidates, key=lambda item: item.confidence, reverse=True
    )
    selected = []
    while remaining and len(selected) < max_candidates:
        candidate = remaining.pop(0)
        selected.append(candidate)
        remaining = [
            other
            for other in remaining
            if _bbox_iou(candidate.bbox_xyxy, other.bbox_xyxy) <= iou_threshold
        ]
    return selected


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Return non-negative image-space bounding-box area."""
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def _bbox_iou(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> float:
    """Return intersection over union for two image-space boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = _bbox_area((x1, y1, x2, y2))
    if intersection == 0:
        return 0.0
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / float(union) if union else 0.0


def _turn_toward(center_x_norm: float) -> str:
    """Turn toward the last observable horizontal image position."""
    return "turn_left" if center_x_norm < 0.0 else "turn_right"


def _finite_float(value: Any) -> Optional[float]:
    """Convert one optional safety cue to a finite float when possible."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _find_privileged_context_path(
    value: Any, path: str = "navigation_context"
) -> Optional[str]:
    """Find a forbidden target-state key in nested policy context."""
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            key_text = str(key)
            nested_path = f"{path}.{key_text}"
            if key_text.casefold() in PRIVILEGED_FIELD_DENYLIST:
                return nested_path
            match = _find_privileged_context_path(nested_value, nested_path)
            if match is not None:
                return match
    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            match = _find_privileged_context_path(
                nested_value,
                f"{path}[{index}]",
            )
            if match is not None:
                return match
    return None


def _validate_navigation_context(context: Any) -> None:
    """Reject hidden target state and unsupported local context fields."""
    if not isinstance(context, Mapping):
        raise ValueError("navigation_context must be a mapping")
    privileged_path = _find_privileged_context_path(context)
    if privileged_path is not None:
        raise ValueError(
            "navigation_context contains privileged target field: "
            f"{privileged_path}"
        )
    unknown_fields = sorted(
        str(key)
        for key in context
        if key not in _ALLOWED_NAVIGATION_CONTEXT_FIELDS
    )
    if unknown_fields:
        raise ValueError(
            "navigation_context contains unsupported or privileged fields: "
            + ", ".join(unknown_fields)
        )
    building_prior = context.get("building_prior")
    if building_prior is None:
        return
    if not isinstance(building_prior, Mapping):
        raise ValueError("navigation_context.building_prior must be a mapping")
    unknown_prior_fields = sorted(
        str(key)
        for key in building_prior
        if key not in ALLOWED_BUILDING_PRIOR_FIELDS
    )
    if unknown_prior_fields:
        raise ValueError(
            "navigation_context.building_prior contains unsupported or "
            "privileged fields: " + ", ".join(unknown_prior_fields)
        )
    for field, value in building_prior.items():
        if field in _B1_TEXT_BUILDING_PRIOR_FIELDS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "navigation_context.building_prior."
                    f"{field} must be non-empty public text"
                )
            continue
        if field in _B1_TEXT_LIST_BUILDING_PRIOR_FIELDS:
            if not isinstance(value, (list, tuple)) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(
                    "navigation_context.building_prior."
                    f"{field} must be a list of public text labels"
                )
            continue
        raise ValueError(
            "navigation_context.building_prior has no B1 value schema for "
            f"{field}"
        )
