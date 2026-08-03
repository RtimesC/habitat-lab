"""Executable observable-only components for the reactive DoorNav baseline."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from ...core import NavigationObservation, PolicyOutput
from .contracts import (
    DoorCandidate,
    DoorNavState,
    DoorNavTerminationReason,
    LocalExecutionDecision,
    VisualTargetTrack,
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

    def __post_init__(self):
        integer_fields = {
            "max_search_steps": self.max_search_steps,
            "max_episode_steps": self.max_episode_steps,
            "arrival_confirm_frames": self.arrival_confirm_frames,
            "target_lost_tolerance_steps": self.target_lost_tolerance_steps,
            "no_progress_tolerance_steps": self.no_progress_tolerance_steps,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        bounded_fields = {
            "arrival_area_ratio_threshold": self.arrival_area_ratio_threshold,
            "arrival_center_tolerance_norm": self.arrival_center_tolerance_norm,
            "grounding_confidence_threshold": self.grounding_confidence_threshold,
        }
        for name, value in bounded_fields.items():
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")

        if not math.isfinite(self.min_area_progress) or self.min_area_progress < 0.0:
            raise ValueError("min_area_progress must be finite and non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        """Build a config from the versioned baseline specification."""
        field_names = set(cls.__dataclass_fields__)
        return cls(
            **{
                key: value
                for key, value in values.items()
                if key in field_names
            }
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
        max_area_ratio: float = 0.85,
        min_aspect_ratio: float = 1.2,
        min_height_ratio: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_candidates: int = 5,
    ):
        if not 0.0 < min_area_ratio < max_area_ratio <= 1.0:
            raise ValueError("area-ratio bounds must satisfy 0 < min < max <= 1")
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
    ):
        if type(lost_tolerance_steps) is not int or lost_tolerance_steps <= 0:
            raise ValueError("lost_tolerance_steps must be a positive integer")
        if not 0.0 < confidence_decay <= 1.0:
            raise ValueError("confidence_decay must be in (0, 1]")
        self.lost_tolerance_steps = lost_tolerance_steps
        self.confidence_decay = confidence_decay
        self._track = None
        self._last_step = None

    def reset(self) -> None:
        self._track = None
        self._last_step = None

    def update(
        self,
        rgb: Any,
        candidates: Sequence[DoorCandidate],
        step: int,
    ) -> Optional[VisualTargetTrack]:
        frame = _as_rgb_uint8(rgb)
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._last_step is not None and step < self._last_step:
            raise ValueError("step must not move backwards")
        self._last_step = step

        candidates = list(candidates)
        if candidates:
            selected = self._select_candidate(candidates)
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

    def _select_candidate(self, candidates: Sequence[DoorCandidate]) -> DoorCandidate:
        if self._track is None:
            return max(
                candidates,
                key=lambda candidate: (
                    candidate.confidence,
                    _bbox_area(candidate.bbox_xyxy),
                ),
            )
        return max(
            candidates,
            key=lambda candidate: (
                0.7 * _bbox_iou(candidate.bbox_xyxy, self._track.bbox_xyxy)
                + 0.3 * candidate.confidence,
                candidate.confidence,
            ),
        )


class ReactiveDoorNavExecutor:
    """Observable image-space state machine for local doorway approach."""

    def __init__(self, config: Optional[ReactiveDoorNavConfig] = None):
        self.config = config or ReactiveDoorNavConfig()
        self.reset()

    def reset(self) -> None:
        self.state = DoorNavState.SEARCH
        self.termination_reason = None
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
        observation = NavigationObservation.from_legacy_inputs(observation)
        self._steps += 1
        if self._steps > self.config.max_episode_steps:
            return self._fail(
                DoorNavTerminationReason.EPISODE_STEP_LIMIT,
                "episode step limit reached",
                target,
            )

        if self._collision_observed(observation.navigation_context):
            self._arrival_frames = 0
            self.state = DoorNavState.AVOID
            return LocalExecutionDecision(
                action=self._safer_turn(observation.navigation_context),
                state=self.state,
                reason="observable collision feedback triggered local avoidance",
                target_track=target,
                obstacle_avoidance_active=True,
            )

        if target is None:
            self._arrival_frames = 0
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

        self._search_steps = 0
        if not target.visible:
            self._arrival_frames = 0
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

        if abs(target.center_x_norm) > self.config.arrival_center_tolerance_norm:
            self._arrival_frames = 0
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
            self._no_progress_steps = 0
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
                return LocalExecutionDecision(
                    action="stop",
                    state=self.state,
                    reason="centered doorway satisfied the image-space arrival rule",
                    target_track=target,
                    obstacle_avoidance_active=False,
                )
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
        if self._last_area_ratio is None:
            self._no_progress_steps = 0
        elif area_ratio <= self._last_area_ratio + self.config.min_area_progress:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        self._last_area_ratio = area_ratio

    def _fail(
        self,
        reason: DoorNavTerminationReason,
        detail: str,
        target: Optional[VisualTargetTrack],
    ) -> LocalExecutionDecision:
        self.state = DoorNavState.FAILED
        self.termination_reason = reason
        return LocalExecutionDecision(
            action="stop",
            state=self.state,
            reason=detail,
            target_track=target,
            obstacle_avoidance_active=False,
        )

    @staticmethod
    def _collision_observed(context: Mapping[str, Any]) -> bool:
        return bool(context.get("collided") or context.get("previous_collision"))

    @staticmethod
    def _safer_turn(context: Mapping[str, Any]) -> str:
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
        self.config = config or ReactiveDoorNavConfig()
        self.grounder = grounder or OpenCVDoorGrounder()
        self.tracker = tracker or IoUVisualTargetTracker(
            lost_tolerance_steps=self.config.target_lost_tolerance_steps
        )
        self.executor = executor or ReactiveDoorNavExecutor(self.config)

    def start_episode(self, episode_id: str) -> None:
        del episode_id
        self.tracker.reset()
        self.executor.reset()

    def predict(
        self,
        observation,
        instruction=None,
        step=None,
        navigation_context=None,
    ) -> PolicyOutput:
        observation = NavigationObservation.from_legacy_inputs(
            observation,
            instruction,
            step,
            navigation_context,
        )
        candidates = list(
            self.grounder.detect(observation.rgb, observation.instruction)
        )
        target = self.tracker.update(
            observation.rgb,
            candidates,
            observation.step or 0,
        )
        decision = self.executor.step(observation, target)
        termination_reason = (
            self.executor.termination_reason.value
            if self.executor.termination_reason is not None
            else None
        )
        metadata = {
            "baseline": "reactive_doornav_b1",
            "state": decision.state.value,
            "reason": decision.reason,
            "candidate_count": len(candidates),
            "candidates": [asdict(candidate) for candidate in candidates],
            "target_track": asdict(target) if target is not None else None,
            "obstacle_avoidance_active": decision.obstacle_avoidance_active,
        }
        raw_text = json.dumps(
            {
                "action": decision.action,
                "state": decision.state.value,
                "reason": decision.reason,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return PolicyOutput(
            action=decision.action,
            raw_text=raw_text,
            is_valid=True,
            termination_reason=termination_reason,
            metadata=metadata,
        )


def _as_rgb_uint8(rgb: Any) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("rgb must have shape H x W x 3 or more channels")
    frame = frame[:, :, :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            maximum = float(np.nanmax(frame)) if frame.size else 0.0
            if maximum <= 1.0:
                frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _candidate_to_track(
    candidate: DoorCandidate,
    frame_shape: Tuple[int, ...],
) -> VisualTargetTrack:
    height, width = frame_shape[:2]
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


def _non_maximum_suppression(
    candidates: Sequence[DoorCandidate],
    iou_threshold: float,
    max_candidates: int,
) -> Sequence[DoorCandidate]:
    remaining = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    selected = []
    while remaining and len(selected) < max_candidates:
        candidate = remaining.pop(0)
        selected.append(candidate)
        remaining = [
            other
            for other in remaining
            if _bbox_iou(candidate.bbox_xyxy, other.bbox_xyxy) < iou_threshold
        ]
    return selected


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def _bbox_iou(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> float:
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
    return "turn_left" if center_x_norm < 0.0 else "turn_right"


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
