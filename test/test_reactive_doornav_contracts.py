import math
import unittest
from dataclasses import FrozenInstanceError

from habitat_vln.baselines.reactive_doornav import (
    DoorCandidate,
    DoorGrounder,
    DoorNavState,
    DoorNavTerminationReason,
    LocalExecutionDecision,
    ReactiveVisualExecutor,
    VisualTargetTrack,
    VisualTargetTracker,
)
from habitat_vln.core import NavigationObservation


def make_visual_track(**overrides):
    """Build a valid image-space track with optional field overrides."""
    values = {
        "bbox_xyxy": (10, 20, 110, 220),
        "center_x_norm": -0.1,
        "center_y_norm": 0.2,
        "area_ratio": 0.18,
        "confidence": 0.85,
        "visible": True,
        "missing_steps": 0,
        "source": "fake_tracker",
    }
    values.update(overrides)
    return VisualTargetTrack(**values)


class DoorNavContractTest(unittest.TestCase):
    def test_door_nav_states_are_stable_string_values(self):
        self.assertEqual(
            [state.value for state in DoorNavState],
            [
                "search",
                "track",
                "approach",
                "avoid",
                "reacquire",
                "verify",
                "stop",
                "failed",
            ],
        )

    def test_termination_reasons_cover_expected_local_outcomes(self):
        self.assertEqual(
            {reason.value for reason in DoorNavTerminationReason},
            {
                "reached_door",
                "target_not_found",
                "target_lost",
                "invalid_visual_track",
                "local_path_blocked",
                "no_progress",
                "search_step_limit",
                "episode_step_limit",
                "invalid_observation",
                "policy_error",
            },
        )

    def test_door_candidate_accepts_valid_observable_detection(self):
        candidate = DoorCandidate(
            bbox_xyxy=(10, 20, 110, 220),
            confidence=0.85,
            label="door",
            source="fake_grounder",
        )

        self.assertEqual(candidate.bbox_xyxy, (10, 20, 110, 220))
        self.assertEqual(candidate.confidence, 0.85)
        self.assertEqual(candidate.label, "door")
        self.assertEqual(candidate.source, "fake_grounder")

    def test_door_candidate_rejects_invalid_bbox(self):
        invalid_bboxes = [
            (10, 20, 10, 220),
            (10, 20, 110, 20),
            (10, 20, 110),
            (10, 20, 110, 220.0),
        ]

        for bbox in invalid_bboxes:
            with self.subTest(bbox=bbox):
                with self.assertRaises(ValueError):
                    DoorCandidate(bbox, 0.8, "door", "fake_grounder")

    def test_door_candidate_rejects_non_finite_confidence(self):
        for confidence in [math.nan, math.inf, -math.inf]:
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    DoorCandidate(
                        (10, 20, 110, 220),
                        confidence,
                        "door",
                        "fake_grounder",
                    )

    def test_door_candidate_rejects_out_of_range_confidence(self):
        for confidence in [-0.01, 1.01]:
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    DoorCandidate(
                        (10, 20, 110, 220),
                        confidence,
                        "door",
                        "fake_grounder",
                    )

    def test_door_candidate_rejects_empty_label_or_source(self):
        invalid_values = [
            {"label": "", "source": "fake_grounder"},
            {"label": "door", "source": "  "},
        ]

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DoorCandidate(
                        (10, 20, 110, 220),
                        0.8,
                        values["label"],
                        values["source"],
                    )

    def test_visual_target_track_accepts_valid_image_space_target(self):
        track = make_visual_track()

        self.assertEqual(track.bbox_xyxy, (10, 20, 110, 220))
        self.assertEqual(track.center_x_norm, -0.1)
        self.assertEqual(track.center_y_norm, 0.2)
        self.assertEqual(track.area_ratio, 0.18)
        self.assertEqual(track.confidence, 0.85)
        self.assertTrue(track.visible)
        self.assertEqual(track.missing_steps, 0)
        self.assertEqual(track.source, "fake_tracker")

    def test_visual_target_track_accepts_recently_missing_target(self):
        track = make_visual_track(visible=False, missing_steps=2)

        self.assertFalse(track.visible)
        self.assertEqual(track.missing_steps, 2)

    def test_visual_target_track_rejects_invalid_bbox(self):
        invalid_bboxes = [
            (10, 20, 10, 220),
            (10, 20, 110, 20),
            (10, 20, 110),
            (10, 20, 110, 220.0),
        ]

        for bbox in invalid_bboxes:
            with self.subTest(bbox=bbox):
                with self.assertRaises(ValueError):
                    make_visual_track(bbox_xyxy=bbox)

    def test_visual_target_track_rejects_invalid_horizontal_center(self):
        for value in [-1.01, 1.01, math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_visual_track(center_x_norm=value)

    def test_visual_target_track_rejects_invalid_vertical_center(self):
        for value in [-1.01, 1.01, math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_visual_track(center_y_norm=value)

    def test_visual_target_track_rejects_invalid_area_ratio(self):
        for value in [0.0, -0.01, 1.01, math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_visual_track(area_ratio=value)

    def test_visual_target_track_rejects_invalid_confidence(self):
        for value in [-0.01, 1.01, math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_visual_track(confidence=value)

    def test_visual_target_track_rejects_invalid_missing_steps(self):
        for value in [-1, 0.5, True]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_visual_track(missing_steps=value)

    def test_visible_track_requires_zero_missing_steps(self):
        with self.assertRaises(ValueError):
            make_visual_track(visible=True, missing_steps=1)

    def test_missing_track_requires_positive_missing_steps(self):
        with self.assertRaises(ValueError):
            make_visual_track(visible=False, missing_steps=0)

    def test_visual_target_track_rejects_invalid_visible_or_source(self):
        for overrides in [
            {"visible": 1},
            {"source": ""},
            {"source": "  "},
        ]:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_visual_track(**overrides)

    def test_local_execution_decision_rejects_unknown_action(self):
        with self.assertRaises(ValueError):
            LocalExecutionDecision(
                action="move_backward",
                state=DoorNavState.APPROACH,
                reason="door remains visible",
                target_track=make_visual_track(),
                obstacle_avoidance_active=False,
            )

    def test_contract_dataclasses_are_frozen(self):
        candidate = DoorCandidate(
            (10, 20, 110, 220), 0.8, "door", "fake_grounder"
        )
        track = make_visual_track()
        decision = LocalExecutionDecision(
            action="move_forward",
            state=DoorNavState.APPROACH,
            reason="door remains centered",
            target_track=track,
            obstacle_avoidance_active=False,
        )

        for instance, field, value in [
            (candidate, "confidence", 0.1),
            (track, "area_ratio", 0.3),
            (decision, "action", "stop"),
        ]:
            with self.subTest(instance=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field, value)

    def test_protocols_accept_minimal_structural_implementations(self):
        class FakeGrounder:
            def detect(self, rgb, instruction):
                del rgb, instruction
                return [
                    DoorCandidate(
                        (10, 20, 110, 220),
                        0.8,
                        "door",
                        "fake_grounder",
                    )
                ]

        class FakeTracker:
            def reset(self):
                return None

            def update(self, rgb, candidates, step):
                del rgb, candidates, step
                return make_visual_track()

        class FakeExecutor:
            def reset(self):
                return None

            def step(self, observation, target):
                del observation
                return LocalExecutionDecision(
                    action="move_forward",
                    state=DoorNavState.APPROACH,
                    reason="door remains centered",
                    target_track=target,
                    obstacle_avoidance_active=False,
                )

        observation = NavigationObservation(
            rgb="rgb",
            depth=None,
            instruction="Approach the visible door.",
        )
        grounder = FakeGrounder()
        tracker = FakeTracker()
        executor = FakeExecutor()

        self.assertIsInstance(grounder, DoorGrounder)
        self.assertIsInstance(tracker, VisualTargetTracker)
        self.assertIsInstance(executor, ReactiveVisualExecutor)
        candidates = grounder.detect("rgb", observation.instruction)
        target = tracker.update("rgb", candidates, step=0)
        self.assertEqual(target.source, "fake_tracker")
        self.assertEqual(
            executor.step(observation, target).action,
            "move_forward",
        )

    def test_executor_accepts_navigation_observation_without_depth(self):
        class FakeExecutor:
            def __init__(self):
                self.received_depth = "not_called"

            def reset(self):
                return None

            def step(self, observation, target):
                self.received_depth = observation.depth
                return LocalExecutionDecision(
                    action="move_forward",
                    state=DoorNavState.APPROACH,
                    reason="RGB target remains observable",
                    target_track=target,
                    obstacle_avoidance_active=False,
                )

        observation = NavigationObservation(
            rgb="rgb",
            instruction="Approach the visible door.",
            depth=None,
        )
        executor = FakeExecutor()

        decision = executor.step(observation, make_visual_track())

        self.assertIsNone(executor.received_depth)
        self.assertEqual(decision.action, "move_forward")

    def test_executor_can_express_search_when_target_is_none(self):
        class FakeExecutor:
            def reset(self):
                return None

            def step(self, observation, target):
                del observation, target
                return LocalExecutionDecision(
                    action="turn_left",
                    state=DoorNavState.SEARCH,
                    reason="no visual target is currently tracked",
                    target_track=None,
                    obstacle_avoidance_active=False,
                )

        decision = FakeExecutor().step(
            NavigationObservation(
                rgb="rgb",
                instruction="Find the visible door.",
                depth=None,
            ),
            None,
        )

        self.assertEqual(decision.state, DoorNavState.SEARCH)
        self.assertIsNone(decision.target_track)


if __name__ == "__main__":
    unittest.main()
