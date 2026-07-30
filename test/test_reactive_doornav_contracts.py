import math
import unittest
from dataclasses import FrozenInstanceError

from habitat_vln.baselines.reactive_doornav import (
    DoorCandidate,
    DoorGrounder,
    DoorNavState,
    DoorNavTerminationReason,
    LocalExecutionDecision,
    LocalNavigationExecutor,
    LocalSubgoal,
)
from habitat_vln.core import NavigationObservation


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
                "invalid_depth",
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

    def test_local_subgoal_accepts_observable_robot_relative_target(self):
        subgoal = LocalSubgoal(
            target_type="doorway",
            relative_x_m=1.8,
            relative_y_m=-0.25,
            desired_heading_rad=0.1,
            stop_distance_m=0.45,
            confidence=0.75,
            source="rgbd_projection",
        )

        self.assertEqual(subgoal.target_type, "doorway")
        self.assertEqual(subgoal.relative_x_m, 1.8)
        self.assertEqual(subgoal.relative_y_m, -0.25)
        self.assertEqual(subgoal.desired_heading_rad, 0.1)
        self.assertEqual(subgoal.stop_distance_m, 0.45)
        self.assertEqual(subgoal.confidence, 0.75)
        self.assertEqual(subgoal.source, "rgbd_projection")

    def test_local_subgoal_allows_target_behind_robot_for_recovery(self):
        subgoal = LocalSubgoal(
            target_type="last_visible_doorway",
            relative_x_m=-0.5,
            relative_y_m=0.2,
            desired_heading_rad=None,
            stop_distance_m=0.4,
            confidence=0.5,
            source="short_rgbd_history",
        )

        self.assertEqual(subgoal.relative_x_m, -0.5)

    def test_local_subgoal_rejects_non_positive_stop_distance(self):
        for stop_distance_m in [0.0, -0.1]:
            with self.subTest(stop_distance_m=stop_distance_m):
                with self.assertRaises(ValueError):
                    LocalSubgoal(
                        "doorway",
                        1.0,
                        0.0,
                        None,
                        stop_distance_m,
                        0.8,
                        "rgbd_projection",
                    )

    def test_local_subgoal_rejects_non_finite_values(self):
        valid_values = {
            "target_type": "doorway",
            "relative_x_m": 1.0,
            "relative_y_m": 0.0,
            "desired_heading_rad": 0.0,
            "stop_distance_m": 0.4,
            "confidence": 0.8,
            "source": "rgbd_projection",
        }

        for field in [
            "relative_x_m",
            "relative_y_m",
            "desired_heading_rad",
            "stop_distance_m",
        ]:
            for value in [math.nan, math.inf, -math.inf]:
                with self.subTest(field=field, value=value):
                    invalid_values = dict(valid_values)
                    invalid_values[field] = value
                    with self.assertRaises(ValueError):
                        LocalSubgoal(**invalid_values)

    def test_local_subgoal_rejects_out_of_range_confidence(self):
        for confidence in [-0.01, 1.01, math.nan]:
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    LocalSubgoal(
                        "doorway",
                        1.0,
                        0.0,
                        None,
                        0.4,
                        confidence,
                        "rgbd_projection",
                    )

    def test_local_subgoal_rejects_empty_target_type_or_source(self):
        invalid_values = [
            {"target_type": "", "source": "rgbd_projection"},
            {"target_type": "doorway", "source": "  "},
        ]

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    LocalSubgoal(
                        values["target_type"],
                        1.0,
                        0.0,
                        None,
                        0.4,
                        0.8,
                        values["source"],
                    )

    def test_local_execution_decision_rejects_unknown_action(self):
        with self.assertRaises(ValueError):
            LocalExecutionDecision(
                action="move_backward",
                state=DoorNavState.APPROACH,
                reason="door remains visible",
                subgoal=None,
                target_visible=True,
                obstacle_avoidance_active=False,
            )

    def test_contract_dataclasses_are_frozen(self):
        candidate = DoorCandidate(
            (10, 20, 110, 220), 0.8, "door", "fake_grounder"
        )
        subgoal = LocalSubgoal(
            "doorway", 1.0, 0.0, None, 0.4, 0.8, "rgbd_projection"
        )
        decision = LocalExecutionDecision(
            "move_forward",
            DoorNavState.APPROACH,
            "safe forward motion",
            subgoal,
            True,
            False,
        )

        for instance, field, value in [
            (candidate, "confidence", 0.1),
            (subgoal, "relative_x_m", 2.0),
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

        class FakeExecutor:
            def reset(self):
                return None

            def step(self, observation, subgoal):
                del observation
                return LocalExecutionDecision(
                    "move_forward",
                    DoorNavState.APPROACH,
                    "safe forward motion",
                    subgoal,
                    True,
                    False,
                )

        observation = NavigationObservation(
            rgb="rgb",
            depth="depth",
            instruction="Approach the visible door.",
        )
        subgoal = LocalSubgoal(
            "doorway", 1.0, 0.0, None, 0.4, 0.8, "rgbd_projection"
        )
        grounder = FakeGrounder()
        executor = FakeExecutor()

        self.assertIsInstance(grounder, DoorGrounder)
        self.assertIsInstance(executor, LocalNavigationExecutor)
        self.assertEqual(
            grounder.detect("rgb", observation.instruction)[0].label, "door"
        )
        self.assertEqual(
            executor.step(observation, subgoal).action, "move_forward"
        )


if __name__ == "__main__":
    unittest.main()
