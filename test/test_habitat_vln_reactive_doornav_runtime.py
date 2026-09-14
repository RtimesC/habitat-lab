import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from habitat_vln.baselines.reactive_doornav import (
    DoorCandidate,
    DoorNavState,
    DoorNavTerminationReason,
    VisualTargetTrack,
)
from habitat_vln.baselines.reactive_doornav.runtime import (
    IoUVisualTargetTracker,
    OpenCVDoorGrounder,
    ReactiveDoorNavConfig,
    ReactiveDoorNavExecutor,
    ReactiveDoorNavPolicy,
    load_reactive_doornav_spec,
)
from habitat_vln.core import NavigationObservation
from habitat_vln.envs import ACTION_MAP
from habitat_vln.habitat_vln_nav import (
    build_navigation_policy,
    main,
    parse_args,
)
from habitat_vln.policies import VALID_ACTIONS
from habitat_vln.runtime import run_navigation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SPEC_PATH = (
    PROJECT_ROOT / "habitat_vln/baselines/reactive_doornav/baseline_spec.yaml"
)
RUNTIME_CONFIG_PATH = (
    PROJECT_ROOT / "habitat_vln/configs/runtime/reactive_doornav_b1.yaml"
)


def doorway_image(height=120, width=160):
    """Create one synthetic RGB frame with a tall doorway-like outline."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (60, 15), (100, 112), (255, 255, 255), 3)
    return image


def door_candidate(bbox=(10, 10, 40, 80), confidence=0.8):
    """Create one valid synthetic doorway candidate."""
    return DoorCandidate(
        bbox_xyxy=bbox,
        confidence=confidence,
        label="doorway",
        source="synthetic_test",
    )


def visual_track(**overrides):
    """Create one valid image-space track with optional overrides."""
    values = {
        "bbox_xyxy": (35, 10, 65, 80),
        "center_x_norm": 0.0,
        "center_y_norm": -0.1,
        "area_ratio": 0.1,
        "confidence": 0.9,
        "visible": True,
        "missing_steps": 0,
        "source": "synthetic_test",
    }
    values.update(overrides)
    return VisualTargetTrack(**values)


def navigation_observation(step=0, **context):
    """Create one target-free RGB observation for executor tests."""
    return NavigationObservation(
        rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        instruction="Approach the visible doorway.",
        step=step,
        navigation_context=context,
    )


class ReactiveDoorNavConfigTest(unittest.TestCase):
    def test_valid_config_and_mapping_loading(self):
        config = ReactiveDoorNavConfig.from_mapping(
            {
                "max_search_steps": 5,
                "arrival_area_ratio_threshold": 0.25,
            }
        )

        self.assertEqual(config.max_search_steps, 5)
        self.assertEqual(config.arrival_area_ratio_threshold, 0.25)

    def test_invalid_config_ranges_are_rejected(self):
        invalid_values = [
            {"max_search_steps": 0},
            {"arrival_area_ratio_threshold": 0.0},
            {"arrival_center_tolerance_norm": 1.1},
            {"grounding_confidence_threshold": float("nan")},
            {"tracker_match_iou_threshold": 0.0},
            {"grounder_max_candidates": True},
        ]

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ReactiveDoorNavConfig.from_mapping(values)

    def test_mapping_rejects_unknown_parameters(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            ReactiveDoorNavConfig.from_mapping({"ignored_threshold": 0.5})

    def test_versioned_spec_loads_all_parameters_and_semantics(self):
        specification = load_reactive_doornav_spec(BASELINE_SPEC_PATH)

        self.assertEqual(specification.name, "reactive_doornav_b1")
        self.assertEqual(specification.status, "engineering_baseline")
        self.assertGreater(specification.config.max_episode_steps, 0)
        self.assertEqual(
            set(specification.parameter_semantics),
            set(ReactiveDoorNavConfig.__dataclass_fields__),
        )
        self.assertTrue(
            all(specification.parameter_semantics.values()),
            "Every B1 parameter needs a non-empty observable engineering meaning",
        )


class OpenCVDoorGrounderTest(unittest.TestCase):
    def test_synthetic_tall_rectangle_produces_bounded_candidate(self):
        image = doorway_image()

        candidates = OpenCVDoorGrounder().detect(
            image,
            "Approach the visible doorway.",
        )

        self.assertGreaterEqual(len(candidates), 1)
        height, width = image.shape[:2]
        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bbox_xyxy
            self.assertTrue(0 <= x1 < x2 <= width)
            self.assertTrue(0 <= y1 < y2 <= height)
            self.assertEqual(candidate.source, "opencv_vertical_contour")

    def test_invalid_rgb_inputs_are_rejected(self):
        invalid_images = [
            None,
            [[[0, 0, 0]]],
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((0, 10, 3), dtype=np.uint8),
            np.full((10, 10, 3), np.nan, dtype=np.float32),
        ]

        for image in invalid_images:
            with self.subTest(image_type=type(image).__name__):
                with self.assertRaises(ValueError):
                    OpenCVDoorGrounder().detect(image, "Find a doorway.")

    def test_blank_image_returns_no_candidates(self):
        candidates = OpenCVDoorGrounder().detect(
            np.zeros((80, 100, 3), dtype=np.uint8),
            "Find a doorway.",
        )

        self.assertEqual(candidates, [])

    def test_frame_scale_contour_cannot_swallow_the_image(self):
        image = np.zeros((160, 96, 3), dtype=np.uint8)
        cv2.rectangle(image, (5, 5), (90, 155), (255, 255, 255), 4)
        cv2.rectangle(image, (35, 45), (60, 150), (255, 255, 255), 3)

        candidates = OpenCVDoorGrounder().detect(image, "Find a doorway.")

        image_area = image.shape[0] * image.shape[1]
        self.assertTrue(candidates)
        self.assertTrue(
            all(
                (candidate.bbox_xyxy[2] - candidate.bbox_xyxy[0])
                * (candidate.bbox_xyxy[3] - candidate.bbox_xyxy[1])
                / image_area
                <= 0.65
                for candidate in candidates
            )
        )


class IoUVisualTargetTrackerTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_initializes_highest_confidence_target(self):
        tracker = IoUVisualTargetTracker()

        track = tracker.update(
            self.rgb,
            [
                door_candidate((10, 10, 30, 80), 0.7),
                door_candidate((50, 10, 75, 85), 0.9),
            ],
            step=0,
        )

        self.assertEqual(track.bbox_xyxy, (50, 10, 75, 85))
        self.assertTrue(track.visible)

    def test_tracks_same_target_with_iou_and_multiple_candidates(self):
        tracker = IoUVisualTargetTracker()
        tracker.update(
            self.rgb,
            [door_candidate((10, 10, 40, 80), 0.8)],
            step=0,
        )

        track = tracker.update(
            self.rgb,
            [
                door_candidate((12, 10, 42, 80), 0.65),
                door_candidate((70, 10, 95, 85), 0.99),
            ],
            step=1,
        )

        self.assertEqual(track.bbox_xyxy, (12, 10, 42, 80))
        self.assertTrue(track.visible)

    def test_zero_iou_candidate_does_not_switch_identity(self):
        tracker = IoUVisualTargetTracker(lost_tolerance_steps=2)
        tracker.update(
            self.rgb,
            [door_candidate((10, 10, 40, 80), 0.8)],
            step=0,
        )

        track = tracker.update(
            self.rgb,
            [door_candidate((70, 10, 95, 85), 0.99)],
            step=1,
        )

        self.assertEqual(track.bbox_xyxy, (10, 10, 40, 80))
        self.assertFalse(track.visible)
        self.assertEqual(track.missing_steps, 1)

    def test_short_loss_decays_confidence_then_reacquires(self):
        tracker = IoUVisualTargetTracker(
            lost_tolerance_steps=2,
            confidence_decay=0.5,
        )
        tracker.update(
            self.rgb,
            [door_candidate((10, 10, 40, 80), 0.8)],
            step=0,
        )

        missing = tracker.update(self.rgb, [], step=1)
        reacquired = tracker.update(
            self.rgb,
            [door_candidate((12, 10, 42, 80), 0.75)],
            step=2,
        )

        self.assertFalse(missing.visible)
        self.assertAlmostEqual(missing.confidence, 0.4)
        self.assertTrue(reacquired.visible)
        self.assertEqual(reacquired.missing_steps, 0)

    def test_clears_target_only_after_missing_tolerance(self):
        tracker = IoUVisualTargetTracker(lost_tolerance_steps=2)
        tracker.update(self.rgb, [door_candidate()], step=0)

        self.assertIsNotNone(tracker.update(self.rgb, [], step=1))
        self.assertIsNotNone(tracker.update(self.rgb, [], step=2))
        self.assertIsNone(tracker.update(self.rgb, [], step=3))

    def test_rejects_duplicate_or_backwards_step(self):
        tracker = IoUVisualTargetTracker()
        tracker.update(self.rgb, [door_candidate()], step=2)

        for invalid_step in [2, 1]:
            with self.subTest(step=invalid_step):
                with self.assertRaisesRegex(ValueError, "increase"):
                    tracker.update(self.rgb, [door_candidate()], invalid_step)

    def test_rejects_candidate_bbox_outside_current_image(self):
        tracker = IoUVisualTargetTracker()

        with self.assertRaisesRegex(ValueError, "image bounds"):
            tracker.update(
                self.rgb,
                [door_candidate((-5, 10, 30, 80))],
                step=0,
            )


class ReactiveDoorNavExecutorTest(unittest.TestCase):
    def config(self, **overrides):
        """Build a compact executor config with optional overrides."""
        values = {
            "max_search_steps": 2,
            "max_episode_steps": 20,
            "arrival_area_ratio_threshold": 0.2,
            "arrival_center_tolerance_norm": 0.1,
            "arrival_confirm_frames": 2,
            "grounding_confidence_threshold": 0.5,
            "target_lost_tolerance_steps": 2,
            "no_progress_tolerance_steps": 2,
            "min_area_progress": 0.005,
        }
        values.update(overrides)
        return ReactiveDoorNavConfig.from_mapping(values)

    def test_search_rotates_without_a_target(self):
        decision = ReactiveDoorNavExecutor(self.config()).step(
            navigation_observation(),
            None,
        )

        self.assertEqual(decision.state, DoorNavState.SEARCH)
        self.assertEqual(decision.action, "turn_left")

    def test_track_centers_left_and_right_targets(self):
        for center, action in [(-0.4, "turn_left"), (0.4, "turn_right")]:
            with self.subTest(center=center):
                executor = ReactiveDoorNavExecutor(self.config())
                decision = executor.step(
                    navigation_observation(),
                    visual_track(center_x_norm=center),
                )
                self.assertEqual(decision.state, DoorNavState.TRACK)
                self.assertEqual(decision.action, action)

    def test_centered_target_enters_approach(self):
        decision = ReactiveDoorNavExecutor(self.config()).step(
            navigation_observation(),
            visual_track(area_ratio=0.1),
        )

        self.assertEqual(decision.state, DoorNavState.APPROACH)
        self.assertEqual(decision.action, "move_forward")

    def test_arrival_rule_verifies_then_stops_after_consecutive_frames(self):
        executor = ReactiveDoorNavExecutor(self.config())

        verify = executor.step(
            navigation_observation(step=0),
            visual_track(area_ratio=0.25),
        )
        stop = executor.step(
            navigation_observation(step=1),
            visual_track(area_ratio=0.26),
        )

        self.assertEqual(verify.state, DoorNavState.VERIFY)
        self.assertEqual(stop.state, DoorNavState.STOP)
        self.assertEqual(stop.action, "stop")
        self.assertEqual(
            stop.termination_reason,
            DoorNavTerminationReason.REACHED_DOOR,
        )
        self.assertEqual(
            executor.termination_reason,
            DoorNavTerminationReason.REACHED_DOOR,
        )
        self.assertIn("unvalidated engineering default", stop.reason)

        repeated = executor.step(navigation_observation(step=2), None)

        self.assertEqual(repeated, stop)

    def test_collision_avoidance_ignores_depth_by_default(self):
        executor = ReactiveDoorNavExecutor(self.config())

        decision = executor.step(
            navigation_observation(
                collided=True,
                depth_left_m=0.1,
                depth_right_m=2.0,
            ),
            visual_track(),
        )

        self.assertEqual(decision.state, DoorNavState.AVOID)
        self.assertEqual(decision.action, "turn_left")
        self.assertTrue(decision.obstacle_avoidance_active)

    def test_optional_target_free_depth_selects_collision_turn(self):
        executor = ReactiveDoorNavExecutor(
            self.config(use_depth_for_avoidance=True)
        )

        decision = executor.step(
            navigation_observation(
                collided=True,
                depth_left_m=0.1,
                depth_right_m=2.0,
            ),
            visual_track(),
        )

        self.assertEqual(decision.action, "turn_right")

    def test_missing_track_enters_reacquire(self):
        decision = ReactiveDoorNavExecutor(self.config()).step(
            navigation_observation(),
            visual_track(
                center_x_norm=-0.3,
                visible=False,
                missing_steps=1,
            ),
        )

        self.assertEqual(decision.state, DoorNavState.REACQUIRE)
        self.assertEqual(decision.action, "turn_left")

    def test_cleared_track_fails_with_target_lost(self):
        executor = ReactiveDoorNavExecutor(self.config())
        executor.step(navigation_observation(step=0), visual_track())

        failed = executor.step(navigation_observation(step=1), None)

        self.assertEqual(failed.state, DoorNavState.FAILED)
        self.assertEqual(
            failed.termination_reason,
            DoorNavTerminationReason.TARGET_LOST,
        )

    def test_search_limit_allows_budget_then_fails(self):
        executor = ReactiveDoorNavExecutor(self.config(max_search_steps=2))

        first = executor.step(navigation_observation(step=0), None)
        second = executor.step(navigation_observation(step=1), None)
        failed = executor.step(navigation_observation(step=2), None)

        self.assertEqual(first.state, DoorNavState.SEARCH)
        self.assertEqual(second.state, DoorNavState.SEARCH)
        self.assertEqual(failed.state, DoorNavState.FAILED)
        self.assertEqual(
            executor.termination_reason,
            DoorNavTerminationReason.SEARCH_STEP_LIMIT,
        )

        repeated = executor.step(
            navigation_observation(step=3), visual_track()
        )
        self.assertEqual(repeated, failed)

    def test_episode_limit_allows_budget_then_fails(self):
        executor = ReactiveDoorNavExecutor(
            self.config(max_episode_steps=2, max_search_steps=10)
        )

        first = executor.step(navigation_observation(step=0), None)
        second = executor.step(navigation_observation(step=1), None)
        failed = executor.step(navigation_observation(step=2), None)

        self.assertEqual(first.state, DoorNavState.SEARCH)
        self.assertEqual(second.state, DoorNavState.SEARCH)
        self.assertEqual(failed.state, DoorNavState.FAILED)
        self.assertEqual(
            executor.termination_reason,
            DoorNavTerminationReason.EPISODE_STEP_LIMIT,
        )

    def test_no_progress_uses_consecutive_centered_image_history(self):
        executor = ReactiveDoorNavExecutor(self.config())

        executor.step(
            navigation_observation(step=0), visual_track(area_ratio=0.1)
        )
        executor.step(
            navigation_observation(step=1), visual_track(area_ratio=0.1)
        )
        tolerated = executor.step(
            navigation_observation(step=2),
            visual_track(area_ratio=0.1),
        )
        recovery = executor.step(
            navigation_observation(step=3),
            visual_track(area_ratio=0.1),
        )

        self.assertEqual(tolerated.state, DoorNavState.APPROACH)
        self.assertEqual(recovery.state, DoorNavState.REACQUIRE)
        self.assertIn("image-space", recovery.reason)

    def test_non_centered_view_resets_no_progress_history(self):
        executor = ReactiveDoorNavExecutor(self.config())
        executor.step(
            navigation_observation(step=0), visual_track(area_ratio=0.1)
        )
        executor.step(
            navigation_observation(step=1), visual_track(area_ratio=0.1)
        )
        executor.step(
            navigation_observation(step=2),
            visual_track(center_x_norm=0.4, area_ratio=0.1),
        )

        after_recentering = executor.step(
            navigation_observation(step=3),
            visual_track(area_ratio=0.1),
        )
        next_centered_frame = executor.step(
            navigation_observation(step=4),
            visual_track(area_ratio=0.1),
        )

        self.assertEqual(after_recentering.state, DoorNavState.APPROACH)
        self.assertEqual(next_centered_frame.state, DoorNavState.APPROACH)


class ReactiveDoorNavPolicyTest(unittest.TestCase):
    def test_real_grounder_tracker_executor_chain_records_metadata(self):
        policy = ReactiveDoorNavPolicy()

        output = policy.predict(
            NavigationObservation(
                rgb=doorway_image(),
                instruction="Approach the visible doorway.",
                step=0,
            )
        )

        self.assertTrue(output.is_valid)
        self.assertIn(
            output.action, {"turn_left", "turn_right", "move_forward"}
        )
        self.assertEqual(output.metadata["baseline"], "reactive_doornav_b1")
        for field in [
            "state",
            "reason",
            "candidate_count",
            "candidates",
            "target_track",
            "obstacle_avoidance_active",
            "termination_reason",
        ]:
            self.assertIn(field, output.metadata)
        self.assertGreater(output.metadata["candidate_count"], 0)
        self.assertIsNotNone(output.metadata["target_track"])

    def test_configured_grounder_thresholds_are_used(self):
        policy = ReactiveDoorNavPolicy(
            config=ReactiveDoorNavConfig.from_mapping(
                {
                    "grounder_min_area_ratio": 0.4,
                    "grounder_max_area_ratio": 0.65,
                }
            )
        )

        output = policy.predict(
            NavigationObservation(
                rgb=doorway_image(),
                instruction="Approach the visible doorway.",
                step=0,
            )
        )

        self.assertEqual(output.metadata["candidate_count"], 0)
        self.assertEqual(output.metadata["state"], DoorNavState.SEARCH.value)

    def test_grounding_confidence_threshold_filters_candidates(self):
        policy = ReactiveDoorNavPolicy(
            config=ReactiveDoorNavConfig.from_mapping(
                {"grounding_confidence_threshold": 0.99}
            )
        )

        output = policy.predict(
            NavigationObservation(
                rgb=doorway_image(),
                instruction="Approach the visible doorway.",
                step=0,
            )
        )

        self.assertEqual(output.metadata["candidate_count"], 0)
        self.assertEqual(output.metadata["state"], DoorNavState.SEARCH.value)

    def test_start_episode_resets_tracker_executor_and_implicit_step(self):
        policy = ReactiveDoorNavPolicy(
            config=ReactiveDoorNavConfig.from_mapping({"max_search_steps": 1})
        )
        blank_observation = NavigationObservation(
            rgb=np.zeros((80, 100, 3), dtype=np.uint8),
            instruction="Find a doorway.",
            step=None,
        )

        first = policy.predict(blank_observation)
        failed = policy.predict(blank_observation)
        policy.start_episode("next-episode")
        reset = policy.predict(blank_observation)

        self.assertEqual(first.metadata["state"], DoorNavState.SEARCH.value)
        self.assertFalse(failed.is_valid)
        self.assertIsNone(failed.action)
        self.assertEqual(failed.metadata["state"], DoorNavState.FAILED.value)
        self.assertTrue(reset.is_valid)
        self.assertEqual(reset.metadata["state"], DoorNavState.SEARCH.value)

    def test_invalid_observation_returns_auditable_failure(self):
        policy = ReactiveDoorNavPolicy()

        output = policy.predict(
            NavigationObservation(
                rgb=None,
                instruction="Find a doorway.",
                step=0,
            )
        )

        self.assertFalse(output.is_valid)
        self.assertIsNone(output.action)
        self.assertEqual(
            output.termination_reason,
            DoorNavTerminationReason.INVALID_OBSERVATION.value,
        )
        self.assertEqual(output.metadata["state"], DoorNavState.FAILED.value)
        self.assertIn("ValueError", output.metadata["error"])

    def test_component_exception_is_not_replaced_with_fallback_action(self):
        class FailingGrounder:
            def detect(self, rgb, instruction):
                del rgb, instruction
                raise RuntimeError("detector unavailable")

        policy = ReactiveDoorNavPolicy(grounder=FailingGrounder())

        output = policy.predict(
            NavigationObservation(
                rgb=doorway_image(),
                instruction="Find a doorway.",
                step=0,
            )
        )

        self.assertFalse(output.is_valid)
        self.assertIsNone(output.action)
        self.assertEqual(
            output.termination_reason,
            DoorNavTerminationReason.POLICY_ERROR.value,
        )
        self.assertIn("detector unavailable", output.metadata["error"])

    def test_executor_failure_is_not_reported_as_successful_stop(self):
        policy = ReactiveDoorNavPolicy(
            config=ReactiveDoorNavConfig.from_mapping({"max_search_steps": 1})
        )
        observation = NavigationObservation(
            rgb=np.zeros((80, 100, 3), dtype=np.uint8),
            instruction="Find a doorway.",
            step=0,
        )
        policy.predict(observation)

        failed = policy.predict(
            NavigationObservation(
                rgb=observation.rgb,
                instruction=observation.instruction,
                step=1,
            )
        )

        self.assertFalse(failed.is_valid)
        self.assertIsNone(failed.action)
        self.assertEqual(
            failed.termination_reason,
            DoorNavTerminationReason.SEARCH_STEP_LIMIT.value,
        )
        self.assertEqual(
            failed.metadata["termination_reason"],
            DoorNavTerminationReason.SEARCH_STEP_LIMIT.value,
        )


class FakeDoorNavEnv:
    """Small target-free environment for the shared runner integration seam."""

    def __init__(self, rgb):
        """Initialize a deterministic RGB-only fake episode."""
        self.rgb = rgb
        self.rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        self.agent_state = SimpleNamespace(
            position=np.zeros(3, dtype=np.float32),
            rotation=self.rotation,
        )
        self.sim = SimpleNamespace(
            previous_step_collided=False,
            get_agent_state=lambda: self.agent_state,
        )
        self.current_episode = None
        self.episode_over = False
        self.actions = []
        self.closed = False

    def reset(self):
        """Start one fake local-navigation episode."""
        self.current_episode = SimpleNamespace(
            episode_id="reactive-b1-fake",
            scene_id="fake-local-scene",
            info={"instruction": "Approach the visible doorway."},
        )
        self.episode_over = False
        return {"rgb": self.rgb.copy()}

    def step(self, action):
        """Record an action and return the unchanged observable RGB frame."""
        self.actions.append(action)
        return {"rgb": self.rgb.copy()}

    def get_metrics(self):
        """Return target-free placeholder metrics for recorder integration."""
        return {"local_semantic_status": "unvalidated"}

    def close(self):
        """Record that the shared runner closed the fake environment."""
        self.closed = True


class ReactiveDoorNavRuntimeIntegrationTest(unittest.TestCase):
    def test_runtime_config_selects_local_reactive_policy_and_spec(self):
        args = parse_args(
            [
                "--experiment-config",
                str(RUNTIME_CONFIG_PATH),
                "--task-config",
                "benchmark/nav/semantic_indoor_mock.yaml",
            ]
        )

        policy = build_navigation_policy(args)

        self.assertEqual(args.policy, "reactive_doornav_b1")
        self.assertIsInstance(policy, ReactiveDoorNavPolicy)
        self.assertEqual(
            Path(args.baseline_spec).resolve(), BASELINE_SPEC_PATH
        )
        self.assertEqual(
            args.instruction,
            load_reactive_doornav_spec(BASELINE_SPEC_PATH).instruction,
        )
        self.assertFalse(args.enable_forward_depth_guard)
        self.assertFalse(policy.config.use_depth_for_avoidance)
        self.assertEqual(
            args.max_steps,
            policy.config.max_episode_steps + 1,
        )

    def test_runtime_config_resolves_spec_independently_of_working_directory(
        self,
    ):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                os.chdir(temporary_directory)
                args = parse_args(
                    [
                        "--experiment-config",
                        str(RUNTIME_CONFIG_PATH),
                        "--task-config",
                        "benchmark/nav/semantic_indoor_mock.yaml",
                    ]
                )
                policy = build_navigation_policy(args)
            finally:
                os.chdir(original_directory)

        self.assertIsInstance(policy, ReactiveDoorNavPolicy)
        self.assertEqual(Path(args.baseline_spec), BASELINE_SPEC_PATH)

    def test_fake_env_runs_shared_loop_until_observable_stop(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (55, 8), (105, 115), (255, 255, 255), 3)
        env = FakeDoorNavEnv(image)
        with tempfile.TemporaryDirectory() as temporary_directory:
            argv = [
                "habitat_vln_nav.py",
                "--experiment-config",
                str(RUNTIME_CONFIG_PATH),
                "--task-config",
                "benchmark/nav/semantic_indoor_mock.yaml",
                "--output-dir",
                temporary_directory,
                "--max-steps",
                "6",
                "--no-artifacts",
                "--no-rate-limit",
                "--quiet",
            ]
            with patch("sys.argv", argv), patch(
                "habitat_vln.habitat_vln_nav.build_env",
                return_value=env,
            ):
                main()

            trajectory = next(
                Path(temporary_directory).glob("run_*/trajectory.csv")
            )
            with trajectory.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        metadata = json.loads(rows[-1]["policy_metadata"])
        self.assertEqual(rows[-1]["action"], "stop")
        self.assertEqual(rows[-1]["termination_reason"], "reached_door")
        self.assertEqual(metadata["state"], DoorNavState.STOP.value)
        self.assertIn("unvalidated engineering default", metadata["reason"])
        self.assertEqual(env.actions[-1], ACTION_MAP["stop"])
        self.assertTrue(env.closed)

    def test_shared_runner_records_failed_without_executing_fallback(self):
        env = FakeDoorNavEnv(np.zeros((80, 100, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = parse_args(
                [
                    "--task-config",
                    "benchmark/nav/semantic_indoor_mock.yaml",
                    "--output-dir",
                    temporary_directory,
                    "--max-steps",
                    "3",
                    "--no-artifacts",
                    "--no-rate-limit",
                    "--quiet",
                ]
            )
            args.building_prior = {}
            args.execution_actions = set(VALID_ACTIONS)
            args.allowed_actions = set(VALID_ACTIONS)
            policy = ReactiveDoorNavPolicy(
                config=ReactiveDoorNavConfig.from_mapping(
                    {"max_search_steps": 1}
                )
            )

            trajectory_path = run_navigation(env, policy, args)
            with Path(trajectory_path).open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        failed_metadata = json.loads(rows[-1]["policy_metadata"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(env.actions, [ACTION_MAP["turn_left"]])
        self.assertEqual(rows[-1]["action"], "")
        self.assertEqual(rows[-1]["done"], "True")
        self.assertEqual(rows[-1]["termination_reason"], "search_step_limit")
        self.assertEqual(failed_metadata["state"], DoorNavState.FAILED.value)


if __name__ == "__main__":
    unittest.main()
