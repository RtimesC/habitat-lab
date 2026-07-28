import csv
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from habitat_vln.core import (
    NavigationObservation,
    NavigationPolicy,
    PolicyOutput,
    load_building_prior,
)
from habitat_vln.envs import (
    ACTION_MAP,
    NavigationStateBuilder,
    step_navigation_action,
)
from habitat_vln.policies import MockVLMPolicy
from habitat_vln.policies.prompts import format_navigation_state
from habitat_vln.runtime import (
    TRAJECTORY_FIELDS,
    TrajectoryRecorder,
    prepare_run_dir,
    write_video,
)


class HabitatVLNInterfaceTest(unittest.TestCase):
    def test_habitat_action_adapter_executes_named_action(self):
        executed_actions = []
        env = SimpleNamespace(
            step=lambda action: executed_actions.append(action)
            or {"rgb": "next"}
        )

        observation = step_navigation_action(env, "turn_left")

        self.assertEqual(executed_actions, [ACTION_MAP["turn_left"]])
        self.assertEqual(observation, {"rgb": "next"})

    def test_state_builder_exposes_only_local_semantic_policy_context(self):
        depth_sensor = SimpleNamespace(
            min_depth=0.0,
            max_depth=10.0,
            normalize_depth=False,
        )
        config = SimpleNamespace(
            habitat=SimpleNamespace(
                simulator=SimpleNamespace(
                    agents=SimpleNamespace(
                        main_agent=SimpleNamespace(
                            sim_sensors={"depth_sensor": depth_sensor}
                        )
                    )
                )
            )
        )
        rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        agent_state = SimpleNamespace(
            position=np.array([1.0, 2.0, 3.0]),
            rotation=rotation,
        )
        env = SimpleNamespace(
            config=config,
            sim=SimpleNamespace(
                get_agent_state=lambda: agent_state,
                previous_step_collided=True,
            ),
        )
        obs = {
            "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "depth": np.array(
                [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
                dtype=np.float32,
            ),
        }

        state = NavigationStateBuilder(env).build(
            obs=obs,
            step=4,
            previous_action="move_forward",
            previous_action_count=2,
            previous_collision=False,
        )
        debug_context = state.as_navigation_context()
        policy_context = state.as_policy_navigation_context(
            {"building_id": "teaching-a", "floor_labels": ["1", "2"]}
        )
        policy_observation = state.as_policy_observation(
            obs,
            "Go to the elevator area.",
            {"building_id": "teaching-a"},
        )

        self.assertEqual(
            (state.depth_left_m, state.depth_center_m, state.depth_right_m),
            (1.0, 2.0, 3.0),
        )
        self.assertTrue(state.collided)
        self.assertIn("agent_position", debug_context)
        self.assertNotIn("agent_position", policy_context)
        self.assertEqual(
            policy_context["building_prior"]["building_id"], "teaching-a"
        )
        self.assertEqual(
            policy_observation.navigation_context["building_prior"],
            {"building_id": "teaching-a"},
        )
        for forbidden in ["goal", "distance", "angle", "success", "pointgoal"]:
            self.assertNotIn(forbidden, " ".join(policy_context))

    def test_prompt_state_keeps_allowed_building_prior_and_hides_target_fields(
        self,
    ):
        prompt_state = format_navigation_state(
            {
                "step": 3,
                "collided": False,
                "depth_center_m": 1.0,
                "building_prior": {"building_id": "teaching-a"},
                "goal_distance_m": 5.0,
            }
        )

        self.assertIn("building_id", prompt_state)
        self.assertIn("depth_center_m: 1.000", prompt_state)
        self.assertNotIn("goal_distance_m", prompt_state)

    def test_building_prior_rejects_unapproved_target_field(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prior.json"
            path.write_text(
                json.dumps(
                    {
                        "building_id": "teaching-a",
                        "target_coordinates": [1.0, 2.0],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                load_building_prior(path)

    def test_mock_policy_accepts_new_navigation_observation(self):
        policy = MockVLMPolicy(allowed_actions={"move_forward", "turn_left"})
        observation = NavigationObservation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            instruction="Go to the elevator area.",
            step=1,
            navigation_context={"collided": False},
        )

        output = policy.predict(observation)

        self.assertIsInstance(policy, NavigationPolicy)
        self.assertIsInstance(output, PolicyOutput)
        self.assertEqual(output.action, "move_forward")

    def test_trajectory_recorder_preserves_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "trajectory.csv"
            row = {field: "" for field in TRAJECTORY_FIELDS}
            row["action"] = "move_forward"
            with TrajectoryRecorder(path) as recorder:
                recorder.write(row)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                saved_rows = list(reader)
            self.assertEqual(reader.fieldnames, TRAJECTORY_FIELDS)
            self.assertEqual(saved_rows[0]["action"], "move_forward")

    def test_prepare_run_dir_places_grouped_runs_inside_output_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir, _ = prepare_run_dir(
                temporary_directory, "semantic_indoor"
            )
            self.assertEqual(Path(run_dir).parent.name, "semantic_indoor")
            with self.assertRaises(ValueError):
                prepare_run_dir(temporary_directory, "../outside")

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg and ffprobe are required for H.264 verification",
    )
    def test_video_writer_outputs_browser_compatible_h264(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            frame_paths = []
            for index, value in enumerate([30, 180]):
                frame_path = directory / f"frame_{index:03d}.jpg"
                frame = np.full((16, 16, 3), value, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(frame_path), frame))
                frame_paths.append(str(frame_path))
            video_path = directory / "video.mp4"
            self.assertTrue(write_video(frame_paths, str(video_path), fps=2.0))
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "h264")


if __name__ == "__main__":
    unittest.main()
