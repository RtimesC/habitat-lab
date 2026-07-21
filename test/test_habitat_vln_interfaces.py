import csv
import shutil
import subprocess
import tempfile
import unittest
from math import radians
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from habitat_vln.core import NavigationObservation, NavigationPolicy, PolicyOutput
from habitat_vln.envs import ACTION_MAP, NavigationStateBuilder, step_navigation_action
from habitat_vln.policies import ActionChunkOutput, MockVLMPolicy, NaVIDAChunkPolicy
from habitat_vln.runtime import TRAJECTORY_FIELDS, TrajectoryRecorder, write_video


class HabitatVLNInterfaceTest(unittest.TestCase):
    def test_habitat_action_adapter_executes_named_action(self):
        executed_actions = []
        env = SimpleNamespace(
            step=lambda action: executed_actions.append(action) or {"rgb": "next"}
        )

        observation = step_navigation_action(env, "turn_left")

        self.assertEqual(executed_actions, [ACTION_MAP["turn_left"]])
        self.assertEqual(observation, {"rgb": "next"})

    def test_habitat_state_builder_assembles_navigation_context(self):
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
                ),
                task=SimpleNamespace(
                    measurements=SimpleNamespace(
                        success=SimpleNamespace(success_distance=0.2)
                    )
                ),
            )
        )
        rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        agent_state = SimpleNamespace(
            position=np.array([1.0, 2.0, 3.0]),
            rotation=rotation,
        )
        env = SimpleNamespace(
            config=config,
            current_episode=SimpleNamespace(
                goals=[SimpleNamespace(position=[4.0, 2.0, 3.0])]
            ),
            sim=SimpleNamespace(
                get_agent_state=lambda: agent_state,
                previous_step_collided=True,
            ),
            get_metrics=lambda: {"distance_to_goal": 5.0},
        )
        obs = {
            "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
            "depth": np.array(
                [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
                dtype=np.float32,
            ),
            "pointgoal_with_gps_compass": np.array(
                [5.0, radians(-30.0)],
                dtype=np.float32,
            ),
        }

        state = NavigationStateBuilder(env).build(
            obs=obs,
            step=4,
            previous_action="move_forward",
            previous_action_count=2,
            previous_collision=False,
            previous_goal_distance=6.0,
            no_progress_steps=1,
        )
        context = state.as_navigation_context()
        policy_observation = state.as_policy_observation(obs, "Move to the goal.")

        self.assertAlmostEqual(state.goal_distance_m, 5.0)
        self.assertAlmostEqual(state.goal_angle_deg, 30.0, places=4)
        self.assertAlmostEqual(state.distance_change_m, -1.0)
        self.assertEqual(
            (state.depth_left_m, state.depth_center_m, state.depth_right_m),
            (1.0, 2.0, 3.0),
        )
        self.assertTrue(state.collided)
        self.assertEqual(context["previous_action"], "move_forward")
        self.assertEqual(policy_observation.navigation_context, context)

    def test_mock_policy_accepts_new_navigation_observation(self):
        policy = MockVLMPolicy(allowed_actions={"move_forward", "turn_left"})
        observation = NavigationObservation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            instruction="Move to the goal.",
            step=1,
            navigation_context={"collided": False},
        )

        output = policy.predict(observation)

        self.assertIsInstance(policy, NavigationPolicy)
        self.assertIsInstance(output, PolicyOutput)
        self.assertEqual(output.action, "move_forward")

    def test_mock_policy_keeps_legacy_predict_signature(self):
        policy = MockVLMPolicy(allowed_actions={"move_forward", "turn_left"})

        output = policy.predict(
            np.zeros((2, 2, 3), dtype=np.uint8),
            "Move to the goal.",
            step=1,
            navigation_context={"collided": False},
        )

        self.assertEqual(output.action, "move_forward")
        self.assertIsInstance(output, PolicyOutput)

    def test_navida_policy_accepts_new_navigation_observation(self):
        class FakeRuntime:
            def generate_action_chunk(self, images, prompt):
                return ActionChunkOutput(
                    atomic_actions=["turn_left", "move_forward"],
                    raw_text='{"actions": [{"action": "turn_left", "count": 1}]}',
                    is_valid=True,
                )

        policy = NaVIDAChunkPolicy.__new__(NaVIDAChunkPolicy)
        policy.runtime = FakeRuntime()
        policy.fallback_action = "move_forward"
        policy.max_history_frames = 2
        policy.max_executed_actions = 2
        policy.history = []
        policy.action_queue = []
        policy.last_chunk_text = ""
        observation = NavigationObservation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            instruction="Move to the goal.",
            step=0,
        )

        first_output = policy.predict(observation)
        second_output = policy.predict(
            NavigationObservation(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                instruction="Move to the goal.",
                step=1,
            )
        )

        self.assertEqual(first_output.action, "turn_left")
        self.assertEqual(second_output.action, "move_forward")

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
            self.assertFalse((directory / "video.mp4v.mp4").exists())


if __name__ == "__main__":
    unittest.main()
