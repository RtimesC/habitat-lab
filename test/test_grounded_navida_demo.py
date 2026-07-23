import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from habitat_vln.core import NavigationObservation, PolicyOutput
from habitat_vln.demos import run_grounded_navida_demo as demo
from habitat_vln.runtime import prepare_run_dir
from habitat_vln.runtime.navigation_runner import run_navigation


class FakeNavigationState:
    """Navigation state containing deliberately privileged fake goal data."""

    def __init__(self, step):
        self.step = step
        self.goal_distance_m = 2.0
        self.goal_angle_deg = 15.0
        self.distance_change_m = 0.0
        self.distance_to_goal = 2.0
        self.depth_left_m = 1.0
        self.depth_center_m = 1.0
        self.depth_right_m = 1.0
        self.depth_min = 1.0
        self.depth_mean = 1.0
        self.agent_state = SimpleNamespace(
            position=np.array([1.0, 2.0, 3.0]),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def as_policy_observation(self, obs, instruction):
        return NavigationObservation(
            rgb=obs["rgb"],
            depth=obs["depth"],
            instruction=instruction,
            step=self.step,
            navigation_context={
                "goal_position": (4.0, 2.0, 3.0),
                "goal_distance_m": self.goal_distance_m,
                "goal_angle_deg": self.goal_angle_deg,
            },
        )


class FakeNavigationStateBuilder:
    """Return deterministic states without loading Habitat-Sim."""

    def __init__(self, env):
        self.env = env

    def build(self, obs, step, **unused):
        return FakeNavigationState(step)


class FakeDemoEnvironment:
    """Small one-episode environment used by grounded runner tests."""

    def __init__(self):
        self.actions = []
        self.episode_over = False
        self.sim = SimpleNamespace(previous_step_collided=False)
        self.current_episode = None

    def reset(self):
        self.episode_over = False
        self.current_episode = SimpleNamespace(
            episode_id="grounded-episode",
            scene_id="skokloster-castle.glb",
            info={
                "instruction": "Instruction from episode info.",
                "instruction_source": "manually_grounded_from_route_preview",
                "dataset_scope": "habitat_test_scene_grounded_demo",
                "benchmark_comparable": False,
            },
        )
        return {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2, 1), dtype=np.float32),
            "instruction": {"text": "Instruction from observation sensor."},
        }

    def get_metrics(self):
        return {"distance_to_goal": 2.0, "success": 0.0, "spl": 0.0}


class CapturingPaperPurePolicy:
    """Capture public policy observations and replay configured outputs."""

    policy_protocol = "paper_pure"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.observations = []

    def start_episode(self, episode_id):
        self.episode_id = episode_id

    def predict(self, observation):
        self.observations.append(observation)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output

    def close(self):
        pass


def grounded_runner_args(
    output_dir: str,
    **overrides: Any,
) -> SimpleNamespace:
    """Build the minimal argument namespace required by run_navigation."""
    values = {
        "output_dir": output_dir,
        "video_fps": None,
        "frequency_mode": "joint",
        "joint_hz": 1.0,
        "vision_hz": 1.0,
        "inference_hz": 1.0,
        "num_episodes": 1,
        "goal": None,
        "instruction": None,
        "max_steps": 1,
        "no_rate_limit": True,
        "advisor_fallback": "follow_goal",
        "grounded_demo_artifacts": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def run_grounded_with_fakes(
    env: FakeDemoEnvironment,
    policy: CapturingPaperPurePolicy,
    args: SimpleNamespace,
    video_side_effect: Any = False,
    step_side_effect: Any = None,
    frame_write_result: bool = True,
    state_builder_class: Any = FakeNavigationStateBuilder,
) -> str:
    """Run the shared loop with simulator, image, and video boundaries mocked."""

    def fake_step(fake_env, action_name):
        """Record one named Habitat action and return the next fake observation."""
        fake_env.actions.append(action_name)
        if action_name == "stop":
            fake_env.episode_over = True
        return {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2, 1), dtype=np.float32),
        }

    video_kwargs = (
        {"side_effect": video_side_effect}
        if isinstance(video_side_effect, BaseException)
        or callable(video_side_effect)
        else {"return_value": video_side_effect}
    )
    step_effect = fake_step if step_side_effect is None else step_side_effect
    with mock.patch(
        "habitat_vln.runtime.navigation_runner.NavigationStateBuilder",
        state_builder_class,
    ), mock.patch(
        "habitat_vln.runtime.navigation_runner.step_navigation_action",
        side_effect=step_effect,
    ), mock.patch(
        "habitat_vln.runtime.navigation_runner.cv2.imwrite",
        return_value=frame_write_result,
    ), mock.patch(
        "habitat_vln.runtime.navigation_runner.write_video",
        **video_kwargs,
    ):
        return run_navigation(env, policy, args, controller=None)


class GroundedClosedLoopTest(unittest.TestCase):
    def test_episode_instruction_and_public_rgb_are_the_only_policy_inputs(
        self,
    ):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir),
            )

        observation = policy.observations[0]
        self.assertEqual(
            observation.instruction,
            "Instruction from episode info.",
        )
        self.assertIsNone(observation.depth)
        self.assertEqual(observation.navigation_context, {})

    def test_episode_summary_preserves_grounded_dataset_metadata(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir),
            )
            summary_path = Path(trajectory_path).with_name(
                "episode_summary.json"
            )
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)

        self.assertEqual(
            summary["instruction"],
            "Instruction from episode info.",
        )
        self.assertEqual(
            summary["instruction_source"],
            "manually_grounded_from_route_preview",
        )
        self.assertEqual(
            summary["dataset_scope"],
            "habitat_test_scene_grounded_demo",
        )
        self.assertFalse(summary["benchmark_comparable"])
        self.assertEqual(summary["episode_id"], "grounded-episode")
        self.assertEqual(summary["scene_id"], "skokloster-castle.glb")
        self.assertEqual(summary["steps"], 1)
        self.assertTrue(summary["stopped"])
        self.assertFalse(summary["failed"])
        self.assertIn("trajectory", summary["output_paths"])
        self.assertIn("frames", summary["output_paths"])
        self.assertIn("video", summary["output_paths"])

    def test_instruction_override_is_sent_and_labeled_in_summary(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(
                    output_dir,
                    instruction="Explicit CLI instruction.",
                ),
            )
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(
            policy.observations[0].instruction,
            "Explicit CLI instruction.",
        )
        self.assertEqual(summary["instruction"], "Explicit CLI instruction.")
        self.assertEqual(summary["instruction_source"], "cli_override")

    def test_empty_instruction_is_still_an_explicit_override(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, instruction=""),
            )
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(policy.observations[0].instruction, "")
        self.assertEqual(summary["instruction"], "")
        self.assertEqual(summary["instruction_source"], "cli_override")

    def test_invalid_policy_output_stops_safely_and_records_error(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [
                PolicyOutput(
                    action=None,
                    raw_text="not an action",
                    is_valid=False,
                    termination_reason="invalid_model_output",
                    metadata={
                        "error": "action could not be parsed",
                        "latency_seconds": 0.25,
                    },
                )
            ]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, max_steps=3),
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        required_fields = {
            "episode_id",
            "step",
            "instruction",
            "policy_action",
            "executed_action",
            "action_valid",
            "inference_latency_sec",
            "position_x",
            "position_y",
            "position_z",
            "rotation_yaw",
            "collision",
            "distance_to_goal",
            "success",
            "spl",
            "done",
            "error",
        }
        self.assertTrue(required_fields.issubset(row))
        self.assertEqual(env.actions, [])
        self.assertEqual(row["policy_action"], "")
        self.assertEqual(row["executed_action"], "")
        self.assertEqual(row["action_valid"], "False")
        self.assertEqual(row["inference_latency_sec"], "0.25")
        self.assertEqual(row["position_x"], "1.0")
        self.assertEqual(row["rotation_yaw"], "0.0")
        self.assertEqual(row["done"], "True")
        self.assertEqual(row["error"], "action could not be parsed")
        self.assertTrue(summary["failed"])
        self.assertEqual(
            summary["failure_reason"],
            "action could not be parsed",
        )

    def test_max_steps_ends_loop_and_is_recorded(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [
                PolicyOutput("move_forward", "step zero", True),
                PolicyOutput("turn_left", "step one", True),
                PolicyOutput("turn_right", "must not execute", True),
            ]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, max_steps=2),
            )
            with open(trajectory_path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(env.actions, ["move_forward", "turn_left"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["done"], "True")
        self.assertEqual(rows[-1]["error"], "max_steps_reached")
        self.assertEqual(summary["steps"], 2)
        self.assertTrue(summary["failed"])
        self.assertEqual(summary["failure_reason"], "max_steps_reached")

    def test_video_writer_failure_does_not_fail_the_navigation_run(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir),
                video_side_effect=RuntimeError("video codec unavailable"),
            )
            run_dir = Path(trajectory_path).parent
            with (run_dir / "episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            markdown = (run_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(Path(trajectory_path).name, "trajectory.csv")
        self.assertFalse(summary["failed"])
        self.assertFalse(summary["video_generated"])
        self.assertEqual(summary["video_error"], "video codec unavailable")
        self.assertIn("benchmark_comparable=false", markdown)
        self.assertIn("Oracle validation", markdown)
        self.assertIn("video_generated", markdown)

    def test_policy_exception_stops_safely_and_is_recorded(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [RuntimeError("unexpected policy failure")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, max_steps=3),
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(env.actions, [])
        self.assertEqual(row["done"], "True")
        self.assertIn("unexpected policy failure", row["error"])
        self.assertTrue(summary["failed"])
        self.assertIn("unexpected policy failure", summary["failure_reason"])

    def test_environment_step_exception_is_recorded_before_termination(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("move_forward", "server action", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, max_steps=3),
                step_side_effect=RuntimeError("simulator step failed"),
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(row["policy_action"], "move_forward")
        self.assertEqual(row["executed_action"], "")
        self.assertEqual(row["done"], "True")
        self.assertIn("simulator step failed", row["error"])
        self.assertTrue(summary["failed"])
        self.assertIn("simulator step failed", summary["failure_reason"])

    def test_failed_rgb_write_is_recorded_and_not_sent_to_video(self):
        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )
        video_frame_paths = []

        def capture_video_frames(frame_paths, video_path, fps):
            """Capture video inputs without encoding a real file."""
            video_frame_paths.extend(frame_paths)
            return False

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir),
                video_side_effect=capture_video_frames,
                frame_write_result=False,
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(row["image"], "")
        self.assertEqual(video_frame_paths, [])
        self.assertEqual(row["done"], "True")
        self.assertIn("RGB frame write failed", row["error"])
        self.assertTrue(summary["failed"])
        self.assertIn("RGB frame write failed", summary["failure_reason"])

    def test_state_build_exception_writes_terminal_artifacts(self):
        class FailingStateBuilder(FakeNavigationStateBuilder):
            """Raise before policy inference to simulate a Habitat state failure."""

            def build(self, obs, step, **unused):
                raise RuntimeError("state assembly failed")

        env = FakeDemoEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("move_forward", "must not run", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, max_steps=3),
                state_builder_class=FailingStateBuilder,
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(policy.observations, [])
        self.assertEqual(row["done"], "True")
        self.assertIn("state assembly failed", row["error"])
        self.assertTrue(summary["failed"])
        self.assertIn("state assembly failed", summary["failure_reason"])

    def test_episode_reset_exception_writes_run_failure_artifacts(self):
        class ResetFailingEnvironment(FakeDemoEnvironment):
            """Fail before Habitat exposes a current episode."""

            def reset(self):
                raise RuntimeError("episode reset failed")

        env = ResetFailingEnvironment()
        policy = CapturingPaperPurePolicy(
            [PolicyOutput("move_forward", "must not run", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_grounded_with_fakes(
                env,
                policy,
                grounded_runner_args(output_dir, max_steps=3),
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))
            with Path(trajectory_path).with_name("episode_summary.json").open(
                encoding="utf-8"
            ) as handle:
                summary = json.load(handle)

        self.assertEqual(policy.observations, [])
        self.assertEqual(row["done"], "True")
        self.assertIn("episode reset failed", row["error"])
        self.assertEqual(summary["steps"], 0)
        self.assertTrue(summary["failed"])
        self.assertIn("episode reset failed", summary["failure_reason"])


class GroundedDatasetPreparationTest(unittest.TestCase):
    def test_missing_default_dataset_is_generated_with_python_api(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = (
                Path(temporary_directory) / "data/datasets/pointnav/"
                "habitat_test_scene_grounded_demo/v1/val/val.json.gz"
            )

            with mock.patch.object(demo, "write_dataset") as write_dataset:
                demo.prepare_dataset(
                    dataset_path,
                    default_dataset_path=dataset_path,
                )

        write_dataset.assert_called_once_with(dataset_path)

    def test_existing_default_dataset_is_not_generated_again(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "val.json.gz"
            dataset_path.write_bytes(b"existing dataset")

            with mock.patch.object(demo, "write_dataset") as write_dataset:
                resolved_path = demo.prepare_dataset(
                    dataset_path,
                    default_dataset_path=dataset_path,
                )

        self.assertEqual(resolved_path, dataset_path)
        write_dataset.assert_not_called()

    def test_missing_custom_dataset_is_rejected_without_generation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            default_path = root / "default/val.json.gz"
            custom_path = root / "custom/val.json.gz"

            with mock.patch.object(demo, "write_dataset") as write_dataset:
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "custom dataset.*does not exist",
                ):
                    demo.prepare_dataset(
                        custom_path,
                        default_dataset_path=default_path,
                    )

            self.assertFalse(custom_path.exists())

        write_dataset.assert_not_called()


class GroundedRunDirectoryTest(unittest.TestCase):
    def test_consecutive_runs_get_independent_directories(self):
        with tempfile.TemporaryDirectory() as output_dir:
            first_run_dir, _ = prepare_run_dir(output_dir)
            second_run_dir, _ = prepare_run_dir(output_dir)

        self.assertNotEqual(first_run_dir, second_run_dir)


class GroundedDemoCliTest(unittest.TestCase):
    def test_cli_defaults_target_grounded_dataset_without_gpu_override(self):
        args = demo.parse_args([])

        self.assertEqual(args.dataset_split, "val")
        self.assertEqual(
            args.dataset_path,
            (
                "data/datasets/pointnav/"
                "habitat_test_scene_grounded_demo/v1/{split}/{split}.json.gz"
            ),
        )
        self.assertEqual(args.num_episodes, 1)
        self.assertIsNone(args.instruction)
        self.assertIsNone(args.gpu_device_id)
        self.assertEqual(args.official_navida_url, "http://127.0.0.1:8008")
        self.assertGreater(args.official_navida_timeout, 0.0)

    def test_main_rejects_missing_custom_dataset_without_generating_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            custom_path = Path(temporary_directory) / "custom/demo.json.gz"

            with mock.patch.object(
                demo,
                "write_dataset",
            ) as write_dataset, mock.patch.object(
                demo,
                "build_env",
            ) as build_env, mock.patch.object(
                demo,
                "run_navigation",
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "custom dataset.*does not exist",
                ):
                    demo.main(["--dataset-path", str(custom_path)])

            self.assertFalse(custom_path.exists())

        write_dataset.assert_not_called()
        build_env.assert_not_called()

    def test_main_passes_gpu_override_and_official_policy_settings(self):
        fake_env = SimpleNamespace(close=mock.Mock())
        fake_policy = SimpleNamespace(policy_protocol="paper_pure")
        with tempfile.TemporaryDirectory() as temporary_directory:
            custom_path = Path(temporary_directory) / "custom/demo.json.gz"
            custom_path.parent.mkdir(parents=True)
            custom_path.write_bytes(b"existing custom dataset")
            argv = [
                "--dataset-path",
                str(custom_path),
                "--dataset-split",
                "test",
                "--gpu-device-id",
                "2",
                "--official-navida-url",
                "http://navida.test:9000",
                "--official-navida-timeout",
                "4.5",
            ]

            with mock.patch.object(
                demo,
                "write_dataset",
            ) as write_dataset, mock.patch.object(
                demo,
                "build_env",
                return_value=fake_env,
            ) as build_env, mock.patch.object(
                demo,
                "OfficialNaVIDAHTTPPolicy",
                return_value=fake_policy,
            ) as policy_class, mock.patch.object(
                demo,
                "run_navigation",
                return_value="output/trajectory.csv",
            ) as run_navigation:
                result = demo.main(argv)

        self.assertEqual(result, "output/trajectory.csv")
        write_dataset.assert_not_called()
        self.assertEqual(build_env.call_args.kwargs["gpu_device_id"], 2)
        self.assertEqual(
            build_env.call_args.kwargs["dataset_path"],
            str(custom_path),
        )
        policy_class.assert_called_once_with(
            base_url="http://navida.test:9000",
            timeout=4.5,
        )
        run_args = run_navigation.call_args.args[2]
        self.assertEqual(run_args.frequency_mode, "joint")
        self.assertTrue(run_args.grounded_demo_artifacts)
        fake_env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
