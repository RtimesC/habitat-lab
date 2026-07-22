import base64
import csv
import json
import sys
import tempfile
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from habitat_vln import habitat_vln_nav
from habitat_vln.core import NavigationObservation, PolicyOutput
from habitat_vln.policies import OfficialNaVIDAHTTPPolicy
from habitat_vln.runtime.navigation_runner import run_navigation


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, payload=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "payload": payload,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeNavigationState:
    def __init__(self, step):
        self.step = step
        self.goal_distance_m = 2.0
        self.goal_angle_deg = 0.0
        self.success_distance_m = 0.2
        self.collided = False
        self.depth_left_m = 1.0
        self.depth_center_m = 1.0
        self.depth_right_m = 1.0
        self.distance_change_m = 0.0
        self.distance_to_goal = 2.0
        self.depth_min = 1.0
        self.depth_mean = 1.0
        self.agent_state = SimpleNamespace(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def as_policy_observation(self, obs, instruction):
        return NavigationObservation(
            rgb=obs["rgb"],
            depth=obs.get("depth"),
            instruction=instruction,
            step=self.step,
            navigation_context={"goal_distance_m": self.goal_distance_m},
        )


class FakeNavigationStateBuilder:
    def __init__(self, env):
        self.env = env

    def build(self, obs, step, **unused):
        return FakeNavigationState(step)


class FakeEnvironment:
    def __init__(self, events):
        self.events = events
        self.reset_count = 0
        self.actions = []
        self.episode_over = False
        self.sim = SimpleNamespace(previous_step_collided=False)
        self.current_episode = None

    def reset(self):
        episode_id = f"episode-{self.reset_count}"
        self.reset_count += 1
        self.current_episode = SimpleNamespace(
            episode_id=episode_id,
            scene_id="fake-scene",
            info={"instruction": "Go forward."},
        )
        self.episode_over = False
        self.events.append(("reset", episode_id))
        return {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2, 1), dtype=np.float32),
            "instruction": {"text": "Go forward."},
        }

    def get_metrics(self):
        return {"distance_to_goal": 2.0, "success": 0.0, "spl": 0.0}


class PassthroughController:
    def __init__(self):
        self.calls = []

    def decide(self, policy_output, navigation_state, **unused):
        self.calls.append(policy_output.action)
        return SimpleNamespace(
            vlm_action=policy_output.action,
            controller_action=policy_output.action,
            action=policy_output.action,
            policy_output=policy_output,
            messages=(),
        )


class RaisingController:
    def decide(self, *args, **kwargs):
        raise AssertionError("paper_pure must not call NavigationController")


class ReplacingController:
    def decide(self, policy_output, navigation_state, **unused):
        return SimpleNamespace(
            vlm_action=policy_output.action,
            controller_action="move_forward",
            action="move_forward",
            policy_output=policy_output,
            messages=(),
        )


class SequencePaperPurePolicy:
    policy_protocol = "paper_pure"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.start_calls = []
        self.close_calls = 0

    def start_episode(self, episode_id):
        self.start_calls.append(episode_id)

    def predict(self, observation):
        return self.outputs.pop(0)

    def close(self):
        self.close_calls += 1


def runner_args(output_dir, **overrides):
    values = {
        "output_dir": output_dir,
        "video_fps": None,
        "frequency_mode": "joint",
        "joint_hz": 1.0,
        "vision_hz": 5.0,
        "inference_hz": 0.5,
        "num_episodes": 1,
        "goal": None,
        "instruction": None,
        "max_steps": 1,
        "no_rate_limit": True,
        "advisor_fallback": "follow_goal",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def run_with_fakes(env, policy, args, controller):
    def fake_step(fake_env, action_name):
        fake_env.actions.append(action_name)
        fake_env.events.append(("step", action_name))
        if action_name == "stop":
            fake_env.episode_over = True
        return {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2, 1), dtype=np.float32),
            "instruction": {"text": "Go forward."},
        }

    with mock.patch(
        "habitat_vln.runtime.navigation_runner.NavigationStateBuilder",
        FakeNavigationStateBuilder,
    ), mock.patch(
        "habitat_vln.runtime.navigation_runner.step_navigation_action",
        side_effect=fake_step,
    ), mock.patch(
        "habitat_vln.runtime.navigation_runner.cv2.imwrite",
        return_value=True,
    ), mock.patch(
        "habitat_vln.runtime.navigation_runner.write_video",
        return_value=False,
    ):
        return run_navigation(env, policy, args, controller=controller)


class PolicyOutputCompatibilityTest(unittest.TestCase):
    def test_existing_constructor_gets_safe_new_field_defaults(self):
        output = PolicyOutput("move_forward", "raw", True)

        self.assertEqual(output.action, "move_forward")
        self.assertEqual(output.raw_text, "raw")
        self.assertTrue(output.is_valid)
        self.assertIsNone(output.termination_reason)
        self.assertEqual(output.metadata, {})


class OfficialNaVIDAHTTPPolicyTest(unittest.TestCase):
    def test_constructor_rejects_invalid_timeout_values(self):
        invalid_values = (
            True,
            False,
            0,
            -1,
            float("nan"),
            float("inf"),
            "30",
            None,
        )
        for timeout in invalid_values:
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    OfficialNaVIDAHTTPPolicy(timeout=timeout)

    def test_start_and_predict_send_only_public_observation_fields(self):
        transport = FakeTransport(
            [
                {"episode_id": "episode-7", "reset": True},
                {
                    "episode_id": "episode-7",
                    "simulator_step": 4,
                    "decision_step": 2,
                    "action": "turn_left",
                    "valid": True,
                    "inferred": True,
                    "raw_output": '{"action":"turn_left"}',
                    "error": None,
                    "termination_reason": None,
                    "latency_seconds": 0.25,
                    "metadata": {"history_frames": 3},
                },
            ]
        )
        policy = OfficialNaVIDAHTTPPolicy(
            base_url="http://navida.test:8008/",
            timeout=7.5,
            transport=transport,
        )
        rgb = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [1, 2, 3]]],
            dtype=np.uint8,
        )

        policy.start_episode("episode-7")
        output = policy.predict(
            NavigationObservation(
                rgb=rgb,
                depth=np.ones((2, 2, 1), dtype=np.float32),
                instruction="Turn left at the doorway.",
                step=4,
                navigation_context={
                    "goal_distance_m": 1.0,
                    "collision": True,
                    "agent_pose": [1.0, 2.0, 3.0],
                },
            )
        )

        self.assertEqual(policy.policy_protocol, "paper_pure")
        self.assertEqual(
            transport.calls[0],
            {
                "method": "POST",
                "url": "http://navida.test:8008/v1/episodes/start",
                "payload": {"episode_id": "episode-7"},
                "timeout": 7.5,
            },
        )
        step_call = transport.calls[1]
        self.assertEqual(step_call["method"], "POST")
        self.assertEqual(step_call["url"], "http://navida.test:8008/v1/steps")
        self.assertEqual(step_call["timeout"], 7.5)
        self.assertEqual(
            set(step_call["payload"]),
            {"episode_id", "simulator_step", "instruction", "rgb_png_base64"},
        )
        self.assertEqual(step_call["payload"]["episode_id"], "episode-7")
        self.assertEqual(step_call["payload"]["simulator_step"], 4)
        self.assertEqual(
            step_call["payload"]["instruction"],
            "Turn left at the doorway.",
        )
        png_bytes = base64.b64decode(step_call["payload"]["rgb_png_base64"])
        decoded_rgb = np.asarray(Image.open(BytesIO(png_bytes)).convert("RGB"))
        np.testing.assert_array_equal(decoded_rgb, rgb)

        self.assertEqual(output.action, "turn_left")
        self.assertEqual(output.raw_text, '{"action":"turn_left"}')
        self.assertTrue(output.is_valid)
        self.assertIsNone(output.termination_reason)
        self.assertEqual(output.metadata["decision_step"], 2)
        self.assertTrue(output.metadata["inferred"])
        self.assertEqual(output.metadata["latency_seconds"], 0.25)
        self.assertEqual(
            output.metadata["server_metadata"], {"history_frames": 3}
        )
        self.assertIn("raw_output", output.metadata)
        self.assertIn("termination_reason", output.metadata)
        self.assertIn("error", output.metadata)

    def test_transport_failure_returns_invalid_output_without_fallback(self):
        transport = FakeTransport(
            [
                {"episode_id": "episode-8", "reset": True},
                ValueError("response was not valid JSON"),
            ]
        )
        policy = OfficialNaVIDAHTTPPolicy(transport=transport)
        policy.start_episode("episode-8")

        output = policy.predict(
            NavigationObservation(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                instruction="Go forward.",
                step=0,
            )
        )

        self.assertIsNone(output.action)
        self.assertFalse(output.is_valid)
        self.assertEqual(output.termination_reason, "http_request_failed")
        self.assertIn("response was not valid JSON", output.metadata["error"])

    def test_malformed_response_returns_invalid_output_without_raising(self):
        transport = FakeTransport(
            [
                {"episode_id": "episode-9", "reset": True},
                {
                    "episode_id": "episode-9",
                    "simulator_step": 0,
                    "decision_step": 0,
                    "action": "move_forward",
                    "valid": True,
                    "inferred": True,
                    "raw_output": "malformed metadata",
                    "error": None,
                    "termination_reason": None,
                    "latency_seconds": 0.1,
                    "metadata": "not-a-json-object",
                },
            ]
        )
        policy = OfficialNaVIDAHTTPPolicy(transport=transport)
        policy.start_episode("episode-9")

        output = policy.predict(
            NavigationObservation(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                instruction="Go forward.",
                step=0,
            )
        )

        self.assertIsNone(output.action)
        self.assertFalse(output.is_valid)
        self.assertEqual(output.raw_text, "malformed metadata")
        self.assertIn(
            "metadata must be a JSON object", output.metadata["error"]
        )

    def test_unhashable_action_returns_invalid_output_without_raising(self):
        transport = FakeTransport(
            [
                {"episode_id": "episode-10", "reset": True},
                {
                    "episode_id": "episode-10",
                    "simulator_step": 0,
                    "decision_step": 0,
                    "action": ["move_forward"],
                    "valid": True,
                    "inferred": True,
                    "raw_output": "malformed action",
                    "error": None,
                    "termination_reason": None,
                    "latency_seconds": 0.1,
                    "metadata": {},
                },
            ]
        )
        policy = OfficialNaVIDAHTTPPolicy(transport=transport)
        policy.start_episode("episode-10")

        output = policy.predict(
            NavigationObservation(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                instruction="Go forward.",
                step=0,
            )
        )

        self.assertIsNone(output.action)
        self.assertFalse(output.is_valid)
        self.assertIn("action must be a string", output.metadata["error"])

    def test_malformed_termination_reason_does_not_escape_output_contract(
        self,
    ):
        transport = FakeTransport(
            [
                {"episode_id": "episode-11", "reset": True},
                {
                    "episode_id": "episode-11",
                    "simulator_step": 0,
                    "decision_step": 0,
                    "action": None,
                    "valid": False,
                    "inferred": True,
                    "raw_output": "bad termination reason",
                    "error": "model error",
                    "termination_reason": ["not", "a", "string"],
                    "latency_seconds": 0.1,
                    "metadata": {},
                },
            ]
        )
        policy = OfficialNaVIDAHTTPPolicy(transport=transport)
        policy.start_episode("episode-11")

        output = policy.predict(
            NavigationObservation(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                instruction="Go forward.",
                step=0,
            )
        )

        self.assertIsNone(output.action)
        self.assertFalse(output.is_valid)
        self.assertIsNone(output.termination_reason)
        self.assertEqual(
            output.metadata["termination_reason"],
            ["not", "a", "string"],
        )
        self.assertEqual(output.metadata["server_error"], "model error")


class NavigationRunnerLifecycleTest(unittest.TestCase):
    def test_episode_start_runs_after_each_reset_and_close_runs_once(self):
        events = []
        env = FakeEnvironment(events)

        class LifecyclePolicy:
            def __init__(self):
                self.start_calls = []
                self.close_calls = 0

            def start_episode(self, episode_id):
                self.start_calls.append(episode_id)
                events.append(("start", episode_id))

            def predict(self, observation):
                events.append(("predict", observation.step))
                return PolicyOutput("move_forward", "raw", True)

            def close(self):
                self.close_calls += 1

        policy = LifecyclePolicy()
        controller = PassthroughController()
        with tempfile.TemporaryDirectory() as output_dir:
            run_with_fakes(
                env,
                policy,
                runner_args(output_dir, num_episodes=2),
                controller,
            )

        self.assertEqual(policy.start_calls, ["episode-0", "episode-1"])
        self.assertEqual(policy.close_calls, 1)
        self.assertEqual(
            events[:3],
            [("reset", "episode-0"), ("start", "episode-0"), ("predict", 0)],
        )

    def test_run_error_is_preserved_when_close_also_fails(self):
        class CloseFailingPolicy:
            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                raise RuntimeError("close failed")

        policy = CloseFailingPolicy()
        with mock.patch(
            "habitat_vln.runtime.navigation_runner._execute_navigation",
            side_effect=ValueError("run failed"),
        ):
            with self.assertRaisesRegex(ValueError, "run failed"):
                run_navigation(None, policy, None)

        self.assertEqual(policy.close_calls, 1)


class PaperPureRunnerTest(unittest.TestCase):
    def test_valid_action_bypasses_controller_unchanged(self):
        events = []
        env = FakeEnvironment(events)
        policy = SequencePaperPurePolicy(
            [PolicyOutput("turn_left", "server output", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            run_with_fakes(
                env,
                policy,
                runner_args(output_dir),
                RaisingController(),
            )

        self.assertEqual(env.actions, ["turn_left"])
        self.assertEqual(policy.close_calls, 1)

    def test_early_stop_is_executed_without_replacement(self):
        events = []
        env = FakeEnvironment(events)
        policy = SequencePaperPurePolicy(
            [PolicyOutput("stop", "server stop", True)]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            run_with_fakes(
                env,
                policy,
                runner_args(output_dir, max_steps=3),
                RaisingController(),
            )

        self.assertEqual(env.actions, ["stop"])

    def test_invalid_output_terminates_without_step_and_records_diagnostics(
        self,
    ):
        events = []
        env = FakeEnvironment(events)
        invalid_output = PolicyOutput(
            action=None,
            raw_text="unparseable server output",
            is_valid=False,
            metadata={
                "error": "action could not be parsed",
                "termination_reason": ["untrusted server value"],
                "decision_step": 3,
                "inferred": True,
                "latency_seconds": 0.4,
                "server_metadata": {"history_frames": 2},
            },
        )
        policy = SequencePaperPurePolicy([invalid_output])

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_with_fakes(
                env,
                policy,
                runner_args(output_dir, max_steps=3),
                RaisingController(),
            )
            with open(trajectory_path, newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(env.actions, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "")
        self.assertEqual(
            rows[0]["raw_vlm_output"], "unparseable server output"
        )
        self.assertEqual(rows[0]["policy_protocol"], "paper_pure")
        self.assertEqual(rows[0]["policy_decision_step"], "3")
        self.assertEqual(rows[0]["policy_inferred"], "True")
        self.assertEqual(rows[0]["termination_reason"], "invalid_model_output")
        self.assertEqual(rows[0]["model_latency_seconds"], "0.4")
        saved_metadata = json.loads(rows[0]["policy_metadata"])
        self.assertEqual(saved_metadata["error"], "action could not be parsed")
        self.assertEqual(
            saved_metadata["termination_reason"],
            ["untrusted server value"],
        )
        self.assertEqual(
            saved_metadata["server_metadata"], {"history_frames": 2}
        )
        self.assertIsNone(invalid_output.termination_reason)

    def test_network_failure_terminates_without_fallback(self):
        events = []
        env = FakeEnvironment(events)
        transport = FakeTransport(
            [
                {"episode_id": "episode-0", "reset": True},
                OSError("connection refused"),
            ]
        )
        policy = OfficialNaVIDAHTTPPolicy(transport=transport)

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_with_fakes(
                env,
                policy,
                runner_args(output_dir, max_steps=3),
                RaisingController(),
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(env.actions, [])
        self.assertEqual(row["termination_reason"], "http_request_failed")
        self.assertIn(
            "connection refused", json.loads(row["policy_metadata"])["error"]
        )

    def test_episode_start_failure_is_recorded_without_fallback(self):
        events = []
        env = FakeEnvironment(events)
        transport = FakeTransport([OSError("start connection refused")])
        policy = OfficialNaVIDAHTTPPolicy(transport=transport)

        with tempfile.TemporaryDirectory() as output_dir:
            trajectory_path = run_with_fakes(
                env,
                policy,
                runner_args(output_dir, max_steps=3),
                RaisingController(),
            )
            with open(trajectory_path, newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(env.actions, [])
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(row["termination_reason"], "episode_start_failed")
        self.assertIn(
            "start connection refused",
            json.loads(row["policy_metadata"])["error"],
        )


class ExistingPolicyRunnerTest(unittest.TestCase):
    def test_existing_policy_still_uses_controller_behavior(self):
        events = []
        env = FakeEnvironment(events)

        class ExistingPolicy:
            def predict(self, observation):
                return PolicyOutput("turn_left", "legacy output", True)

        with tempfile.TemporaryDirectory() as output_dir:
            run_with_fakes(
                env,
                ExistingPolicy(),
                runner_args(output_dir),
                ReplacingController(),
            )

        self.assertEqual(env.actions, ["move_forward"])


class OfficialNaVIDACliTest(unittest.TestCase):
    def test_official_arguments_have_localhost_defaults(self):
        with mock.patch.object(
            sys,
            "argv",
            ["habitat_vln_nav.py", "--official-navida-http"],
        ):
            args = habitat_vln_nav.parse_args()

        self.assertTrue(args.official_navida_http)
        self.assertEqual(args.official_navida_url, "http://127.0.0.1:8008")
        self.assertGreater(args.official_navida_timeout, 0.0)
        self.assertIsNone(args.policy_protocol)
        self.assertIsNone(args.gpu_device_id)

    def test_official_main_uses_joint_http_policy_without_loading_qwen(self):
        fake_policy = SimpleNamespace(policy_protocol="paper_pure")
        fake_env = SimpleNamespace(close=mock.Mock())
        argv = [
            "habitat_vln_nav.py",
            "--official-navida-http",
            "--official-navida-url",
            "http://navida.test:9000",
            "--official-navida-timeout",
            "4.5",
            "--gpu-device-id",
            "-1",
        ]

        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            habitat_vln_nav,
            "OfficialNaVIDAHTTPPolicy",
            return_value=fake_policy,
        ) as http_policy_class, mock.patch.object(
            habitat_vln_nav,
            "QwenVLMPolicy",
            side_effect=AssertionError(
                "Qwen must not load in Official HTTP mode"
            ),
        ) as qwen_policy_class, mock.patch.object(
            habitat_vln_nav,
            "NavigationController",
            side_effect=AssertionError(
                "NavigationController must not be constructed in Official mode"
            ),
        ) as controller_class, mock.patch.object(
            habitat_vln_nav,
            "build_env",
            return_value=fake_env,
        ) as build_env_mock, mock.patch.object(
            habitat_vln_nav,
            "run_navigation",
        ) as run_navigation_mock:
            habitat_vln_nav.main()

        http_policy_class.assert_called_once_with(
            base_url="http://navida.test:9000",
            timeout=4.5,
        )
        qwen_policy_class.assert_not_called()
        controller_class.assert_not_called()
        self.assertEqual(
            build_env_mock.call_args.kwargs["gpu_device_id"],
            -1,
        )
        run_args = run_navigation_mock.call_args.args[2]
        self.assertEqual(run_args.frequency_mode, "joint")
        self.assertEqual(run_args.policy_protocol, "paper_pure")
        fake_env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
