import unittest
from types import SimpleNamespace

from habitat_vln.control import ControllerConfig, NavigationController
from habitat_vln.core import PolicyOutput
from habitat_vln.habitat_vln_nav import parse_args


def navigation_state(**overrides):
    values = {
        "depth_left_m": 1.0,
        "depth_center_m": 1.0,
        "depth_right_m": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class NavigationControllerTest(unittest.TestCase):
    def test_model_stop_is_not_checked_against_hidden_goal_geometry(self):
        controller = NavigationController(ControllerConfig())

        decision = controller.decide(
            PolicyOutput("stop", "raw", True),
            navigation_state(),
            step=0,
            previous_action="none",
            previous_action_count=0,
            previous_collision=False,
        )

        self.assertEqual(decision.action, "stop")
        self.assertTrue(decision.policy_output.is_valid)

    def test_forward_depth_guard_is_disabled_by_default(self):
        controller = NavigationController(ControllerConfig())

        decision = controller.decide(
            PolicyOutput("move_forward", "raw", True),
            navigation_state(
                depth_left_m=0.8,
                depth_center_m=0.2,
                depth_right_m=0.3,
            ),
            step=8,
            previous_action="none",
            previous_action_count=0,
            previous_collision=False,
        )

        self.assertEqual(decision.action, "move_forward")
        self.assertTrue(decision.policy_output.is_valid)

    def test_forward_depth_guard_uses_only_clearer_local_side(self):
        controller = NavigationController(
            ControllerConfig(enable_forward_depth_guard=True)
        )

        decision = controller.decide(
            PolicyOutput("move_forward", "raw", True),
            navigation_state(
                depth_left_m=0.8,
                depth_center_m=0.2,
                depth_right_m=0.3,
            ),
            step=8,
            previous_action="none",
            previous_action_count=0,
            previous_collision=False,
        )

        self.assertEqual(decision.vlm_action, "move_forward")
        self.assertEqual(decision.action, "turn_left")
        self.assertFalse(decision.policy_output.is_valid)
        self.assertEqual(
            decision.policy_output.metadata["forward_depth_guard"][
                "replacement_action"
            ],
            "turn_left",
        )

    def test_parser_exposes_only_direct_controller_mode(self):
        args = parse_args(
            ["--task-config", "benchmark/nav/semantic_indoor_mock.yaml"]
        )

        self.assertEqual(args.qwen_role, "controller")
        self.assertEqual(args.frequency_mode, "joint")
        self.assertFalse(hasattr(args, "navigation_task_mode"))


if __name__ == "__main__":
    unittest.main()
