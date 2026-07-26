import unittest
from types import SimpleNamespace
from unittest.mock import patch

from habitat_vln.control import ControllerConfig, NavigationController
from habitat_vln.core import PolicyOutput
from habitat_vln.habitat_vln_nav import main, parse_args


def navigation_state(**overrides):
    values = {
        "goal_distance_m": 2.0,
        "goal_angle_deg": 0.0,
        "success_distance_m": 0.2,
        "collided": False,
        "depth_left_m": 1.0,
        "depth_center_m": 1.0,
        "depth_right_m": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class NavigationControllerTest(unittest.TestCase):
    def test_advisor_action_is_converted_by_geometry(self):
        controller = NavigationController(
            ControllerConfig(qwen_role="advisor")
        )

        decision = controller.decide(
            PolicyOutput("follow_goal", "raw", True),
            navigation_state(goal_angle_deg=30.0),
            step=8,
            previous_action="none",
            previous_action_count=0,
            previous_collision=False,
            no_progress_steps=0,
        )

        self.assertEqual(decision.vlm_action, "follow_goal")
        self.assertEqual(decision.action, "turn_right")

    def test_anti_stuck_override_moves_forward(self):
        controller = NavigationController(
            ControllerConfig(qwen_role="controller")
        )

        decision = controller.decide(
            PolicyOutput("turn_left", "raw", True),
            navigation_state(),
            step=8,
            previous_action="turn_left",
            previous_action_count=2,
            previous_collision=False,
            no_progress_steps=2,
        )

        self.assertEqual(decision.action, "move_forward")
        self.assertFalse(decision.policy_output.is_valid)
        self.assertIn("anti_stuck_override", decision.policy_output.raw_text)

    def test_forward_depth_guard_is_disabled_by_default(self):
        controller = NavigationController(
            ControllerConfig(qwen_role="controller")
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
            no_progress_steps=0,
        )

        self.assertEqual(decision.action, "move_forward")
        self.assertTrue(decision.policy_output.is_valid)

    def test_forward_depth_guard_turns_to_clearer_side(self):
        controller = NavigationController(
            ControllerConfig(
                qwen_role="controller",
                enable_forward_depth_guard=True,
            )
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
            no_progress_steps=0,
        )

        self.assertEqual(decision.vlm_action, "move_forward")
        self.assertEqual(decision.controller_action, "turn_left")
        self.assertEqual(decision.action, "turn_left")
        self.assertFalse(decision.policy_output.is_valid)
        self.assertEqual(
            decision.policy_output.metadata["forward_depth_guard"],
            {
                "threshold_m": 0.35,
                "depth_left_m": 0.8,
                "depth_center_m": 0.2,
                "depth_right_m": 0.3,
                "replacement_action": "turn_left",
            },
        )

    def test_forward_depth_guard_flag_is_opt_in(self):
        default_args = parse_args([])
        guarded_args = parse_args(["--enable-forward-depth-guard"])

        self.assertFalse(default_args.enable_forward_depth_guard)
        self.assertTrue(guarded_args.enable_forward_depth_guard)
        self.assertEqual(guarded_args.forward_depth_guard_threshold, 0.35)

    def test_output_group_is_optional(self):
        default_args = parse_args([])
        grouped_args = parse_args(
            ["--output-group", "07_six_wheel_visual"]
        )

        self.assertIsNone(default_args.output_group)
        self.assertEqual(grouped_args.output_group, "07_six_wheel_visual")

    def test_forward_depth_guard_rejects_paper_pure_protocol(self):
        argv = [
            "habitat_vln_nav.py",
            "--official-navida-http",
            "--enable-forward-depth-guard",
        ]

        with patch("sys.argv", argv), self.assertRaisesRegex(
            ValueError, "paper_pure runs bypass the controller"
        ):
            main()

    def test_early_stop_is_rejected(self):
        controller = NavigationController(
            ControllerConfig(qwen_role="controller")
        )

        decision = controller.decide(
            PolicyOutput("stop", "raw", True),
            navigation_state(goal_distance_m=0.1),
            step=0,
            previous_action="none",
            previous_action_count=0,
            previous_collision=False,
            no_progress_steps=0,
        )

        self.assertEqual(decision.action, "move_forward")
        self.assertIn("early_stop_rejected", decision.policy_output.raw_text)

    def test_privileged_success_guard_is_labeled_invalid(self):
        config = ControllerConfig(
            qwen_role="controller",
            force_stop_within_success_radius=True,
            allow_early_stop=True,
        )
        controller = NavigationController(config)

        decision = controller.decide(
            PolicyOutput("move_forward", "raw", True),
            navigation_state(goal_distance_m=0.1),
            step=8,
            previous_action="move_forward",
            previous_action_count=1,
            previous_collision=False,
            no_progress_steps=0,
        )

        self.assertEqual(decision.action, "stop")
        self.assertFalse(decision.policy_output.is_valid)
        self.assertIn("success_radius_guard", decision.policy_output.raw_text)


if __name__ == "__main__":
    unittest.main()
