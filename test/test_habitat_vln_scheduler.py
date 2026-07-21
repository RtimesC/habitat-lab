import unittest
from types import SimpleNamespace

import numpy as np

from habitat_vln.core import NavigationObservation, PolicyOutput
from habitat_vln.runtime.scheduler import (
    BackgroundPolicyInference,
    active_frequencies,
    advance_inference_time,
    inference_is_due,
    resolve_frequency_mode,
    validate_frequencies,
)


def scheduler_args(**overrides):
    values = {
        "frequency_mode": "layered",
        "qwen_role": "advisor",
        "vision_hz": 5.0,
        "inference_hz": 0.5,
        "joint_hz": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class NavigationSchedulerTest(unittest.TestCase):
    def test_background_inference_separates_submit_and_collect(self):
        class FakePolicy:
            def predict(self, observation):
                return PolicyOutput("follow_goal", str(observation.step), True)

        scheduler = BackgroundPolicyInference(FakePolicy())
        observation = NavigationObservation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            instruction="Move to the goal.",
            step=3,
        )

        self.assertTrue(scheduler.submit(observation))
        scheduler.future.result(timeout=1.0)
        self.assertFalse(scheduler.submit(observation))
        output, duration_sec, source_step = scheduler.collect_if_ready()
        scheduler.close()

        self.assertEqual(output.action, "follow_goal")
        self.assertGreaterEqual(duration_sec, 0.0)
        self.assertEqual(source_step, 3)

    def test_auto_mode_follows_policy_role(self):
        self.assertEqual(resolve_frequency_mode("auto", "advisor"), "layered")
        self.assertEqual(resolve_frequency_mode("auto", "controller"), "joint")

    def test_layered_and_joint_frequencies(self):
        self.assertEqual(active_frequencies(scheduler_args()), (5.0, 0.5))
        self.assertEqual(
            active_frequencies(scheduler_args(frequency_mode="joint")),
            (1.0, 1.0),
        )

    def test_layered_controller_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --qwen-role advisor"):
            validate_frequencies(scheduler_args(qwen_role="controller"))

    def test_inference_deadline_advances_past_current_tick(self):
        self.assertTrue(inference_is_due(2.0, 2.0))
        next_time = advance_inference_time(0.0, inference_hz=0.5, logical_time_sec=2.0)
        self.assertEqual(next_time, 4.0)


if __name__ == "__main__":
    unittest.main()
