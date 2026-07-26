import csv
import json
import tempfile
import unittest
from pathlib import Path

from habitat_vln.evaluation.analyze_closed_loop_failures import (
    analyze_trajectory,
    read_trajectory,
    write_report,
)


class ClosedLoopFailureReportTest(unittest.TestCase):
    def test_report_separates_actions_and_failure_signals(self):
        rows = [
            {
                "episode_index": "0",
                "episode_id": "episode-a",
                "scene_id": "scene-a",
                "distance_to_goal": "5.0",
                "vlm_action": "turn_left",
                "controller_action": "turn_left",
                "executed_action": "turn_left",
                "collision": "False",
                "valid_action": "True",
                "no_progress_steps": "0",
                "success": "0.0",
                "spl": "0.0",
            },
            {
                "episode_index": "0",
                "episode_id": "episode-a",
                "scene_id": "scene-a",
                "distance_to_goal": "4.0",
                "vlm_action": "move_forward",
                "controller_action": "move_forward",
                "executed_action": "move_forward",
                "collision": "True",
                "valid_action": "True",
                "no_progress_steps": "1",
                "success": "0.0",
                "spl": "0.0",
            },
            {
                "episode_index": "0",
                "episode_id": "episode-a",
                "scene_id": "scene-a",
                "distance_to_goal": "6.0",
                "vlm_action": "stop",
                "controller_action": "move_forward",
                "executed_action": "move_forward",
                "collision": "False",
                "valid_action": "False",
                "termination_reason": "invalid_action",
                "error": "fallback used",
                "no_progress_steps": "2",
                "success": "0.0",
                "spl": "0.0",
            },
        ]

        report = analyze_trajectory(rows)
        episode = report["episodes"][0]

        self.assertEqual(episode["steps"], 3)
        self.assertEqual(episode["collision_count"], 1)
        self.assertEqual(episode["controller_override_steps"], 1)
        self.assertEqual(episode["model_execution_override_steps"], 1)
        self.assertEqual(episode["invalid_policy_action_count"], 1)
        self.assertEqual(episode["policy_error_count"], 1)
        self.assertEqual(episode["model_stop_action_count"], 1)
        self.assertEqual(episode["max_no_progress_steps"], 2.0)
        self.assertAlmostEqual(episode["distance_improvement_m"], -1.0)
        self.assertEqual(
            set(episode["failure_reasons"]),
            {
                "collisions",
                "controller_override",
                "model_execution_override",
                "invalid_policy_action",
                "policy_or_execution_error",
                "model_requested_stop_outside_goal",
                "distance_increased",
            },
        )
        self.assertEqual(report["summary"]["episodes"], 1)
        self.assertEqual(report["summary"]["total_collisions"], 1)

    def test_report_reads_old_schema_and_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            trajectory_path = directory / "trajectory.csv"
            with trajectory_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "episode_index",
                        "episode_id",
                        "action",
                        "collided",
                        "goal_distance_m",
                        "success",
                        "spl",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "episode_index": "1",
                        "episode_id": "old-episode",
                        "action": "move_forward",
                        "collided": "False",
                        "goal_distance_m": "2.0",
                        "success": "1.0",
                        "spl": "0.8",
                    }
                )

            report = analyze_trajectory(read_trajectory(trajectory_path))
            output_dir = directory / "report"
            episode_path, summary_path = write_report(report, output_dir)

            self.assertTrue(episode_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(report["summary"]["success_rate"], 1.0)
            self.assertEqual(json.loads(summary_path.read_text())["mean_spl"], 0.8)
            with self.assertRaises(FileExistsError):
                write_report(report, output_dir)


if __name__ == "__main__":
    unittest.main()
