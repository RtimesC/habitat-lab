import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from habitat_vln.envs import ACTION_MAP
from habitat_vln.habitat_vln_nav import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOCK_CONFIG = PROJECT_ROOT / "habitat_vln/configs/runtime/mock_hm3d_smoke.yaml"


class FakeHabitatEnv:
    """Small deterministic environment that exercises the complete runtime."""

    def __init__(self):
        depth_sensor = SimpleNamespace(
            min_depth=0.0,
            max_depth=10.0,
            normalize_depth=False,
        )
        self.config = SimpleNamespace(
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
        self.rotation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        self.agent_state = SimpleNamespace(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=self.rotation,
        )
        self.sim = SimpleNamespace(
            previous_step_collided=False,
            get_agent_state=lambda: self.agent_state,
        )
        self.current_episode = None
        self.episode_over = False
        self.closed = False
        self.distance = 2.0
        self.steps = 0

    def _observation(self):
        value = min(255, 40 + self.steps * 30)
        return {
            "rgb": np.full((48, 64, 3), value, dtype=np.uint8),
            "depth": np.full((48, 64, 1), 2.0, dtype=np.float32),
            "pointgoal_with_gps_compass": np.array(
                [self.distance, 0.0], dtype=np.float32
            ),
        }

    def reset(self):
        self.distance = 2.0
        self.steps = 0
        self.episode_over = False
        self.current_episode = SimpleNamespace(
            episode_id="mock-episode",
            scene_id="mock-scene",
            info={"instruction": "Move to the mock goal."},
            goals=[SimpleNamespace(position=[0.0, 0.0, -2.0])],
        )
        return self._observation()

    def step(self, action):
        if action == ACTION_MAP["move_forward"]:
            self.distance = max(0.0, self.distance - 0.25)
            self.agent_state.position[2] -= 0.25
        self.steps += 1
        return self._observation()

    def get_metrics(self):
        return {
            "distance_to_goal": self.distance,
            "success": float(self.distance < 0.2),
            "spl": 0.0,
        }

    def close(self):
        self.closed = True


class HabitatVLNEndToEndTest(unittest.TestCase):
    def test_yaml_mock_run_writes_trajectory_frames_and_video(self):
        env = FakeHabitatEnv()
        with tempfile.TemporaryDirectory() as temporary_directory:
            argv = [
                "habitat_vln_nav.py",
                "--experiment-config",
                str(MOCK_CONFIG),
                "--output-dir",
                temporary_directory,
                "--max-steps",
                "3",
            ]
            with patch("sys.argv", argv), patch(
                "habitat_vln.habitat_vln_nav.build_env",
                return_value=env,
            ):
                main()

            run_directories = list(Path(temporary_directory).glob("run_*"))
            self.assertEqual(len(run_directories), 1)
            run_directory = run_directories[0]

            with (run_directory / "trajectory.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["frequency_mode"], "joint")
            self.assertEqual(rows[0]["vlm_action"], "turn_left")
            self.assertEqual(rows[1]["vlm_action"], "move_forward")
            self.assertEqual(rows[1]["action"], "move_forward")
            self.assertTrue(all(Path(row["image"]).is_file() for row in rows))
            self.assertTrue((run_directory / "episode_000.mp4").is_file())
            self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
