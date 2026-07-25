import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from habitat_vln.data.create_grounded_demo_episode import (
    EPISODE_ID,
    GOAL_POSITION,
    INSTRUCTION,
    SCENE_ID,
    build_payload,
    main as create_demo_main,
)
from habitat_vln.evaluation.evaluate_pointnav_policies import parse_args


class GroundedDemoEpisodeTests(unittest.TestCase):
    def test_payload_contains_expected_grounded_episode(self):
        payload = build_payload()

        self.assertFalse(payload["metadata"]["benchmark_comparable"])
        self.assertEqual(payload["metadata"]["episode_count"], 1)

        episode = payload["episodes"][0]

        self.assertEqual(episode["episode_id"], EPISODE_ID)
        self.assertEqual(episode["scene_id"], SCENE_ID)
        self.assertEqual(episode["goals"][0]["position"], GOAL_POSITION)
        self.assertEqual(episode["info"]["instruction"], INSTRUCTION)
        self.assertEqual(episode["info"]["route_turn_count"], 2)
        self.assertFalse(episode["info"]["benchmark_comparable"])
        self.assertIn("horse paintings", episode["info"]["instruction"])

    def test_generator_writes_valid_gzip_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "val.json.gz"

            argv = [
                "create_grounded_demo_episode",
                "--output",
                str(output),
            ]

            with patch.object(sys, "argv", argv):
                create_demo_main()

            self.assertTrue(output.is_file())

            with gzip.open(output, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(len(payload["episodes"]), 1)
            self.assertEqual(
                payload["episodes"][0]["episode_id"],
                EPISODE_ID,
            )
            self.assertFalse(
                payload["metadata"]["benchmark_comparable"]
            )

    def test_pointnav_cli_accepts_gpu_device_override(self):
        argv = [
            "evaluate_pointnav_policies",
            "--gpu-device-id",
            "-1",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.gpu_device_id, -1)


if __name__ == "__main__":
    unittest.main()
