import inspect
import unittest
from dataclasses import fields
from pathlib import Path

import yaml

from habitat_vln.baselines.reactive_doornav import (
    PRIVILEGED_FIELD_DENYLIST,
    DoorCandidate,
    DoorGrounder,
    LocalExecutionDecision,
    LocalNavigationExecutor,
    LocalSubgoal,
)
from habitat_vln.core import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "habitat_vln/baselines/reactive_doornav"
README_PATH = PACKAGE_ROOT / "README.md"
BASELINE_SPEC_PATH = PACKAGE_ROOT / "baseline_spec.yaml"
CONFIG_PATH = (
    PROJECT_ROOT / "habitat_vln/configs/runtime/reactive_doornav_b1.yaml"
)


class DoorNavNoPrivilegedLeakageTest(unittest.TestCase):
    def assert_names_do_not_expose_privileged_fields(self, names):
        leaked_names = sorted(set(names) & PRIVILEGED_FIELD_DENYLIST)
        self.assertEqual(leaked_names, [])

    def test_denylist_covers_hidden_target_geometry_and_oracle_routes(self):
        self.assertTrue(
            {
                "goal",
                "goal_position",
                "target_position",
                "goal_distance",
                "goal_distance_m",
                "goal_angle",
                "goal_angle_deg",
                "success_distance",
                "success_distance_m",
                "success_radius",
                "shortest_path",
                "geodesic_distance",
                "oracle_waypoint",
            }.issubset(PRIVILEGED_FIELD_DENYLIST)
        )

    def test_public_dataclasses_do_not_expose_privileged_fields(self):
        for contract in [
            DoorCandidate,
            LocalSubgoal,
            LocalExecutionDecision,
        ]:
            with self.subTest(contract=contract.__name__):
                self.assert_names_do_not_expose_privileged_fields(
                    field.name for field in fields(contract)
                )

    def test_grounder_signature_does_not_accept_privileged_fields(self):
        parameters = inspect.signature(DoorGrounder.detect).parameters

        self.assert_names_do_not_expose_privileged_fields(parameters)

    def test_executor_signature_does_not_accept_privileged_fields(self):
        parameters = inspect.signature(LocalNavigationExecutor.step).parameters

        self.assert_names_do_not_expose_privileged_fields(parameters)

    def test_readme_explicitly_forbids_hidden_target_geometry(self):
        readme = README_PATH.read_text(encoding="utf-8").casefold()

        for forbidden_input in [
            "hidden target coordinates",
            "target distance",
            "target bearing",
            "success radius",
            "shortest path",
            "oracle waypoint",
        ]:
            with self.subTest(forbidden_input=forbidden_input):
                self.assertIn(forbidden_input, readme)
        self.assertIn(
            "must never enter the grounder, target estimator, state machine, "
            "or local executor",
            readme,
        )

    def test_future_config_has_no_privileged_target_fields(self):
        for path in [CONFIG_PATH, BASELINE_SPEC_PATH]:
            config_text = path.read_text(encoding="utf-8").casefold()
            for forbidden_field in [
                "goal_distance",
                "goal_angle",
                "success_radius",
                "shortest_path",
                "oracle_waypoint",
                "oracle waypoint",
            ]:
                with self.subTest(
                    path=path.name, forbidden_field=forbidden_field
                ):
                    self.assertNotIn(forbidden_field, config_text)

    def test_future_config_keeps_current_arguments_loader_compatible(self):
        config = load_experiment_config(
            CONFIG_PATH,
            valid_arguments={
                "instruction",
                "output_dir",
            },
        )

        self.assertEqual(config.name, "reactive_doornav_b1")
        self.assertEqual(set(config.arguments), {"instruction", "output_dir"})

    def test_versioned_baseline_spec_preserves_future_b1_settings(self):
        with BASELINE_SPEC_PATH.open(encoding="utf-8") as handle:
            specification = yaml.safe_load(handle)

        self.assertEqual(
            {
                key: specification[key]
                for key in [
                    "max_search_steps",
                    "max_episode_steps",
                    "observable_stop_distance_m",
                    "grounding_confidence_threshold",
                    "target_lost_tolerance_steps",
                    "no_progress_tolerance_steps",
                    "depth_safety_threshold_m",
                ]
            },
            {
                "max_search_steps": 24,
                "max_episode_steps": 200,
                "observable_stop_distance_m": 0.45,
                "grounding_confidence_threshold": 0.55,
                "target_lost_tolerance_steps": 3,
                "no_progress_tolerance_steps": 8,
                "depth_safety_threshold_m": 0.35,
            },
        )
        self.assertEqual(specification["status"], "scaffold_only")


if __name__ == "__main__":
    unittest.main()
