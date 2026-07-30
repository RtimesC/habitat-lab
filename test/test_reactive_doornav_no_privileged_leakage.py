import inspect
import unittest
from dataclasses import fields
from pathlib import Path

import yaml

import habitat_vln.baselines.reactive_doornav as doornav
from habitat_vln.baselines.reactive_doornav import (
    PRIVILEGED_FIELD_DENYLIST,
    DoorCandidate,
    DoorGrounder,
    LocalExecutionDecision,
    ReactiveVisualExecutor,
    VisualTargetTrack,
    VisualTargetTracker,
)
from habitat_vln.core import NavigationObservation, load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "habitat_vln/baselines/reactive_doornav"
README_PATH = PACKAGE_ROOT / "README.md"
CONTRACTS_PATH = PACKAGE_ROOT / "contracts.py"
BASELINE_SPEC_PATH = PACKAGE_ROOT / "baseline_spec.yaml"
CONFIG_PATH = (
    PROJECT_ROOT / "habitat_vln/configs/runtime/reactive_doornav_b1.yaml"
)
B1_FORBIDDEN_METRIC_FIELDS = {
    "relative_x_m",
    "relative_y_m",
    "desired_heading_rad",
    "stop_distance_m",
}


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
                "global_x",
                "global_y",
                "waypoint_position",
            }.issubset(PRIVILEGED_FIELD_DENYLIST)
        )

    def test_b1_package_does_not_export_metric_target_contracts(self):
        for legacy_name in ["LocalSubgoal", "LocalNavigationExecutor"]:
            with self.subTest(legacy_name=legacy_name):
                self.assertFalse(hasattr(doornav, legacy_name))
                self.assertNotIn(legacy_name, doornav.__all__)

    def test_public_dataclasses_do_not_expose_privileged_fields(self):
        for contract in [
            DoorCandidate,
            VisualTargetTrack,
            LocalExecutionDecision,
        ]:
            with self.subTest(contract=contract.__name__):
                self.assert_names_do_not_expose_privileged_fields(
                    field.name for field in fields(contract)
                )

    def test_public_dataclasses_do_not_expose_metric_target_fields(self):
        for contract in [
            DoorCandidate,
            VisualTargetTrack,
            LocalExecutionDecision,
        ]:
            with self.subTest(contract=contract.__name__):
                leaked_names = sorted(
                    {
                        field.name for field in fields(contract)
                    }
                    & B1_FORBIDDEN_METRIC_FIELDS
                )
                self.assertEqual(leaked_names, [])

    def test_contract_source_removes_metric_target_fields(self):
        contracts = CONTRACTS_PATH.read_text(encoding="utf-8")

        for forbidden_field in B1_FORBIDDEN_METRIC_FIELDS:
            with self.subTest(forbidden_field=forbidden_field):
                self.assertNotIn(forbidden_field, contracts)

    def test_protocol_signatures_do_not_accept_privileged_fields(self):
        for protocol_method in [
            DoorGrounder.detect,
            VisualTargetTracker.update,
            ReactiveVisualExecutor.step,
        ]:
            with self.subTest(method=protocol_method.__qualname__):
                parameters = inspect.signature(protocol_method).parameters
                self.assert_names_do_not_expose_privileged_fields(parameters)

    def test_readme_explicitly_forbids_hidden_target_geometry(self):
        readme = " ".join(
            README_PATH.read_text(encoding="utf-8").casefold().split()
        )

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
            "must never enter the grounder, visual tracker",
            readme,
        )
        self.assertIn("state machine, or reactive visual executor", readme)

    def test_readme_documents_image_space_track_without_distance_claim(self):
        readme = " ".join(
            README_PATH.read_text(encoding="utf-8").casefold().split()
        )

        self.assertIn(
            "visualtargettrack` is an image-space observation record, not a "
            "physical distance estimate",
            readme,
        )
        self.assertIn("it is not absolute physical distance truth", readme)
        self.assertIn(
            "plain rgb bounding boxes do not provide metric robot-local "
            "position or absolute distance",
            readme,
        )
        self.assertNotIn("localsubgoal", readme)
        self.assertNotIn("localnavigationexecutor", readme)
        for forbidden_field in B1_FORBIDDEN_METRIC_FIELDS:
            with self.subTest(forbidden_field=forbidden_field):
                self.assertNotIn(forbidden_field, readme)

    def test_b1_docs_and_configs_are_rgb_first_and_sensor_neutral(self):
        readme = README_PATH.read_text(encoding="utf-8").casefold()

        self.assertIn("rgb is b1's core visual input", readme)
        self.assertIn("b1 does not require an rgb-d camera", readme)
        self.assertEqual(readme.count("rgb-d"), 1)
        self.assertIn(
            "optional target-free platform safety adapter",
            readme,
        )
        for forbidden_requirement in [
            "instruction + rgb-d",
            "rgb-d local target",
            "current rgb and depth",
            "depth_safety_threshold_m",
        ]:
            with self.subTest(forbidden_requirement=forbidden_requirement):
                self.assertNotIn(forbidden_requirement, readme)

        for path in [CONFIG_PATH, BASELINE_SPEC_PATH]:
            config_text = path.read_text(encoding="utf-8").casefold()
            for forbidden_requirement in [
                "rgb-d",
                "depth camera",
                "depth_safety_threshold_m",
            ]:
                with self.subTest(
                    path=path.name,
                    forbidden_requirement=forbidden_requirement,
                ):
                    self.assertNotIn(forbidden_requirement, config_text)

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

    def test_versioned_baseline_spec_uses_image_space_arrival_settings(self):
        with BASELINE_SPEC_PATH.open(encoding="utf-8") as handle:
            specification = yaml.safe_load(handle)

        self.assertEqual(
            {
                key: specification[key]
                for key in [
                    "max_search_steps",
                    "max_episode_steps",
                    "arrival_area_ratio_threshold",
                    "arrival_center_tolerance_norm",
                    "arrival_confirm_frames",
                    "grounding_confidence_threshold",
                    "target_lost_tolerance_steps",
                    "no_progress_tolerance_steps",
                ]
            },
            {
                "max_search_steps": 24,
                "max_episode_steps": 200,
                "arrival_area_ratio_threshold": 0.18,
                "arrival_center_tolerance_norm": 0.15,
                "arrival_confirm_frames": 3,
                "grounding_confidence_threshold": 0.55,
                "target_lost_tolerance_steps": 3,
                "no_progress_tolerance_steps": 8,
            },
        )
        self.assertNotIn("observable_stop_distance_m", specification)
        self.assertEqual(specification["status"], "scaffold_only")

    def test_navigation_observation_still_allows_optional_depth(self):
        observation = NavigationObservation(
            rgb="rgb",
            instruction="Approach the visible door.",
            depth=None,
        )

        self.assertIsNone(observation.depth)


if __name__ == "__main__":
    unittest.main()
