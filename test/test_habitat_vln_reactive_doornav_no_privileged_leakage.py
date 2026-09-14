import inspect
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

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
from habitat_vln.baselines.reactive_doornav.runtime import (
    ReactiveDoorNavConfig,
    ReactiveDoorNavExecutor,
    ReactiveDoorNavPolicy,
    load_reactive_doornav_spec,
)
from habitat_vln.core import NavigationObservation, load_experiment_config
from habitat_vln.habitat_vln_nav import parse_args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "habitat_vln/baselines/reactive_doornav"
PROJECT_DIRECTION_PATH = PROJECT_ROOT / "habitat_vln/PROJECT_DIRECTION.md"
CONTRACTS_PATH = PACKAGE_ROOT / "contracts.py"
RUNTIME_PATH = PACKAGE_ROOT / "runtime.py"
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
                    {field.name for field in fields(contract)}
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

    def test_project_direction_defines_b1_hidden_target_boundary(self):
        direction = " ".join(
            PROJECT_DIRECTION_PATH.read_text(encoding="utf-8")
            .casefold()
            .split()
        )

        for forbidden_input in [
            "target distance",
            "target bearing",
            "success radius",
            "shortest path",
            "oracle waypoint",
        ]:
            with self.subTest(forbidden_input=forbidden_input):
                self.assertIn(forbidden_input, direction)
        self.assertIn("reactive visual doornav b1", direction)
        self.assertIn("不得进入 grounder、tracker、executor 或 policy", direction)

    def test_project_direction_documents_image_space_cues_honestly(self):
        direction = " ".join(
            PROJECT_DIRECTION_PATH.read_text(encoding="utf-8")
            .casefold()
            .split()
        )

        self.assertIn("image-space observable cues", direction)
        self.assertIn("bbox area ratio 不是真实物理距离", direction)
        self.assertIn("unvalidated engineering default", direction)
        self.assertIn(
            "future belief-graph and active-verification method",
            direction,
        )
        self.assertNotIn("future #8", direction)
        for forbidden_field in B1_FORBIDDEN_METRIC_FIELDS:
            with self.subTest(forbidden_field=forbidden_field):
                self.assertNotIn(forbidden_field, direction)

    def test_b1_docs_and_configs_are_rgb_first_and_sensor_neutral(self):
        direction = PROJECT_DIRECTION_PATH.read_text(
            encoding="utf-8"
        ).casefold()

        self.assertIn("b1 的核心视觉输入只有当前 rgb", direction)
        self.assertIn("默认不启用 depth", direction)
        self.assertIn("target-free", direction)
        for forbidden_requirement in [
            "instruction + rgb-d",
            "rgb-d local target",
            "current rgb and depth",
            "depth_safety_threshold_m",
        ]:
            with self.subTest(forbidden_requirement=forbidden_requirement):
                self.assertNotIn(forbidden_requirement, direction)

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
                "policy",
                "baseline_spec",
                "frequency_mode",
                "no_rate_limit",
            },
        )

        self.assertEqual(config.name, "reactive_doornav_b1")
        self.assertEqual(
            set(config.arguments),
            {"policy", "baseline_spec", "frequency_mode", "no_rate_limit"},
        )
        self.assertEqual(config.arguments["policy"], "reactive_doornav_b1")

    def test_versioned_baseline_spec_uses_image_space_arrival_settings(self):
        specification = load_reactive_doornav_spec(BASELINE_SPEC_PATH)

        self.assertEqual(
            {
                key: getattr(specification.config, key)
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
        self.assertNotIn(
            "observable_stop_distance_m",
            ReactiveDoorNavConfig.__dataclass_fields__,
        )
        self.assertEqual(specification.status, "engineering_baseline")

    def test_public_runtime_and_cli_do_not_expose_privileged_fields(self):
        config_fields = set(ReactiveDoorNavConfig.__dataclass_fields__)
        policy_parameters = set(
            inspect.signature(ReactiveDoorNavPolicy.predict).parameters
        )
        cli_fields = vars(parse_args([]))

        self.assert_names_do_not_expose_privileged_fields(config_fields)
        self.assert_names_do_not_expose_privileged_fields(policy_parameters)
        self.assert_names_do_not_expose_privileged_fields(cli_fields)
        self.assertNotIn("goal", cli_fields)

    def test_policy_rejects_privileged_navigation_context_at_runtime(self):
        forbidden_contexts = [
            {"goal_distance": 1.0},
            {"building_prior": {"target_position": [1.0, 2.0, 3.0]}},
            {"goal_bearing": 0.2},
            {"target_distance": 3.0},
            {"target_bearing": -0.2},
            {"geodesic_route": ["left", "forward"]},
            {"oracle_route": ["forward"]},
            {"target_coordinates": [1.0, 2.0, 3.0]},
            {"goal_steering": "turn_right"},
        ]

        for step, navigation_context in enumerate(forbidden_contexts):
            with self.subTest(navigation_context=navigation_context):
                policy = ReactiveDoorNavPolicy()
                output = policy.predict(
                    NavigationObservation(
                        rgb=np.zeros((40, 60, 3), dtype=np.uint8),
                        instruction="Approach a visible doorway.",
                        step=step,
                        navigation_context=navigation_context,
                    )
                )

                self.assertFalse(output.is_valid)
                self.assertIsNone(output.action)
                self.assertEqual(
                    output.termination_reason,
                    "invalid_observation",
                )
                self.assertIn("privileged", output.metadata["error"])

    def test_executor_rejects_context_outside_target_free_allowlist(self):
        executor = ReactiveDoorNavExecutor()
        observation = NavigationObservation(
            rgb=np.zeros((40, 60, 3), dtype=np.uint8),
            instruction="Approach a visible doorway.",
            step=0,
            navigation_context={"goal_distance": 1.0},
        )

        with self.assertRaisesRegex(ValueError, "privileged"):
            executor.step(observation, None)

    def test_b1_building_prior_accepts_text_but_rejects_nested_state(self):
        allowed_output = ReactiveDoorNavPolicy().predict(
            NavigationObservation(
                rgb=np.zeros((40, 60, 3), dtype=np.uint8),
                instruction="Approach a visible doorway.",
                step=0,
                navigation_context={
                    "building_prior": {
                        "building_id": "teaching-a",
                        "floor_labels": ["1", "2"],
                    }
                },
            )
        )
        self.assertTrue(allowed_output.is_valid)

        forbidden_priors = [
            {
                "public_structure_summary": {
                    "destination": {
                        "position": [1, 2, 3],
                        "bearing": 0.5,
                        "route": ["left"],
                    }
                }
            },
            {"public_structure_summary": [1.0, 2.0, 3.0]},
        ]
        for building_prior in forbidden_priors:
            with self.subTest(building_prior=building_prior):
                output = ReactiveDoorNavPolicy().predict(
                    NavigationObservation(
                        rgb=np.zeros((40, 60, 3), dtype=np.uint8),
                        instruction="Approach a visible doorway.",
                        step=0,
                        navigation_context={"building_prior": building_prior},
                    )
                )
                self.assertFalse(output.is_valid)
                self.assertIn("building_prior", output.metadata["error"])

    def test_runtime_import_does_not_load_habitat_sim(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import habitat_vln.baselines.reactive_doornav.runtime; "
                    "assert 'habitat_sim' not in sys.modules"
                ),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_project_direction_is_only_tracked_habitat_vln_markdown(self):
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "habitat_vln/*.md",
                "habitat_vln/**/*.md",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        existing_paths = {
            path
            for path in result.stdout.splitlines()
            if (PROJECT_ROOT / path).is_file()
        }

        self.assertEqual(existing_paths, {"habitat_vln/PROJECT_DIRECTION.md"})
        self.assertFalse((PACKAGE_ROOT / "README.md").exists())

    def test_navigation_observation_still_allows_optional_depth(self):
        observation = NavigationObservation(
            rgb="rgb",
            instruction="Approach the visible door.",
            depth=None,
        )

        self.assertIsNone(observation.depth)


if __name__ == "__main__":
    unittest.main()
