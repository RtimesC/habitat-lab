"""Habitat environment construction and executable action mapping."""

import os
from dataclasses import dataclass
from typing import Optional

import habitat
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions


DEFAULT_TASK_CONFIG = "benchmark/nav/vln_r2r.yaml"

ACTION_MAP = {
    "turn_left": HabitatSimActions.turn_left,
    "turn_right": HabitatSimActions.turn_right,
    "move_forward": HabitatSimActions.move_forward,
    "stop": HabitatSimActions.stop,
}


@dataclass(frozen=True)
class HabitatEnvironmentConfig:
    """Inputs required to construct one Habitat navigation environment."""

    task_config: str = DEFAULT_TASK_CONFIG
    width: int = 640
    height: int = 480
    hfov: int = 90
    scene: Optional[str] = None
    dataset_split: Optional[str] = None
    dataset_path: Optional[str] = None
    scenes_dir: Optional[str] = None
    gpu_device_id: Optional[int] = None


def create_habitat_env(settings):
    """Create Habitat with dataset and visual-sensor overrides applied."""
    config = habitat.get_config(settings.task_config)
    with read_write(config):
        if settings.dataset_split is not None:
            config.habitat.dataset.split = settings.dataset_split
        if settings.dataset_path is not None:
            config.habitat.dataset.data_path = os.path.expanduser(
                settings.dataset_path
            )
        if settings.scenes_dir is not None:
            config.habitat.dataset.scenes_dir = os.path.expanduser(
                settings.scenes_dir
            )
        if settings.gpu_device_id is not None:
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                settings.gpu_device_id
            )
        if settings.scene is not None:
            config.habitat.simulator.scene = os.path.expanduser(settings.scene)
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        sensors.rgb_sensor.width = settings.width
        sensors.rgb_sensor.height = settings.height
        sensors.rgb_sensor.hfov = settings.hfov
        sensors.depth_sensor.width = settings.width
        sensors.depth_sensor.height = settings.height
        sensors.depth_sensor.hfov = settings.hfov
    return habitat.Env(config=config)


def build_env(
    task_config=DEFAULT_TASK_CONFIG,
    width=640,
    height=480,
    hfov=90,
    scene=None,
    dataset_split=None,
    dataset_path=None,
    scenes_dir=None,
    gpu_device_id=None,
):
    """Compatibility wrapper around :func:`create_habitat_env`."""
    return create_habitat_env(
        HabitatEnvironmentConfig(
            task_config=task_config,
            width=width,
            height=height,
            hfov=hfov,
            scene=scene,
            dataset_split=dataset_split,
            dataset_path=dataset_path,
            scenes_dir=scenes_dir,
            gpu_device_id=gpu_device_id,
        )
    )


def step_navigation_action(env, action_name):
    """Execute one project-standard action name in Habitat."""
    try:
        action = ACTION_MAP[action_name]
    except KeyError as exc:
        raise ValueError(f"Unknown navigation action: {action_name}") from exc
    return env.step(action)
