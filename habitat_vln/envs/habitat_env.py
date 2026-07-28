"""Habitat environment construction and executable action mapping."""

import os
from dataclasses import dataclass
from typing import Optional

import habitat
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from .six_wheel_robot import (
    ROBOT_BODY_NONE,
    ROBOT_BODY_SIX_WHEEL,
    SixWheelRobotConfig,
    SixWheelRobotEnvironment,
    configure_six_wheel_sensors,
)

ACTION_MAP = {
    "turn_left": HabitatSimActions.turn_left,
    "turn_right": HabitatSimActions.turn_right,
    "move_forward": HabitatSimActions.move_forward,
    "stop": HabitatSimActions.stop,
}


@dataclass(frozen=True)
class HabitatEnvironmentConfig:
    """Inputs required to construct one Habitat navigation environment."""

    task_config: str
    width: int = 640
    height: int = 480
    hfov: int = 90
    scene: Optional[str] = None
    dataset_split: Optional[str] = None
    dataset_path: Optional[str] = None
    scenes_dir: Optional[str] = None
    gpu_device_id: Optional[int] = None
    robot_body: str = ROBOT_BODY_NONE
    robot_camera_height: float = 0.62
    robot_debug_view: bool = False


def create_habitat_env(settings):
    """Create Habitat with dataset and visual-sensor overrides applied."""
    if not settings.task_config:
        raise ValueError("A semantic-indoor task config must be supplied")
    if settings.robot_body not in {ROBOT_BODY_NONE, ROBOT_BODY_SIX_WHEEL}:
        raise ValueError(
            "robot_body must be one of "
            f"{ROBOT_BODY_NONE!r} or {ROBOT_BODY_SIX_WHEEL!r}"
        )
    if settings.robot_camera_height <= 0:
        raise ValueError("robot_camera_height must be greater than zero")

    robot_config = None
    if settings.robot_body == ROBOT_BODY_SIX_WHEEL:
        robot_config = SixWheelRobotConfig(
            camera_height=settings.robot_camera_height,
            debug_view=settings.robot_debug_view,
            debug_view_width=settings.width,
            debug_view_height=settings.height,
            debug_view_hfov=settings.hfov,
        )

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
        if robot_config is not None:
            configure_six_wheel_sensors(sensors, robot_config)

    environment = habitat.Env(config=config)
    if robot_config is not None:
        return SixWheelRobotEnvironment(environment, robot_config)
    return environment


def build_env(
    task_config,
    width=640,
    height=480,
    hfov=90,
    scene=None,
    dataset_split=None,
    dataset_path=None,
    scenes_dir=None,
    gpu_device_id=None,
    robot_body=ROBOT_BODY_NONE,
    robot_camera_height=0.62,
    robot_debug_view=False,
):
    """Build an environment from an explicit semantic-indoor task config."""
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
            robot_body=robot_body,
            robot_camera_height=robot_camera_height,
            robot_debug_view=robot_debug_view,
        )
    )


def step_navigation_action(env, action_name):
    """Execute one project-standard action name in Habitat."""
    try:
        action = ACTION_MAP[action_name]
    except KeyError as exc:
        raise ValueError(f"Unknown navigation action: {action_name}") from exc
    return env.step(action)
