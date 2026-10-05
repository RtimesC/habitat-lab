#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
XJTLUCarEnv: Low-chassis robot environment wrapper for embodied VLN research.

Designed for low-viewpoint spatial reasoning and 3D Gaussian Splatting (3DGS)
pointcloud back-projection on the differential drive mobile robot.
"""

import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import quaternion
from omegaconf import DictConfig

import habitat
from habitat.config.default import get_config
from habitat.core.env import Env


class XJTLUCarEnv:
    r"""Environment wrapper specifically tailored for the XJTLU low-chassis VLN car.

    Physical & Sensor Specifications:
    - Chassis: agent_radius = 0.38625 m, agent_height = 0.5 m
    - Camera: centered forward-facing at height = 0.45 m
    - Sensor resolution: 640 x 480 (width x height), HFOV = 90.0 deg
    - Control: differential drive (linear_velocity v, angular_velocity w), 10Hz (dt = 0.1s)
    """

    DEFAULT_CONFIG_PATHS = [
        "configs/xjtlu/xjtlu_car_agent.yaml",
        "benchmark/nav/xjtlu_car_agent.yaml",
        "xjtlu/xjtlu_car_agent.yaml",
    ]

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Union[DictConfig, Any]] = None,
        env: Optional[Env] = None,
        overrides: Optional[List[str]] = None,
        dt: float = 0.1,
    ):
        r"""Initialize XJTLUCarEnv.

        Args:
            config_path: Optional path to the environment YAML configuration.
            config: Optional pre-loaded DictConfig.
            env: Optional existing habitat.Env instance to wrap.
            overrides: Optional list of Hydra configuration overrides.
            dt: Control cycle time step in seconds (default 0.1s for 10Hz).
        """
        self.dt = float(dt)
        self._owns_env = False

        if env is not None:
            self._env = env
        else:
            if config is None:
                if config_path is None:
                    # Search default config paths
                    for p in self.DEFAULT_CONFIG_PATHS:
                        if osp.exists(p):
                            config_path = p
                            break
                    if config_path is None:
                        config_path = self.DEFAULT_CONFIG_PATHS[0]
                config = get_config(config_path, overrides=overrides)
            self._env = habitat.Env(config=config)
            self._owns_env = True

        self._last_raw_obs: Optional[Dict[str, Any]] = None
        self._last_is_collision: bool = False

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def env(self) -> Env:
        return self._env

    @property
    def sim(self) -> Any:
        return self._env.sim

    @property
    def task(self) -> Any:
        return self._env.task

    @property
    def current_episode(self) -> Any:
        return self._env.current_episode

    @property
    def episodes(self) -> List[Any]:
        return self._env.episodes

    @property
    def episode_over(self) -> bool:
        return self._env.episode_over

    def reset(self) -> Dict[str, Any]:
        r"""Resets the environment and returns the standardized observation dict.

        Returns:
            Dict containing:
            - 'rgb': np.ndarray of shape (480, 640, 3), dtype uint8
            - 'depth': np.ndarray of shape (480, 640, 1), dtype float32
            - 'camera_pose': np.ndarray of shape (4, 4), dtype float32 (T_wc)
            - 'instruction': str navigation instruction for current episode
        """
        raw_obs = self._env.reset()
        self._last_raw_obs = raw_obs
        self._last_is_collision = False
        return self._format_observation(raw_obs)

    def step(
        self, linear_vel: float, angular_vel: float
    ) -> Tuple[Dict[str, Any], bool, float]:
        r"""Executes a differential drive step and advances simulation by dt (0.1s).

        Args:
            linear_vel: Linear velocity v in m/s (forward > 0, backward < 0).
            angular_vel: Angular velocity w in rad/s (left/counter-clockwise > 0, right < 0).

        Returns:
            Tuple of:
            - obs: Standardized observation dict (rgb, depth, camera_pose, instruction).
            - is_collision: Boolean indicating if a collision occurred in this step.
            - distance_to_goal: Geodesic distance to goal in meters (float or NaN).
        """
        # Execute differential drive action
        raw_obs = self._execute_velocity_action(linear_vel, angular_vel)
        self._last_raw_obs = raw_obs

        # Detect collision
        is_collision = self._check_collision()
        self._last_is_collision = is_collision

        # Compute geodesic distance to goal
        distance_to_goal = self._compute_distance_to_goal()

        obs = self._format_observation(raw_obs)
        return obs, is_collision, distance_to_goal

    def _execute_velocity_action(
        self, linear_vel: float, angular_vel: float
    ) -> Dict[str, Any]:
        r"""Executes velocity control via Habitat's action manager or direct sim integration."""
        task = getattr(self._env, "task", None)
        has_velocity_action = (
            task is not None
            and hasattr(task, "actions")
            and "velocity_control" in task.actions
        )

        if has_velocity_action:
            action_obj = task.actions["velocity_control"]
            action_cfg = action_obj._config
            min_lin, max_lin = action_cfg.lin_vel_range
            min_ang, max_ang = action_cfg.ang_vel_range  # In degrees/sec

            # Convert angular velocity from rad/s to deg/s
            ang_deg = float(np.rad2deg(angular_vel))
            lin_val = float(linear_vel)

            # Map to [-1, 1] range expected by Habitat's VelocityAction
            norm_lin = float(
                np.clip(
                    2.0 * (lin_val - min_lin) / (max_lin - min_lin) - 1.0,
                    -1.0,
                    1.0,
                )
            )
            norm_ang = float(
                np.clip(
                    2.0 * (ang_deg - min_ang) / (max_ang - min_ang) - 1.0,
                    -1.0,
                    1.0,
                )
            )

            raw_obs = self._env.step(
                {
                    "action": "velocity_control",
                    "action_args": {
                        "linear_velocity": norm_lin,
                        "angular_velocity": norm_ang,
                        "time_step": self.dt,
                    },
                }
            )
            return raw_obs
        else:
            # Fallback to direct simulation integration using habitat_sim.physics.VelocityControl
            return self._step_sim_direct(linear_vel, angular_vel)

    def _step_sim_direct(
        self, linear_vel: float, angular_vel: float
    ) -> Dict[str, Any]:
        r"""Directly integrates differential drive velocity via simulator physics and navmesh."""
        import habitat_sim
        import magnum as mn

        sim = self._env.sim
        vc = habitat_sim.physics.VelocityControl()
        vc.controlling_lin_vel = True
        vc.controlling_ang_vel = True
        vc.lin_vel_is_local = True
        vc.ang_vel_is_local = True

        # In Habitat coordinate system: -Z is forward, +Y is up (rotation around +Y turns left)
        vc.linear_velocity = np.array([0.0, 0.0, -float(linear_vel)])
        vc.angular_velocity = np.array([0.0, float(angular_vel), 0.0])

        agent_state = sim.get_agent_state()
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = habitat_sim.RigidState(
            agent_mn_quat, agent_state.position
        )

        goal_rigid_state = vc.integrate_transform(self.dt, current_rigid_state)

        # Navmesh step with sliding or no-sliding
        allow_sliding = getattr(sim.config.sim_cfg, "allow_sliding", True)
        step_fn = (
            sim.pathfinder.try_step
            if allow_sliding
            else sim.pathfinder.try_step_no_sliding
        )

        final_position = step_fn(
            agent_state.position, goal_rigid_state.translation
        )
        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        dist_moved_before = (
            goal_rigid_state.translation - agent_state.position
        ).dot()
        dist_moved_after = (final_position - agent_state.position).dot()
        self._last_is_collision = (dist_moved_after + 1e-5) < dist_moved_before

        sim.set_agent_state(final_position, final_rotation, reset_sensors=False)
        return sim.get_sensor_observations()

    def _check_collision(self) -> bool:
        r"""Checks whether a collision occurred during the previous step."""
        sim = getattr(self._env, "sim", None)
        if sim is not None and getattr(sim, "previous_step_collided", False):
            return True

        metrics = self.get_metrics()
        if "collisions" in metrics:
            coll_data = metrics["collisions"]
            if isinstance(coll_data, dict):
                return bool(coll_data.get("is_collision", False))
            elif isinstance(coll_data, (int, float)):
                return coll_data > 0

        return getattr(self, "_last_is_collision", False)

    def _compute_distance_to_goal(self) -> float:
        r"""Computes geodesic distance from current agent position to episode goal(s)."""
        metrics = self.get_metrics()
        if "distance_to_goal" in metrics and metrics["distance_to_goal"] is not None:
            return float(metrics["distance_to_goal"])

        episode = getattr(self._env, "current_episode", None)
        sim = getattr(self._env, "sim", None)
        if episode is not None and hasattr(episode, "goals") and episode.goals and sim is not None:
            try:
                agent_pos = sim.get_agent_state().position
                goal_pos = [g.position for g in episode.goals]
                dist = sim.geodesic_distance(agent_pos, goal_pos, episode)
                return float(dist)
            except Exception:
                pass

        return float("nan")

    def _format_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        r"""Formats raw simulator observations into the standardized XJTLU contract."""
        # 1. RGB image: (480, 640, 3) uint8
        rgb = None
        for key in ["rgb", "color_sensor", "rgb_sensor"]:
            if key in raw_obs and raw_obs[key] is not None:
                rgb = raw_obs[key]
                break

        if rgb is None:
            # Generate placeholder if not found
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            rgb = np.asarray(rgb, dtype=np.uint8)

        # 2. Depth map: (480, 640, 1) float32
        depth = None
        for key in ["depth", "depth_sensor"]:
            if key in raw_obs and raw_obs[key] is not None:
                depth = raw_obs[key]
                break

        if depth is None:
            depth = np.zeros((480, 640, 1), dtype=np.float32)
        else:
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 2:
                depth = depth[..., np.newaxis]

        # 3. Camera Pose: 4x4 matrix in world coordinates (SE(3) transformation matrix)
        camera_pose = self._get_camera_pose_matrix()

        # 4. Instruction text
        instruction = self._extract_instruction(raw_obs)

        return {
            "rgb": rgb,
            "depth": depth,
            "camera_pose": camera_pose,
            "instruction": instruction,
        }

    def _get_camera_pose_matrix(self) -> np.ndarray:
        r"""Computes 4x4 camera-to-world transformation matrix (T_wc)."""
        sim = getattr(self._env, "sim", None)
        if sim is None:
            return np.eye(4, dtype=np.float32)

        agent_state = sim.get_agent_state()
        sensor_state = None
        # Look for the primary camera sensor state
        for key in ["rgb", "rgb_sensor", "color_sensor", "depth", "depth_sensor"]:
            if key in agent_state.sensor_states:
                sensor_state = agent_state.sensor_states[key]
                break

        if sensor_state is not None:
            position = sensor_state.position
            rotation = sensor_state.rotation
        else:
            position = agent_state.position
            rotation = agent_state.rotation

        rot_mat = quaternion.as_rotation_matrix(rotation)
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :3] = rot_mat.astype(np.float32)
        pose_4x4[:3, 3] = np.asarray(position, dtype=np.float32)

        return pose_4x4

    def _extract_instruction(self, raw_obs: Dict[str, Any]) -> str:
        r"""Extracts the language navigation instruction from episode or observation."""
        episode = getattr(self._env, "current_episode", None)
        if episode is not None:
            if hasattr(episode, "instruction") and episode.instruction is not None:
                inst = episode.instruction
                if hasattr(inst, "instruction_text"):
                    return str(inst.instruction_text)
                elif isinstance(inst, str):
                    return inst
                elif isinstance(inst, dict) and "text" in inst:
                    return str(inst["text"])
            if hasattr(episode, "question") and episode.question is not None:
                return str(getattr(episode.question, "question_text", episode.question))

        if "instruction" in raw_obs and raw_obs["instruction"] is not None:
            inst = raw_obs["instruction"]
            if isinstance(inst, dict) and "text" in inst:
                return str(inst["text"])
            return str(inst)

        return ""

    def get_metrics(self) -> Dict[str, Any]:
        r"""Retrieves current episode metrics from the underlying Habitat environment."""
        if hasattr(self._env, "get_metrics"):
            return self._env.get_metrics()
        return {}

    def render(self, mode: str = "rgb") -> np.ndarray:
        r"""Renders observation image."""
        if hasattr(self._env, "render"):
            return self._env.render(mode=mode)
        if self._last_raw_obs is not None:
            obs = self._format_observation(self._last_raw_obs)
            return obs["rgb"]
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        r"""Closes environment and releases simulator resources."""
        if self._owns_env and hasattr(self._env, "close"):
            self._env.close()

    def __enter__(self) -> "XJTLUCarEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
