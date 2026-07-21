"""Build navigation state from Habitat observations and simulator state."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

try:
    from ..core import NavigationObservation
except ImportError:
    from core import NavigationObservation


def env_config(env):
    """Return the public Habitat config, with compatibility for older versions."""
    return getattr(env, "config", None) or getattr(env, "_config", None)


def optional_float(value):
    """Convert a metric to float while preserving missing values as ``None``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def depth_sensor_config(env):
    """Read the depth conversion settings used by the active Habitat sensor."""
    try:
        config = env_config(env)
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        depth_sensor = sensors.get("depth_sensor", None)
    except (AttributeError, TypeError):
        depth_sensor = None

    return {
        "min_depth": float(getattr(depth_sensor, "min_depth", 0.0)),
        "max_depth": float(getattr(depth_sensor, "max_depth", 10.0)),
        "normalize_depth": bool(getattr(depth_sensor, "normalize_depth", True)),
    }


def depth_to_meters(depth, config):
    """Convert a Habitat depth image to a two-dimensional meter array."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if config.get("normalize_depth", False):
        min_depth = config["min_depth"]
        max_depth = config["max_depth"]
        depth = depth * (max_depth - min_depth) + min_depth
    return depth


def depth_percentile(values):
    """Return a robust near-obstacle distance for one image region."""
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return ""
    return float(np.percentile(valid, 10))


def depth_stats(obs, config=None):
    """Return the near-depth percentile and mean depth for logging."""
    depth = obs.get("depth")
    if depth is None:
        return "", ""
    if config is None:
        depth = np.asarray(depth, dtype=np.float32)
    else:
        depth = depth_to_meters(depth, config)
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        return "", ""
    return float(np.percentile(valid, 10)), float(np.mean(valid))


def depth_region_summary(obs, config):
    """Return near-obstacle depth for the left, center, and right thirds."""
    depth = obs.get("depth")
    if depth is None:
        return "", "", ""

    depth_m = depth_to_meters(depth, config)
    width = depth_m.shape[1]
    left_end = width // 3
    right_start = (2 * width) // 3
    return (
        depth_percentile(depth_m[:, :left_end]),
        depth_percentile(depth_m[:, left_end:right_start]),
        depth_percentile(depth_m[:, right_start:]),
    )


def normalize_angle_deg(angle):
    """Normalize an angle to the inclusive-left, exclusive-right 180 range."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def get_goal_position(env):
    """Read the first Habitat episode goal as a NumPy position."""
    goals = getattr(env.current_episode, "goals", None) or []
    if not goals:
        return None
    position = getattr(goals[0], "position", None)
    if position is None:
        return None
    return np.asarray(position, dtype=np.float32)


def pointgoal_from_observation(obs):
    """Read distance and the project-standard signed angle from PointGoal."""
    pointgoal = obs.get("pointgoal_with_gps_compass")
    if pointgoal is None:
        return None

    pointgoal = np.asarray(pointgoal, dtype=np.float32).reshape(-1)
    if pointgoal.size < 2:
        return None

    goal_distance_m = float(pointgoal[0])
    # Habitat uses positive-left/negative-right. This project logs the opposite.
    goal_angle_deg = normalize_angle_deg(-math.degrees(float(pointgoal[1])))
    return goal_distance_m, goal_angle_deg


def pointgoal_from_agent_state(env, agent_state, goal_position):
    """Calculate point-goal state when the observation sensor is unavailable."""
    if goal_position is None:
        return None

    direction_world = goal_position - np.asarray(agent_state.position, dtype=np.float32)
    direction_agent = quaternion_rotate_vector(
        agent_state.rotation.inverse(),
        direction_world,
    )
    goal_distance_m, _ = cartesian_to_polar(
        -direction_agent[2],
        direction_agent[0],
    )
    goal_angle_deg = normalize_angle_deg(
        math.degrees(math.atan2(direction_agent[0], -direction_agent[2]))
    )
    return float(goal_distance_m), goal_angle_deg


def pointgoal_state(env, obs, agent_state, goal_position):
    """Prefer PointGoal observations and fall back to simulator geometry."""
    from_observation = pointgoal_from_observation(obs)
    if from_observation is not None:
        return from_observation
    return pointgoal_from_agent_state(env, agent_state, goal_position)


def success_distance(env):
    """Read the configured success radius, defaulting to 0.2 meters."""
    try:
        config = env_config(env)
        return float(config.habitat.task.measurements.success.success_distance)
    except (AttributeError, TypeError, ValueError):
        return 0.2


@dataclass(frozen=True)
class HabitatNavigationState:
    """One assembled Habitat navigation state for policy and controller use."""

    step: int
    agent_state: Any
    goal_position: Any
    goal_distance_m: Optional[float]
    goal_angle_deg: Optional[float]
    success_distance_m: float
    distance_change_m: Optional[float]
    collided: bool
    depth_left_m: Any
    depth_center_m: Any
    depth_right_m: Any
    previous_action: str
    previous_action_count: int
    previous_collision: bool
    distance_to_goal: Optional[float]
    no_progress_steps: int
    depth_min: Any
    depth_mean: Any

    def as_navigation_context(self):
        """Return the dictionary format consumed by existing prompts and logs."""
        return {
            "step": self.step,
            "agent_position": tuple(float(x) for x in self.agent_state.position),
            "agent_rotation": (
                float(self.agent_state.rotation.x),
                float(self.agent_state.rotation.y),
                float(self.agent_state.rotation.z),
                float(self.agent_state.rotation.w),
            ),
            "goal_position": (
                None
                if self.goal_position is None
                else tuple(float(x) for x in self.goal_position)
            ),
            "goal_distance_m": self.goal_distance_m,
            "goal_angle_deg": self.goal_angle_deg,
            "success_distance_m": self.success_distance_m,
            "distance_change_m": self.distance_change_m,
            "collided": self.collided,
            "depth_left_m": self.depth_left_m,
            "depth_center_m": self.depth_center_m,
            "depth_right_m": self.depth_right_m,
            "previous_action": self.previous_action,
            "previous_action_count": self.previous_action_count,
            "previous_collision": self.previous_collision,
            "distance_to_goal": self.distance_to_goal,
            "distance_delta": self.distance_change_m,
            "no_progress_steps": self.no_progress_steps,
            "depth_min": self.depth_min,
            "depth_mean": self.depth_mean,
        }

    def as_policy_observation(self, obs: Mapping[str, Any], instruction: str):
        """Combine this state with RGB/depth data for a navigation policy."""
        return NavigationObservation(
            rgb=obs["rgb"],
            depth=obs.get("depth"),
            instruction=instruction,
            step=self.step,
            navigation_context=self.as_navigation_context(),
        )


class NavigationStateBuilder:
    """Assemble the Habitat navigation state used on every control tick."""

    def __init__(self, env, depth_config=None, success_distance_m=None):
        self.env = env
        self.depth_config = (
            depth_sensor_config(env) if depth_config is None else depth_config
        )
        self.success_distance_m = (
            success_distance(env)
            if success_distance_m is None
            else float(success_distance_m)
        )

    def build(
        self,
        obs,
        step,
        previous_action,
        previous_action_count,
        previous_collision,
        previous_goal_distance,
        no_progress_steps,
        metrics=None,
    ):
        """Build distance, angle, depth, collision, and progress state."""
        agent_state = self.env.sim.get_agent_state()
        goal_position = get_goal_position(self.env)
        goal_relative = pointgoal_state(
            self.env,
            obs,
            agent_state,
            goal_position,
        )
        if goal_relative is None:
            goal_distance_m = None
            goal_angle_deg = None
        else:
            goal_distance_m, goal_angle_deg = goal_relative

        distance_change_m = (
            None
            if goal_distance_m is None or previous_goal_distance is None
            else goal_distance_m - previous_goal_distance
        )
        depth_left_m, depth_center_m, depth_right_m = depth_region_summary(
            obs,
            self.depth_config,
        )
        depth_min, depth_mean = depth_stats(obs, self.depth_config)
        if metrics is None:
            metrics = self.env.get_metrics()

        return HabitatNavigationState(
            step=step,
            agent_state=agent_state,
            goal_position=goal_position,
            goal_distance_m=goal_distance_m,
            goal_angle_deg=goal_angle_deg,
            success_distance_m=self.success_distance_m,
            distance_change_m=distance_change_m,
            collided=bool(self.env.sim.previous_step_collided),
            depth_left_m=depth_left_m,
            depth_center_m=depth_center_m,
            depth_right_m=depth_right_m,
            previous_action=previous_action,
            previous_action_count=previous_action_count,
            previous_collision=previous_collision,
            distance_to_goal=optional_float(metrics.get("distance_to_goal", "")),
            no_progress_steps=no_progress_steps,
            depth_min=depth_min,
            depth_mean=depth_mean,
        )


def build_navigation_context(
    env,
    obs,
    depth_config,
    goal_success_distance,
    step,
    previous_action,
    previous_action_count,
    previous_collision,
    previous_goal_distance,
    no_progress_steps,
):
    """Compatibility wrapper for callers that still expect a context dictionary."""
    builder = NavigationStateBuilder(
        env,
        depth_config=depth_config,
        success_distance_m=goal_success_distance,
    )
    state = builder.build(
        obs=obs,
        step=step,
        previous_action=previous_action,
        previous_action_count=previous_action_count,
        previous_collision=previous_collision,
        previous_goal_distance=previous_goal_distance,
        no_progress_steps=no_progress_steps,
    )
    return state.as_navigation_context()
