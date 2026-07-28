"""Build target-free, policy-visible state for indoor semantic navigation."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    from ..core import NavigationObservation
except ImportError:
    from core import NavigationObservation


def env_config(env):
    """Return the public Habitat config, with compatibility for older versions."""
    return getattr(env, "config", None) or getattr(env, "_config", None)


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
        "normalize_depth": bool(
            getattr(depth_sensor, "normalize_depth", True)
        ),
    }


def depth_to_meters(depth, config):
    """Convert a Habitat depth image to a two-dimensional metre array."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if config.get("normalize_depth", False):
        depth = (
            depth * (config["max_depth"] - config["min_depth"])
            + config["min_depth"]
        )
    return depth


def depth_percentile(values):
    """Return a robust near-obstacle distance for one image region."""
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return ""
    return float(np.percentile(valid, 10))


def depth_stats(obs, config=None):
    """Return near-depth percentile and mean depth for local safety logging."""
    depth = obs.get("depth")
    if depth is None:
        return "", ""
    depth = (
        np.asarray(depth, dtype=np.float32)
        if config is None
        else depth_to_meters(depth, config)
    )
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        return "", ""
    return float(np.percentile(valid, 10)), float(np.mean(valid))


def depth_region_summary(obs, config):
    """Return local near-obstacle depth for the left, centre, and right thirds."""
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


@dataclass(frozen=True)
class HabitatNavigationState:
    """One target-free state assembled from observable local navigation signals."""

    step: int
    agent_state: Any
    collided: bool
    depth_left_m: Any
    depth_center_m: Any
    depth_right_m: Any
    previous_action: str
    previous_action_count: int
    previous_collision: bool
    depth_min: Any
    depth_mean: Any

    def as_navigation_context(self):
        """Return debug state for logs; pose is never forwarded to the policy."""
        return {
            "step": self.step,
            "agent_position": tuple(
                float(value) for value in self.agent_state.position
            ),
            "agent_rotation": (
                float(self.agent_state.rotation.x),
                float(self.agent_state.rotation.y),
                float(self.agent_state.rotation.z),
                float(self.agent_state.rotation.w),
            ),
            "collided": self.collided,
            "depth_left_m": self.depth_left_m,
            "depth_center_m": self.depth_center_m,
            "depth_right_m": self.depth_right_m,
            "previous_action": self.previous_action,
            "previous_action_count": self.previous_action_count,
            "previous_collision": self.previous_collision,
            "depth_min": self.depth_min,
            "depth_mean": self.depth_mean,
        }

    def as_policy_navigation_context(self, building_prior=None):
        """Return only local observations and allowed building weak priors."""
        context = {
            "step": self.step,
            "collided": self.collided,
            "depth_left_m": self.depth_left_m,
            "depth_center_m": self.depth_center_m,
            "depth_right_m": self.depth_right_m,
            "previous_action": self.previous_action,
            "previous_action_count": self.previous_action_count,
            "previous_collision": self.previous_collision,
        }
        if building_prior:
            context["building_prior"] = building_prior
        return context

    def as_policy_observation(
        self, obs: Mapping[str, Any], instruction: str, building_prior=None
    ):
        """Combine RGB-D with target-free local state for the language policy."""
        return NavigationObservation(
            rgb=obs["rgb"],
            depth=obs.get("depth"),
            instruction=instruction,
            step=self.step,
            navigation_context=self.as_policy_navigation_context(
                building_prior
            ),
        )


class NavigationStateBuilder:
    """Assemble target-free state on every indoor semantic-navigation tick."""

    def __init__(self, env, depth_config=None):
        self.env = env
        self.depth_config = (
            depth_sensor_config(env) if depth_config is None else depth_config
        )

    def build(
        self,
        obs,
        step,
        previous_action,
        previous_action_count,
        previous_collision,
    ):
        """Build local depth, collision, action-history, and log-only pose state."""
        depth_left_m, depth_center_m, depth_right_m = depth_region_summary(
            obs,
            self.depth_config,
        )
        depth_min, depth_mean = depth_stats(obs, self.depth_config)
        return HabitatNavigationState(
            step=step,
            agent_state=self.env.sim.get_agent_state(),
            collided=bool(self.env.sim.previous_step_collided),
            depth_left_m=depth_left_m,
            depth_center_m=depth_center_m,
            depth_right_m=depth_right_m,
            previous_action=previous_action,
            previous_action_count=previous_action_count,
            previous_collision=previous_collision,
            depth_min=depth_min,
            depth_mean=depth_mean,
        )
