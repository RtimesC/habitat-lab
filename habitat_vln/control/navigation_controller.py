"""Validate direct semantic-navigation actions and apply local safety only."""

from dataclasses import dataclass, replace
from typing import FrozenSet, Tuple

try:
    from ..core import PolicyOutput
    from ..policies.prompts import VALID_ACTIONS
except ImportError:
    from core import PolicyOutput
    from policies.prompts import VALID_ACTIONS


def enough_depth(depth_m, threshold):
    """Treat missing depth as unknown rather than an automatic obstacle."""
    return not isinstance(depth_m, float) or depth_m >= threshold


def local_safety_turn(depth_left_m, depth_right_m):
    """Choose a turn using only visible local depth, never a hidden target."""
    if isinstance(depth_left_m, float) and isinstance(depth_right_m, float):
        return "turn_left" if depth_left_m >= depth_right_m else "turn_right"
    return "turn_left"


@dataclass(frozen=True)
class ControllerConfig:
    """Local execution safety settings for the semantic-indoor task."""

    execution_actions: FrozenSet[str] = frozenset(VALID_ACTIONS)
    enable_forward_depth_guard: bool = False
    forward_depth_guard_threshold: float = 0.35

    @classmethod
    def from_args(cls, args):
        """Build local safety settings from the command-line namespace."""
        return cls(
            execution_actions=frozenset(args.execution_actions),
            enable_forward_depth_guard=args.enable_forward_depth_guard,
            forward_depth_guard_threshold=(args.forward_depth_guard_threshold),
        )


@dataclass(frozen=True)
class ControlDecision:
    """Keep requested, safety-adjusted, and executed actions auditable."""

    vlm_action: str
    controller_action: str
    action: str
    policy_output: PolicyOutput
    messages: Tuple[str, ...] = ()


class NavigationController:
    """Apply only target-free local safety overrides to direct model actions."""

    def __init__(self, config):
        self.config = config

    def decide(
        self,
        policy_output,
        navigation_state,
        step,
        previous_action,
        previous_action_count,
        previous_collision,
    ):
        """Return one executable action without consulting a hidden goal state."""
        del step, previous_action, previous_action_count, previous_collision
        output = replace(policy_output)
        vlm_action = output.action
        action = output.action
        messages = []

        if (
            self.config.enable_forward_depth_guard
            and action == "move_forward"
            and not enough_depth(
                navigation_state.depth_center_m,
                self.config.forward_depth_guard_threshold,
            )
        ):
            action = local_safety_turn(
                navigation_state.depth_left_m,
                navigation_state.depth_right_m,
            )
            messages.append(
                "Forward depth guard; centre depth is below the safety threshold"
            )
            metadata = dict(output.metadata)
            metadata["forward_depth_guard"] = {
                "threshold_m": self.config.forward_depth_guard_threshold,
                "depth_left_m": navigation_state.depth_left_m,
                "depth_center_m": navigation_state.depth_center_m,
                "depth_right_m": navigation_state.depth_right_m,
                "replacement_action": action,
            }
            output.action = action
            output.raw_text = (
                f"{output.raw_text}\n" f'{{"forward_depth_guard": "{action}"}}'
            )
            output.metadata = metadata
            output.is_valid = False

        if action == "stop" and "stop" not in self.config.execution_actions:
            action = local_safety_turn(
                navigation_state.depth_left_m,
                navigation_state.depth_right_m,
            )
            output.action = action
            output.raw_text = (
                f"{output.raw_text}\n" f'{{"stop_disabled": "{action}"}}'
            )
            output.is_valid = False

        if action not in self.config.execution_actions:
            raise RuntimeError(f"Policy returned invalid action: {action}")

        return ControlDecision(
            vlm_action=vlm_action,
            controller_action=action,
            action=action,
            policy_output=output,
            messages=tuple(messages),
        )
