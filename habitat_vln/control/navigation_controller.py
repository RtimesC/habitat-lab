"""Convert policy output into safe executable navigation actions."""

from dataclasses import dataclass, replace
from typing import FrozenSet, Tuple

try:
    from ..core import PolicyOutput
    from ..policies.prompts import VALID_ACTIONS
except ImportError:
    from core import PolicyOutput
    from policies.prompts import VALID_ACTIONS


def navigation_fallback_action(goal_angle_deg, previous_collision, depth_center_m):
    """Choose a conservative action when a requested action cannot be used."""
    if goal_angle_deg is not None:
        if goal_angle_deg < -15.0:
            return "turn_left"
        if goal_angle_deg > 15.0:
            return "turn_right"

    if previous_collision or (
        isinstance(depth_center_m, float) and depth_center_m < 0.35
    ):
        if goal_angle_deg is not None and goal_angle_deg > 0.0:
            return "turn_right"
        return "turn_left"

    return "move_forward"


def enough_depth(depth_m, threshold):
    """Treat missing depth as unknown rather than an automatic obstacle."""
    return not isinstance(depth_m, float) or depth_m >= threshold


def geometric_navigation_action(
    goal_distance_m,
    goal_angle_deg,
    success_distance_m,
    previous_collision,
    depth_left_m,
    depth_center_m,
    depth_right_m,
    angle_threshold_deg=15.0,
    forward_safe_depth_m=0.35,
):
    """Turn toward the goal, avoid close obstacles, and stop when successful."""
    if goal_distance_m is not None and goal_distance_m < success_distance_m:
        return "stop"

    if previous_collision:
        if isinstance(depth_left_m, float) and isinstance(depth_right_m, float):
            return "turn_left" if depth_left_m >= depth_right_m else "turn_right"
        return navigation_fallback_action(
            goal_angle_deg,
            previous_collision,
            depth_center_m,
        )

    if goal_angle_deg is not None:
        if goal_angle_deg < -angle_threshold_deg:
            return "turn_left"
        if goal_angle_deg > angle_threshold_deg:
            return "turn_right"

    if not enough_depth(depth_center_m, forward_safe_depth_m):
        if isinstance(depth_left_m, float) and isinstance(depth_right_m, float):
            return "turn_left" if depth_left_m >= depth_right_m else "turn_right"
        return navigation_fallback_action(
            goal_angle_deg,
            previous_collision,
            depth_center_m,
        )

    return "move_forward"


def action_from_advice(
    advice,
    goal_distance_m,
    goal_angle_deg,
    success_distance_m,
    previous_collision,
    depth_left_m,
    depth_center_m,
    depth_right_m,
):
    """Convert Qwen advisor output into one atomic controller action."""
    if advice == "stop_if_reached":
        if goal_distance_m is not None and goal_distance_m < success_distance_m:
            return "stop"
        return geometric_navigation_action(
            goal_distance_m,
            goal_angle_deg,
            success_distance_m,
            previous_collision,
            depth_left_m,
            depth_center_m,
            depth_right_m,
        )

    if advice == "turn_left_to_avoid":
        return "turn_left"
    if advice == "turn_right_to_avoid":
        return "turn_right"

    return geometric_navigation_action(
        goal_distance_m,
        goal_angle_deg,
        success_distance_m,
        previous_collision,
        depth_left_m,
        depth_center_m,
        depth_right_m,
    )


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime safety and action-conversion settings."""

    qwen_role: str = "advisor"
    execution_actions: FrozenSet[str] = frozenset(VALID_ACTIONS)
    force_stop_within_success_radius: bool = False
    allow_early_stop: bool = False
    min_stop_step: int = 8
    disable_anti_stuck: bool = False
    max_repeated_turns: int = 2
    max_no_progress_steps: int = 2
    anti_stuck_forward_depth: float = 0.35
    anti_stuck_max_goal_angle: float = 30.0

    @classmethod
    def from_args(cls, args):
        """Build controller settings from the existing command-line namespace."""
        return cls(
            qwen_role=args.qwen_role,
            execution_actions=frozenset(args.execution_actions),
            force_stop_within_success_radius=(
                args.force_stop_within_success_radius
            ),
            allow_early_stop=args.allow_early_stop,
            min_stop_step=args.min_stop_step,
            disable_anti_stuck=args.disable_anti_stuck,
            max_repeated_turns=args.max_repeated_turns,
            max_no_progress_steps=args.max_no_progress_steps,
            anti_stuck_forward_depth=args.anti_stuck_forward_depth,
            anti_stuck_max_goal_angle=args.anti_stuck_max_goal_angle,
        )


@dataclass(frozen=True)
class ControlDecision:
    """Policy, controller, and final action values for one control tick."""

    vlm_action: str
    controller_action: str
    action: str
    policy_output: PolicyOutput
    messages: Tuple[str, ...] = ()


class NavigationController:
    """Apply advisor conversion and deterministic safety overrides."""

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
        no_progress_steps,
    ):
        """Return one validated executable action without changing model state."""
        output = replace(policy_output)
        vlm_action = output.action
        if self.config.qwen_role == "advisor":
            controller_action = action_from_advice(
                vlm_action,
                navigation_state.goal_distance_m,
                navigation_state.goal_angle_deg,
                navigation_state.success_distance_m,
                navigation_state.collided,
                navigation_state.depth_left_m,
                navigation_state.depth_center_m,
                navigation_state.depth_right_m,
            )
        else:
            controller_action = output.action
        action = controller_action
        messages = []

        if (
            self.config.force_stop_within_success_radius
            and navigation_state.goal_distance_m is not None
            and navigation_state.goal_distance_m
            < navigation_state.success_distance_m
        ):
            action = "stop"
            controller_action = "stop"
            output.raw_text = (
                f"{output.raw_text}\n" '{"success_radius_guard":"stop"}'
            )
            output.is_valid = False

        if (
            not self.config.disable_anti_stuck
            and action in {"turn_left", "turn_right"}
            and action == previous_action
            and previous_action_count >= self.config.max_repeated_turns
            and no_progress_steps >= self.config.max_no_progress_steps
            and enough_depth(
                navigation_state.depth_center_m,
                self.config.anti_stuck_forward_depth,
            )
            and (
                navigation_state.goal_angle_deg is None
                or abs(navigation_state.goal_angle_deg)
                <= self.config.anti_stuck_max_goal_angle
            )
            and not previous_collision
        ):
            messages.append(
                "Anti-stuck override; repeated turn with no progress, "
                "fallback to move_forward"
            )
            action = "move_forward"
            controller_action = action
            output.action = action
            output.raw_text = (
                f"{output.raw_text}\n"
                '{"anti_stuck_override": "move_forward"}'
            )
            output.is_valid = False

        if (
            action == "stop"
            and not self.config.allow_early_stop
            and step < self.config.min_stop_step
        ):
            replacement_action = navigation_fallback_action(
                navigation_state.goal_angle_deg,
                navigation_state.collided,
                navigation_state.depth_center_m,
            )
            messages.append(
                "Stop rejected before minimum stop step; "
                f"fallback to {replacement_action}"
            )
            action = replacement_action
            controller_action = action
            output.action = action
            output.raw_text = (
                f"{output.raw_text}\n"
                f'{{"early_stop_rejected": "{replacement_action}"}}'
            )
            output.is_valid = False

        if action == "stop" and (
            navigation_state.goal_distance_m is None
            or navigation_state.goal_distance_m
            >= navigation_state.success_distance_m
        ):
            replacement_action = navigation_fallback_action(
                navigation_state.goal_angle_deg,
                navigation_state.collided,
                navigation_state.depth_center_m,
            )
            messages.append(
                "Stop rejected outside success radius; "
                f"fallback to {replacement_action}"
            )
            action = replacement_action
            controller_action = action
            output.action = action
            output.raw_text = (
                f"{output.raw_text}\n"
                f'{{"stop_rejected": "{replacement_action}"}}'
            )
            output.is_valid = False

        if action == "stop" and "stop" not in self.config.execution_actions:
            action = navigation_fallback_action(
                navigation_state.goal_angle_deg,
                navigation_state.collided,
                navigation_state.depth_center_m,
            )
            controller_action = action
            output.raw_text = (
                f"{output.raw_text}\n" f'{{"stop_disabled": "{action}"}}'
            )
            output.is_valid = False

        if action not in self.config.execution_actions:
            raise RuntimeError(f"Policy returned invalid action: {action}")

        return ControlDecision(
            vlm_action=vlm_action,
            controller_action=controller_action,
            action=action,
            policy_output=output,
            messages=tuple(messages),
        )
