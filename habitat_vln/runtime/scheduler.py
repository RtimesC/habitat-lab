"""Inference scheduling for layered advisor and joint controller modes."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from ..core import NavigationObservation
except ImportError:
    from core import NavigationObservation


def resolve_frequency_mode(frequency_mode, qwen_role):
    """Resolve automatic timing from the advisor/controller role."""
    if frequency_mode == "auto":
        return "layered" if qwen_role == "advisor" else "joint"
    return frequency_mode


def active_frequencies(args):
    """Return the visual/control tick rate and model refresh rate in Hz."""
    if args.frequency_mode == "joint":
        return args.joint_hz, args.joint_hz
    return args.vision_hz, args.inference_hz


def validate_frequencies(args):
    """Reject invalid or incompatible frequency schedules."""
    values = {
        "--vision-hz": args.vision_hz,
        "--inference-hz": args.inference_hz,
        "--joint-hz": args.joint_hz,
    }
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
    if args.frequency_mode == "layered" and args.inference_hz > args.vision_hz:
        raise ValueError("--inference-hz cannot exceed --vision-hz in layered mode")
    if args.frequency_mode == "layered" and args.qwen_role != "advisor":
        raise ValueError(
            "layered frequency mode requires --qwen-role advisor; "
            "use joint mode when the model directly controls navigation"
        )


def inference_is_due(logical_time_sec, next_inference_time_sec):
    """Return true when logical simulator time reaches the next model update."""
    return logical_time_sec + 1e-9 >= next_inference_time_sec


def advance_inference_time(next_inference_time_sec, inference_hz, logical_time_sec):
    """Advance the inference deadline beyond the current logical time."""
    period_sec = 1.0 / inference_hz
    while next_inference_time_sec <= logical_time_sec + 1e-9:
        next_inference_time_sec += period_sec
    return next_inference_time_sec


def limit_loop_rate(step_start_time, target_hz, enabled=True):
    """Prevent one visual/control iteration from running faster than target_hz."""
    if not enabled:
        return
    remaining_sec = (1.0 / target_hz) - (time.perf_counter() - step_start_time)
    if remaining_sec > 0.0:
        time.sleep(remaining_sec)


def timed_policy_predict(policy, observation):
    """Run one policy inference and return its output plus wall-clock duration."""
    start_time = time.perf_counter()
    output = policy.predict(observation)
    return output, time.perf_counter() - start_time


class BackgroundPolicyInference:
    """Keep layered visual/control ticks running while Qwen is generating."""

    def __init__(self, policy):
        self.policy = policy
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.source_step = None

    def submit(self, observation):
        """Submit one inference only when the previous request has finished."""
        if self.future is not None:
            return False
        self.source_step = observation.step
        background_observation = NavigationObservation(
            rgb=np.asarray(observation.rgb).copy(),
            depth=(
                None
                if observation.depth is None
                else np.asarray(observation.depth).copy()
            ),
            instruction=observation.instruction,
            step=observation.step,
            navigation_context=dict(observation.navigation_context),
        )
        self.future = self.executor.submit(
            timed_policy_predict,
            self.policy,
            background_observation,
        )
        return True

    def collect_if_ready(self):
        """Return a completed output without blocking the visual loop."""
        if self.future is None or not self.future.done():
            return None
        output, duration_sec = self.future.result()
        source_step = self.source_step
        self.future = None
        self.source_step = None
        return output, duration_sec, source_step

    def close(self):
        """Wait for an outstanding request so model errors are not hidden."""
        self.executor.shutdown(wait=True)
        if self.future is not None:
            self.future.result()
