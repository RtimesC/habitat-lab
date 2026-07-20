import argparse
import csv
import glob
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import cv2
import habitat
import numpy as np
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector

try:
    from .navida_policy import NaVIDAChunkPolicy
    from .prompts import ADVISORY_ACTIONS, EXPLORATION_ACTIONS, VALID_ACTIONS
    from .vlm_policy import (
        DEFAULT_MODEL_ID,
        MockVLMPolicy,
        PolicyOutput,
        QwenVLMPolicy,
    )
except ImportError:
    from navida_policy import NaVIDAChunkPolicy
    from prompts import ADVISORY_ACTIONS, EXPLORATION_ACTIONS, VALID_ACTIONS
    from vlm_policy import (
        DEFAULT_MODEL_ID,
        MockVLMPolicy,
        PolicyOutput,
        QwenVLMPolicy,
    )


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DEFAULT_TASK_CONFIG = "benchmark/nav/vln_r2r.yaml"

ACTION_MAP = {
    "turn_left": HabitatSimActions.turn_left,
    "turn_right": HabitatSimActions.turn_right,
    "move_forward": HabitatSimActions.move_forward,
    "stop": HabitatSimActions.stop,
}

TRAJECTORY_FIELDS = [
    "episode_index",
    "episode_id",
    "scene_id",
    "instruction",
    "step",
    "frequency_mode",
    "target_vision_hz",
    "target_inference_hz",
    "logical_time_sec",
    "wall_time_sec",
    "inference_ran",
    "inference_completed",
    "inference_duration_sec",
    "decision_age_steps",
    "decision_age_sec",
    "action",
    "valid_action",
    "raw_vlm_output",
    "vlm_action",
    "controller_action",
    "previous_action",
    "goal_distance_m",
    "goal_angle_deg",
    "distance_change_m",
    "collided",
    "depth_left_m",
    "depth_center_m",
    "depth_right_m",
    "previous_action_count",
    "previous_collision",
    "collision",
    "distance_delta",
    "no_progress_steps",
    "agent_x",
    "agent_y",
    "agent_z",
    "agent_rotation_x",
    "agent_rotation_y",
    "agent_rotation_z",
    "agent_rotation_w",
    "depth_min",
    "depth_mean",
    "distance_to_goal",
    "success",
    "spl",
    "image",
]


def build_env(
    task_config=DEFAULT_TASK_CONFIG,
    width=640,
    height=480,
    hfov=90,
    scene=None,
    dataset_split=None,
    dataset_path=None,
    scenes_dir=None,
):
    config = habitat.get_config(task_config)
    with read_write(config):
        if dataset_split is not None:
            config.habitat.dataset.split = dataset_split
        if dataset_path is not None:
            config.habitat.dataset.data_path = os.path.expanduser(dataset_path)
        if scenes_dir is not None:
            config.habitat.dataset.scenes_dir = os.path.expanduser(scenes_dir)
        if scene is not None:
            config.habitat.simulator.scene = os.path.expanduser(scene)
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        sensors.rgb_sensor.width = width
        sensors.rgb_sensor.height = height
        sensors.rgb_sensor.hfov = hfov
        sensors.depth_sensor.width = width
        sensors.depth_sensor.height = height
        sensors.depth_sensor.hfov = hfov
    return habitat.Env(config=config)


def prepare_run_dir(output_dir):
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_name)
    frame_dir = os.path.join(run_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for old_frame in glob.glob(os.path.join(frame_dir, "frame_*.jpg")):
        os.remove(old_frame)
    return run_dir, frame_dir


def rgb_to_bgr(rgb):
    return rgb[:, :, [2, 1, 0]].copy()


def draw_status(bgr, step, action, valid_action, collision):
    color = (255, 255, 255) if valid_action else (0, 165, 255)
    lines = [
        f"step={step:03d}",
        f"action={action}",
        f"valid={valid_action} collision={collision}",
    ]
    height, width = bgr.shape[:2]
    font_scale = min(0.65, max(0.38, width / 900.0))
    thickness = 2 if width >= 480 else 1
    padding = max(6, width // 100)
    text_height = cv2.getTextSize(
        "Ag",
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )[0][1]
    line_height = text_height + max(7, text_height // 2)
    overlay_height = min(height, padding * 2 + line_height * len(lines))
    status_area = bgr[:overlay_height, :].copy()
    black = np.zeros_like(status_area)
    bgr[:overlay_height, :] = cv2.addWeighted(status_area, 0.35, black, 0.65, 0.0)

    for idx, line in enumerate(lines):
        cv2.putText(
            bgr,
            line,
            (padding, padding + text_height + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def depth_stats(obs, config=None):
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


def depth_sensor_config(env):
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
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if config.get("normalize_depth", False):
        min_depth = config["min_depth"]
        max_depth = config["max_depth"]
        depth = depth * (max_depth - min_depth) + min_depth
    return depth


def depth_percentile(values):
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return ""
    return float(np.percentile(valid, 10))


def depth_region_summary(obs, config):
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


def instruction_text(obs, fallback=None):
    instruction = obs.get("instruction")
    if isinstance(instruction, dict):
        text = instruction.get("text")
        if text:
            return str(text)
    if hasattr(instruction, "text"):
        return str(instruction.text)
    if instruction is not None:
        return str(instruction)
    return fallback or ""


def episode_instruction_text(episode):
    """Read an engineering instruction stored in Habitat episode metadata."""
    info = getattr(episode, "info", None)
    if info is not None and hasattr(info, "get"):
        value = info.get("instruction")
        if value:
            return str(value)
    instruction = getattr(episode, "instruction", None)
    if isinstance(instruction, dict):
        instruction = instruction.get("text")
    elif hasattr(instruction, "text"):
        instruction = instruction.text
    return str(instruction) if instruction else ""


def metric_values(metrics):
    return {
        "distance_to_goal": metrics.get("distance_to_goal", ""),
        "success": metrics.get("success", ""),
        "spl": metrics.get("spl", ""),
    }


def env_config(env):
    return getattr(env, "config", None) or getattr(env, "_config", None)


def optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_angle_deg(angle):
    return ((float(angle) + 180.0) % 360.0) - 180.0


def get_goal_position(env):
    goals = getattr(env.current_episode, "goals", None) or []
    if not goals:
        return None
    position = getattr(goals[0], "position", None)
    if position is None:
        return None
    return np.asarray(position, dtype=np.float32)


def pointgoal_from_observation(obs):
    pointgoal = obs.get("pointgoal_with_gps_compass")
    if pointgoal is None:
        return None

    pointgoal = np.asarray(pointgoal, dtype=np.float32).reshape(-1)
    if pointgoal.size < 2:
        return None

    goal_distance_m = float(pointgoal[0])
    # Habitat's polar pointgoal angle is positive-left/negative-right.
    # The prompt and CSV use the opposite convention requested here:
    # negative means left, positive means right, zero means straight ahead.
    goal_angle_deg = normalize_angle_deg(-math.degrees(float(pointgoal[1])))
    return goal_distance_m, goal_angle_deg


def pointgoal_from_agent_state(env, agent_state, goal_position):
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
    from_observation = pointgoal_from_observation(obs)
    if from_observation is not None:
        return from_observation
    return pointgoal_from_agent_state(env, agent_state, goal_position)


def success_distance(env):
    try:
        config = env_config(env)
        return float(config.habitat.task.measurements.success.success_distance)
    except (AttributeError, TypeError, ValueError):
        return 0.2


def navigation_fallback_action(goal_angle_deg, previous_collision, depth_center_m):
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


def episode_values(env, episode_index, instruction):
    episode = env.current_episode
    return {
        "episode_index": episode_index,
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,
        "instruction": instruction,
    }


def agent_values_from_state(state):
    pos = state.position
    rot = state.rotation
    return {
        "agent_x": float(pos[0]),
        "agent_y": float(pos[1]),
        "agent_z": float(pos[2]),
        "agent_rotation_x": float(rot.x),
        "agent_rotation_y": float(rot.y),
        "agent_rotation_z": float(rot.z),
        "agent_rotation_w": float(rot.w),
    }


def resolve_frequency_mode(frequency_mode, qwen_role):
    """Resolve auto timing from the existing layered/direct-control split."""
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


def timed_policy_predict(policy, rgb, instruction, step, navigation_context):
    """Run one policy inference and return its output plus wall-clock duration."""
    start_time = time.perf_counter()
    output = policy.predict(
        rgb,
        instruction,
        step=step,
        navigation_context=navigation_context,
    )
    return output, time.perf_counter() - start_time


class BackgroundPolicyInference:
    """Keep layered visual/control ticks running while Qwen is generating."""

    def __init__(self, policy):
        self.policy = policy
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.source_step = None

    def submit(self, rgb, instruction, step, navigation_context):
        """Submit one inference only when the previous request has finished."""
        if self.future is not None:
            return False
        self.source_step = step
        self.future = self.executor.submit(
            timed_policy_predict,
            self.policy,
            np.asarray(rgb).copy(),
            instruction,
            step,
            dict(navigation_context),
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


def write_video(frame_paths, video_path, fps):
    if not frame_paths:
        return False

    first = cv2.imread(frame_paths[0])
    if first is None:
        return False

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()
    return os.path.exists(video_path)


def run_navigation(env, policy, args):
    run_dir, frame_dir = prepare_run_dir(args.output_dir)
    trajectory_path = os.path.join(run_dir, "trajectory.csv")
    depth_config = depth_sensor_config(env)
    goal_success_distance = success_distance(env)
    vision_hz, inference_hz = active_frequencies(args)
    video_fps = args.video_fps if args.video_fps is not None else vision_hz

    start_time = time.perf_counter()
    total_steps = 0
    total_collisions = 0
    episode_summaries = []

    with open(trajectory_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()

        for episode_index in range(args.num_episodes):
            obs = env.reset()
            episode_fallback = episode_instruction_text(env.current_episode) or args.goal
            instruction = args.instruction or instruction_text(obs, episode_fallback)
            episode_frame_dir = os.path.join(frame_dir, f"episode_{episode_index:03d}")
            os.makedirs(episode_frame_dir, exist_ok=True)
            episode_video_path = os.path.join(
                run_dir,
                f"episode_{episode_index:03d}.mp4",
            )
            frame_paths = []
            collision_count = 0
            stopped = False
            previous_action = "none"
            previous_action_count = 0
            previous_collision = False
            previous_goal_distance = None
            no_progress_steps = 0
            use_background_inference = (
                args.frequency_mode == "layered" and not args.no_rate_limit
            )
            background_inference = (
                BackgroundPolicyInference(policy)
                if use_background_inference
                else None
            )
            last_policy_output = (
                PolicyOutput(
                    action=args.advisor_fallback,
                    raw_text='{"scheduler_fallback":"waiting_for_first_inference"}',
                    is_valid=False,
                )
                if use_background_inference
                else None
            )
            last_inference_step = None
            next_inference_time_sec = 0.0
            episode_start_time = time.perf_counter()

            print(
                f"episode={episode_index:03d} "
                f"id={env.current_episode.episode_id} "
                f"instruction={instruction!r}"
            )
            print(
                f"frequency_mode={args.frequency_mode} "
                f"vision_hz={vision_hz:g} inference_hz={inference_hz:g} "
                f"video_fps={video_fps:g} rate_limit={not args.no_rate_limit}"
            )

            for step in range(args.max_steps):
                step_start_time = time.perf_counter()
                wall_time_sec = step_start_time - episode_start_time
                logical_time_sec = step / vision_hz
                rgb = obs["rgb"]
                metrics_before_action = env.get_metrics()
                agent_state = env.sim.get_agent_state()
                goal_position = get_goal_position(env)
                goal_relative = pointgoal_state(
                    env,
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
                    depth_config,
                )
                depth_min, depth_mean = depth_stats(obs, depth_config)
                current_distance = optional_float(
                    metrics_before_action.get("distance_to_goal", "")
                )
                previous_step_collided = bool(env.sim.previous_step_collided)
                navigation_context = {
                    "step": step,
                    "agent_position": tuple(float(x) for x in agent_state.position),
                    "agent_rotation": (
                        float(agent_state.rotation.x),
                        float(agent_state.rotation.y),
                        float(agent_state.rotation.z),
                        float(agent_state.rotation.w),
                    ),
                    "goal_position": (
                        None
                        if goal_position is None
                        else tuple(float(x) for x in goal_position)
                    ),
                    "goal_distance_m": goal_distance_m,
                    "goal_angle_deg": goal_angle_deg,
                    "success_distance_m": goal_success_distance,
                    "distance_change_m": distance_change_m,
                    "collided": previous_step_collided,
                    "depth_left_m": depth_left_m,
                    "depth_center_m": depth_center_m,
                    "depth_right_m": depth_right_m,
                    "previous_action": previous_action,
                    "previous_action_count": previous_action_count,
                    "previous_collision": previous_collision,
                    "distance_to_goal": current_distance,
                    "distance_delta": distance_change_m,
                    "no_progress_steps": no_progress_steps,
                    "depth_min": depth_min,
                    "depth_mean": depth_mean,
                }
                inference_ran = inference_is_due(
                    logical_time_sec,
                    next_inference_time_sec,
                )
                inference_completed = False
                inference_duration_sec = 0.0
                if background_inference is not None:
                    completed = background_inference.collect_if_ready()
                    if completed is not None:
                        (
                            last_policy_output,
                            inference_duration_sec,
                            last_inference_step,
                        ) = completed
                        inference_completed = True
                    if inference_ran:
                        inference_ran = background_inference.submit(
                            rgb,
                            instruction,
                            step,
                            navigation_context,
                        )
                        if inference_ran:
                            next_inference_time_sec = advance_inference_time(
                                next_inference_time_sec,
                                inference_hz,
                                logical_time_sec,
                            )
                elif inference_ran:
                    inference_start_time = time.perf_counter()
                    last_policy_output = policy.predict(
                        rgb,
                        instruction,
                        step=step,
                        navigation_context=navigation_context,
                    )
                    inference_completed = True
                    inference_duration_sec = (
                        time.perf_counter() - inference_start_time
                    )
                    last_inference_step = step
                    next_inference_time_sec = advance_inference_time(
                        next_inference_time_sec,
                        inference_hz,
                        logical_time_sec,
                    )
                policy_output = replace(last_policy_output)
                decision_age_steps = (
                    "" if last_inference_step is None else step - last_inference_step
                )
                decision_age_sec = (
                    ""
                    if decision_age_steps == ""
                    else decision_age_steps / vision_hz
                )
                vlm_action = policy_output.action
                if args.qwen_role == "advisor":
                    controller_action = action_from_advice(
                        vlm_action,
                        goal_distance_m,
                        goal_angle_deg,
                        goal_success_distance,
                        previous_step_collided,
                        depth_left_m,
                        depth_center_m,
                        depth_right_m,
                    )
                    action_name = controller_action
                else:
                    controller_action = policy_output.action
                    action_name = policy_output.action
                if (
                    args.force_stop_within_success_radius
                    and goal_distance_m is not None
                    and goal_distance_m < goal_success_distance
                ):
                    action_name = "stop"
                    controller_action = "stop"
                    policy_output.raw_text = (
                        f"{policy_output.raw_text}\n"
                        '{"success_radius_guard":"stop"}'
                    )
                    policy_output.is_valid = False
                if (
                    not args.disable_anti_stuck
                    and action_name in {"turn_left", "turn_right"}
                    and action_name == previous_action
                    and previous_action_count >= args.max_repeated_turns
                    and no_progress_steps >= args.max_no_progress_steps
                    and enough_depth(depth_center_m, args.anti_stuck_forward_depth)
                    and (
                        goal_angle_deg is None
                        or abs(goal_angle_deg) <= args.anti_stuck_max_goal_angle
                    )
                    and not previous_collision
                ):
                    print(
                        "Anti-stuck override; repeated turn with no progress, "
                        "fallback to move_forward"
                    )
                    action_name = "move_forward"
                    controller_action = action_name
                    policy_output.action = action_name
                    policy_output.raw_text = (
                        f"{policy_output.raw_text}\n"
                        '{"anti_stuck_override": "move_forward"}'
                    )
                    policy_output.is_valid = False
                if (
                    action_name == "stop"
                    and not args.allow_early_stop
                    and step < args.min_stop_step
                ):
                    replacement_action = navigation_fallback_action(
                        goal_angle_deg,
                        previous_step_collided,
                        depth_center_m,
                    )
                    print(
                        "Stop rejected before minimum stop step; "
                        f"fallback to {replacement_action}"
                    )
                    action_name = replacement_action
                    controller_action = action_name
                    policy_output.action = action_name
                    policy_output.raw_text = (
                        f"{policy_output.raw_text}\n"
                        f'{{"early_stop_rejected": "{replacement_action}"}}'
                    )
                    policy_output.is_valid = False
                if (
                    action_name == "stop"
                    and (
                        goal_distance_m is None
                        or goal_distance_m >= goal_success_distance
                    )
                ):
                    replacement_action = navigation_fallback_action(
                        goal_angle_deg,
                        previous_step_collided,
                        depth_center_m,
                    )
                    print(
                        "Stop rejected outside success radius; "
                        f"fallback to {replacement_action}"
                    )
                    action_name = replacement_action
                    controller_action = action_name
                    policy_output.action = action_name
                    policy_output.raw_text = (
                        f"{policy_output.raw_text}\n"
                        f'{{"stop_rejected": "{replacement_action}"}}'
                    )
                    policy_output.is_valid = False
                if action_name == "stop" and "stop" not in args.execution_actions:
                    action_name = navigation_fallback_action(
                        goal_angle_deg,
                        previous_step_collided,
                        depth_center_m,
                    )
                    controller_action = action_name
                    policy_output.raw_text = (
                        f"{policy_output.raw_text}\n"
                        f'{{"stop_disabled": "{action_name}"}}'
                    )
                    policy_output.is_valid = False
                if action_name not in args.execution_actions:
                    raise RuntimeError(f"Policy returned invalid action: {action_name}")
                if action_name != vlm_action and hasattr(policy, "clear_action_queue"):
                    policy.clear_action_queue()

                action = ACTION_MAP[action_name]
                obs = env.step(action)
                collision = bool(env.sim.previous_step_collided)
                collision_count += int(collision)
                stopped = action_name == "stop"
                metrics = env.get_metrics()
                next_distance = optional_float(metrics.get("distance_to_goal", ""))
                action_distance_delta = (
                    None
                    if current_distance is None or next_distance is None
                    else next_distance - current_distance
                )

                bgr = rgb_to_bgr(rgb)
                draw_status(bgr, step, action_name, policy_output.is_valid, collision)
                image_path = os.path.join(episode_frame_dir, f"frame_{step:03d}.jpg")
                cv2.imwrite(image_path, bgr)
                frame_paths.append(image_path)

                row = {
                    "step": step,
                    "frequency_mode": args.frequency_mode,
                    "target_vision_hz": vision_hz,
                    "target_inference_hz": inference_hz,
                    "logical_time_sec": logical_time_sec,
                    "wall_time_sec": wall_time_sec,
                    "inference_ran": inference_ran,
                    "inference_completed": inference_completed,
                    "inference_duration_sec": inference_duration_sec,
                    "decision_age_steps": decision_age_steps,
                    "decision_age_sec": decision_age_sec,
                    "action": action_name,
                    "valid_action": policy_output.is_valid,
                    "raw_vlm_output": policy_output.raw_text,
                    "vlm_action": vlm_action,
                    "controller_action": controller_action,
                    "previous_action": previous_action,
                    "goal_distance_m": goal_distance_m,
                    "goal_angle_deg": goal_angle_deg,
                    "distance_change_m": distance_change_m,
                    "collided": collision,
                    "depth_left_m": depth_left_m,
                    "depth_center_m": depth_center_m,
                    "depth_right_m": depth_right_m,
                    "previous_action_count": previous_action_count,
                    "previous_collision": previous_collision,
                    "collision": collision,
                    "distance_delta": distance_change_m,
                    "no_progress_steps": no_progress_steps,
                    "depth_min": depth_min,
                    "depth_mean": depth_mean,
                    "image": image_path,
                }
                row.update(episode_values(env, episode_index, instruction))
                row.update(agent_values_from_state(agent_state))
                row.update(metric_values(metrics))
                writer.writerow(row)

                print(
                    f"episode={episode_index:03d} step={step:03d} "
                    f"inference={'started' if inference_ran else 'idle'} "
                    f"completed={inference_completed} "
                    f"vlm={vlm_action} action={action_name} "
                    f"valid={policy_output.is_valid} "
                    f"collision={collision} "
                    f"goal_distance={goal_distance_m if goal_distance_m is not None else ''} "
                    f"goal_angle={goal_angle_deg if goal_angle_deg is not None else ''} "
                    f"delta={distance_change_m if distance_change_m is not None else ''} "
                    f"distance={metrics.get('distance_to_goal', '')} "
                    f"success={metrics.get('success', '')} "
                    f"spl={metrics.get('spl', '')}"
                )

                if stopped or env.episode_over:
                    break

                if action_name == previous_action:
                    previous_action_count += 1
                else:
                    previous_action_count = 1
                previous_action = action_name
                previous_collision = collision
                previous_goal_distance = goal_distance_m
                progress_delta = (
                    distance_change_m
                    if distance_change_m is not None
                    else action_distance_delta
                )
                if progress_delta is not None and progress_delta < -0.05:
                    no_progress_steps = 0
                else:
                    no_progress_steps += 1

                limit_loop_rate(
                    step_start_time,
                    vision_hz,
                    enabled=not args.no_rate_limit,
                )

            if background_inference is not None:
                background_inference.close()

            video_written = write_video(frame_paths, episode_video_path, video_fps)
            final_metrics = env.get_metrics()
            total_steps += len(frame_paths)
            total_collisions += collision_count
            episode_summaries.append(final_metrics)
            print(
                f"episode_summary: episode={episode_index:03d} "
                f"steps={len(frame_paths)} stopped={stopped} "
                f"collisions={collision_count} "
                f"success={final_metrics.get('success', '')} "
                f"spl={final_metrics.get('spl', '')} "
                f"distance={final_metrics.get('distance_to_goal', '')}"
            )
            print(
                f"saved video to {episode_video_path}"
                if video_written
                else "video was not written"
            )

    runtime_sec = time.perf_counter() - start_time
    print(f"saved trajectory to {trajectory_path}")
    avg_success = ""
    avg_spl = ""
    if episode_summaries:
        success_values = [
            float(m.get("success", 0.0))
            for m in episode_summaries
            if m.get("success", "") != ""
        ]
        spl_values = [
            float(m.get("spl", 0.0))
            for m in episode_summaries
            if m.get("spl", "") != ""
        ]
        if success_values:
            avg_success = sum(success_values) / len(success_values)
        if spl_values:
            avg_spl = sum(spl_values) / len(spl_values)
    print(
        f"summary: episodes={len(episode_summaries)} steps={total_steps} "
        f"collisions={total_collisions} success={avg_success} spl={avg_spl} "
        f"runtime_sec={runtime_sec:.2f}"
    )
    return trajectory_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-path")
    parser.add_argument("--scenes-dir")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        help="Optional trained LoRA adapter directory to load on top of --model-id.",
    )
    parser.add_argument(
        "--instruction",
        help="Override the episode instruction. By default, use the VLN dataset instruction.",
    )
    parser.add_argument(
        "--goal",
        help="Alias fallback for --instruction when an episode has no instruction sensor.",
    )
    parser.add_argument("--scene")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the VLM with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="float16",
        help="Compute dtype for bitsandbytes 4-bit quantization.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--video-fps",
        type=float,
        help="Saved-video playback rate. Defaults to the active visual frequency.",
    )
    parser.add_argument(
        "--frequency-mode",
        choices=["auto", "layered", "joint"],
        default="auto",
        help=(
            "auto uses layered timing for advisor mode and joint timing for "
            "controller mode."
        ),
    )
    parser.add_argument(
        "--vision-hz",
        type=float,
        default=5.0,
        help="Layered mode visual/control update frequency.",
    )
    parser.add_argument(
        "--inference-hz",
        type=float,
        default=0.5,
        help="Layered mode Qwen inference frequency.",
    )
    parser.add_argument(
        "--joint-hz",
        type=float,
        default=1.0,
        help="Joint model visual-plus-inference frequency.",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help=(
            "Run without wall-clock sleeps while preserving the same logical "
            "inference schedule. Useful only for fast tests."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=90)
    parser.add_argument(
        "--fallback-action",
        choices=sorted(VALID_ACTIONS),
        default="move_forward",
    )
    parser.add_argument(
        "--qwen-role",
        choices=["advisor", "controller"],
        default="advisor",
        help="Use Qwen as a high-level advisor or direct low-level controller.",
    )
    parser.add_argument(
        "--advisor-fallback",
        choices=sorted(ADVISORY_ACTIONS),
        default="follow_goal",
        help="Fallback advisory action when Qwen output cannot be parsed.",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Disable stop. Useful only for open-ended exploration, not standard VLN.",
    )
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Allow stop before --min-stop-step.",
    )
    parser.add_argument(
        "--force-stop-within-success-radius",
        action="store_true",
        help="Guarded evaluation: force stop when privileged goal distance is successful.",
    )
    parser.add_argument(
        "--min-stop-step",
        type=int,
        default=8,
        help="Reject stop before this step unless --allow-early-stop is set.",
    )
    parser.add_argument(
        "--disable-anti-stuck",
        action="store_true",
        help="Disable repeated-turn overrides.",
    )
    parser.add_argument(
        "--max-repeated-turns",
        type=int,
        default=2,
        help="Override repeated same-direction turns after this count.",
    )
    parser.add_argument(
        "--max-no-progress-steps",
        type=int,
        default=2,
        help="Override repeated turns after this many no-progress steps.",
    )
    parser.add_argument(
        "--anti-stuck-forward-depth",
        type=float,
        default=0.35,
        help="Require this much center depth before anti-stuck moves forward.",
    )
    parser.add_argument(
        "--anti-stuck-max-goal-angle",
        type=float,
        default=30.0,
        help="Only anti-stuck forward when the target angle is within this range.",
    )
    parser.add_argument("--mock-policy", action="store_true")
    parser.add_argument(
        "--navida-chunk-policy",
        action="store_true",
        help="Use a NaVIDA adapter to generate and execute structured action chunks.",
    )
    parser.add_argument("--navida-max-history-frames", type=int, default=8)
    parser.add_argument(
        "--navida-max-executed-actions",
        type=int,
        default=1,
        help=(
            "Actions kept from each NaVIDA chunk. Joint 1 Hz replanning requires 1."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.frequency_mode = resolve_frequency_mode(args.frequency_mode, args.qwen_role)
    validate_frequencies(args)
    os.makedirs(args.output_dir, exist_ok=True)
    args.execution_actions = (
        set(EXPLORATION_ACTIONS) if args.no_stop else set(VALID_ACTIONS)
    )
    args.allowed_actions = (
        set(ADVISORY_ACTIONS)
        if args.qwen_role == "advisor"
        else set(args.execution_actions)
    )
    policy_fallback_action = (
        args.advisor_fallback
        if args.qwen_role == "advisor"
        else args.fallback_action
    )
    if args.fallback_action not in args.execution_actions:
        raise ValueError(
            f"--fallback-action must be one of {sorted(args.execution_actions)} "
            "with the current --no-stop setting."
        )

    if args.navida_chunk_policy:
        if args.mock_policy:
            raise ValueError("--navida-chunk-policy cannot be combined with --mock-policy")
        if args.qwen_role != "controller":
            raise ValueError("--navida-chunk-policy requires --qwen-role controller")
        if not args.adapter_path:
            raise ValueError("--navida-chunk-policy requires --adapter-path")
        if args.navida_max_executed_actions != 1:
            raise ValueError(
                "joint NaVIDA timing requires --navida-max-executed-actions 1 "
                "so the model replans from the latest image at every 1 Hz tick"
            )
        policy = NaVIDAChunkPolicy(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            fallback_action=policy_fallback_action,
            max_history_frames=args.navida_max_history_frames,
            max_executed_actions=args.navida_max_executed_actions,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            max_new_tokens=max(args.max_new_tokens, 64),
        )
    elif args.mock_policy:
        policy = MockVLMPolicy(allowed_actions=args.allowed_actions)
    else:
        policy = QwenVLMPolicy(
            model_id=args.model_id,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            fallback_action=policy_fallback_action,
            allowed_actions=args.allowed_actions,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            adapter_path=args.adapter_path,
        )

    env = build_env(
        task_config=args.task_config,
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        scene=args.scene,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        scenes_dir=args.scenes_dir,
    )
    try:
        run_navigation(env, policy, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
