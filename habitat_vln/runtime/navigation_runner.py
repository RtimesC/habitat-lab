"""Closed-loop Habitat navigation execution independent from CLI parsing."""

import os
import time

import cv2

try:
    from ..control import ControllerConfig, NavigationController
    from ..core import PolicyOutput
    from ..envs import NavigationStateBuilder, optional_float, step_navigation_action
except ImportError:
    from control import ControllerConfig, NavigationController
    from core import PolicyOutput
    from envs import NavigationStateBuilder, optional_float, step_navigation_action

from .artifacts import (
    draw_status,
    prepare_run_dir,
    rgb_to_bgr,
    write_video,
)
from .recorder import TrajectoryRecorder
from .scheduler import (
    BackgroundPolicyInference,
    active_frequencies,
    advance_inference_time,
    inference_is_due,
    limit_loop_rate,
)


def instruction_text(obs, fallback=None):
    """Read an instruction sensor value with a plain-text fallback."""
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
    """Select standard navigation metrics for one trajectory row."""
    return {
        "distance_to_goal": metrics.get("distance_to_goal", ""),
        "success": metrics.get("success", ""),
        "spl": metrics.get("spl", ""),
    }


def episode_values(env, episode_index, instruction):
    """Select episode identifiers for one trajectory row."""
    episode = env.current_episode
    return {
        "episode_index": episode_index,
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,
        "instruction": instruction,
    }


def agent_values_from_state(state):
    """Flatten a Habitat agent pose for CSV output."""
    position = state.position
    rotation = state.rotation
    return {
        "agent_x": float(position[0]),
        "agent_y": float(position[1]),
        "agent_z": float(position[2]),
        "agent_rotation_x": float(rotation.x),
        "agent_rotation_y": float(rotation.y),
        "agent_rotation_z": float(rotation.z),
        "agent_rotation_w": float(rotation.w),
    }


def run_navigation(env, policy, args, controller=None):
    """Run closed-loop navigation and return the saved trajectory path."""
    run_dir, frame_dir = prepare_run_dir(args.output_dir)
    trajectory_path = os.path.join(run_dir, "trajectory.csv")
    state_builder = NavigationStateBuilder(env)
    controller = controller or NavigationController(ControllerConfig.from_args(args))
    vision_hz, inference_hz = active_frequencies(args)
    video_fps = args.video_fps if args.video_fps is not None else vision_hz

    start_time = time.perf_counter()
    total_steps = 0
    total_collisions = 0
    episode_summaries = []

    with TrajectoryRecorder(trajectory_path) as recorder:
        for episode_index in range(args.num_episodes):
            obs = env.reset()
            episode_fallback = (
                episode_instruction_text(env.current_episode) or args.goal
            )
            instruction = args.instruction or instruction_text(obs, episode_fallback)
            episode_frame_dir = os.path.join(
                frame_dir,
                f"episode_{episode_index:03d}",
            )
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
                navigation_state = state_builder.build(
                    obs=obs,
                    step=step,
                    previous_action=previous_action,
                    previous_action_count=previous_action_count,
                    previous_collision=previous_collision,
                    previous_goal_distance=previous_goal_distance,
                    no_progress_steps=no_progress_steps,
                    metrics=metrics_before_action,
                )
                policy_observation = navigation_state.as_policy_observation(
                    obs,
                    instruction,
                )
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
                            policy_observation
                        )
                        if inference_ran:
                            next_inference_time_sec = advance_inference_time(
                                next_inference_time_sec,
                                inference_hz,
                                logical_time_sec,
                            )
                elif inference_ran:
                    inference_start_time = time.perf_counter()
                    last_policy_output = policy.predict(policy_observation)
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
                decision_age_steps = (
                    "" if last_inference_step is None else step - last_inference_step
                )
                decision_age_sec = (
                    ""
                    if decision_age_steps == ""
                    else decision_age_steps / vision_hz
                )
                control_decision = controller.decide(
                    last_policy_output,
                    navigation_state,
                    step=step,
                    previous_action=previous_action,
                    previous_action_count=previous_action_count,
                    previous_collision=previous_collision,
                    no_progress_steps=no_progress_steps,
                )
                for message in control_decision.messages:
                    print(message)
                vlm_action = control_decision.vlm_action
                controller_action = control_decision.controller_action
                action_name = control_decision.action
                policy_output = control_decision.policy_output
                if action_name != vlm_action and hasattr(policy, "clear_action_queue"):
                    policy.clear_action_queue()

                obs = step_navigation_action(env, action_name)
                collision = bool(env.sim.previous_step_collided)
                collision_count += int(collision)
                stopped = action_name == "stop"
                metrics = env.get_metrics()
                next_distance = optional_float(metrics.get("distance_to_goal", ""))
                action_distance_delta = (
                    None
                    if navigation_state.distance_to_goal is None
                    or next_distance is None
                    else next_distance - navigation_state.distance_to_goal
                )

                bgr = rgb_to_bgr(rgb)
                draw_status(
                    bgr,
                    step,
                    action_name,
                    policy_output.is_valid,
                    collision,
                )
                image_path = os.path.join(
                    episode_frame_dir,
                    f"frame_{step:03d}.jpg",
                )
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
                    "goal_distance_m": navigation_state.goal_distance_m,
                    "goal_angle_deg": navigation_state.goal_angle_deg,
                    "distance_change_m": navigation_state.distance_change_m,
                    "collided": collision,
                    "depth_left_m": navigation_state.depth_left_m,
                    "depth_center_m": navigation_state.depth_center_m,
                    "depth_right_m": navigation_state.depth_right_m,
                    "previous_action_count": previous_action_count,
                    "previous_collision": previous_collision,
                    "collision": collision,
                    "distance_delta": navigation_state.distance_change_m,
                    "no_progress_steps": no_progress_steps,
                    "depth_min": navigation_state.depth_min,
                    "depth_mean": navigation_state.depth_mean,
                    "image": image_path,
                }
                row.update(episode_values(env, episode_index, instruction))
                row.update(agent_values_from_state(navigation_state.agent_state))
                row.update(metric_values(metrics))
                recorder.write(row)

                print(
                    f"episode={episode_index:03d} step={step:03d} "
                    f"inference={'started' if inference_ran else 'idle'} "
                    f"completed={inference_completed} "
                    f"vlm={vlm_action} action={action_name} "
                    f"valid={policy_output.is_valid} "
                    f"collision={collision} "
                    f"goal_distance={_display(navigation_state.goal_distance_m)} "
                    f"goal_angle={_display(navigation_state.goal_angle_deg)} "
                    f"delta={_display(navigation_state.distance_change_m)} "
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
                previous_goal_distance = navigation_state.goal_distance_m
                progress_delta = (
                    navigation_state.distance_change_m
                    if navigation_state.distance_change_m is not None
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

            video_written = write_video(
                frame_paths,
                episode_video_path,
                video_fps,
            )
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
    success_values = [
        float(metrics.get("success", 0.0))
        for metrics in episode_summaries
        if metrics.get("success", "") != ""
    ]
    spl_values = [
        float(metrics.get("spl", 0.0))
        for metrics in episode_summaries
        if metrics.get("spl", "") != ""
    ]
    average_success = (
        sum(success_values) / len(success_values) if success_values else ""
    )
    average_spl = sum(spl_values) / len(spl_values) if spl_values else ""
    print(
        f"summary: episodes={len(episode_summaries)} steps={total_steps} "
        f"collisions={total_collisions} success={average_success} "
        f"spl={average_spl} runtime_sec={runtime_sec:.2f}"
    )
    return trajectory_path


def _display(value):
    return "" if value is None else value
