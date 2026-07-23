"""Closed-loop Habitat navigation execution independent from CLI parsing."""

from dataclasses import replace
import json
import math
import os
import time
from typing import Any

import cv2

try:
    from ..control import ControllerConfig, NavigationController
    from ..core import NavigationObservation, PolicyOutput
    from ..envs import (
        NavigationStateBuilder,
        optional_float,
        step_navigation_action,
    )
except ImportError:
    from control import ControllerConfig, NavigationController
    from core import NavigationObservation, PolicyOutput
    from envs import (
        NavigationStateBuilder,
        optional_float,
        step_navigation_action,
    )

from .artifacts import (
    draw_status,
    prepare_run_dir,
    rgb_to_bgr,
    write_video,
)
from .grounded_demo_artifacts import write_grounded_demo_artifacts
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


def append_failure_reason(existing: str, new_reason: str) -> str:
    """Combine terminal failure details without losing an earlier cause."""
    return f"{existing}; {new_reason}" if existing else new_reason


def save_rgb_frame(
    rgb: Any,
    image_path: str,
    step: int,
    action: Any,
    valid_action: bool,
    collision: bool,
) -> tuple[str, str]:
    """Save one review frame and return its path plus any write error."""
    try:
        bgr = rgb_to_bgr(rgb)
        draw_status(
            bgr,
            step,
            action,
            valid_action,
            collision,
        )
        if not cv2.imwrite(image_path, bgr):
            raise RuntimeError("cv2.imwrite returned False")
    except Exception as exc:
        return (
            "",
            f"RGB frame write failed: {type(exc).__name__}: {exc}",
        )
    return image_path, ""


def terminal_runtime_error_row(
    episode: Any,
    episode_index: int,
    instruction: str,
    step: int,
    error: str,
    metrics: Any,
    image_path: str,
    frequency_mode: str,
    policy_protocol: str,
    vision_hz: float,
    inference_hz: float,
    wall_time_sec: float,
) -> dict:
    """Build a terminal CSV row when navigation state assembly failed."""
    row = {
        "step": step,
        "frequency_mode": frequency_mode,
        "policy_protocol": policy_protocol,
        "target_vision_hz": vision_hz,
        "target_inference_hz": inference_hz,
        "logical_time_sec": step / vision_hz,
        "wall_time_sec": wall_time_sec,
        "inference_ran": False,
        "inference_completed": False,
        "action": None,
        "valid_action": False,
        "raw_vlm_output": "",
        "termination_reason": "runtime_exception",
        "policy_metadata": json.dumps(
            {"error": error},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "collision": False,
        "image": image_path,
        "policy_action": None,
        "executed_action": None,
        "action_valid": False,
        "inference_latency_sec": "",
        "done": True,
        "error": error,
    }
    row.update(
        {
            "episode_index": episode_index,
            "episode_id": str(getattr(episode, "episode_id", "")),
            "scene_id": str(getattr(episode, "scene_id", "")),
            "instruction": instruction,
        }
    )
    row.update(metric_values(metrics))
    return row


def agent_values_from_state(state):
    """Flatten a Habitat agent pose for CSV output."""
    position = state.position
    rotation = state.rotation
    yaw_numerator = 2.0 * (
        float(rotation.w) * float(rotation.y)
        + float(rotation.x) * float(rotation.z)
    )
    yaw_radians = math.atan2(
        yaw_numerator,
        1.0
        - 2.0
        * (
            float(rotation.y) * float(rotation.y)
            + float(rotation.z) * float(rotation.z)
        ),
    )
    return {
        "agent_x": float(position[0]),
        "agent_y": float(position[1]),
        "agent_z": float(position[2]),
        "agent_rotation_x": float(rotation.x),
        "agent_rotation_y": float(rotation.y),
        "agent_rotation_z": float(rotation.z),
        "agent_rotation_w": float(rotation.w),
        "position_x": float(position[0]),
        "position_y": float(position[1]),
        "position_z": float(position[2]),
        "rotation_yaw": math.degrees(yaw_radians),
    }


def run_navigation(
    env: Any,
    policy: Any,
    args: Any,
    controller: Any = None,
) -> str:
    """Run navigation and close an optional policy resource exactly once."""
    try:
        trajectory_path = _execute_navigation(
            env,
            policy,
            args,
            controller=controller,
        )
    except BaseException:
        try:
            _close_policy(policy)
        except BaseException as close_error:
            print(
                f"warning: policy close failed after run error: {close_error}"
            )
        raise
    _close_policy(policy)
    return trajectory_path


def _close_policy(policy: Any) -> None:
    """Call the optional policy close hook without requiring it in the protocol."""
    close_policy = getattr(policy, "close", None)
    if callable(close_policy):
        close_policy()


def _execute_navigation(
    env: Any,
    policy: Any,
    args: Any,
    controller: Any = None,
) -> str:
    """Run closed-loop navigation and return the saved trajectory path."""
    run_dir, frame_dir = prepare_run_dir(args.output_dir)
    trajectory_path = os.path.join(run_dir, "trajectory.csv")
    state_builder = None
    policy_protocol = getattr(policy, "policy_protocol", None)
    paper_pure = policy_protocol == "paper_pure"
    if paper_pure and args.frequency_mode != "joint":
        raise ValueError("paper_pure policies require joint frequency mode")
    if not paper_pure:
        controller = controller or NavigationController(
            ControllerConfig.from_args(args)
        )
    vision_hz, inference_hz = active_frequencies(args)
    video_fps = args.video_fps if args.video_fps is not None else vision_hz

    start_time = time.perf_counter()
    total_steps = 0
    total_collisions = 0
    episode_summaries = []
    grounded_episode_summaries = []
    grounded_demo_artifacts = getattr(args, "grounded_demo_artifacts", False)

    with TrajectoryRecorder(trajectory_path) as recorder:
        for episode_index in range(args.num_episodes):
            episode_frame_dir = os.path.join(
                frame_dir,
                f"episode_{episode_index:03d}",
            )
            os.makedirs(episode_frame_dir, exist_ok=True)
            episode_video_name = (
                "run.mp4"
                if grounded_demo_artifacts and episode_index == 0
                else f"episode_{episode_index:03d}.mp4"
            )
            episode_video_path = os.path.join(run_dir, episode_video_name)
            episode_start_time = time.perf_counter()
            instruction = (
                args.instruction if args.instruction is not None else ""
            )
            try:
                if state_builder is None:
                    state_builder = NavigationStateBuilder(env)
                obs = env.reset()
                start_episode = getattr(policy, "start_episode", None)
                episode_start_error = ""
                if callable(start_episode):
                    try:
                        start_episode(str(env.current_episode.episode_id))
                    except Exception as exc:
                        if not paper_pure:
                            raise
                        episode_start_error = (
                            "episode_start_exception: "
                            f"{type(exc).__name__}: {exc}"
                        )
                episode_fallback = (
                    episode_instruction_text(env.current_episode) or args.goal
                )
                if grounded_demo_artifacts:
                    instruction = (
                        args.instruction
                        if args.instruction is not None
                        else episode_instruction_text(env.current_episode)
                    )
                else:
                    instruction = args.instruction or instruction_text(
                        obs,
                        episode_fallback,
                    )
            except Exception as exc:
                if not grounded_demo_artifacts:
                    raise
                bootstrap_error = (
                    "episode_bootstrap_exception: "
                    f"{type(exc).__name__}: {exc}"
                )
                recorder.write(
                    terminal_runtime_error_row(
                        episode=None,
                        episode_index=episode_index,
                        instruction=instruction,
                        step=0,
                        error=bootstrap_error,
                        metrics={},
                        image_path="",
                        frequency_mode=args.frequency_mode,
                        policy_protocol=policy_protocol or "",
                        vision_hz=vision_hz,
                        inference_hz=inference_hz,
                        wall_time_sec=(
                            time.perf_counter() - episode_start_time
                        ),
                    )
                )
                failed_metrics = {"success": 0.0, "spl": 0.0}
                episode_summaries.append(failed_metrics)
                grounded_episode_summaries.append(
                    {
                        "episode_id": "",
                        "scene_id": "",
                        "instruction": instruction,
                        "instruction_source": (
                            "cli_override"
                            if args.instruction is not None
                            else 'episode.info["instruction"]'
                        ),
                        "dataset_scope": ("habitat_test_scene_grounded_demo"),
                        "benchmark_comparable": False,
                        "steps": 0,
                        "stopped": False,
                        "failed": True,
                        "collisions": 0,
                        "success": 0.0,
                        "spl": 0.0,
                        "final_distance": None,
                        "runtime_sec": (
                            time.perf_counter() - episode_start_time
                        ),
                        "output_paths": {
                            "trajectory": trajectory_path,
                            "episode_summary": os.path.join(
                                run_dir,
                                "episode_summary.json",
                            ),
                            "summary_markdown": os.path.join(
                                run_dir,
                                "summary.md",
                            ),
                            "frames": episode_frame_dir,
                            "video": episode_video_path,
                        },
                        "video_generated": False,
                        "video_error": "episode did not start",
                        "failure_reason": bootstrap_error,
                    }
                )
                print(f"episode bootstrap failed: {bootstrap_error}")
                break
            frame_paths = []
            episode_steps = 0
            collision_count = 0
            stopped = False
            episode_failed = False
            failure_reason = ""
            previous_action = "none"
            previous_action_count = 0
            previous_collision = False
            previous_goal_distance = None
            no_progress_steps = 0
            latest_metrics = {}
            use_background_inference = (
                not paper_pure
                and args.frequency_mode == "layered"
                and not args.no_rate_limit
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
                rgb = obs.get("rgb")
                metrics_before_action = {}
                try:
                    rgb = obs["rgb"]
                    metrics_before_action = env.get_metrics()
                    latest_metrics = dict(metrics_before_action)
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
                except Exception as exc:
                    if not grounded_demo_artifacts:
                        raise
                    runtime_error = (
                        "runtime_exception: " f"{type(exc).__name__}: {exc}"
                    )
                    candidate_image_path = os.path.join(
                        episode_frame_dir,
                        f"frame_{step:03d}.jpg",
                    )
                    image_path, frame_error = save_rgb_frame(
                        rgb,
                        candidate_image_path,
                        step,
                        None,
                        False,
                        False,
                    )
                    if image_path:
                        frame_paths.append(image_path)
                    if frame_error:
                        runtime_error = append_failure_reason(
                            runtime_error,
                            frame_error,
                        )
                    episode_failed = True
                    failure_reason = append_failure_reason(
                        failure_reason,
                        runtime_error,
                    )
                    recorder.write(
                        terminal_runtime_error_row(
                            episode=env.current_episode,
                            episode_index=episode_index,
                            instruction=instruction,
                            step=step,
                            error=runtime_error,
                            metrics=metrics_before_action,
                            image_path=image_path,
                            frequency_mode=args.frequency_mode,
                            policy_protocol=policy_protocol or "",
                            vision_hz=vision_hz,
                            inference_hz=inference_hz,
                            wall_time_sec=wall_time_sec,
                        )
                    )
                    episode_steps += 1
                    break
                if paper_pure:
                    policy_observation = NavigationObservation(
                        rgb=obs["rgb"],
                        depth=None,
                        instruction=instruction,
                        step=step,
                        navigation_context={},
                    )
                else:
                    policy_observation = (
                        navigation_state.as_policy_observation(
                            obs,
                            instruction,
                        )
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
                    try:
                        if episode_start_error:
                            raise RuntimeError(episode_start_error)
                        last_policy_output = policy.predict(policy_observation)
                    except Exception as exc:
                        if not paper_pure:
                            raise
                        termination_reason = (
                            "episode_start_exception"
                            if episode_start_error
                            else "policy_exception"
                        )
                        error = (
                            episode_start_error
                            or f"policy_exception: {type(exc).__name__}: {exc}"
                        )
                        last_policy_output = PolicyOutput(
                            action=None,
                            raw_text="",
                            is_valid=False,
                            termination_reason=termination_reason,
                            metadata={"error": error},
                        )
                    inference_completed = True
                    inference_duration_sec = (
                        time.perf_counter() - inference_start_time
                    )
                    if not last_policy_output.is_valid:
                        invalid_metadata = dict(last_policy_output.metadata)
                        invalid_metadata.setdefault(
                            "latency_seconds",
                            inference_duration_sec,
                        )
                        last_policy_output = replace(
                            last_policy_output,
                            metadata=invalid_metadata,
                        )
                    last_inference_step = step
                    next_inference_time_sec = advance_inference_time(
                        next_inference_time_sec,
                        inference_hz,
                        logical_time_sec,
                    )
                decision_age_steps = (
                    ""
                    if last_inference_step is None
                    else step - last_inference_step
                )
                decision_age_sec = (
                    ""
                    if decision_age_steps == ""
                    else decision_age_steps / vision_hz
                )
                if paper_pure:
                    policy_output = last_policy_output
                    vlm_action = policy_output.action
                    controller_action = policy_output.action
                    action_name = policy_output.action
                else:
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
                    if action_name != vlm_action and hasattr(
                        policy, "clear_action_queue"
                    ):
                        policy.clear_action_queue()

                terminate_without_step = paper_pure and (
                    not policy_output.is_valid or action_name is None
                )
                policy_metadata = dict(policy_output.metadata)
                execution_error = ""
                execution_failed = False
                executed_action_name = None
                if terminate_without_step:
                    action_name = None
                    termination_reason = policy_output.termination_reason
                    metadata_reason = policy_metadata.get("termination_reason")
                    if (
                        not isinstance(termination_reason, str)
                        or not termination_reason
                    ):
                        termination_reason = (
                            metadata_reason
                            if isinstance(metadata_reason, str)
                            and metadata_reason
                            else "invalid_model_output"
                        )
                    policy_output = replace(
                        policy_output,
                        action=None,
                        termination_reason=termination_reason,
                    )
                    failure_reason = str(
                        policy_metadata.get("error") or termination_reason
                    )
                    collision = False
                    metrics = metrics_before_action
                    action_distance_delta = None
                    episode_failed = True
                else:
                    try:
                        obs = step_navigation_action(env, action_name)
                        executed_action_name = action_name
                        collision = bool(env.sim.previous_step_collided)
                        collision_count += int(collision)
                        metrics = env.get_metrics()
                        latest_metrics = dict(metrics)
                        next_distance = optional_float(
                            metrics.get("distance_to_goal", "")
                        )
                        action_distance_delta = (
                            None
                            if navigation_state.distance_to_goal is None
                            or next_distance is None
                            else next_distance
                            - navigation_state.distance_to_goal
                        )
                    except Exception as exc:
                        execution_failed = True
                        execution_error = (
                            "action_execution_exception: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        episode_failed = True
                        failure_reason = append_failure_reason(
                            failure_reason,
                            execution_error,
                        )
                        collision = False
                        metrics = metrics_before_action
                        action_distance_delta = None
                stopped = action_name == "stop" and not execution_failed
                max_steps_reached = bool(
                    step + 1 >= args.max_steps
                    and not terminate_without_step
                    and not execution_failed
                    and not stopped
                    and not env.episode_over
                )
                if max_steps_reached and grounded_demo_artifacts:
                    episode_failed = True
                    failure_reason = "max_steps_reached"

                candidate_image_path = os.path.join(
                    episode_frame_dir,
                    f"frame_{step:03d}.jpg",
                )
                image_path, frame_error = save_rgb_frame(
                    rgb,
                    candidate_image_path,
                    step,
                    action_name,
                    policy_output.is_valid,
                    collision,
                )
                if image_path:
                    frame_paths.append(image_path)
                if frame_error:
                    episode_failed = True
                    failure_reason = append_failure_reason(
                        failure_reason,
                        frame_error,
                    )
                done = bool(
                    terminate_without_step
                    or execution_failed
                    or bool(frame_error)
                    or stopped
                    or env.episode_over
                    or max_steps_reached
                )
                row_error = (
                    policy_metadata.get("error", "")
                    or execution_error
                    or frame_error
                    or (
                        "max_steps_reached"
                        if max_steps_reached and grounded_demo_artifacts
                        else ""
                    )
                )

                row = {
                    "step": step,
                    "frequency_mode": args.frequency_mode,
                    "policy_protocol": policy_protocol or "",
                    "target_vision_hz": vision_hz,
                    "target_inference_hz": inference_hz,
                    "logical_time_sec": logical_time_sec,
                    "wall_time_sec": wall_time_sec,
                    "inference_ran": inference_ran,
                    "inference_completed": inference_completed,
                    "inference_duration_sec": inference_duration_sec,
                    "policy_decision_step": policy_metadata.get(
                        "decision_step", ""
                    ),
                    "policy_inferred": policy_metadata.get("inferred", ""),
                    "decision_age_steps": decision_age_steps,
                    "decision_age_sec": decision_age_sec,
                    "action": action_name,
                    "valid_action": policy_output.is_valid,
                    "raw_vlm_output": policy_output.raw_text,
                    "termination_reason": policy_output.termination_reason
                    or "",
                    "model_latency_seconds": policy_metadata.get(
                        "latency_seconds", ""
                    ),
                    "policy_metadata": json.dumps(
                        policy_metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
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
                    "policy_action": vlm_action,
                    "executed_action": executed_action_name,
                    "action_valid": policy_output.is_valid,
                    "inference_latency_sec": policy_metadata.get(
                        "latency_seconds",
                        inference_duration_sec,
                    ),
                    "done": done,
                    "error": row_error,
                }
                row.update(episode_values(env, episode_index, instruction))
                row.update(
                    agent_values_from_state(navigation_state.agent_state)
                )
                row.update(metric_values(metrics))
                recorder.write(row)
                episode_steps += 1

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

                if (
                    terminate_without_step
                    or execution_failed
                    or frame_error
                    or stopped
                    or env.episode_over
                ):
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

            video_error = ""
            try:
                video_written = write_video(
                    frame_paths,
                    episode_video_path,
                    video_fps,
                )
            except Exception as exc:
                video_written = False
                video_error = str(exc)
                print(f"warning: video was not written: {video_error}")
            try:
                final_metrics = dict(env.get_metrics())
            except Exception as exc:
                final_metrics_error = (
                    "final_metrics_exception: " f"{type(exc).__name__}: {exc}"
                )
                episode_failed = True
                failure_reason = append_failure_reason(
                    failure_reason,
                    final_metrics_error,
                )
                final_metrics = dict(latest_metrics)
            if episode_failed:
                final_metrics["success"] = 0.0
                final_metrics["spl"] = 0.0
            total_steps += episode_steps
            total_collisions += collision_count
            episode_summaries.append(final_metrics)
            if grounded_demo_artifacts:
                episode = env.current_episode
                episode_info = getattr(episode, "info", None) or {}
                instruction_source = (
                    "cli_override"
                    if args.instruction is not None
                    else episode_info.get(
                        "instruction_source",
                        'episode.info["instruction"]',
                    )
                )
                grounded_episode_summaries.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "scene_id": str(episode.scene_id),
                        "instruction": instruction,
                        "instruction_source": str(instruction_source),
                        "dataset_scope": str(
                            episode_info.get(
                                "dataset_scope",
                                "habitat_test_scene_grounded_demo",
                            )
                        ),
                        "benchmark_comparable": bool(
                            episode_info.get("benchmark_comparable", False)
                        ),
                        "steps": episode_steps,
                        "stopped": stopped,
                        "failed": episode_failed,
                        "collisions": collision_count,
                        "success": optional_float(
                            final_metrics.get("success", "")
                        ),
                        "spl": optional_float(final_metrics.get("spl", "")),
                        "final_distance": optional_float(
                            final_metrics.get("distance_to_goal", "")
                        ),
                        "runtime_sec": time.perf_counter()
                        - episode_start_time,
                        "output_paths": {
                            "trajectory": trajectory_path,
                            "episode_summary": os.path.join(
                                run_dir,
                                "episode_summary.json",
                            ),
                            "summary_markdown": os.path.join(
                                run_dir,
                                "summary.md",
                            ),
                            "frames": episode_frame_dir,
                            "video": episode_video_path,
                        },
                        "video_generated": video_written,
                        "video_error": video_error,
                        "failure_reason": failure_reason,
                    }
                )
            print(
                f"episode_summary: episode={episode_index:03d} "
                f"steps={episode_steps} stopped={stopped} failed={episode_failed} "
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

    if grounded_demo_artifacts:
        write_grounded_demo_artifacts(
            run_dir,
            grounded_episode_summaries,
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
