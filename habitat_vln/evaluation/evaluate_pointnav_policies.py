"""Compare geometric, Qwen-advisor, and Oracle PointNav policies."""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from ..control import action_from_advice, geometric_navigation_action
from ..core import NavigationObservation
from ..envs import (
    ACTION_MAP,
    NavigationStateBuilder,
    build_env,
    build_navigation_context,
    get_goal_position,
    optional_float,
)
from ..policies import DEFAULT_MODEL_ID, QwenVLMPolicy
from ..runtime import instruction_text

_STATE_API_COMPATIBILITY_EXPORTS = (build_navigation_context,)


DEFAULT_TASK_CONFIG = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
DEFAULT_DATASET_PATH = (
    "data/datasets/pointnav/hm3d_smoke/v1/{split}/{split}.json.gz"
)
DEFAULT_SCENES_DIR = "data/versioned_data/hm3d-0.2"
DEFAULT_INSTRUCTION = (
    "Navigate to the target location and stop when you reach it."
)
DEFAULT_OUTPUT_DIR = "habitat_vln/outputs/pointnav_policy_eval"

ACTION_NAMES = {value: key for key, value in ACTION_MAP.items()}

EPISODE_FIELDS = [
    "policy",
    "episode_index",
    "episode_id",
    "geodesic_distance",
    "steps",
    "collisions",
    "success",
    "spl",
    "final_distance",
    "stopped",
    "invalid_policy_outputs",
    "runtime_sec",
]


def episode_geodesic_distance(episode):
    info = getattr(episode, "info", None) or {}
    try:
        return float(info.get("geodesic_distance", np.nan))
    except (TypeError, ValueError):
        return float("nan")


def make_run_dir(output_dir):
    run_dir = os.path.join(
        output_dir, datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def action_name_from_habitat_action(action):
    if action is None:
        return "stop"
    return ACTION_NAMES.get(action, str(action).lower())


def choose_action(policy_name, env, policy, obs, instruction, context):
    if policy_name == "oracle":
        follower = policy
        goal_position = get_goal_position(env)
        return (
            action_name_from_habitat_action(
                follower.get_next_action(goal_position)
            ),
            True,
        )

    if policy_name == "geometric":
        return (
            geometric_navigation_action(
                context["goal_distance_m"],
                context["goal_angle_deg"],
                context["success_distance_m"],
                context["previous_collision"],
                context["depth_left_m"],
                context["depth_center_m"],
                context["depth_right_m"],
            ),
            True,
        )

    policy_output = policy.predict(
        NavigationObservation(
            rgb=obs["rgb"],
            depth=obs.get("depth"),
            instruction=instruction,
            step=context["step"],
            navigation_context=context,
        ),
    )
    action_name = action_from_advice(
        policy_output.action,
        context["goal_distance_m"],
        context["goal_angle_deg"],
        context["success_distance_m"],
        context["previous_collision"],
        context["depth_left_m"],
        context["depth_center_m"],
        context["depth_right_m"],
    )
    return action_name, policy_output.is_valid


def run_policy(env, policy_name, policy, args):
    state_builder = NavigationStateBuilder(env)
    goal_success_distance = state_builder.success_distance_m
    rows = []

    for episode_index in range(args.num_episodes):
        episode_start = time.perf_counter()
        obs = env.reset()
        instruction = args.instruction or instruction_text(obs, args.goal)
        if policy_name == "oracle":
            policy = ShortestPathFollower(
                sim=env.sim,
                goal_radius=goal_success_distance,
                return_one_hot=False,
            )

        collisions = 0
        stopped = False
        invalid_policy_outputs = 0
        previous_action = "none"
        previous_action_count = 0
        previous_collision = False
        previous_goal_distance = None
        no_progress_steps = 0
        steps_taken = 0

        for step in range(args.max_steps):
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
            current_distance = navigation_state.distance_to_goal
            context = navigation_state.as_navigation_context()
            action_name, is_valid = choose_action(
                policy_name,
                env,
                policy,
                obs,
                instruction,
                context,
            )
            if not is_valid:
                invalid_policy_outputs += 1
            if action_name not in ACTION_MAP:
                raise RuntimeError(
                    f"{policy_name} returned invalid action {action_name!r}"
                )

            obs = env.step(ACTION_MAP[action_name])
            steps_taken += 1
            collision = bool(env.sim.previous_step_collided)
            collisions += int(collision)
            stopped = action_name == "stop"
            metrics = env.get_metrics()
            next_distance = optional_float(metrics.get("distance_to_goal", ""))
            action_distance_delta = (
                None
                if current_distance is None or next_distance is None
                else next_distance - current_distance
            )

            if stopped or env.episode_over:
                break

            if action_name == previous_action:
                previous_action_count += 1
            else:
                previous_action_count = 1
            previous_action = action_name
            previous_collision = collision
            previous_goal_distance = context["goal_distance_m"]
            progress_delta = (
                context["distance_change_m"]
                if context["distance_change_m"] is not None
                else action_distance_delta
            )
            if progress_delta is not None and progress_delta < -0.05:
                no_progress_steps = 0
            else:
                no_progress_steps += 1

        final_metrics = env.get_metrics()
        row = {
            "policy": policy_name,
            "episode_index": episode_index,
            "episode_id": env.current_episode.episode_id,
            "geodesic_distance": episode_geodesic_distance(
                env.current_episode
            ),
            "steps": steps_taken,
            "collisions": collisions,
            "success": float(final_metrics.get("success", 0.0)),
            "spl": float(final_metrics.get("spl", 0.0)),
            "final_distance": float(
                final_metrics.get("distance_to_goal", np.nan)
            ),
            "stopped": stopped,
            "invalid_policy_outputs": invalid_policy_outputs,
            "runtime_sec": time.perf_counter() - episode_start,
        }
        rows.append(row)
        print(
            "episode_summary: "
            f"policy={policy_name} episode={episode_index:03d} "
            f"steps={steps_taken} collisions={collisions} "
            f"success={row['success']:.3f} spl={row['spl']:.3f} "
            f"final_distance={row['final_distance']:.3f}"
        )

    return rows


def summarize(rows, notes):
    summaries = []
    for policy_name in ["geometric", "qwen_advisor", "oracle"]:
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        if not policy_rows:
            summaries.append(
                {
                    "policy": policy_name,
                    "episodes": 0,
                    "success": "",
                    "spl": "",
                    "avg_geodesic_dist": "",
                    "avg_collision": "",
                    "avg_final_dist": "",
                    "note": notes.get(policy_name, ""),
                }
            )
            continue

        summaries.append(
            {
                "policy": policy_name,
                "episodes": len(policy_rows),
                "success": float(
                    np.mean([row["success"] for row in policy_rows])
                ),
                "spl": float(np.mean([row["spl"] for row in policy_rows])),
                "avg_geodesic_dist": float(
                    np.nanmean(
                        [row["geodesic_distance"] for row in policy_rows]
                    )
                ),
                "avg_collision": float(
                    np.mean([row["collisions"] for row in policy_rows])
                ),
                "avg_final_dist": float(
                    np.mean([row["final_distance"] for row in policy_rows])
                ),
                "note": notes.get(policy_name, ""),
            }
        )
    return summaries


def format_metric(value):
    if value == "":
        return "?"
    return f"{float(value):.3f}"


def write_outputs(run_dir, episode_rows, summary_rows):
    episode_path = os.path.join(run_dir, "episode_metrics.csv")
    with open(episode_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)

    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "episodes",
                "success",
                "spl",
                "avg_geodesic_dist",
                "avg_collision",
                "avg_final_dist",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_path = os.path.join(run_dir, "summary.md")
    with open(markdown_path, "w") as f:
        f.write(
            "| policy | episodes | success | SPL | avg geodesic dist | avg collision | avg final dist | note |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in summary_rows:
            f.write(
                f"| {row['policy']} | {row['episodes']} | "
                f"{format_metric(row['success'])} | {format_metric(row['spl'])} | "
                f"{format_metric(row['avg_geodesic_dist'])} | "
                f"{format_metric(row['avg_collision'])} | "
                f"{format_metric(row['avg_final_dist'])} | {row['note']} |\n"
            )

    return episode_path, summary_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-split", default="val")
    parser.add_argument("--scenes-dir", default=DEFAULT_SCENES_DIR)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--hfov", type=int, default=90)
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        help=(
            "Override the Habitat-Sim renderer GPU device id. "
            "Use -1 to let EGL select the available device."
        ),
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--goal")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["geometric", "qwen_advisor", "oracle"],
        default=["geometric", "qwen_advisor", "oracle"],
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        help="Optional trained LoRA adapter directory to load on top of --model-id.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", default="float16")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = make_run_dir(args.output_dir)
    all_rows = []
    notes = {
        "geometric": "baseline",
        "qwen_advisor": "advisor gain",
        "oracle": "upper bound",
    }

    for policy_name in args.policies:
        print(f"running policy={policy_name}")
        env = build_env(
            task_config=args.task_config,
            width=args.width,
            height=args.height,
            hfov=args.hfov,
            dataset_split=args.dataset_split,
            dataset_path=args.dataset_path,
            scenes_dir=args.scenes_dir,
            gpu_device_id=args.gpu_device_id,
        )
        try:
            if policy_name == "qwen_advisor":
                policy = QwenVLMPolicy(
                    model_id=args.model_id,
                    device_map=args.device_map,
                    torch_dtype=args.torch_dtype,
                    max_new_tokens=args.max_new_tokens,
                    fallback_action="follow_goal",
                    allowed_actions={
                        "follow_goal",
                        "turn_left_to_avoid",
                        "turn_right_to_avoid",
                        "stop_if_reached",
                    },
                    load_in_4bit=args.load_in_4bit,
                    bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
                    adapter_path=args.adapter_path,
                )
            else:
                policy = None
            all_rows.extend(run_policy(env, policy_name, policy, args))
        finally:
            env.close()

    summary_rows = summarize(all_rows, notes)
    episode_path, summary_path, markdown_path = write_outputs(
        run_dir,
        all_rows,
        summary_rows,
    )
    print(f"saved episode metrics to {episode_path}")
    print(f"saved summary csv to {summary_path}")
    print(f"saved summary table to {markdown_path}")
    with open(markdown_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
