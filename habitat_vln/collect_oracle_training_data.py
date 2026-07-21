"""Collect RGB/instruction/Oracle-action samples from Habitat episodes."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

try:
    from .evaluate_pointnav_policies import (
        action_name_from_habitat_action,
    )
    from .envs import (
        ACTION_MAP,
        build_env,
        build_navigation_context,
        depth_sensor_config,
        success_distance,
    )
    from .runtime import (
        episode_instruction_text,
        instruction_text,
        rgb_to_bgr,
    )
    from .training_data import make_record
except ImportError:
    from evaluate_pointnav_policies import (
        action_name_from_habitat_action,
    )
    from envs import (
        ACTION_MAP,
        build_env,
        build_navigation_context,
        depth_sensor_config,
        success_distance,
    )
    from runtime import (
        episode_instruction_text,
        instruction_text,
        rgb_to_bgr,
    )
    from training_data import make_record


DEFAULT_TASK_CONFIG = "benchmark/nav/vln_r2r.yaml"
DEFAULT_DATASET_PATH = "data/datasets/vln/mp3d/r2r/v1/{split}/{split}.json.gz"
DEFAULT_SCENES_DIR = "data/scene_datasets/mp3d"
DEFAULT_OUTPUT_DIR = "habitat_vln/outputs/oracle_training_data"

EPISODE_FIELDS = [
    "episode_index",
    "episode_id",
    "scene_id",
    "route_profile",
    "coverage_labels",
    "geodesic_distance",
    "route_turn_count",
    "route_low_clearance_fraction",
    "samples",
    "steps",
    "success",
    "spl",
    "final_distance",
    "valid_for_training",
    "recovery_samples",
    "collision_recovery_samples",
]


def make_run_dir(output_dir):
    run_dir = Path(output_dir) / datetime.now().strftime("collect_%Y%m%d_%H%M%S")
    (run_dir / "images").mkdir(parents=True, exist_ok=False)
    return run_dir


def append_records(manifest_path, records):
    with manifest_path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def collect_episode(env, episode_index, args, run_dir):
    obs = env.reset()
    episode = env.current_episode
    episode_id = str(episode.episode_id)
    scene_id = str(episode.scene_id)
    episode_fallback = episode_instruction_text(episode)
    instruction = args.instruction or instruction_text(obs, episode_fallback)
    if not instruction:
        raise RuntimeError(f"episode {episode_id} has no navigation instruction")

    goal_radius = success_distance(env)
    follower = ShortestPathFollower(
        sim=env.sim,
        goal_radius=goal_radius,
        return_one_hot=False,
    )
    depth_config = depth_sensor_config(env)
    episode_dir = run_dir / "images" / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=False)

    previous_action = "none"
    previous_action_count = 0
    previous_collision = False
    previous_goal_distance = None
    no_progress_steps = 0
    records = []
    stopped = False
    recovery_samples = 0
    collision_recovery_samples = 0

    for step in range(args.max_steps):
        context = build_navigation_context(
            env,
            obs,
            depth_config,
            goal_radius,
            step,
            previous_action,
            previous_action_count,
            previous_collision,
            previous_goal_distance,
            no_progress_steps,
        )
        is_recovery = bool(
            context.get("collided", False)
            or int(context.get("no_progress_steps", 0)) >= 2
        )
        recovery_samples += int(is_recovery)
        collision_recovery_samples += int(context.get("collided", False))
        goal_position = getattr(episode.goals[0], "position", None)
        action_name = action_name_from_habitat_action(
            follower.get_next_action(goal_position)
        )
        if action_name not in ACTION_MAP:
            raise RuntimeError(f"Oracle returned unsupported action {action_name!r}")

        image_path = episode_dir / f"step_{step:04d}.jpg"
        next_image_path = episode_dir / f"step_{step + 1:04d}.jpg"
        if not cv2.imwrite(str(image_path), rgb_to_bgr(obs["rgb"])):
            raise RuntimeError(f"failed to write image: {image_path}")
        relative_image = image_path.relative_to(run_dir)
        relative_next_image = next_image_path.relative_to(run_dir)
        sample_id = f"{episode_id}:{step}"
        records.append(
            make_record(
                sample_id=sample_id,
                episode_id=episode_id,
                scene_id=scene_id,
                step=step,
                instruction=instruction,
                image=relative_image,
                action=action_name,
                navigation_context=context,
                next_image=relative_next_image,
            )
        )

        obs = env.step(ACTION_MAP[action_name])
        if not cv2.imwrite(str(next_image_path), rgb_to_bgr(obs["rgb"])):
            raise RuntimeError(f"failed to write image: {next_image_path}")
        collision = bool(env.sim.previous_step_collided)
        stopped = action_name == "stop"
        if stopped or env.episode_over:
            break

        if action_name == previous_action:
            previous_action_count += 1
        else:
            previous_action_count = 1
        previous_action = action_name
        previous_collision = collision
        previous_goal_distance = context["goal_distance_m"]
        distance_change = context["distance_change_m"]
        no_progress_steps = (
            0
            if distance_change is not None and distance_change < -0.05
            else no_progress_steps + 1
        )

    metrics = env.get_metrics()
    episode_info = getattr(episode, "info", {}) or {}
    success = float(metrics.get("success", 0.0))
    valid_for_training = bool(stopped and success > 0.0)
    row = {
        "episode_index": episode_index,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "route_profile": str(episode_info.get("route_profile", "unknown")),
        "coverage_labels": ";".join(episode_info.get("coverage_labels", [])),
        "geodesic_distance": episode_info.get("geodesic_distance", ""),
        "route_turn_count": episode_info.get("route_turn_count", ""),
        "route_low_clearance_fraction": episode_info.get(
            "route_low_clearance_fraction", ""
        ),
        "samples": len(records),
        "steps": len(records),
        "success": success,
        "spl": float(metrics.get("spl", 0.0)),
        "final_distance": float(metrics.get("distance_to_goal", float("nan"))),
        "valid_for_training": valid_for_training,
        "recovery_samples": recovery_samples,
        "collision_recovery_samples": collision_recovery_samples,
    }
    return records, row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--scenes-dir", default=DEFAULT_SCENES_DIR)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--hfov", type=int, default=90)
    parser.add_argument("--instruction", help="Override every episode instruction.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep-failed-episodes",
        action="store_true",
        help="Include incomplete Oracle trajectories in the training manifest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_episodes < 1 or args.max_steps < 1:
        raise ValueError("--num-episodes and --max-steps must be positive")

    env = build_env(
        task_config=args.task_config,
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        scenes_dir=args.scenes_dir,
    )
    try:
        run_dir = make_run_dir(args.output_dir)
        manifest_path = run_dir / "manifest.jsonl"
        manifest_path.touch()
        episode_path = run_dir / "episodes.csv"
        included_samples = 0
        with episode_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            for episode_index in range(args.num_episodes):
                records, row = collect_episode(env, episode_index, args, run_dir)
                include = row["valid_for_training"] or args.keep_failed_episodes
                if include:
                    append_records(manifest_path, records)
                    included_samples += len(records)
                writer.writerow(row)
                print(
                    f"episode={episode_index:04d} id={row['episode_id']} "
                    f"samples={row['samples']} success={row['success']:.3f} "
                    f"included={include}"
                )
    finally:
        env.close()

    print(f"saved manifest to {manifest_path}")
    print(f"saved episode report to {episode_path}")
    print(f"included training samples: {included_samples}")


if __name__ == "__main__":
    main()
