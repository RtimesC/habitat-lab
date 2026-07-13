import argparse
from collections import Counter
import gzip
import json
import math
import random
from pathlib import Path

import habitat_sim
import numpy as np


DEFAULT_SCENE_ROOT = Path("data/versioned_data/hm3d-0.2")
DEFAULT_OUTPUT = Path("data/datasets/pointnav/hm3d_smoke/v1/val/val.json.gz")
COVERAGE_PROFILES = [
    "standard",
    "long_route",
    "multi_turn",
    "low_clearance",
    "reorientation",
]
HM3D_CITATION = {
    "citation_key": "ramakrishnan2021hm3d",
    "title": (
        "Habitat-Matterport 3D Dataset (HM3D): 1000 Large-scale 3D "
        "Environments for Embodied AI"
    ),
    "year": 2021,
    "url": "https://arxiv.org/abs/2109.08238",
    "bibtex": (
        "@inproceedings{ramakrishnan2021hm3d,\n"
        "  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale "
        "3D Environments for Embodied {AI}},\n"
        "  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik "
        "Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner "
        "and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel "
        "X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},\n"
        "  booktitle={Thirty-fifth Conference on Neural Information Processing "
        "Systems Datasets and Benchmarks Track},\n"
        "  year={2021},\n"
        "  url={https://arxiv.org/abs/2109.08238}\n"
        "}"
    ),
}


def absolute_without_resolving(path):
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def find_hm3d_scenes(scene_root):
    """Return all Habitat-ready HM3D meshes under the configured root."""
    candidates = sorted(scene_root.glob("hm3d/**/**/*.basis.glb"))
    if not candidates:
        candidates = sorted(scene_root.glob("hm3d/**/*.glb"))
    if not candidates:
        raise FileNotFoundError(
            f"No HM3D .glb scene found under {scene_root / 'hm3d'}"
        )
    return candidates


def find_hm3d_scene(scene_root):
    """Keep the original single-scene behavior for existing callers."""
    return find_hm3d_scenes(scene_root)[0]


def create_sim(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def random_yaw_rotation():
    yaw = random.uniform(-math.pi, math.pi)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def yaw_rotation(yaw):
    """Return a Habitat quaternion list for a yaw angle in radians."""
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def shortest_path(pathfinder, start, goal):
    """Find a geodesic path and keep its waypoints for route analysis."""
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start, dtype=np.float32)
    path.requested_end = np.array(goal, dtype=np.float32)
    if not pathfinder.find_path(path):
        return None
    return path


def interpolated_path_points(points, spacing=0.5):
    """Sample route segments so clearance is not measured only at corners."""
    samples = []
    for start, end in zip(points, points[1:]):
        segment_length = float(np.linalg.norm(end - start))
        intervals = max(1, int(math.ceil(segment_length / spacing)))
        for index in range(intervals):
            samples.append(start + (end - start) * (index / intervals))
    if points:
        samples.append(points[-1])
    return samples


def route_metrics(pathfinder, path, turn_threshold_deg, clearance_threshold):
    """Measure distance, turns, and a navmesh-clearance proxy for one route."""
    points = [np.asarray(point, dtype=np.float32) for point in path.points]
    turn_angles = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        incoming = (middle - first)[[0, 2]]
        outgoing = (last - middle)[[0, 2]]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        if incoming_norm <= 1e-5 or outgoing_norm <= 1e-5:
            continue
        cosine = float(
            np.clip(
                np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm),
                -1.0,
                1.0,
            )
        )
        turn_angles.append(math.degrees(math.acos(cosine)))

    route_samples = interpolated_path_points(points)
    clearances = [
        float(pathfinder.distance_to_closest_obstacle(point, 2.0))
        for point in route_samples
    ]
    low_clearance_fraction = (
        sum(value <= clearance_threshold for value in clearances) / len(clearances)
        if clearances
        else 0.0
    )
    return {
        "geodesic_distance": float(path.geodesic_distance),
        "route_waypoints": len(points),
        "route_turn_count": sum(
            angle >= turn_threshold_deg for angle in turn_angles
        ),
        "route_total_turn_deg": float(sum(turn_angles)),
        "route_min_clearance_m": min(clearances) if clearances else None,
        "route_median_clearance_m": (
            float(np.median(clearances)) if clearances else None
        ),
        "route_low_clearance_fraction": low_clearance_fraction,
    }


def profile_matches(
    profile,
    metrics,
    long_distance,
    min_route_turns,
    min_low_clearance_fraction,
):
    """Return whether a sampled route satisfies the requested profile."""
    if profile == "long_route":
        return metrics["geodesic_distance"] >= long_distance
    if profile == "multi_turn":
        return metrics["route_turn_count"] >= min_route_turns
    if profile == "low_clearance":
        return (
            metrics["route_low_clearance_fraction"]
            >= min_low_clearance_fraction
        )
    return True


def start_rotation(profile, path_points):
    """Create random yaw, or face away from the path for recovery examples."""
    if profile != "reorientation" or len(path_points) < 2:
        return random_yaw_rotation()
    direction = np.asarray(path_points[1]) - np.asarray(path_points[0])
    horizontal_norm = float(np.linalg.norm(direction[[0, 2]]))
    if horizontal_norm <= 1e-5:
        return random_yaw_rotation()
    # Habitat's local forward axis is -Z. This yaw points it away from the route.
    away_yaw = math.atan2(
        float(direction[0]) / horizontal_norm,
        float(direction[2]) / horizontal_norm,
    )
    return yaw_rotation(away_yaw + random.uniform(-math.pi / 12, math.pi / 12))


def engineering_instruction(distance, template_index):
    """Build a truthful language command for HM3D engineering experiments."""
    templates = [
        "Navigate through the indoor space to the target location and stop there.",
        (
            "Follow a navigable indoor route for about "
            f"{distance:.1f} meters to reach the target, then stop."
        ),
        "Find a safe route to the target point and stop when you reach it.",
    ]
    return templates[template_index % len(templates)]


def sample_episode(
    sim,
    episode_id,
    scene_id,
    min_distance,
    max_distance,
    max_tries,
    dataset_scope,
    instruction_index,
    route_profile,
    long_distance,
    min_route_turns,
    turn_threshold_deg,
    clearance_threshold,
    min_low_clearance_fraction,
):
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        raise RuntimeError(
            "No navmesh is loaded for the scene. HM3D habitat assets should "
            "include a matching .navmesh file."
        )

    for _ in range(max_tries):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        path = shortest_path(pathfinder, start, goal)
        if path is None:
            continue
        distance = float(path.geodesic_distance)
        within_min = distance >= min_distance
        within_max = max_distance is None or distance <= max_distance
        if not (np.isfinite(distance) and within_min and within_max):
            continue
        metrics = route_metrics(
            pathfinder,
            path,
            turn_threshold_deg,
            clearance_threshold,
        )
        if profile_matches(
            route_profile,
            metrics,
            long_distance,
            min_route_turns,
            min_low_clearance_fraction,
        ):
            coverage_labels = [route_profile]
            if metrics["geodesic_distance"] >= long_distance:
                coverage_labels.append("long_route")
            if metrics["route_turn_count"] >= min_route_turns:
                coverage_labels.append("multi_turn")
            if (
                metrics["route_low_clearance_fraction"]
                >= min_low_clearance_fraction
            ):
                coverage_labels.append("low_clearance")
            coverage_labels = list(dict.fromkeys(coverage_labels))
            return {
                "episode_id": str(episode_id),
                "scene_id": scene_id,
                "start_position": [float(v) for v in start],
                "start_rotation": start_rotation(route_profile, path.points),
                "info": {
                    **metrics,
                    "difficulty": "engineering",
                    "instruction": engineering_instruction(
                        distance, instruction_index
                    ),
                    "instruction_source": "hm3d_pointnav_template_v1",
                    "dataset_scope": dataset_scope,
                    "route_profile": route_profile,
                    "coverage_labels": coverage_labels,
                    "reorientation_required": route_profile == "reorientation",
                },
                "goals": [
                    {
                        "position": [float(v) for v in goal],
                        "radius": None,
                    }
                ],
                "shortest_paths": None,
                "start_room": None,
            }

    raise RuntimeError(
        "Could not sample a start/goal pair with "
        f"{min_distance} <= distance <= {max_distance}"
    )


def selected_scene_paths(args):
    """Resolve explicit, all-scene, or backward-compatible first-scene mode."""
    if args.scene and args.scenes:
        raise ValueError("Use either --scene or --scenes, not both.")
    if args.all_scenes and (args.scene or args.scenes):
        raise ValueError("--all-scenes cannot be combined with --scene/--scenes.")
    if args.scene:
        return [args.scene]
    if args.scenes:
        return list(args.scenes)
    candidates = find_hm3d_scenes(args.scene_root)
    return candidates if args.all_scenes else candidates[:1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene", type=Path)
    parser.add_argument(
        "--scenes",
        type=Path,
        nargs="+",
        help="Explicit list of HM3D scene meshes to sample evenly.",
    )
    parser.add_argument(
        "--all-scenes",
        action="store_true",
        help="Use every HM3D .glb found under --scene-root.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min-distance", type=float, default=2.0)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=6.0,
        help="Maximum geodesic distance for smoke episodes. Use <= 0 to disable.",
    )
    parser.add_argument("--max-tries", type=int, default=1000)
    parser.add_argument(
        "--sampling-profile",
        choices=["random", "coverage"],
        default="random",
        help="Cycle through route-coverage profiles instead of pure random sampling.",
    )
    parser.add_argument("--long-distance", type=float, default=8.0)
    parser.add_argument("--min-route-turns", type=int, default=2)
    parser.add_argument("--turn-threshold-deg", type=float, default=30.0)
    parser.add_argument("--clearance-threshold", type=float, default=0.65)
    parser.add_argument("--min-low-clearance-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dataset-scope",
        default="hm3d_engineering",
        help="Label saved with every episode to prevent benchmark confusion.",
    )
    args = parser.parse_args()

    max_distance = None if args.max_distance <= 0 else args.max_distance
    if args.min_distance <= 0:
        raise ValueError("--min-distance must be greater than 0.")
    if max_distance is not None and max_distance < args.min_distance:
        raise ValueError(
            "--max-distance must be greater than or equal to --min-distance."
        )
    if args.sampling_profile == "coverage":
        if args.long_distance < args.min_distance:
            raise ValueError("--long-distance must be at least --min-distance")
        if max_distance is not None and args.long_distance > max_distance:
            raise ValueError("--long-distance must not exceed --max-distance")
    if args.min_route_turns < 1 or args.turn_threshold_deg <= 0:
        raise ValueError("route turn thresholds must be positive")
    if args.clearance_threshold <= 0:
        raise ValueError("--clearance-threshold must be positive")
    if not 0.0 <= args.min_low_clearance_fraction <= 1.0:
        raise ValueError("--min-low-clearance-fraction must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    scene_root = absolute_without_resolving(args.scene_root)
    scene_paths = [
        absolute_without_resolving(path) for path in selected_scene_paths(args)
    ]
    if args.episodes < len(scene_paths):
        raise ValueError(
            "--episodes must be at least the number of selected scenes so every "
            "scene receives an episode"
        )
    for scene_path in scene_paths:
        if not scene_path.is_file():
            raise FileNotFoundError(f"HM3D scene does not exist: {scene_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    episodes = []
    base_count, remainder = divmod(args.episodes, len(scene_paths))
    for scene_index, scene_path in enumerate(scene_paths):
        scene_id = str(scene_path.relative_to(scene_root))
        scene_count = base_count + int(scene_index < remainder)
        sim = create_sim(scene_path)
        try:
            for local_index in range(scene_count):
                episode_index = len(episodes)
                episode_id = f"{scene_path.parent.name}-{local_index:06d}"
                route_profile = (
                    COVERAGE_PROFILES[episode_index % len(COVERAGE_PROFILES)]
                    if args.sampling_profile == "coverage"
                    else "random"
                )
                episodes.append(
                    sample_episode(
                        sim,
                        episode_id,
                        scene_id,
                        args.min_distance,
                        max_distance,
                        args.max_tries,
                        args.dataset_scope,
                        episode_index,
                        route_profile,
                        args.long_distance,
                        args.min_route_turns,
                        args.turn_threshold_deg,
                        args.clearance_threshold,
                        args.min_low_clearance_fraction,
                    )
                )
        finally:
            sim.close()

    payload = {
        "episodes": episodes,
        "metadata": {
            "dataset_scope": args.dataset_scope,
            "instruction_source": "hm3d_pointnav_template_v1",
            "sampling_profile": args.sampling_profile,
            "route_profile_counts": dict(
                sorted(
                    Counter(
                        episode["info"]["route_profile"] for episode in episodes
                    ).items()
                )
            ),
            "coverage_thresholds": {
                "long_distance_m": args.long_distance,
                "min_route_turns": args.min_route_turns,
                "turn_threshold_deg": args.turn_threshold_deg,
                "clearance_threshold_m": args.clearance_threshold,
                "min_low_clearance_fraction": args.min_low_clearance_fraction,
            },
            "hm3d_citation": HM3D_CITATION,
            "scene_count": len(scene_paths),
            "seed": args.seed,
        },
    }
    with gzip.open(args.output, "wt") as f:
        json.dump(payload, f)

    print(f"scenes: {len(scene_paths)}")
    for scene_path in scene_paths:
        print(f"  - {scene_path}")
    print(f"episodes: {len(episodes)}")
    print(f"route_profiles: {payload['metadata']['route_profile_counts']}")
    print(f"distance_range: {args.min_distance} <= geodesic <= {max_distance}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
