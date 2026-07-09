import argparse
import gzip
import json
import math
import random
from pathlib import Path

import habitat_sim
import numpy as np


DEFAULT_SCENE_ROOT = Path("data/versioned_data/hm3d-0.2")
DEFAULT_OUTPUT = Path("data/datasets/pointnav/hm3d_smoke/v1/val/val.json.gz")


def absolute_without_resolving(path):
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def find_hm3d_scene(scene_root):
    candidates = sorted(scene_root.glob("hm3d/**/**/*.basis.glb"))
    if not candidates:
        candidates = sorted(scene_root.glob("hm3d/**/*.glb"))
    if not candidates:
        raise FileNotFoundError(
            f"No HM3D .glb scene found under {scene_root / 'hm3d'}"
        )
    return candidates[0]


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


def geodesic_distance(pathfinder, start, goal):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start, dtype=np.float32)
    path.requested_end = np.array(goal, dtype=np.float32)
    if not pathfinder.find_path(path):
        return math.inf
    return float(path.geodesic_distance)


def sample_episode(sim, episode_id, scene_id, min_distance, max_distance, max_tries):
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        raise RuntimeError(
            "No navmesh is loaded for the scene. HM3D habitat assets should "
            "include a matching .navmesh file."
        )

    for _ in range(max_tries):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        distance = geodesic_distance(pathfinder, start, goal)
        within_min = distance >= min_distance
        within_max = max_distance is None or distance <= max_distance
        if np.isfinite(distance) and within_min and within_max:
            return {
                "episode_id": str(episode_id),
                "scene_id": scene_id,
                "start_position": [float(v) for v in start],
                "start_rotation": random_yaw_rotation(),
                "info": {
                    "geodesic_distance": distance,
                    "difficulty": "smoke",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene", type=Path)
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
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    max_distance = None if args.max_distance <= 0 else args.max_distance
    if args.min_distance <= 0:
        raise ValueError("--min-distance must be greater than 0.")
    if max_distance is not None and max_distance < args.min_distance:
        raise ValueError(
            "--max-distance must be greater than or equal to --min-distance."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    scene_path = args.scene or find_hm3d_scene(args.scene_root)
    scene_path = absolute_without_resolving(scene_path)
    scene_root = absolute_without_resolving(args.scene_root)
    scene_id = str(scene_path.relative_to(scene_root))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    sim = create_sim(scene_path)
    try:
        episodes = [
            sample_episode(
                sim,
                episode_id,
                scene_id,
                args.min_distance,
                max_distance,
                args.max_tries,
            )
            for episode_id in range(args.episodes)
        ]
    finally:
        sim.close()

    payload = {"episodes": episodes}
    with gzip.open(args.output, "wt") as f:
        json.dump(payload, f)

    print(f"scene: {scene_path}")
    print(f"scene_id: {scene_id}")
    print(f"episodes: {len(episodes)}")
    print(f"distance_range: {args.min_distance} <= geodesic <= {max_distance}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
