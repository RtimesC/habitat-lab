"""Create the primary grounded NaVIDA engineering-demo episode.

The episode uses Habitat's bundled ``skokloster-castle`` test scene.
It is manually grounded from rendered route previews and is not
comparable with R2R, RxR, MP3D, or HM3D benchmark results.
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "data/datasets/pointnav/"
    "habitat_test_scene_grounded_demo/v1/val/val.json.gz"
)

EPISODE_ID = "skokloster-grounded-demo-000"
SCENE_ID = "skokloster-castle.glb"

START_POSITION = [
    3.218496799468994,
    0.20991861820220947,
    14.824010848999023,
]

START_ROTATION = [
    0.0,
    -0.3303377187646299,
    0.0,
    0.9438628033572359,
]

GOAL_POSITION = [
    -3.120556592941284,
    0.20991861820220947,
    13.496004104614258,
]

INSTRUCTION = (
    "Move away from the fireplace into the center of the hall. "
    "Pass the large wooden table, continue along the wall with the "
    "large horse paintings, and stop near the tall window at the far end."
)

GEODESIC_DISTANCE_M = 7.291404724121094
ROUTE_WAYPOINTS = 6
ROUTE_TURN_COUNT = 2
ROUTE_TOTAL_TURN_DEG = 65.32106101538186


def build_payload() -> dict[str, Any]:
    """Return the complete one-episode dataset payload."""

    episode = {
        "episode_id": EPISODE_ID,
        "scene_id": SCENE_ID,
        "start_position": START_POSITION,
        "start_rotation": START_ROTATION,
        "goals": [
            {
                "position": GOAL_POSITION,
                "radius": 3.0,
            }
        ],
        "shortest_paths": None,
        "info": {
            "instruction": INSTRUCTION,
            "geodesic_distance": GEODESIC_DISTANCE_M,
            "route_waypoints": ROUTE_WAYPOINTS,
            "route_turn_count": ROUTE_TURN_COUNT,
            "route_total_turn_deg": ROUTE_TOTAL_TURN_DEG,
            "instruction_source": "manually_grounded_from_route_preview",
            "dataset_scope": "habitat_test_scene_grounded_demo",
            "benchmark_comparable": False,
            "source_candidate_index": 2,
            "scene_family": "habitat_test_scene",
            "demo_role": "primary_engineering_demo",
            "notes": (
                "Instruction manually grounded from rendered route previews. "
                "This episode is not an R2R, RxR, MP3D, or HM3D benchmark sample."
            ),
        },
    }

    return {
        "episodes": [episode],
        "metadata": {
            "dataset_scope": "habitat_test_scene_grounded_demo",
            "benchmark_comparable": False,
            "episode_count": 1,
            "scene_count": 1,
            "instruction_source": "manually_grounded_from_route_preview",
            "source_candidate_index": 2,
            "scene_family": "habitat_test_scene",
            "scene_id": SCENE_ID,
        },
    }


def write_dataset(output: Path) -> dict[str, Any]:
    """Write the grounded demo dataset and return its payload."""
    output = Path(output)
    payload = build_payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = write_dataset(args.output)
    episode = payload["episodes"][0]

    print(f"wrote={args.output}")
    print(f"episode_id={episode['episode_id']}")
    print(f"scene_id={episode['scene_id']}")
    print(f"start_position={episode['start_position']}")
    print(f"start_rotation={episode['start_rotation']}")
    print(f"goal_position={episode['goals'][0]['position']}")
    print(f"instruction={episode['info']['instruction']}")
    print(f"geodesic_distance={episode['info']['geodesic_distance']}")
    print(f"route_turn_count={episode['info']['route_turn_count']}")
    print(
        f"benchmark_comparable={payload['metadata']['benchmark_comparable']}"
    )


if __name__ == "__main__":
    main()
