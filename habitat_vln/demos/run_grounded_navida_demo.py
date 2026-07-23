"""Run Official NaVIDA on the grounded Habitat test-scene episode."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ..data.create_grounded_demo_episode import write_dataset
from ..envs import build_env
from ..policies import (
    DEFAULT_OFFICIAL_NAVIDA_URL,
    OfficialNaVIDAHTTPPolicy,
)
from ..runtime import run_navigation


DEFAULT_TASK_CONFIG = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
DEFAULT_DATASET_PATH = (
    "data/datasets/pointnav/"
    "habitat_test_scene_grounded_demo/v1/{split}/{split}.json.gz"
)
DEFAULT_DATASET_SPLIT = "val"
DEFAULT_SCENES_DIR = "data/scene_datasets/habitat-test-scenes"
DEFAULT_OUTPUT_DIR = "habitat_vln/outputs/grounded_navida_demo"


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse the grounded Official NaVIDA demo command line."""
    parser = argparse.ArgumentParser(
        description="Run Official NaVIDA on the grounded Habitat test-scene demo."
    )
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--scenes-dir", default=DEFAULT_SCENES_DIR)
    parser.add_argument("--gpu-device-id", type=int)
    parser.add_argument(
        "--official-navida-url",
        default=DEFAULT_OFFICIAL_NAVIDA_URL,
    )
    parser.add_argument("--official-navida-timeout", type=float, default=30.0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=90)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--instruction")
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--joint-hz", type=float, default=1.0)
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="Disable wall-clock pacing for lightweight integration tests.",
    )
    return parser.parse_args(argv)


def prepare_dataset(dataset_path: Path, default_dataset_path: Path) -> Path:
    """Generate a missing default grounded dataset without touching custom data."""
    dataset_path = Path(dataset_path)
    default_dataset_path = Path(default_dataset_path)
    if dataset_path.is_file():
        return dataset_path
    if dataset_path == default_dataset_path:
        write_dataset(dataset_path)
        return dataset_path
    raise FileNotFoundError(f"custom dataset does not exist: {dataset_path}")


def _resolved_dataset_path(path_template: str, split: str) -> Path:
    """Resolve Habitat's conventional ``{split}`` dataset placeholder."""
    return Path(path_template.replace("{split}", split))


def _validate_args(args: argparse.Namespace) -> None:
    """Reject values that cannot produce a valid bounded demo run."""
    positive_values = {
        "--num-episodes": args.num_episodes,
        "--max-steps": args.max_steps,
        "--width": args.width,
        "--height": args.height,
        "--hfov": args.hfov,
        "--official-navida-timeout": args.official_navida_timeout,
        "--joint-hz": args.joint_hz,
    }
    if args.video_fps is not None:
        positive_values["--video-fps"] = args.video_fps
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")


def main(argv: Optional[Sequence[str]] = None) -> str:
    """Build and run the grounded Official NaVIDA engineering demo."""
    args = parse_args(argv)
    _validate_args(args)

    dataset_path = _resolved_dataset_path(
        args.dataset_path,
        args.dataset_split,
    )
    default_dataset_path = _resolved_dataset_path(
        DEFAULT_DATASET_PATH,
        args.dataset_split,
    )
    prepare_dataset(
        dataset_path,
        default_dataset_path=default_dataset_path,
    )

    args.dataset_path = str(dataset_path)
    args.frequency_mode = "joint"
    args.vision_hz = args.joint_hz
    args.inference_hz = args.joint_hz
    args.qwen_role = "controller"
    args.goal = None
    args.advisor_fallback = "follow_goal"
    args.grounded_demo_artifacts = True

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
        policy = OfficialNaVIDAHTTPPolicy(
            base_url=args.official_navida_url,
            timeout=args.official_navida_timeout,
        )
        return run_navigation(env, policy, args, controller=None)
    finally:
        env.close()


if __name__ == "__main__":
    main()
