"""Check whether R2R episode files and referenced MP3D scenes are available."""

import argparse
import gzip
import json
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("data/datasets/vln/mp3d/r2r/v1")
DEFAULT_SCENES_ROOT = Path("data/scene_datasets/mp3d")


def scene_id_from_path(scene_path):
    path = Path(scene_path)
    if path.suffix == ".glb":
        return path.stem
    return path.name


def load_split(dataset_root, split):
    path = dataset_root / split / f"{split}.json.gz"
    if not path.exists():
        return path, []

    with gzip.open(path, "rt") as f:
        payload = json.load(f)
    return path, payload.get("episodes", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val_seen", "val_unseen"],
    )
    args = parser.parse_args()

    all_required_scenes = set()
    missing_splits = []

    for split in args.splits:
        path, episodes = load_split(args.dataset_root, split)
        if not episodes:
            missing_splits.append(str(path))
            print(f"{split}: missing or empty ({path})")
            continue

        scenes = sorted({scene_id_from_path(ep["scene_id"]) for ep in episodes})
        all_required_scenes.update(scenes)
        print(f"{split}: episodes={len(episodes)} scenes={len(scenes)}")

    missing_scenes = sorted(
        scene
        for scene in all_required_scenes
        if not (args.scenes_root / scene / f"{scene}.glb").exists()
    )

    print(f"required_scenes={len(all_required_scenes)}")
    print(f"missing_scenes={len(missing_scenes)}")

    if missing_scenes:
        for scene in missing_scenes[:40]:
            print(f"missing: {args.scenes_root / scene / f'{scene}.glb'}")
        if len(missing_scenes) > 40:
            print(f"... {len(missing_scenes) - 40} more missing scenes")

    if missing_splits or missing_scenes:
        raise SystemExit(1)

    print("VLN data looks ready.")


if __name__ == "__main__":
    main()
