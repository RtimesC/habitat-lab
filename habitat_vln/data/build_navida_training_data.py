"""Convert Oracle manifests into NaVIDA VLN and IDS samples."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .analyze_navida_coverage import bearing_bin, distance_bin
from .navida_data import build_episode_samples, dataset_summary


def load_source_records(manifest_path):
    """Load the existing single-step Oracle manifest grouped by episode."""
    episodes = defaultdict(list)
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: {exc}") from exc
            for field in ["episode_id", "step", "image", "action"]:
                if field not in record:
                    raise ValueError(f"{manifest_path}:{line_number}: missing {field}")
            episodes[str(record["episode_id"])].append(record)
    if not episodes:
        raise ValueError("source manifest contains no records")
    return episodes


def limit_samples(samples, max_samples, strategy):
    """Limit paired VLN/IDS records without silently favoring early episodes."""
    if max_samples is None:
        return samples
    if strategy == "source-order":
        return samples[:max_samples]

    if strategy == "coverage-balanced":
        return coverage_balanced_samples(samples, max_samples)

    by_episode = defaultdict(list)
    episode_order = []
    for sample in samples:
        episode_id = str(sample["episode_id"])
        if episode_id not in by_episode:
            episode_order.append(episode_id)
        by_episode[episode_id].append(sample)

    target_pairs = max_samples // 2
    base_quota, remainder = divmod(target_pairs, len(episode_order))
    selected_pairs = {}
    for episode_index, episode_id in enumerate(episode_order):
        episode_samples = by_episode[episode_id]
        pairs = [
            episode_samples[offset : offset + 2]
            for offset in range(0, len(episode_samples), 2)
        ]
        quota = min(len(pairs), base_quota + int(episode_index < remainder))
        if quota == 0:
            selected_pairs[episode_id] = []
        elif quota == 1:
            selected_pairs[episode_id] = [pairs[-1]]
        else:
            indices = [
                round(index * (len(pairs) - 1) / (quota - 1))
                for index in range(quota)
            ]
            selected_pairs[episode_id] = [pairs[index] for index in indices]

    selected = []
    for pair_index in range(max(len(value) for value in selected_pairs.values())):
        for episode_id in episode_order:
            pairs = selected_pairs[episode_id]
            if pair_index < len(pairs):
                selected.extend(pairs[pair_index])
            if len(selected) >= max_samples:
                return selected[:max_samples]
    return selected


def coverage_balanced_samples(samples, max_samples):
    """Greedily favor rare state-action cells, episodes, and scenes."""
    pairs = [samples[offset : offset + 2] for offset in range(0, len(samples), 2)]
    candidates = []
    for index, pair in enumerate(pairs):
        vln = next((sample for sample in pair if sample["task"] == "vln"), None)
        if vln is None:
            continue
        context = vln.get("navigation_context") or {}
        key = (
            bearing_bin(context.get("goal_angle_deg")),
            distance_bin(
                context.get("goal_distance_m"), context.get("success_distance_m")
            ),
            vln["atomic_actions"][0],
        )
        candidates.append(
            {
                "index": index,
                "pair": pair,
                "cell": key,
                "episode": str(vln["episode_id"]),
                "scene": str(vln.get("scene_id", "")),
                "contains_stop": "stop" in vln["atomic_actions"],
            }
        )

    target_pairs = min(max_samples // 2, len(candidates))
    selected = []
    cell_counts = Counter()
    episode_counts = Counter()
    scene_counts = Counter()
    for episode_id in dict.fromkeys(item["episode"] for item in candidates):
        terminal = next(
            (
                item
                for item in reversed(candidates)
                if item["episode"] == episode_id and item["contains_stop"]
            ),
            None,
        )
        if terminal is None or len(selected) >= target_pairs:
            continue
        candidates.remove(terminal)
        selected.append(terminal["pair"])
        cell_counts[terminal["cell"]] += 1
        episode_counts[terminal["episode"]] += 1
        scene_counts[terminal["scene"]] += 1
    while candidates and len(selected) < target_pairs:
        best = min(
            candidates,
            key=lambda item: (
                cell_counts[item["cell"]],
                episode_counts[item["episode"]],
                scene_counts[item["scene"]],
                item["index"],
            ),
        )
        candidates.remove(best)
        selected.append(best["pair"])
        cell_counts[best["cell"]] += 1
        episode_counts[best["episode"]] += 1
        scene_counts[best["scene"]] += 1
    return [sample for pair in selected for sample in pair][:max_samples]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--merge-probability", type=float, default=0.7)
    parser.add_argument("--max-level2-chunks", type=int, default=3)
    parser.add_argument("--max-history-frames", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Keep at most this many paired VLN/IDS samples; must be even.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["source-order", "round-robin-episodes", "coverage-balanced"],
        default="source-order",
        help="How to select --max-samples. The default preserves old behavior.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples is not None and (
        args.max_samples < 2 or args.max_samples % 2 != 0
    ):
        raise ValueError("--max-samples must be an even integer of at least 2")
    source_manifest = args.source_manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source_manifest.parent / "navida"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_source_records(source_manifest)
    all_samples = []
    episode_rows = []
    for episode_id, episode_records in episodes.items():
        samples = build_episode_samples(
            episode_records,
            source_manifest,
            merge_probability=args.merge_probability,
            max_level2_chunks=args.max_level2_chunks,
            max_history_frames=args.max_history_frames,
            seed=args.seed,
        )
        all_samples.extend(samples)
        task_counts = {
            task: sum(sample["task"] == task for sample in samples)
            for task in ["vln", "ids"]
        }
        episode_rows.append(
            {
                "episode_id": episode_id,
                "source_steps": len(episode_records),
                "vln_samples": task_counts["vln"],
                "ids_samples": task_counts["ids"],
            }
        )

    if args.max_samples is not None:
        all_samples = limit_samples(
            all_samples,
            args.max_samples,
            args.sample_strategy,
        )
        for row in episode_rows:
            episode_id = row["episode_id"]
            row["vln_samples"] = sum(
                sample["episode_id"] == episode_id and sample["task"] == "vln"
                for sample in all_samples
            )
            row["ids_samples"] = sum(
                sample["episode_id"] == episode_id and sample["task"] == "ids"
                for sample in all_samples
            )
        episode_rows = [
            row
            for row in episode_rows
            if row["vln_samples"] > 0 or row["ids_samples"] > 0
        ]

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w") as handle:
        for sample in all_samples:
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
    episode_path = output_dir / "episodes.csv"
    with episode_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)
    summary = dataset_summary(all_samples)
    summary.update(
        {
            "source_manifest": str(source_manifest),
            "merge_probability": args.merge_probability,
            "max_level2_chunks": args.max_level2_chunks,
            "max_history_frames": args.max_history_frames,
            "max_samples": args.max_samples,
            "sample_strategy": args.sample_strategy,
            "seed": args.seed,
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved mixed manifest to {manifest_path}")
    print(f"saved episode report to {episode_path}")


if __name__ == "__main__":
    main()
