"""Generate and score VLN/IDS action chunks from a NaVIDA manifest."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

try:
    from .navida_data import load_manifest, navigation_prompt
    from .navida_policy import QwenNaVIDAModel
except ImportError:
    from navida_data import load_manifest, navigation_prompt
    from navida_policy import QwenNaVIDAModel


FIELDS = [
    "sample_id",
    "episode_id",
    "task",
    "valid_json",
    "exact_match",
    "first_action_match",
    "target_actions",
    "predicted_actions",
    "raw_text",
]


def select_balanced(records, samples_per_task):
    selected = []
    for task in ["vln", "ids"]:
        selected.extend(
            [record for record in records if record["task"] == task][
                :samples_per_task
            ]
        )
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=128 * 128)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_manifest(args.manifest)
    selected = select_balanced(records, args.samples_per_task)
    runtime = QwenNaVIDAModel(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    rows = []
    with prediction_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, record in enumerate(selected):
            images = [Image.open(path).convert("RGB") for path in record["_image_paths"]]
            try:
                output = runtime.generate_action_chunk(
                    images, navigation_prompt(record)
                )
            finally:
                for image in images:
                    image.close()
            target = record["atomic_actions"]
            predicted = output.atomic_actions
            row = {
                "sample_id": record["sample_id"],
                "episode_id": record["episode_id"],
                "task": record["task"],
                "valid_json": int(output.is_valid),
                "exact_match": int(predicted == target),
                "first_action_match": int(
                    bool(predicted) and predicted[0] == target[0]
                ),
                "target_actions": json.dumps(target, separators=(",", ":")),
                "predicted_actions": json.dumps(predicted, separators=(",", ":")),
                "raw_text": output.raw_text,
            }
            writer.writerow(row)
            rows.append(row)
            print(
                f"sample={index + 1}/{len(selected)} task={record['task']} "
                f"valid={row['valid_json']} exact={row['exact_match']} "
                f"first={row['first_action_match']}"
            )

    summary = {}
    for task in ["all", "vln", "ids"]:
        task_rows = rows if task == "all" else [row for row in rows if row["task"] == task]
        summary[task] = {
            "samples": len(task_rows),
            "valid_json_rate": sum(row["valid_json"] for row in task_rows)
            / len(task_rows),
            "exact_match_rate": sum(row["exact_match"] for row in task_rows)
            / len(task_rows),
            "first_action_accuracy": sum(
                row["first_action_match"] for row in task_rows
            )
            / len(task_rows),
        }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved predictions to {prediction_path}")


if __name__ == "__main__":
    main()
