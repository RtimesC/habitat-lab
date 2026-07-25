"""Analyze navigation-state and Oracle-action coverage in JSONL data."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BEARING_BINS = ["strong_left", "left", "ahead", "right", "strong_right"]
DISTANCE_BINS = ["arrived", "near", "mid", "far", "unknown"]
ACTIONS = ["move_forward", "turn_left", "turn_right", "stop"]


def bearing_bin(value):
    """Bucket target bearing; negative angles mean left."""
    if value is None:
        return "unknown"
    value = float(value)
    if value < -45.0:
        return "strong_left"
    if value < -15.0:
        return "left"
    if value <= 15.0:
        return "ahead"
    if value <= 45.0:
        return "right"
    return "strong_right"


def distance_bin(distance, success_distance=None):
    """Bucket target distance using the episode's success radius when present."""
    if distance is None:
        return "unknown"
    distance = float(distance)
    success_distance = (
        float(success_distance) if success_distance is not None else 0.2
    )
    if distance <= success_distance:
        return "arrived"
    if distance < 1.0:
        return "near"
    if distance < 3.0:
        return "mid"
    return "far"


def record_state_action(record):
    """Normalize either atomic Oracle or mixed NaVIDA records."""
    if record.get("task") not in (None, "vln"):
        return None
    context = record.get("navigation_context") or {}
    if record.get("task") == "vln":
        actions = record.get("atomic_actions") or []
        action = actions[0] if actions else None
        contains_stop = "stop" in actions
    else:
        action = record.get("action")
        contains_stop = action == "stop"
    if action not in ACTIONS:
        return None
    return {
        "episode_id": str(record.get("episode_id", "")),
        "scene_id": str(record.get("scene_id", "")),
        "bearing_bin": bearing_bin(context.get("goal_angle_deg")),
        "distance_bin": distance_bin(
            context.get("goal_distance_m"), context.get("success_distance_m")
        ),
        "action": action,
        "contains_stop": contains_stop,
        "previous_action": str(context.get("previous_action", "unknown")),
        "collided": bool(context.get("collided", False)),
        "no_progress_steps": int(context.get("no_progress_steps") or 0),
        "previous_action_count": int(
            context.get("previous_action_count") or 0
        ),
    }


def canonical_action(bearing, distance):
    """Return the direct-control action expected for the core coverage grid."""
    if distance == "arrived":
        return "stop"
    if bearing in {"strong_left", "left"}:
        return "turn_left"
    if bearing == "ahead":
        return "move_forward"
    if bearing in {"right", "strong_right"}:
        return "turn_right"
    return None


def load_rows(manifest):
    """Load the relevant VLN state-action rows from a JSONL manifest."""
    rows = []
    with manifest.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest}:{line_number}: {exc}") from exc
            row = record_state_action(record)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError(f"No VLN state-action rows found in {manifest}")
    return rows


def build_report(rows, manifest, minimum_cell_count):
    """Build coverage counts and identify underrepresented observed cells."""
    cells = defaultdict(list)
    for row in rows:
        key = (row["bearing_bin"], row["distance_bin"], row["action"])
        cells[key].append(row)
    cell_rows = []
    for (bearing, distance, action), values in sorted(cells.items()):
        cell_rows.append(
            {
                "bearing_bin": bearing,
                "distance_bin": distance,
                "action": action,
                "samples": len(values),
                "episodes": len({value["episode_id"] for value in values}),
                "scenes": len({value["scene_id"] for value in values}),
                "under_target": len(values) < minimum_cell_count,
            }
        )
    observed_cells = len(cell_rows)
    possible_cells = len(BEARING_BINS) * len(DISTANCE_BINS[:-1]) * len(ACTIONS)
    core_keys = [
        (bearing, distance, canonical_action(bearing, distance))
        for bearing in BEARING_BINS
        for distance in DISTANCE_BINS[:-1]
    ]
    core_counts = {key: len(cells.get(key, [])) for key in core_keys}
    recovery_rows = [
        row for row in rows if row["collided"] or row["no_progress_steps"] >= 2
    ]
    collision_recovery_rows = [row for row in rows if row["collided"]]
    no_progress_recovery_rows = [
        row for row in rows if row["no_progress_steps"] >= 2
    ]
    stop_rows = [row for row in rows if row["contains_stop"]]
    first_action_stop_rows = [row for row in rows if row["action"] == "stop"]
    return {
        "manifest": str(manifest),
        "samples": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "scenes": len({row["scene_id"] for row in rows}),
        "actions": dict(
            sorted(Counter(row["action"] for row in rows).items())
        ),
        "bearing_bins": dict(
            sorted(Counter(row["bearing_bin"] for row in rows).items())
        ),
        "distance_bins": dict(
            sorted(Counter(row["distance_bin"] for row in rows).items())
        ),
        "collision_states": dict(
            sorted(Counter(str(row["collided"]) for row in rows).items())
        ),
        "stop_coverage": {
            "samples": len(stop_rows),
            "first_action_samples": len(first_action_stop_rows),
            "episodes": len({row["episode_id"] for row in stop_rows}),
            "scenes": len({row["scene_id"] for row in stop_rows}),
        },
        "recovery_coverage": {
            "samples": len(recovery_rows),
            "episodes": len({row["episode_id"] for row in recovery_rows}),
            "scenes": len({row["scene_id"] for row in recovery_rows}),
            "actions": dict(
                sorted(Counter(row["action"] for row in recovery_rows).items())
            ),
            "collision_samples": len(collision_recovery_rows),
            "no_progress_samples": len(no_progress_recovery_rows),
        },
        "observed_cells": observed_cells,
        "possible_cells": possible_cells,
        "observed_cell_rate": observed_cells / possible_cells,
        "minimum_cell_count": minimum_cell_count,
        "under_target_cells": sum(row["under_target"] for row in cell_rows),
        "core_observed_cells": sum(
            count > 0 for count in core_counts.values()
        ),
        "core_possible_cells": len(core_keys),
        "core_observed_cell_rate": sum(
            count > 0 for count in core_counts.values()
        )
        / len(core_keys),
        "core_under_target_cells": sum(
            count < minimum_cell_count for count in core_counts.values()
        ),
        "core_cells": [
            {
                "bearing_bin": key[0],
                "distance_bin": key[1],
                "action": key[2],
                "samples": core_counts[key],
                "under_target": core_counts[key] < minimum_cell_count,
            }
            for key in core_keys
        ],
        "cells": cell_rows,
    }


def write_report(report, output_dir):
    """Write JSON, CSV, and a compact Markdown coverage summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "coverage.json"
    csv_path = output_dir / "coverage_cells.csv"
    markdown_path = output_dir / "coverage.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report["cells"][0].keys())
        writer.writeheader()
        writer.writerows(report["cells"])
    lines = [
        "# VLN State-Action Coverage",
        "",
        f"- Samples: {report['samples']}",
        f"- Episodes: {report['episodes']}",
        f"- Scenes: {report['scenes']}",
        f"- Observed cells: {report['observed_cells']} / {report['possible_cells']} ({report['observed_cell_rate']:.1%})",
        f"- Under-target observed cells: {report['under_target_cells']} (target >= {report['minimum_cell_count']})",
        f"- Core control cells: {report['core_observed_cells']} / {report['core_possible_cells']} ({report['core_observed_cell_rate']:.1%})",
        f"- Under-target core cells: {report['core_under_target_cells']}",
        f"- STOP coverage: {report['stop_coverage']['samples']} chunks containing STOP / {report['stop_coverage']['first_action_samples']} first-action STOP / {report['stop_coverage']['episodes']} episodes / {report['stop_coverage']['scenes']} scenes",
        f"- Recovery coverage: {report['recovery_coverage']['samples']} samples / {report['recovery_coverage']['episodes']} episodes / {report['recovery_coverage']['scenes']} scenes",
        f"- Collision / no-progress recovery samples: {report['recovery_coverage']['collision_samples']} / {report['recovery_coverage']['no_progress_samples']}",
        "",
        "| Bearing | Distance | Action | Samples | Episodes | Scenes | Under target |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["cells"]:
        lines.append(
            f"| {row['bearing_bin']} | {row['distance_bin']} | {row['action']} | "
            f"{row['samples']} | {row['episodes']} | {row['scenes']} | "
            f"{str(row['under_target']).lower()} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    return json_path, csv_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-cell-count", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    if args.minimum_cell_count < 1:
        raise ValueError("--minimum-cell-count must be positive")
    rows = load_rows(manifest)
    report = build_report(rows, manifest, args.minimum_cell_count)
    paths = write_report(report, args.output_dir.expanduser().resolve())
    summary = {
        key: report[key]
        for key in report
        if key not in {"cells", "core_cells"}
    }
    print(json.dumps(summary, indent=2))
    print(f"saved coverage JSON to {paths[0]}")
    print(f"saved coverage cells to {paths[1]}")
    print(f"saved coverage Markdown to {paths[2]}")


if __name__ == "__main__":
    main()
