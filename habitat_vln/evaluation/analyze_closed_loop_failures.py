"""Summarize closed-loop navigation failures from a saved trajectory CSV."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EPISODE_FIELDS = [
    "episode_index",
    "episode_id",
    "scene_id",
    "steps",
    "start_distance_m",
    "min_distance_m",
    "final_distance_m",
    "distance_improvement_m",
    "success",
    "spl",
    "collision_count",
    "collision_rate",
    "model_action_steps",
    "controller_override_steps",
    "model_execution_override_steps",
    "invalid_policy_action_count",
    "policy_error_count",
    "model_stop_action_count",
    "stop_action_count",
    "max_no_progress_steps",
    "termination_reasons",
    "failure_reasons",
]


def parse_bool(value):
    """Convert common CSV boolean values to ``True`` or ``False``."""
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value):
    """Convert a CSV value to float, keeping absent or invalid values as None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_value(row, names):
    """Return the first non-empty value among compatible CSV column names."""
    for name in names:
        value = row.get(name, "")
        if str(value).strip():
            return str(value).strip()
    return ""


def episode_key(row):
    """Build a stable group key for one episode, including old trajectory files."""
    return (
        first_value(row, ["episode_index"]),
        first_value(row, ["episode_id"]),
        first_value(row, ["scene_id"]),
    )


def distance_value(row):
    """Read the available distance-to-goal column from current or older logs."""
    return parse_float(first_value(row, ["distance_to_goal", "goal_distance_m"]))


def action_values(row):
    """Read model, controller, and executed actions from one trajectory row."""
    return (
        first_value(row, ["vlm_action", "policy_action"]),
        first_value(row, ["controller_action", "action"]),
        first_value(row, ["executed_action", "action"]),
    )


def optional_mean(values):
    """Return a mean only when at least one numeric value is available."""
    return sum(values) / len(values) if values else None


def failure_reasons(episode):
    """Describe observable failure signals without guessing an unrecorded cause."""
    reasons = []
    if episode["collision_count"]:
        reasons.append("collisions")
    if episode["controller_override_steps"]:
        reasons.append("controller_override")
    if episode["model_execution_override_steps"]:
        reasons.append("model_execution_override")
    if episode["invalid_policy_action_count"]:
        reasons.append("invalid_policy_action")
    if episode["policy_error_count"]:
        reasons.append("policy_or_execution_error")
    if episode["model_stop_action_count"] and episode["success"] != 1.0:
        reasons.append("model_requested_stop_outside_goal")
    if episode["stop_action_count"] and episode["success"] != 1.0:
        reasons.append("stopped_outside_goal")
    improvement = episode["distance_improvement_m"]
    if improvement is not None and improvement < 0.0:
        reasons.append("distance_increased")
    if "max_steps_reached" in episode["termination_reasons"]:
        reasons.append("max_steps_reached")
    return reasons


def analyze_trajectory(rows):
    """Group trajectory rows by episode and calculate diagnostic measurements."""
    grouped = {}
    for row in rows:
        grouped.setdefault(episode_key(row), []).append(row)

    episodes = []
    for key, episode_rows in grouped.items():
        distances = [
            distance
            for row in episode_rows
            for distance in [distance_value(row)]
            if distance is not None
        ]
        model_actions = []
        controller_overrides = 0
        execution_overrides = 0
        collisions = 0
        invalid_actions = 0
        policy_errors = 0
        model_stop_actions = 0
        stop_actions = 0
        no_progress_values = []
        termination_reasons = []

        for row in episode_rows:
            model_action, controller_action, executed_action = action_values(row)
            if model_action:
                model_actions.append(model_action)
            if model_action == "stop":
                model_stop_actions += 1
            if model_action and controller_action and model_action != controller_action:
                controller_overrides += 1
            if model_action and executed_action and model_action != executed_action:
                execution_overrides += 1
            if parse_bool(first_value(row, ["collision", "collided"])):
                collisions += 1
            if first_value(row, ["valid_action", "action_valid"]) and not parse_bool(
                first_value(row, ["valid_action", "action_valid"])
            ):
                invalid_actions += 1
            if first_value(row, ["error"]):
                policy_errors += 1
            reason = first_value(row, ["termination_reason"])
            if reason:
                termination_reasons.append(reason)
            if executed_action == "stop":
                stop_actions += 1
            no_progress = parse_float(first_value(row, ["no_progress_steps"]))
            if no_progress is not None:
                no_progress_values.append(no_progress)

        final_row = episode_rows[-1]
        success = parse_float(final_row.get("success", ""))
        spl = parse_float(final_row.get("spl", ""))
        episode = {
            "episode_index": key[0],
            "episode_id": key[1],
            "scene_id": key[2],
            "steps": len(episode_rows),
            "start_distance_m": distances[0] if distances else None,
            "min_distance_m": min(distances) if distances else None,
            "final_distance_m": distances[-1] if distances else None,
            "distance_improvement_m": (
                distances[0] - distances[-1] if distances else None
            ),
            "success": success,
            "spl": spl,
            "collision_count": collisions,
            "collision_rate": collisions / len(episode_rows),
            "model_action_steps": len(model_actions),
            "controller_override_steps": controller_overrides,
            "model_execution_override_steps": execution_overrides,
            "invalid_policy_action_count": invalid_actions,
            "policy_error_count": policy_errors,
            "model_stop_action_count": model_stop_actions,
            "stop_action_count": stop_actions,
            "max_no_progress_steps": max(no_progress_values, default=0),
            "termination_reasons": sorted(set(termination_reasons)),
        }
        episode["failure_reasons"] = failure_reasons(episode)
        episodes.append(episode)

    successful = [episode for episode in episodes if episode["success"] == 1.0]
    failure_counts = Counter(
        reason for episode in episodes for reason in episode["failure_reasons"]
    )
    summary = {
        "episodes": len(episodes),
        "successful_episodes": len(successful),
        "success_rate": len(successful) / len(episodes) if episodes else 0.0,
        "mean_spl": optional_mean(
            [episode["spl"] for episode in episodes if episode["spl"] is not None]
        ),
        "total_steps": sum(episode["steps"] for episode in episodes),
        "total_collisions": sum(
            episode["collision_count"] for episode in episodes
        ),
        "controller_override_steps": sum(
            episode["controller_override_steps"] for episode in episodes
        ),
        "model_execution_override_steps": sum(
            episode["model_execution_override_steps"] for episode in episodes
        ),
        "invalid_policy_action_count": sum(
            episode["invalid_policy_action_count"] for episode in episodes
        ),
        "policy_error_count": sum(
            episode["policy_error_count"] for episode in episodes
        ),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
    }
    return {"episodes": episodes, "summary": summary}


def read_trajectory(path):
    """Load a non-empty trajectory CSV without requiring Habitat dependencies."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"trajectory contains no rows: {path}")
    return rows


def csv_value(value):
    """Serialize list-valued diagnostic fields into a readable CSV cell."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value


def write_report(report, output_dir, overwrite=False):
    """Save the per-episode CSV and machine-readable summary without overwriting by default."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "episode_diagnostics.csv"
    summary_path = output_dir / "summary.json"
    existing = [path for path in [episode_path, summary_path] if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"refusing to overwrite {names}; pass --overwrite to replace them"
        )

    with episode_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for episode in report["episodes"]:
            writer.writerow(
                {field: csv_value(episode[field]) for field in EPISODE_FIELDS}
            )
    summary_path.write_text(
        json.dumps(report["summary"], indent=2, sort_keys=True) + "\n"
    )
    return episode_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize closed-loop failures from a Habitat VLN trajectory."
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for episode_diagnostics.csv and summary.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing report in --output-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = analyze_trajectory(read_trajectory(args.trajectory))
    episode_path, summary_path = write_report(
        report, args.output_dir, overwrite=args.overwrite
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"saved episode diagnostics to {episode_path}")
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
