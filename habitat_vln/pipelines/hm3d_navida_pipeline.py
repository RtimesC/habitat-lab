"""Run the HM3D NaVIDA engineering workflow in explicit, inspectable stages."""

import argparse
import csv
import gzip
import json
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "habitat_vln"
DEFAULT_WORKSPACE = SCRIPT_DIR / "outputs" / "hm3d_navida_system"
DEFAULT_SCENE_ROOT = PROJECT_ROOT / "data" / "versioned_data" / "hm3d-0.2"
DEFAULT_SCENES_DIR = PROJECT_ROOT / "data" / "scene_datasets"
DEFAULT_TASK_CONFIG = "benchmark/nav/pointnav/pointnav_hm3d.yaml"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_state(workspace):
    state_path = workspace / "pipeline_state.json"
    if not state_path.is_file():
        return {"schema_version": 1, "workspace": str(workspace), "stages": {}}
    return json.loads(state_path.read_text())


def save_state(workspace, state):
    workspace.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now().astimezone().isoformat()
    (workspace / "pipeline_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n"
    )


def record_stage(workspace, stage, **values):
    state = load_state(workspace)
    state.setdefault("stages", {})[stage] = {
        "completed_at": datetime.now().astimezone().isoformat(),
        **values,
    }
    save_state(workspace, state)


def run_command(command):
    print("running:", shlex.join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command], cwd=PROJECT_ROOT, check=True
    )


def find_scenes(scene_root):
    scenes = sorted(scene_root.glob("hm3d/**/*.basis.glb"))
    if not scenes:
        scenes = sorted(scene_root.glob("hm3d/**/*.glb"))
    return scenes


def read_episode_file(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def inspect_dataset(path):
    payload = read_episode_file(path)
    episodes = payload.get("episodes", [])
    metadata = payload.get("metadata", {})
    route_profiles = Counter(
        str((item.get("info") or {}).get("route_profile", "unknown"))
        for item in episodes
    )
    coverage_labels = Counter(
        label
        for item in episodes
        for label in (item.get("info") or {}).get("coverage_labels", [])
    )
    distances = [
        float((item.get("info") or {}).get("geodesic_distance"))
        for item in episodes
        if (item.get("info") or {}).get("geodesic_distance") is not None
    ]
    return {
        "path": str(path),
        "episodes": len(episodes),
        "scenes": sorted({str(item.get("scene_id", "")) for item in episodes}),
        "instruction_sources": sorted(
            {
                str((item.get("info") or {}).get("instruction_source", ""))
                for item in episodes
            }
        ),
        "route_profiles": dict(sorted(route_profiles.items())),
        "coverage_labels": dict(sorted(coverage_labels.items())),
        "distance_m": {
            "min": min(distances) if distances else None,
            "mean": sum(distances) / len(distances) if distances else None,
            "max": max(distances) if distances else None,
        },
        "hm3d_citation": metadata.get("hm3d_citation"),
    }


def latest_stage_path(workspace, stage, field):
    state = load_state(workspace)
    value = state.get("stages", {}).get(stage, {}).get(field)
    if not value:
        raise RuntimeError(
            f"No {stage}.{field} in {workspace / 'pipeline_state.json'}; "
            f"run the {stage} stage first."
        )
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"Recorded pipeline artifact is missing: {path}"
        )
    return path


def command_check(args):
    scenes = find_scenes(args.scene_root)
    report = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "scene_root": str(args.scene_root),
        "scene_count": len(scenes),
        "scenes": [str(path) for path in scenes],
    }
    try:
        import habitat
        import habitat_sim

        report["habitat_version"] = getattr(habitat, "__version__", "unknown")
        report["habitat_sim_version"] = getattr(
            habitat_sim, "__version__", "unknown"
        )
    except ImportError as exc:
        report["environment_error"] = str(exc)
    print(json.dumps(report, indent=2, sort_keys=True))
    if len(scenes) < 3:
        raise RuntimeError(
            "The default train/eval protocol requires at least three HM3D scenes."
        )
    record_stage(args.workspace, "check", report=report)


def command_prepare(args):
    scenes = find_scenes(args.scene_root)
    required_scenes = args.train_scene_count + args.val_scene_count
    if len(scenes) < required_scenes:
        raise RuntimeError(
            f"Need {required_scenes} scenes for disjoint train/val, "
            f"but found {len(scenes)}."
        )
    train_scenes = scenes[: args.train_scene_count]
    val_scenes = scenes[args.train_scene_count : required_scenes]
    train_path = args.workspace / "datasets" / "train" / "train.json.gz"
    val_path = args.workspace / "datasets" / "val" / "val.json.gz"
    common = [
        sys.executable,
        "-m",
        "habitat_vln.data.generate_hm3d_pointnav_smoke",
        "--scene-root",
        args.scene_root,
        "--min-distance",
        args.min_distance,
        "--max-distance",
        args.max_distance,
        "--seed",
        args.seed,
        "--sampling-profile",
        args.sampling_profile,
        "--long-distance",
        args.long_distance,
        "--min-route-turns",
        args.min_route_turns,
        "--turn-threshold-deg",
        args.turn_threshold_deg,
        "--clearance-threshold",
        args.clearance_threshold,
        "--min-low-clearance-fraction",
        args.min_low_clearance_fraction,
    ]
    run_command(
        common
        + [
            "--scenes",
            *train_scenes,
            "--episodes",
            args.train_episodes,
            "--dataset-scope",
            f"{args.dataset_label}_train",
            "--output",
            train_path,
        ]
    )
    run_command(
        common
        + [
            "--scenes",
            *val_scenes,
            "--episodes",
            args.val_episodes,
            "--dataset-scope",
            f"{args.dataset_label}_val",
            "--output",
            val_path,
        ]
    )
    train_report = inspect_dataset(train_path)
    val_report = inspect_dataset(val_path)
    overlap = sorted(set(train_report["scenes"]) & set(val_report["scenes"]))
    if overlap:
        raise RuntimeError(f"train/val scene leakage detected: {overlap}")
    report = {
        "dataset_label": args.dataset_label,
        "benchmark_comparable": False,
        "train": train_report,
        "val": val_report,
        "scene_overlap": overlap,
    }
    report_path = args.workspace / "datasets" / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    record_stage(
        args.workspace,
        "prepare",
        train_dataset=str(train_path),
        val_dataset=str(val_path),
        report=str(report_path),
    )


def command_collect(args):
    train_dataset = latest_stage_path(
        args.workspace, "prepare", "train_dataset"
    )
    dataset_template = str(
        train_dataset.parent.parent / "{split}" / "{split}.json.gz"
    )
    output_dir = args.workspace / "oracle"
    before = (
        set(output_dir.glob("collect_*")) if output_dir.exists() else set()
    )
    run_command(
        [
            sys.executable,
            "-m",
            "habitat_vln.data.collect_oracle_training_data",
            "--task-config",
            args.task_config,
            "--dataset-path",
            dataset_template,
            "--dataset-split",
            "train",
            "--scenes-dir",
            args.scenes_dir,
            "--num-episodes",
            args.num_episodes,
            "--max-steps",
            args.max_steps,
            "--output-dir",
            output_dir,
        ]
    )
    created = sorted(set(output_dir.glob("collect_*")) - before)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one new collection directory, found {created}"
        )
    manifest = created[0] / "manifest.jsonl"
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise RuntimeError("Oracle collection produced no training records.")
    with (created[0] / "episodes.csv").open() as handle:
        episode_rows = list(csv.DictReader(handle))
    successful = sum(float(row["success"]) > 0.0 for row in episode_rows)
    if not args.allow_failed_episodes and successful != args.num_episodes:
        raise RuntimeError(
            f"Required {args.num_episodes} successful episodes, got {successful}. "
            "Inspect episodes.csv or use --allow-failed-episodes."
        )
    record_stage(
        args.workspace,
        "collect",
        run_dir=str(created[0]),
        source_manifest=str(manifest),
        episodes_report=str(created[0] / "episodes.csv"),
        requested_episodes=args.num_episodes,
        successful_episodes=successful,
    )


def command_coverage(args):
    """Analyze state-action coverage before or after mixed-data selection."""
    if args.data == "oracle":
        manifest = latest_stage_path(
            args.workspace, "collect", "source_manifest"
        )
    else:
        manifest = latest_stage_path(args.workspace, "build", "mixed_manifest")
    output_dir = args.workspace / "coverage" / f"{args.data}_{timestamp()}"
    run_command(
        [
            sys.executable,
            "-m",
            "habitat_vln.data.analyze_navida_coverage",
            "--manifest",
            manifest,
            "--output-dir",
            output_dir,
            "--minimum-cell-count",
            args.minimum_cell_count,
        ]
    )
    record_stage(
        args.workspace,
        f"coverage_{args.data}",
        output_dir=str(output_dir),
        json_report=str(output_dir / "coverage.json"),
        csv_report=str(output_dir / "coverage_cells.csv"),
        markdown_report=str(output_dir / "coverage.md"),
    )


def command_build(args):
    source_manifest = latest_stage_path(
        args.workspace, "collect", "source_manifest"
    )
    output_dir = args.workspace / "mixed" / f"build_{timestamp()}"
    run_command(
        [
            sys.executable,
            "-m",
            "habitat_vln.data.build_navida_training_data",
            "--source-manifest",
            source_manifest,
            "--output-dir",
            output_dir,
            "--max-samples",
            args.max_samples,
            "--sample-strategy",
            args.sample_strategy,
            "--seed",
            args.seed,
        ]
    )
    manifest = output_dir / "manifest.jsonl"
    record_stage(
        args.workspace,
        "build",
        output_dir=str(output_dir),
        mixed_manifest=str(manifest),
        summary=str(output_dir / "summary.json"),
    )


def command_train(args):
    manifest = latest_stage_path(args.workspace, "build", "mixed_manifest")
    output_dir = args.workspace / "training" / f"train_{timestamp()}"
    command = [
        sys.executable,
        "-m",
        "habitat_vln.training.train_navida_qlora",
        "--manifest",
        manifest,
        "--model-id",
        args.model_id,
        "--output-dir",
        output_dir,
        "--validation-ratio",
        args.validation_ratio,
        "--epochs",
        args.epochs,
        "--gradient-accumulation-steps",
        args.gradient_accumulation_steps,
        "--lora-target-modules",
        "all-linear",
        "--seed",
        args.seed,
    ]
    if args.dry_run:
        command.append("--dry-run")
    run_command(command)
    values = {"output_dir": str(output_dir), "dry_run": args.dry_run}
    if not args.dry_run:
        values["adapter_path"] = str(output_dir / "final_adapter")
    record_stage(args.workspace, "train", **values)


def command_offline_eval(args):
    manifest = latest_stage_path(args.workspace, "build", "mixed_manifest")
    adapter = latest_stage_path(args.workspace, "train", "adapter_path")
    output_dir = args.workspace / "offline_eval" / f"eval_{timestamp()}"
    run_command(
        [
            sys.executable,
            "-m",
            "habitat_vln.evaluation.evaluate_navida_outputs",
            "--manifest",
            manifest,
            "--adapter-path",
            adapter,
            "--model-id",
            args.model_id,
            "--output-dir",
            output_dir,
            "--samples-per-task",
            args.samples_per_task,
        ]
    )
    record_stage(
        args.workspace,
        "offline_eval",
        output_dir=str(output_dir),
        summary=str(output_dir / "summary.json"),
    )


def command_closed_loop(args):
    val_dataset = latest_stage_path(args.workspace, "prepare", "val_dataset")
    adapter = latest_stage_path(args.workspace, "train", "adapter_path")
    dataset_template = str(
        val_dataset.parent.parent / "{split}" / "{split}.json.gz"
    )
    output_dir = (
        args.workspace
        / "closed_loop"
        / ("guarded" if args.guarded else "pure")
    )
    command = [
        sys.executable,
        SCRIPT_DIR / "habitat_vln_nav.py",
        "--task-config",
        args.task_config,
        "--dataset-path",
        dataset_template,
        "--dataset-split",
        "val",
        "--scenes-dir",
        args.scenes_dir,
        "--qwen-role",
        "controller",
        "--frequency-mode",
        "joint",
        "--joint-hz",
        args.joint_hz,
        "--navida-chunk-policy",
        "--navida-max-executed-actions",
        args.max_executed_actions,
        "--adapter-path",
        adapter,
        "--model-id",
        args.model_id,
        "--load-in-4bit",
        "--num-episodes",
        args.num_episodes,
        "--max-steps",
        args.max_steps,
        "--output-dir",
        output_dir,
    ]
    if args.guarded:
        command.append("--force-stop-within-success-radius")
    before = set(output_dir.glob("run_*")) if output_dir.exists() else set()
    run_command(command)
    created = sorted(set(output_dir.glob("run_*")) - before)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one closed-loop run directory, found {created}"
        )
    stage = "closed_loop_guarded" if args.guarded else "closed_loop_pure"
    record_stage(
        args.workspace,
        stage,
        run_dir=str(created[0]),
        trajectory=str(created[0] / "trajectory.csv"),
        privileged_guard=args.guarded,
        frequency_mode="joint",
        joint_hz=args.joint_hz,
        max_executed_actions=args.max_executed_actions,
    )


def command_report(args):
    """Create a compact report from the latest recorded pipeline artifacts."""
    dataset_report = json.loads(
        latest_stage_path(args.workspace, "prepare", "report").read_text()
    )
    mixed_summary = json.loads(
        latest_stage_path(args.workspace, "build", "summary").read_text()
    )
    offline_summary = json.loads(
        latest_stage_path(
            args.workspace, "offline_eval", "summary"
        ).read_text()
    )
    trajectory_path = latest_stage_path(
        args.workspace, "closed_loop_pure", "trajectory"
    )
    with trajectory_path.open() as handle:
        trajectory = list(csv.DictReader(handle))
    distances = [
        float(row["distance_to_goal"])
        for row in trajectory
        if row.get("distance_to_goal") not in (None, "")
    ]
    final_row = trajectory[-1] if trajectory else {}
    closed_loop = {
        "episodes": len({row["episode_id"] for row in trajectory}),
        "steps": len(trajectory),
        "collisions": sum(
            row.get("collision") == "True" for row in trajectory
        ),
        "success": float(final_row.get("success") or 0.0),
        "spl": float(final_row.get("spl") or 0.0),
        "start_distance": distances[0] if distances else None,
        "minimum_distance": min(distances) if distances else None,
        "final_distance": distances[-1] if distances else None,
        "trajectory": str(trajectory_path),
        "privileged_guard": False,
    }
    report = {
        "scope": "HM3D example subset engineering protocol",
        "benchmark_comparable": False,
        "dataset": dataset_report,
        "mixed_training_data": mixed_summary,
        "offline_evaluation": offline_summary,
        "pure_closed_loop": closed_loop,
        "adapter_path": str(
            latest_stage_path(args.workspace, "train", "adapter_path")
        ),
    }
    json_path = args.workspace / "latest_report.json"
    markdown_path = args.workspace / "latest_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(
        "\n".join(
            [
                "# HM3D NaVIDA Engineering Report",
                "",
                "This is an HM3D example-subset engineering result, not an R2R/RxR benchmark result.",
                "",
                "| Item | Result |",
                "|---|---:|",
                f"| Train scenes / episodes | {len(dataset_report['train']['scenes'])} / {dataset_report['train']['episodes']} |",
                f"| Val scenes / episodes | {len(dataset_report['val']['scenes'])} / {dataset_report['val']['episodes']} |",
                f"| Mixed samples (VLN / IDS) | {mixed_summary['samples']} ({mixed_summary['tasks']['vln']} / {mixed_summary['tasks']['ids']}) |",
                f"| VLN valid JSON / first action / exact chunk | {offline_summary['vln']['valid_json_rate']:.3f} / {offline_summary['vln']['first_action_accuracy']:.3f} / {offline_summary['vln']['exact_match_rate']:.3f} |",
                f"| IDS valid JSON / first action / exact chunk | {offline_summary['ids']['valid_json_rate']:.3f} / {offline_summary['ids']['first_action_accuracy']:.3f} / {offline_summary['ids']['exact_match_rate']:.3f} |",
                f"| Pure closed-loop success / SPL | {closed_loop['success']:.3f} / {closed_loop['spl']:.3f} |",
                f"| Pure closed-loop steps / collisions | {closed_loop['steps']} / {closed_loop['collisions']} |",
                f"| Start / minimum / final distance (m) | {closed_loop['start_distance']:.3f} / {closed_loop['minimum_distance']:.3f} / {closed_loop['final_distance']:.3f} |",
                "",
            ]
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    record_stage(
        args.workspace,
        "report",
        json_report=str(json_path),
        markdown_report=str(markdown_path),
    )


def add_shared_args(parser):
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HM3D-only NaVIDA training and closed-loop engineering pipeline."
    )
    add_shared_args(parser)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    subparsers.add_parser("check")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train-scene-count", type=int, default=2)
    prepare.add_argument("--val-scene-count", type=int, default=1)
    prepare.add_argument("--train-episodes", type=int, default=12)
    prepare.add_argument("--val-episodes", type=int, default=6)
    prepare.add_argument("--min-distance", type=float, default=2.0)
    prepare.add_argument("--max-distance", type=float, default=15.0)
    prepare.add_argument(
        "--sampling-profile",
        choices=["random", "coverage"],
        default="coverage",
    )
    prepare.add_argument("--long-distance", type=float, default=8.0)
    prepare.add_argument("--min-route-turns", type=int, default=2)
    prepare.add_argument("--turn-threshold-deg", type=float, default=30.0)
    prepare.add_argument("--clearance-threshold", type=float, default=0.65)
    prepare.add_argument(
        "--min-low-clearance-fraction", type=float, default=0.7
    )
    prepare.add_argument(
        "--dataset-label", default="hm3d_engineering_scale_v1"
    )

    collect = subparsers.add_parser("collect")
    collect.add_argument("--num-episodes", type=int, default=12)
    collect.add_argument("--max-steps", type=int, default=80)
    collect.add_argument("--allow-failed-episodes", action="store_true")

    coverage = subparsers.add_parser("coverage")
    coverage.add_argument(
        "--data", choices=["oracle", "mixed"], default="oracle"
    )
    coverage.add_argument("--minimum-cell-count", type=int, default=10)

    build = subparsers.add_parser("build")
    build.add_argument("--max-samples", type=int, default=100)
    build.add_argument(
        "--sample-strategy",
        choices=["source-order", "round-robin-episodes", "coverage-balanced"],
        default="coverage-balanced",
    )

    train = subparsers.add_parser("train")
    train.add_argument("--epochs", type=float, default=1.0)
    train.add_argument("--validation-ratio", type=float, default=0.1)
    train.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train.add_argument("--dry-run", action="store_true")

    offline = subparsers.add_parser("offline-eval")
    offline.add_argument("--samples-per-task", type=int, default=20)

    closed = subparsers.add_parser("closed-loop")
    closed.add_argument("--num-episodes", type=int, default=2)
    closed.add_argument("--max-steps", type=int, default=80)
    closed.add_argument("--joint-hz", type=float, default=1.0)
    closed.add_argument("--max-executed-actions", type=int, default=1)
    closed.add_argument("--guarded", action="store_true")
    subparsers.add_parser("report")
    return parser.parse_args()


def main():
    args = parse_args()
    args.workspace = args.workspace.expanduser().resolve()
    args.scene_root = args.scene_root.expanduser().resolve()
    args.scenes_dir = args.scenes_dir.expanduser().resolve()
    commands = {
        "check": command_check,
        "prepare": command_prepare,
        "collect": command_collect,
        "coverage": command_coverage,
        "build": command_build,
        "train": command_train,
        "offline-eval": command_offline_eval,
        "closed-loop": command_closed_loop,
        "report": command_report,
    }
    commands[args.stage](args)


if __name__ == "__main__":
    main()
