import argparse
import os
from pathlib import Path

try:
    from .baselines.reactive_doornav import (
        ReactiveDoorNavPolicy,
        load_reactive_doornav_spec,
    )
    from .control import (
        ControllerConfig,
        NavigationController,
        enough_depth,
        local_safety_turn,
    )
    from .core import load_building_prior, load_experiment_config
    from .envs import (
        ACTION_MAP,
        NavigationStateBuilder,
        build_env,
        depth_percentile,
        depth_region_summary,
        depth_sensor_config,
        depth_stats,
        depth_to_meters,
        env_config,
    )
    from .policies import (
        DEFAULT_MODEL_ID,
        EXPLORATION_ACTIONS,
        VALID_ACTIONS,
        MockVLMPolicy,
        QwenVLMPolicy,
    )
    from .runtime import (
        BackgroundPolicyInference,
        TrajectoryRecorder,
        active_frequencies,
        advance_inference_time,
        agent_values_from_state,
        draw_status,
        episode_instruction_text,
        episode_values,
        inference_is_due,
        instruction_text,
        limit_loop_rate,
        metric_values,
        prepare_run_dir,
        resolve_frequency_mode,
        rgb_to_bgr,
        run_navigation,
        timed_policy_predict,
        validate_frequencies,
        write_video,
    )
except ImportError:
    from baselines.reactive_doornav import (
        ReactiveDoorNavPolicy,
        load_reactive_doornav_spec,
    )
    from control import (
        ControllerConfig,
        NavigationController,
        enough_depth,
        local_safety_turn,
    )
    from core import load_building_prior, load_experiment_config
    from envs import (
        ACTION_MAP,
        NavigationStateBuilder,
        build_env,
        depth_percentile,
        depth_region_summary,
        depth_sensor_config,
        depth_stats,
        depth_to_meters,
        env_config,
    )
    from policies import (
        DEFAULT_MODEL_ID,
        EXPLORATION_ACTIONS,
        VALID_ACTIONS,
        MockVLMPolicy,
        QwenVLMPolicy,
    )
    from runtime import (
        BackgroundPolicyInference,
        TrajectoryRecorder,
        active_frequencies,
        advance_inference_time,
        agent_values_from_state,
        draw_status,
        episode_instruction_text,
        episode_values,
        inference_is_due,
        instruction_text,
        limit_loop_rate,
        metric_values,
        prepare_run_dir,
        resolve_frequency_mode,
        rgb_to_bgr,
        run_navigation,
        timed_policy_predict,
        validate_frequencies,
        write_video,
    )


_STATE_API_COMPATIBILITY_EXPORTS = (
    NavigationStateBuilder,
    depth_percentile,
    depth_region_summary,
    depth_sensor_config,
    depth_stats,
    depth_to_meters,
    env_config,
)

_CONTROL_API_COMPATIBILITY_EXPORTS = (
    enough_depth,
    local_safety_turn,
)

_ENV_API_COMPATIBILITY_EXPORTS = (ACTION_MAP, build_env)

_SCHEDULER_API_COMPATIBILITY_EXPORTS = (
    BackgroundPolicyInference,
    active_frequencies,
    advance_inference_time,
    inference_is_due,
    limit_loop_rate,
    resolve_frequency_mode,
    timed_policy_predict,
    validate_frequencies,
)

_RUNTIME_API_COMPATIBILITY_EXPORTS = (
    TrajectoryRecorder,
    agent_values_from_state,
    draw_status,
    episode_instruction_text,
    episode_values,
    instruction_text,
    metric_values,
    prepare_run_dir,
    rgb_to_bgr,
    run_navigation,
    write_video,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DEFAULT_REACTIVE_DOORNAV_SPEC = os.path.join(
    PROJECT_DIR,
    "baselines",
    "reactive_doornav",
    "baseline_spec.yaml",
)
POLICY_CHOICES = ("qwen", "mock", "reactive_doornav_b1")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=POLICY_CHOICES,
        default="qwen",
        help=(
            "Select qwen, mock, or reactive_doornav_b1. The B1 option is a "
            "local reactive visual engineering baseline, not a building-scale policy."
        ),
    )
    parser.add_argument(
        "--baseline-spec",
        help="Versioned parameter specification used by reactive_doornav_b1.",
    )
    parser.add_argument(
        "--experiment-config",
        help=(
            "YAML experiment preset. Values become defaults and explicit "
            "command-line arguments take precedence."
        ),
    )
    parser.add_argument(
        "--task-config",
        help="Semantic-indoor Habitat task config. This must be supplied explicitly.",
    )
    parser.add_argument(
        "--building-prior-file",
        help=(
            "Optional target-free JSON weak prior. Allowed fields are validated "
            "before the policy can see them."
        ),
    )
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-path")
    parser.add_argument("--scenes-dir")
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        help=(
            "Override the Habitat-Sim renderer GPU device id. "
            "Use -1 to let EGL select the default device."
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--episode-start-index",
        type=int,
        default=0,
        help=(
            "Skip this many dataset episodes before running. Use with a new "
            "output directory to resume an interrupted batch."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        help="Optional trained LoRA adapter directory to load on top of --model-id.",
    )
    parser.add_argument(
        "--instruction",
        help="Override the episode instruction. By default, use the VLN dataset instruction.",
    )
    parser.add_argument(
        "--episode-instruction-fallback",
        help="Fallback task text when an episode has no instruction sensor.",
    )
    parser.add_argument("--scene")
    parser.add_argument(
        "--robot-body",
        choices=["none", "six_wheel"],
        default="none",
        help=(
            "Optional visible robot body. six_wheel keeps the existing "
            "navigation actions and mounts RGB-D on a six-wheel chassis."
        ),
    )
    parser.add_argument(
        "--robot-camera-height",
        type=float,
        default=0.62,
        help="Six-wheel robot RGB-D camera height above the floor, in metres.",
    )
    parser.add_argument(
        "--robot-debug-view",
        action="store_true",
        help=(
            "Add a rear third-person robot_view camera and save robot_view "
            "frames/video when --robot-body six_wheel is enabled."
        ),
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the VLM with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="float16",
        help="Compute dtype for bitsandbytes 4-bit quantization.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--output-group",
        help=(
            "Optional one-level experiment group inside --output-dir. "
            "For example: 07_six_wheel_visual."
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        help="Saved-video playback rate. Defaults to the active visual frequency.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help=(
            "Do not save per-step frames or videos. Keep trajectory CSV logs "
            "for batch benchmark runs."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-step logs while retaining episode and final summaries.",
    )
    parser.add_argument(
        "--frequency-mode",
        choices=["joint"],
        default="joint",
        help="Direct semantic decisions use one joint visual-plus-inference loop.",
    )
    parser.add_argument(
        "--vision-hz",
        type=float,
        default=5.0,
        help="Layered mode visual/control update frequency.",
    )
    parser.add_argument(
        "--inference-hz",
        type=float,
        default=0.5,
        help="Layered mode Qwen inference frequency.",
    )
    parser.add_argument(
        "--joint-hz",
        type=float,
        default=1.0,
        help="Joint model visual-plus-inference frequency.",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help=(
            "Run without wall-clock sleeps while preserving the same logical "
            "inference schedule. Useful only for fast tests."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=90)
    parser.add_argument(
        "--fallback-action",
        choices=sorted(VALID_ACTIONS),
        default="move_forward",
    )
    parser.add_argument(
        "--qwen-role",
        choices=["controller"],
        default="controller",
        help="The active semantic task uses direct model actions plus local safety.",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Disable stop. Useful only for open-ended exploration, not standard VLN.",
    )
    parser.add_argument(
        "--enable-forward-depth-guard",
        action="store_true",
        help=(
            "Guarded evaluation: replace unsafe forward actions using the "
            "current depth observation."
        ),
    )
    parser.add_argument(
        "--forward-depth-guard-threshold",
        type=float,
        default=0.35,
        help=(
            "Minimum center depth in metres for guarded move_forward actions."
        ),
    )
    parser.add_argument("--mock-policy", action="store_true")
    config_args, _ = parser.parse_known_args(argv)
    experiment_config = None
    if config_args.experiment_config:
        valid_arguments = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "experiment_config"}
        }
        experiment_config = load_experiment_config(
            config_args.experiment_config,
            valid_arguments,
        )
        experiment_arguments = dict(experiment_config.arguments)
        baseline_spec = experiment_arguments.get("baseline_spec")
        if (
            baseline_spec
            and not Path(baseline_spec).expanduser().is_absolute()
        ):
            experiment_arguments["baseline_spec"] = str(
                (experiment_config.path.parent / baseline_spec).resolve()
            )
        parser.set_defaults(**experiment_arguments)

    args = parser.parse_args(argv)
    args.experiment_name = (
        experiment_config.name if experiment_config is not None else ""
    )
    args.experiment_description = (
        experiment_config.description if experiment_config is not None else ""
    )
    return args


def build_navigation_policy(args):
    """Build the explicitly selected policy and apply its versioned defaults."""
    selected_policy = args.policy
    if args.mock_policy:
        if selected_policy not in {"qwen", "mock"}:
            raise ValueError(
                "--mock-policy cannot be combined with --policy "
                f"{selected_policy}"
            )
        selected_policy = "mock"
    args.policy = selected_policy

    if selected_policy == "reactive_doornav_b1":
        if args.no_stop:
            raise ValueError("reactive_doornav_b1 requires observable STOP")
        spec = load_reactive_doornav_spec(
            args.baseline_spec or DEFAULT_REACTIVE_DOORNAV_SPEC
        )
        args.baseline_spec = str(spec.path)
        if args.instruction is None:
            args.instruction = spec.instruction
        if args.output_dir is None:
            args.output_dir = spec.output_dir
        if args.max_steps is None:
            # The extra runner tick records FAILED after the configured number
            # of non-terminal local actions instead of ending silently.
            args.max_steps = spec.config.max_episode_steps + 1
        print(
            "policy=reactive_doornav_b1 scope=local_reactive_visual "
            f"status={spec.status} spec={spec.path}"
        )
        return ReactiveDoorNavPolicy(config=spec.config)

    if args.baseline_spec:
        raise ValueError(
            "--baseline-spec is only valid for reactive_doornav_b1"
        )
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR
    if args.max_steps is None:
        args.max_steps = 40
    if selected_policy == "mock":
        return MockVLMPolicy(allowed_actions=args.allowed_actions)
    return QwenVLMPolicy(
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        fallback_action=args.fallback_action,
        allowed_actions=args.allowed_actions,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        adapter_path=args.adapter_path,
    )


def main():
    args = parse_args()
    if args.robot_camera_height <= 0:
        raise ValueError("--robot-camera-height must be greater than zero")
    if args.episode_start_index < 0:
        raise ValueError("--episode-start-index must be zero or greater")
    if not args.task_config:
        raise ValueError(
            "--task-config is required. Supply the semantic-indoor task adapter "
            "instead of a legacy PointNav configuration."
        )
    legacy_config_markers = ("pointnav", "r2r", "hm3d", "navida")
    if any(
        marker in args.task_config.lower() for marker in legacy_config_markers
    ):
        raise ValueError(
            "--task-config points to an archived target-navigation workflow. "
            "Use a semantic-indoor task adapter instead."
        )
    args.building_prior = (
        load_building_prior(args.building_prior_file)
        if args.building_prior_file
        else {}
    )
    if args.experiment_name:
        print(
            f"experiment={args.experiment_name} "
            f"config={os.path.abspath(args.experiment_config)}"
        )
    args.frequency_mode = "joint"
    validate_frequencies(args)
    args.execution_actions = (
        set(EXPLORATION_ACTIONS) if args.no_stop else set(VALID_ACTIONS)
    )
    args.allowed_actions = set(args.execution_actions)
    if args.fallback_action not in args.execution_actions:
        raise ValueError(
            f"--fallback-action must be one of {sorted(args.execution_actions)} "
            "with the current --no-stop setting."
        )

    policy = build_navigation_policy(args)
    os.makedirs(args.output_dir, exist_ok=True)

    controller = NavigationController(ControllerConfig.from_args(args))
    env = build_env(
        task_config=args.task_config,
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        scene=args.scene,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        scenes_dir=args.scenes_dir,
        gpu_device_id=args.gpu_device_id,
        robot_body=args.robot_body,
        robot_camera_height=args.robot_camera_height,
        robot_debug_view=args.robot_debug_view,
    )
    try:
        run_navigation(env, policy, args, controller=controller)
    finally:
        env.close()


if __name__ == "__main__":
    main()
