import argparse
import os

try:
    from .control import (
        ControllerConfig,
        NavigationController,
        action_from_advice,
        enough_depth,
        geometric_navigation_action,
        navigation_fallback_action,
    )
    from .core import load_experiment_config
    from .envs import (
        ACTION_MAP,
        DEFAULT_TASK_CONFIG,
        NavigationStateBuilder,
        build_env,
        depth_percentile,
        depth_region_summary,
        depth_sensor_config,
        depth_stats,
        depth_to_meters,
        env_config,
        get_goal_position,
        normalize_angle_deg,
        optional_float,
        pointgoal_from_agent_state,
        pointgoal_from_observation,
        pointgoal_state,
        success_distance,
    )
    from .policies import (
        ADVISORY_ACTIONS,
        DEFAULT_MODEL_ID,
        DEFAULT_OFFICIAL_NAVIDA_URL,
        EXPLORATION_ACTIONS,
        VALID_ACTIONS,
        MockVLMPolicy,
        NaVIDAChunkPolicy,
        OfficialNaVIDAHTTPPolicy,
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
    from control import (
        ControllerConfig,
        NavigationController,
        action_from_advice,
        enough_depth,
        geometric_navigation_action,
        navigation_fallback_action,
    )
    from core import load_experiment_config
    from envs import (
        ACTION_MAP,
        DEFAULT_TASK_CONFIG,
        NavigationStateBuilder,
        build_env,
        depth_percentile,
        depth_region_summary,
        depth_sensor_config,
        depth_stats,
        depth_to_meters,
        env_config,
        get_goal_position,
        normalize_angle_deg,
        optional_float,
        pointgoal_from_agent_state,
        pointgoal_from_observation,
        pointgoal_state,
        success_distance,
    )
    from policies import (
        ADVISORY_ACTIONS,
        DEFAULT_MODEL_ID,
        DEFAULT_OFFICIAL_NAVIDA_URL,
        EXPLORATION_ACTIONS,
        VALID_ACTIONS,
        MockVLMPolicy,
        NaVIDAChunkPolicy,
        OfficialNaVIDAHTTPPolicy,
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
    get_goal_position,
    normalize_angle_deg,
    optional_float,
    pointgoal_from_agent_state,
    pointgoal_from_observation,
    pointgoal_state,
    success_distance,
)

_CONTROL_API_COMPATIBILITY_EXPORTS = (
    action_from_advice,
    enough_depth,
    geometric_navigation_action,
    navigation_fallback_action,
)

_ENV_API_COMPATIBILITY_EXPORTS = (ACTION_MAP, DEFAULT_TASK_CONFIG, build_env)

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        help=(
            "YAML experiment preset. Values become defaults and explicit "
            "command-line arguments take precedence."
        ),
    )
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
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
        "--goal",
        help="Alias fallback for --instruction when an episode has no instruction sensor.",
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
    parser.add_argument("--max-steps", type=int, default=40)
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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
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
        "--frequency-mode",
        choices=["auto", "layered", "joint"],
        default="auto",
        help=(
            "auto uses layered timing for advisor mode and joint timing for "
            "controller mode."
        ),
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
        choices=["advisor", "controller"],
        default="advisor",
        help="Use Qwen as a high-level advisor or direct low-level controller.",
    )
    parser.add_argument(
        "--advisor-fallback",
        choices=sorted(ADVISORY_ACTIONS),
        default="follow_goal",
        help="Fallback advisory action when Qwen output cannot be parsed.",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Disable stop. Useful only for open-ended exploration, not standard VLN.",
    )
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Allow stop before --min-stop-step.",
    )
    parser.add_argument(
        "--force-stop-within-success-radius",
        action="store_true",
        help="Guarded evaluation: force stop when privileged goal distance is successful.",
    )
    parser.add_argument(
        "--min-stop-step",
        type=int,
        default=8,
        help="Reject stop before this step unless --allow-early-stop is set.",
    )
    parser.add_argument(
        "--disable-anti-stuck",
        action="store_true",
        help="Disable repeated-turn overrides.",
    )
    parser.add_argument(
        "--max-repeated-turns",
        type=int,
        default=2,
        help="Override repeated same-direction turns after this count.",
    )
    parser.add_argument(
        "--max-no-progress-steps",
        type=int,
        default=2,
        help="Override repeated turns after this many no-progress steps.",
    )
    parser.add_argument(
        "--anti-stuck-forward-depth",
        type=float,
        default=0.35,
        help="Require this much center depth before anti-stuck moves forward.",
    )
    parser.add_argument(
        "--anti-stuck-max-goal-angle",
        type=float,
        default=30.0,
        help="Only anti-stuck forward when the target angle is within this range.",
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
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--mock-policy", action="store_true")
    policy_group.add_argument(
        "--navida-chunk-policy",
        action="store_true",
        help="Use a NaVIDA adapter to generate and execute structured action chunks.",
    )
    policy_group.add_argument(
        "--official-navida-http",
        action="store_true",
        help="Use the separately hosted Official NaVIDA HTTP policy.",
    )
    parser.add_argument(
        "--official-navida-url",
        default=DEFAULT_OFFICIAL_NAVIDA_URL,
        help="Base URL of the Official NaVIDA policy server.",
    )
    parser.add_argument(
        "--official-navida-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each Official NaVIDA request.",
    )
    parser.add_argument(
        "--policy-protocol",
        choices=["paper_pure"],
        help="Execution protocol for Official NaVIDA HTTP mode.",
    )
    parser.add_argument("--navida-max-history-frames", type=int, default=8)
    parser.add_argument(
        "--navida-max-executed-actions",
        type=int,
        default=1,
        help=(
            "Actions kept from each NaVIDA chunk. Joint 1 Hz replanning requires 1."
        ),
    )
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
        parser.set_defaults(**experiment_config.arguments)

    args = parser.parse_args(argv)
    args.experiment_name = (
        experiment_config.name if experiment_config is not None else ""
    )
    args.experiment_description = (
        experiment_config.description if experiment_config is not None else ""
    )
    return args


def main():
    args = parse_args()
    if args.robot_camera_height <= 0:
        raise ValueError("--robot-camera-height must be greater than zero")
    if args.experiment_name:
        print(
            f"experiment={args.experiment_name} "
            f"config={os.path.abspath(args.experiment_config)}"
        )
    if args.official_navida_http:
        if args.enable_forward_depth_guard:
            raise ValueError(
                "--enable-forward-depth-guard cannot be used with "
                "--official-navida-http because paper_pure runs bypass "
                "the controller."
            )
        if args.frequency_mode == "layered":
            raise ValueError(
                "--official-navida-http requires joint frequency mode"
            )
        if args.adapter_path:
            raise ValueError(
                "--official-navida-http cannot be combined with --adapter-path"
            )
        if args.no_stop:
            raise ValueError("--official-navida-http cannot be combined with --no-stop")
        args.frequency_mode = "joint"
        args.policy_protocol = args.policy_protocol or "paper_pure"
    else:
        if args.policy_protocol is not None:
            raise ValueError(
                "--policy-protocol is only valid with --official-navida-http"
            )
        args.frequency_mode = resolve_frequency_mode(
            args.frequency_mode,
            args.qwen_role,
        )
    validate_frequencies(args)
    os.makedirs(args.output_dir, exist_ok=True)
    args.execution_actions = (
        set(EXPLORATION_ACTIONS) if args.no_stop else set(VALID_ACTIONS)
    )
    args.allowed_actions = (
        set(ADVISORY_ACTIONS)
        if args.qwen_role == "advisor"
        else set(args.execution_actions)
    )
    policy_fallback_action = (
        args.advisor_fallback
        if args.qwen_role == "advisor"
        else args.fallback_action
    )
    if args.fallback_action not in args.execution_actions:
        raise ValueError(
            f"--fallback-action must be one of {sorted(args.execution_actions)} "
            "with the current --no-stop setting."
        )

    if args.official_navida_http:
        policy = OfficialNaVIDAHTTPPolicy(
            base_url=args.official_navida_url,
            timeout=args.official_navida_timeout,
        )
    elif args.navida_chunk_policy:
        if args.qwen_role != "controller":
            raise ValueError(
                "--navida-chunk-policy requires --qwen-role controller"
            )
        if not args.adapter_path:
            raise ValueError("--navida-chunk-policy requires --adapter-path")
        if args.navida_max_executed_actions != 1:
            raise ValueError(
                "joint NaVIDA timing requires --navida-max-executed-actions 1 "
                "so the model replans from the latest image at every 1 Hz tick"
            )
        policy = NaVIDAChunkPolicy(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            fallback_action=policy_fallback_action,
            max_history_frames=args.navida_max_history_frames,
            max_executed_actions=args.navida_max_executed_actions,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            max_new_tokens=max(args.max_new_tokens, 64),
        )
    elif args.mock_policy:
        policy = MockVLMPolicy(allowed_actions=args.allowed_actions)
    else:
        policy = QwenVLMPolicy(
            model_id=args.model_id,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            fallback_action=policy_fallback_action,
            allowed_actions=args.allowed_actions,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            adapter_path=args.adapter_path,
        )

    controller = (
        None
        if args.official_navida_http
        else NavigationController(ControllerConfig.from_args(args))
    )
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
