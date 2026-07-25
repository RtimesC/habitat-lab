# Habitat VLN Architecture

本目录是项目自己的导航应用层，底层继续使用 Habitat-Lab 和
Habitat-Sim。当前框架的目标是让仿真环境、导航状态、模型策略、确定性控制、
运行调度和实验产物具有清楚边界，同时保持主导航命令稳定。

## Runtime data flow

```text
Habitat dataset / scene
        |
        v
envs/habitat_env.py             create environment, execute named action
        |
        v
envs/habitat_state.py           RGB/depth + simulator state -> navigation state
        |
        v
policies/                       Mock / Qwen / NaVIDA / Official NaVIDA HTTP
        |
        v
control/navigation_controller.py
                                advice conversion + deterministic safety overrides
        |
        v
runtime/navigation_runner.py    closed-loop execution
        |
        +--> runtime/recorder.py       trajectory.csv
        +--> runtime/artifacts.py      frames and MP4
        +--> runtime/scheduler.py      layered/joint inference timing
```

`habitat_vln_nav.py` is the primary CLI. It parses arguments, constructs the policy,
controller, and Habitat environment, then calls `run_navigation()`.

Runtime defaults can be loaded from versioned YAML files under `configs/runtime/`.
The YAML `arguments` mapping uses the same names as CLI options, and explicit CLI
arguments override the preset values.

## Module contracts

### `core/`

- `NavigationObservation` is the policy input: RGB, optional depth, instruction,
  step number, and structured navigation context.
- `PolicyOutput` is the raw policy result: optional parsed action, original text,
  validity flag, optional termination reason, and policy metadata.
- `NavigationPolicy` documents the shared `predict(observation)` interface.

### `envs/`

- `HabitatEnvironmentConfig` contains Habitat dataset, scene, and sensor settings.
- `build_env()` preserves the original function interface.
- `step_navigation_action()` is the only runtime conversion from the four project
  action names to `HabitatSimActions`.
- `NavigationStateBuilder.build()` assembles distance, signed goal angle, depth
  regions, collision, progress, and agent pose.
- `HabitatNavigationState.as_navigation_context()` produces the dictionary used by
  prompts and logs.

This is the platform boundary. A future XJTLU robot adapter should expose the same
four action names and build an equivalent navigation context from camera, depth,
odometry, and collision sensors. It must not expose privileged Habitat goal state
unless an experiment is explicitly labeled as guarded.

### `policies/`

This package is the stable import surface for:

- `MockVLMPolicy`, used for plumbing tests;
- `QwenVLMPolicy`, used as advisor or direct controller;
- `NaVIDAChunkPolicy`, used for local multi-frame action-chunk experiments;
- `OfficialNaVIDAHTTPPolicy`, used to send only instruction, simulator step, and
  lossless RGB PNG to a separately hosted Official NaVIDA process.

The implementations and prompt templates live in `policies/vlm_policy.py`,
`policies/navida_policy.py`, `policies/official_navida_http_policy.py`, and
`policies/prompts.py`.

### `control/`

`NavigationController.decide()` receives a `PolicyOutput` and one assembled
navigation state. It returns `ControlDecision`, which keeps three values separate:

- `vlm_action`: model output;
- `controller_action`: geometric/advisor conversion;
- `action`: final action after safety overrides.

The success-radius guard remains privileged diagnostic behavior. Pure-policy and
guarded results must be reported separately.

Policies that explicitly declare the `paper_pure` protocol bypass this controller.
Their valid atomic action is executed unchanged; invalid output terminates the
episode without a simulator step or fallback action.

### `runtime/`

- `scheduler.py` owns layered 5 Hz/0.5 Hz and joint 1 Hz timing behavior.
- `navigation_runner.py` owns the episode and step loops.
- The runner calls optional policy `start_episode()` and `close()` lifecycle hooks.
- `recorder.py` owns the stable trajectory schema. The executed-action column is
  `action`; policy protocol, server decision metadata, latency, and termination
  reasons are recorded alongside the existing fields.
- `artifacts.py` owns run directories, frame overlays, and browser-compatible
  H.264 MP4 generation through ffmpeg, with an mp4v fallback.

### Offline workflow packages

- `data/` owns dataset checks, HM3D smoke generation, Oracle collection, record
  schemas, coverage analysis, and NaVIDA sample construction.
- `training/` owns the single-step Qwen and mixed VLN/IDS NaVIDA QLoRA
  entrypoints.
- `evaluation/` owns offline action-chunk scoring and closed-loop PointNav policy
  comparison.
- `pipelines/` owns multi-stage orchestration and artifact-path tracking.
- `legacy/` archives standalone experiments that bypass the current framework.

Offline commands use module entrypoints, for example:

```bash
conda run -n habitat_vlm python -m habitat_vln.data.check_vln_data --help
conda run -n habitat_vlm python -m habitat_vln.training.train_navida_qlora --help
conda run -n habitat_vlm python -m habitat_vln.evaluation.evaluate_navida_outputs --help
conda run -n habitat_vlm python -m habitat_vln.pipelines.hm3d_navida_pipeline --help
```

## Training and evaluation flow

Training remains an explicit artifact-producing pipeline:

```text
Oracle collection
    -> coverage analysis
    -> VLN/IDS sample construction
    -> QLoRA training
    -> offline evaluation
    -> pure or guarded closed-loop evaluation
    -> report
```

`pipelines/hm3d_navida_pipeline.py` is the stage orchestrator. HM3D is used for
legal engineering and smoke validation; R2R/RxR benchmark claims still require
MP3D.

## Validation

The standard command remains:

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py ...
```

Framework tests use Python's standard `unittest`, so they do not require pytest:

```bash
conda run -n habitat_vlm python -m unittest discover -v \
  -s test -p 'test_habitat_vln_*.py'
```

`test_habitat_vln_end_to_end.py` runs the real CLI assembly, state builder, Mock
policy, controller, recorder, frame writer, and video writer against a deterministic
lightweight environment. It does not load a model or start a long simulation.

Fast validation order:

1. compile/import checks;
2. controller, state, scheduler, and recorder unit tests;
3. one Mock-policy HM3D episode;
4. only then load Qwen or start training.
