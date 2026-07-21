# Habitat VLN Architecture

本目录是项目自己的导航应用层，底层继续使用 Habitat-Lab 和
Habitat-Sim。当前框架的目标是让仿真环境、导航状态、模型策略、确定性控制、
运行调度和实验产物具有清楚边界，同时保留原有命令。

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
policies/                       Mock / Qwen advisor / NaVIDA controller
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

`habitat_vln_nav.py` is the compatibility CLI. It parses arguments, constructs the
policy, controller, and Habitat environment, then calls `run_navigation()`.

## Module contracts

### `core/`

- `NavigationObservation` is the policy input: RGB, optional depth, instruction,
  step number, and structured navigation context.
- `PolicyOutput` is the raw policy result: parsed action, original text, and a
  validity flag.
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
- `NaVIDAChunkPolicy`, used for multi-frame action-chunk experiments.

The original `vlm_policy.py` and `navida_policy.py` paths remain available for
existing scripts.

### `control/`

`NavigationController.decide()` receives a `PolicyOutput` and one assembled
navigation state. It returns `ControlDecision`, which keeps three values separate:

- `vlm_action`: model output;
- `controller_action`: geometric/advisor conversion;
- `action`: final action after safety overrides.

The success-radius guard remains privileged diagnostic behavior. Pure-policy and
guarded results must be reported separately.

### `runtime/`

- `scheduler.py` owns layered 5 Hz/0.5 Hz and joint 1 Hz timing behavior.
- `navigation_runner.py` owns the episode and step loops.
- `recorder.py` owns the stable trajectory schema. The executed-action column is
  `action`.
- `artifacts.py` owns run directories, frame overlays, and MP4 generation.

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

`hm3d_navida_pipeline.py` remains the stage orchestrator. HM3D is used for legal
engineering and smoke validation; R2R/RxR benchmark claims still require MP3D.

## Compatibility and validation

The standard command remains:

```bash
conda run -n habitat_vlm python habitat_vln/habitat_vln_nav.py ...
```

Framework tests use Python's standard `unittest`, so they do not require pytest:

```bash
conda run -n habitat_vlm python -m unittest discover -v \
  -s test -p 'test_habitat_vln_*.py'
```

Fast validation order:

1. compile/import checks;
2. controller, state, scheduler, and recorder unit tests;
3. one Mock-policy HM3D episode;
4. only then load Qwen or start training.
