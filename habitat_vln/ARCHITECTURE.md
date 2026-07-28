# Indoor Semantic Navigation Architecture

项目的目标和输入边界见 [`PROJECT_DIRECTION.md`](PROJECT_DIRECTION.md)。本架构只服务于
单栋室内楼宇的低先验语义导航，不为 PointNav 目标跟随保留活动兼容层。

## Active data flow

```text
semantic indoor task adapter
        |
        |-- language goal + target-free building weak prior
        v
envs/habitat_state.py
        |-- RGB-D + collision + local depth + action history
        v
core/NavigationObservation
        v
policies/QwenVLMPolicy
        |-- action + location/evidence/topology/verification statement
        v
control/NavigationController
        |-- direct action validation + local forward-depth safety only
        v
runtime/navigation_runner.py
        |-- closed loop, trajectory.csv, frames, MP4
        v
semantic task evaluator
```

## Input boundary

`NavigationStateBuilder` may retain simulator pose for local debug logging, but
`as_policy_observation()` forwards only RGB-D, collision, local depth, action
history and an allowed `building_prior`. It never forwards or computes:

- goal position, distance or angle;
- success radius or shortest path;
- target-derived progress;
- target-based STOP, geometric steering or recovery.

`core/semantic_task.py` validates the optional building-prior JSON against a
small field whitelist. This prevents arbitrary route answers from entering the
policy prompt through an unreviewed mapping.

## Module responsibilities

- `core/`: platform-independent observation, policy contracts, experiment YAML
  loading and building-prior validation.
- `envs/`: Habitat sensor/action adapters and target-free state assembly.
- `policies/`: Mock/Qwen direct-action policies and semantic prompt formatting.
- `control/`: validates direct actions; its only optional override is local
  forward-depth avoidance based on current RGB-D.
- `runtime/`: schedules inference, executes actions, writes review artifacts and
  records generic task-adapter metrics without using them for decisions.
- `data/`, `training/`, `evaluation/`, `pipelines/`: historical PointNav-era
  material until replaced by semantic-building equivalents; they are not active
  dependencies of the runtime.
- `legacy/`: historical scripts and records only.

## Model output and current gap

The prompt requires the model to accompany its action with a location hypothesis,
confidence, observed evidence, topology hypothesis and next verification step.
The raw response is recorded now; the next core implementation is a strict
structured parser and a persistent semantic belief record. Until that exists,
we must not claim that a valid low-level action alone proves understanding.

## Validation

Use the `habitat_vlm` environment from the repository root:

```bash
conda run -n habitat_vlm python -m black habitat_vln test/test_habitat_vln_*.py
conda run -n habitat_vlm env PYTHONPATH=/home/sousuke/Desktop/habitat-lab \
  python test/test_habitat_vln_interfaces.py
conda run -n habitat_vlm env PYTHONPATH=/home/sousuke/Desktop/habitat-lab \
  python test/test_habitat_vln_control.py
```

Then run only the YAML-backed `semantic_indoor_mock` smoke test. Do not start a
large simulation, model download or training run until the semantic task adapter
and scenario contract are reviewed.
