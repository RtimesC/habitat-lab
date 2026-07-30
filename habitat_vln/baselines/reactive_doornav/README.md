# Reactive Visual DoorNav B1

## Research question

B1 asks one deliberately local question:

> When the target door can be detected in the current or recent RGB
> observations, or discovered by a local rotation scan, can the robot select it
> as a local visual target, approach it reliably, and stop based on observable
> evidence?

B1 does not answer building-scale semantic localization, topology inference,
cross-floor planning, long-horizon recovery, or the full low-prior indoor
navigation problem. A successful local approach must not be described as stable
building navigation.

## Input boundary

Allowed inputs are:

- a natural-language local objective;
- the current RGB observation;
- a short recent RGB history;
- observable collision feedback;
- recent action history;
- whitelist-validated building weak priors;
- optional platform-local safety signals when available.

RGB is B1's core visual input. B1 does not require an RGB-D camera. It does not
require Depth or LiDAR. Depth, LiDAR, proximity sensors, and similar local
obstacle sensing may later be connected through an optional safety adapter.
Such an adapter must remain target-free and cannot provide hidden target
geometry to the policy. Phase A does not select a real-robot depth sensor.

Hidden target coordinates, target distance, target bearing, success radius,
shortest path, geodesic route, and Oracle waypoint are forbidden. These values,
along with any progress, automatic STOP, steering, or recovery derived from
hidden target state, must never enter the grounder, visual tracker, state
machine, or reactive visual executor.

An evaluator may read privileged simulator information offline for diagnostics.
That information cannot be returned to the policy, grounder, visual tracker,
state machine, or controller.

## Proposed data flow

```text
instruction + current/recent RGB
        |
        v
DoorGrounder
        |
        v
DoorCandidate
        |
        v
VisualTargetTracker
        |
        v
VisualTargetTrack
        |
        v                 optional target-free platform safety adapter
ReactiveVisualExecutor <--------- collision / local obstacle signal
        |
        v
move_forward / turn_left / turn_right / stop
```

`VisualTargetTrack` is an image-space observation record, not a physical
distance estimate. Its normalized horizontal center indicates whether the
tracked doorway lies left or right of the image center and can support choosing
`turn_left` or `turn_right`. Its bounding-box area ratio is an observable
approach cue: growth may indicate that the doorway occupies more of the current
view. It is not absolute physical distance truth and must not be treated as one
across different cameras, scenes, or doorway shapes.

Current visibility, consecutive missing frames, and short-term track stability
can support later `VERIFY` logic. Stage B and Stage C must experimentally
calibrate any centering, area, and confirmation rules before they are treated as
working arrival behavior. The Phase A thresholds are initial specifications
only; they have not been validated.

Plain RGB bounding boxes do not provide metric robot-local position or absolute
distance. B1 therefore does not derive a metric waypoint or metric stop
threshold from `DoorCandidate`.

## Planned reactive state machine

```text
SEARCH
  -> TRACK
  -> APPROACH
  -> AVOID
  -> REACQUIRE
  -> VERIFY
  -> STOP / FAILED
```

These names are contracts for later stages. This scaffold does not implement
their transition logic.

## Relationship to the future #8 method

```text
B1:
local instruction
  -> RGB doorway grounding
  -> visual tracking
  -> reactive visual executor

Future #8:
semantic belief + topology + active verification + recovery
  -> selected local semantic target description
  -> same RGB grounding/tracking/reactive executor

Future waypoint-model integration is a separate adapter:
RGB navigation model
  -> local waypoint trajectory
  -> future waypoint executor
```

The future #8 method can reuse B1 by selecting a local semantic target
description for the same RGB grounding, tracking, and reactive execution path.
A model-produced robot-local waypoint adapter is outside B1 Phase A. B1 must not
take on a metric contract merely to anticipate a future waypoint model.

## Phase A status

This phase contains contracts, validation, documentation, a future configuration
specification, and fast leakage tests only. It does not implement a real door
detector, visual target tracker, reactive visual executor, state machine, or
DoorNav simulation. Mock objects and contract tests are not benchmark results,
and no real navigation or stable navigation capability is claimed.

The runtime YAML keeps currently supported common CLI values under `arguments`.
The complete future settings live in [`baseline_spec.yaml`](baseline_spec.yaml),
which is the versioned B1 source of truth and is not consumed by the current
experiment loader. Its dedicated fields are:

- `max_search_steps`;
- `max_episode_steps`;
- `arrival_area_ratio_threshold`;
- `arrival_center_tolerance_norm`;
- `arrival_confirm_frames`;
- `grounding_confidence_threshold`;
- `target_lost_tolerance_steps`;
- `no_progress_tolerance_steps`.

The runtime file is therefore a small set of supported common defaults, not a
claim that complete DoorNav can run today.

The active action vocabulary also exists in the policy-prompt layer. This
baseline defines the same four strings locally so its platform-independent
contracts do not depend on model prompting. A later cross-project action
contract should move into `core` and replace both definitions; Phase A does not
refactor the active runtime.
