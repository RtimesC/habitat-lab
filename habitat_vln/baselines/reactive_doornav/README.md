# Reactive RGB-D DoorNav B1

## Research question

B1 asks one deliberately local question:

> When a door is detectable in the current RGB-D observation, or can be found
> by a local rotation scan, can the robot safely approach the doorway, avoid
> local obstacles, and stop under observable conditions?

B1 does not answer building-scale semantic localization, topology inference,
cross-floor planning, long-horizon recovery, or the full low-prior indoor
navigation problem. A successful local approach must not be described as stable
building navigation.

## Input boundary

Allowed inputs are:

- a natural-language local objective;
- current RGB and depth;
- a short recent RGB or RGB-D history;
- observable collision feedback and recent action history;
- building weak priors accepted by the project's field whitelist.

Hidden target coordinates, target distance, target bearing, success radius,
shortest path, geodesic route, and Oracle waypoint are forbidden. These values,
along with any progress, automatic STOP, steering, or recovery derived from
hidden target state, must never enter the grounder, target estimator, state machine, or local executor.

An evaluator may read privileged simulator information offline for diagnostics.
That information cannot be returned to the policy, grounder, target estimator,
state machine, or controller.

## Proposed data flow

```text
instruction + RGB-D
        |
        v
DoorGrounder
        |
        v
DoorCandidate
        |
        v
RGB-D local target estimator
        |
        v
LocalSubgoal
        |
        v
shared LocalNavigationExecutor
        |
        v
observable arrival verification
```

`LocalSubgoal` uses the robot's local coordinate frame. `relative_x_m` is
forward distance. `relative_y_m` is lateral displacement, positive to the
robot's left and negative to its right. A normal doorway approach usually has a
positive `relative_x_m`; negative values remain valid for local recovery from
short observable history. The contract contains no global position, Habitat
episode target, or route information.

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
Reactive local target selector
  -> shared local executor

Future method:
Semantic belief + topology + active verification + recovery
  -> same shared local executor
```

B1 isolates the local execution question so the future full method can share a
target-free executor without inheriting a hidden-goal shortcut.

## Phase A status

This phase contains contracts, validation, documentation, a future configuration
specification, and fast leakage tests only. It does not implement a real door
detector, RGB-D target estimator, obstacle-avoidance executor, state machine, or
DoorNav benchmark. Mock objects and contract tests are not benchmark results,
and no stable navigation capability is claimed.

The runtime YAML keeps currently supported common CLI values under `arguments`.
The complete future settings live in [`baseline_spec.yaml`](baseline_spec.yaml),
which is the versioned B1 source of truth and is not consumed by the current
experiment loader. Its dedicated fields are:

- `max_search_steps`;
- `max_episode_steps`;
- `observable_stop_distance_m`;
- `grounding_confidence_threshold`;
- `target_lost_tolerance_steps`;
- `no_progress_tolerance_steps`;
- `depth_safety_threshold_m`.

The runtime file is therefore a small set of supported common defaults, not a
claim that complete DoorNav can run today.

The active action vocabulary also exists in the policy-prompt layer. This
baseline defines the same four strings locally so its platform-independent
contracts do not depend on model prompting. A later cross-project action
contract should move into `core` and replace both definitions; Phase A does not
refactor the active runtime.
