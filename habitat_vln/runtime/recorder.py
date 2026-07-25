"""CSV trajectory recording kept separate from the navigation control loop."""

import csv

TRAJECTORY_FIELDS = [
    "episode_index",
    "episode_id",
    "scene_id",
    "instruction",
    "step",
    "frequency_mode",
    "target_vision_hz",
    "target_inference_hz",
    "logical_time_sec",
    "wall_time_sec",
    "inference_ran",
    "inference_completed",
    "inference_duration_sec",
    "decision_age_steps",
    "decision_age_sec",
    "action",
    "valid_action",
    "raw_vlm_output",
    "vlm_action",
    "controller_action",
    "previous_action",
    "goal_distance_m",
    "goal_angle_deg",
    "distance_change_m",
    "collided",
    "depth_left_m",
    "depth_center_m",
    "depth_right_m",
    "previous_action_count",
    "previous_collision",
    "collision",
    "distance_delta",
    "no_progress_steps",
    "agent_x",
    "agent_y",
    "agent_z",
    "agent_rotation_x",
    "agent_rotation_y",
    "agent_rotation_z",
    "agent_rotation_w",
    "depth_min",
    "depth_mean",
    "distance_to_goal",
    "success",
    "spl",
    "image",
]


class TrajectoryRecorder:
    """Write trajectory rows with one stable, centrally defined CSV schema."""

    def __init__(self, path, fieldnames=None):
        self.path = path
        self.fieldnames = list(fieldnames or TRAJECTORY_FIELDS)
        self._handle = None
        self._writer = None

    def __enter__(self):
        # This class is itself the context manager that owns the file handle.
        self._handle = open(self.path, "w", newline="")  # noqa: SIM115
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()
        return self

    def write(self, row):
        if self._writer is None:
            raise RuntimeError(
                "TrajectoryRecorder must be used as a context manager"
            )
        self._writer.writerow(row)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None
