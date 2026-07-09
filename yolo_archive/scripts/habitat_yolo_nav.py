import argparse
import csv
import glob
import os
import time

import cv2
import habitat
import numpy as np
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from ultralytics import YOLO


OUT_DIR = "outputs/habitat_yolo_nav_fixed_start"

# Replace these four values with the row from:
# grep '^4,' outputs/start_views/start_views.csv
START_POSITION = [-2.7726662158966064, 0.04812638461589813, 7.372976779937744]
START_YAW = -3.111943205703981

TARGET_CLASSES = {"dining table"}
CONF_THRES = 0.15
IMG_SIZE = 960
MAX_STEPS = 120

STOP_AREA_THRESHOLD = 0.12
TURN_CONFIRM_FRAMES = 3
TURN_COOLDOWN_STEPS = 2
MAX_LOST_FRAMES = 5
TARGET_SWITCH_DISTANCE_RATIO = 0.25

ACTION_MAP = {
    "turn_left": HabitatSimActions.turn_left,
    "turn_right": HabitatSimActions.turn_right,
    "move_forward": HabitatSimActions.move_forward,
    "stop": HabitatSimActions.stop,
}

EXPERIMENTS = {
    "A": {
        "name": "original_rules",
        "use_state_machine": False,
        "use_center_deadzone": False,
        "use_turn_stability": False,
        "use_target_lock": False,
        "use_lost_tolerance": False,
    },
    "B": {
        "name": "state_deadzone_cooldown",
        "use_state_machine": True,
        "use_center_deadzone": True,
        "use_turn_stability": True,
        "use_target_lock": False,
        "use_lost_tolerance": False,
    },
    "C": {
        "name": "state_deadzone_cooldown_lock",
        "use_state_machine": True,
        "use_center_deadzone": True,
        "use_turn_stability": True,
        "use_target_lock": True,
        "use_lost_tolerance": True,
    },
}

TRAJECTORY_FIELDS = [
    "step",
    "state",
    "action",
    "target_detected",
    "target_class",
    "confidence",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_area_ratio",
    "lost_frames",
    "collision",
    "agent_x",
    "agent_y",
    "agent_z",
    "target_switch_count",
    "left_right_switch_count",
    "target_locked",
    "candidate_count",
    "lock_score",
    "position_similarity",
    "left_count",
    "right_count",
    "turn_cooldown",
    "last_turn",
    "image",
]

SUMMARY_FIELDS = [
    "experiment",
    "run",
    "success_stop",
    "total_steps",
    "collision_count",
    "left_right_switch_count",
    "target_switch_count",
    "target_lost_frames",
    "final_bbox_area_ratio",
    "runtime_sec",
    "trajectory_csv",
]


def build_env():
    config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

    with read_write(config):
        rgb_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        rgb_sensor.width = 640
        rgb_sensor.height = 480
        rgb_sensor.hfov = 70

    return habitat.Env(config=config)


def set_start_state(env):
    rotation = [0, np.sin(START_YAW / 2), 0, np.cos(START_YAW / 2)]
    env.sim.set_agent_state(START_POSITION, rotation)
    return env.sim.get_sensor_observations()


def reset_output_dir(path):
    os.makedirs(path, exist_ok=True)
    for old_frame in glob.glob(os.path.join(path, "frame_*.jpg")):
        os.remove(old_frame)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_distance(box1, box2):
    cx1, cy1 = box_center(box1)
    cx2, cy2 = box_center(box2)
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def find_target_candidates(result, image_width, image_height):
    names = result.names
    candidates = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = names[cls_id]
        conf = float(box.conf[0])

        if name not in TARGET_CLASSES or conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / (image_width * image_height)
        candidate_box = [x1, y1, x2, y2]
        cx, cy = box_center(candidate_box)

        candidates.append({
            "name": name,
            "conf": conf,
            "cx": cx,
            "cy": cy,
            "area_ratio": area_ratio,
            "box": candidate_box,
            "lock_score": "",
            "position_similarity": "",
            "visible": True,
        })

    return candidates


def score_candidate(candidate, locked_box, image_width, image_height):
    area_score = min(candidate["area_ratio"] / STOP_AREA_THRESHOLD, 1.0)

    if locked_box is None:
        position_similarity = 0.0
    else:
        image_diagonal = (image_width ** 2 + image_height ** 2) ** 0.5
        distance = box_distance(candidate["box"], locked_box)
        position_similarity = max(0.0, 1.0 - distance / image_diagonal)

    score = (
        0.5 * candidate["conf"]
        + 0.3 * area_score
        + 0.2 * position_similarity
    )

    return score, position_similarity


def choose_target(
    candidates,
    locked_target,
    lost_frames,
    image_width,
    image_height,
    experiment_config,
):
    if not experiment_config["use_target_lock"]:
        if candidates:
            target = max(candidates, key=lambda candidate: candidate["conf"])
            target["visible"] = True
            return target, None, 0
        return None, None, 0

    locked_box = locked_target["box"] if locked_target is not None else None

    if candidates:
        for candidate in candidates:
            score, position_similarity = score_candidate(
                candidate,
                locked_box,
                image_width,
                image_height,
            )
            candidate["lock_score"] = score
            candidate["position_similarity"] = position_similarity

        target = max(candidates, key=lambda candidate: candidate["lock_score"])
        target["visible"] = True
        return target, target.copy(), 0

    if locked_target is None or not experiment_config["use_lost_tolerance"]:
        return None, None, 0

    lost_frames += 1
    if lost_frames >= MAX_LOST_FRAMES:
        return None, None, lost_frames

    target = locked_target.copy()
    target["visible"] = False
    target["lock_score"] = ""
    target["position_similarity"] = ""
    return target, locked_target, lost_frames


def update_target_switch_count(target, previous_target_box, image_width, image_height, count):
    if target is None or not target.get("visible", False):
        return previous_target_box, count

    if previous_target_box is not None:
        image_diagonal = (image_width ** 2 + image_height ** 2) ** 0.5
        distance = box_distance(target["box"], previous_target_box)
        if distance > TARGET_SWITCH_DISTANCE_RATIO * image_diagonal:
            count += 1

    return target["box"], count


def update_turn_switch_count(action_name, previous_turn_action, count):
    if action_name not in {"turn_left", "turn_right"}:
        return previous_turn_action, count

    if previous_turn_action is not None and previous_turn_action != action_name:
        count += 1

    return action_name, count


def deadzone_thresholds(image_width, experiment_config):
    if experiment_config["use_center_deadzone"]:
        return 0.40 * image_width, 0.60 * image_width
    center = 0.50 * image_width
    return center, center


def confirmed_turn(direction, nav, experiment_config):
    if not experiment_config["use_turn_stability"]:
        nav["last_turn"] = direction
        nav["turn_cooldown"] = TURN_COOLDOWN_STEPS
        return f"turn_{direction}"

    count_key = f"{direction}_count"
    opposite = "right" if direction == "left" else "left"

    if nav[count_key] < TURN_CONFIRM_FRAMES:
        return "stop"
    if nav["last_turn"] == opposite and nav["turn_cooldown"] > 0:
        return "stop"

    nav[count_key] = 0
    nav["last_turn"] = direction
    nav["turn_cooldown"] = TURN_COOLDOWN_STEPS
    return f"turn_{direction}"


def update_turn_counts(target, target_visible, image_width, nav, experiment_config):
    if nav["turn_cooldown"] > 0:
        nav["turn_cooldown"] -= 1

    if not experiment_config["use_turn_stability"]:
        nav["left_count"] = 0
        nav["right_count"] = 0
        return

    if not target_visible:
        nav["left_count"] = 0
        nav["right_count"] = 0
        return

    left_threshold, right_threshold = deadzone_thresholds(image_width, experiment_config)

    if target["cx"] < left_threshold:
        nav["left_count"] += 1
        nav["right_count"] = 0
    elif target["cx"] > right_threshold:
        nav["right_count"] += 1
        nav["left_count"] = 0
    else:
        nav["left_count"] = 0
        nav["right_count"] = 0


def choose_original_action(target, target_visible, image_width):
    if not target_visible:
        return "SEARCH", "turn_left"
    if target["area_ratio"] >= STOP_AREA_THRESHOLD:
        return "STOP", "stop"
    center_x = 0.50 * image_width
    if target["cx"] < center_x:
        return "TRACK", "turn_left"
    if target["cx"] > center_x:
        return "TRACK", "turn_right"
    return "TRACK", "move_forward"


def choose_state_action(target, target_visible, image_width, nav, experiment_config):
    target_locked = target is not None
    left_threshold, right_threshold = deadzone_thresholds(image_width, experiment_config)
    state = nav["state"]

    if state == "SEARCH":
        if target_visible:
            nav["state"] = "ALIGN"
            return "stop"
        return "turn_left"

    if state == "ALIGN":
        if not target_locked:
            nav["state"] = "SEARCH"
            return "turn_left"
        if not target_visible:
            return "stop"
        if target["cx"] < left_threshold:
            return confirmed_turn("left", nav, experiment_config)
        if target["cx"] > right_threshold:
            return confirmed_turn("right", nav, experiment_config)

        nav["state"] = "APPROACH"
        return "move_forward"

    if state == "APPROACH":
        if not target_locked:
            nav["state"] = "SEARCH"
            return "turn_left"
        if not target_visible:
            return "stop"
        if target["area_ratio"] >= STOP_AREA_THRESHOLD:
            nav["state"] = "STOP"
            return "stop"
        if target["cx"] < left_threshold:
            nav["state"] = "ALIGN"
            return confirmed_turn("left", nav, experiment_config)
        if target["cx"] > right_threshold:
            nav["state"] = "ALIGN"
            return confirmed_turn("right", nav, experiment_config)

        return "move_forward"

    if state == "STOP":
        return "stop"

    raise ValueError(f"Unknown navigation state: {state}")


def choose_action(target, target_visible, image_width, nav, experiment_config):
    update_turn_counts(target, target_visible, image_width, nav, experiment_config)

    if not experiment_config["use_state_machine"]:
        state, action_name = choose_original_action(target, target_visible, image_width)
        nav["state"] = state
    else:
        action_name = choose_state_action(
            target,
            target_visible,
            image_width,
            nav,
            experiment_config,
        )

    if action_name == "stop" and nav["state"] != "STOP":
        action = None
    else:
        action = ACTION_MAP[action_name]

    return action, action_name


def draw_locked_target(image, target):
    if target is None:
        return

    x1, y1, x2, y2 = [int(value) for value in target["box"]]
    visible = target.get("visible", False)
    color = (0, 255, 0) if visible else (0, 165, 255)
    label = "locked" if visible else "locked lost"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_status(image, step, nav, action_name, target, lost_frames):
    cv2.putText(
        image,
        f"step={step} state={nav['state']} action={action_name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if target is None:
        text = "target=None"
    else:
        lock_status = "visible" if target.get("visible", False) else "lost"
        text = (
            f"target={target['name']} "
            f"conf={target['conf']:.2f} "
            f"area={target['area_ratio']:.3f} "
            f"lock={lock_status} "
            f"lost={lost_frames}"
        )

    cv2.putText(
        image,
        text,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def empty_target_values():
    return {
        "target_class": "",
        "confidence": "",
        "bbox_center_x": "",
        "bbox_center_y": "",
        "bbox_area_ratio": "",
        "lock_score": "",
        "position_similarity": "",
    }


def target_values(target):
    if target is None:
        return empty_target_values()

    return {
        "target_class": target["name"],
        "confidence": target["conf"],
        "bbox_center_x": target["cx"],
        "bbox_center_y": target["cy"],
        "bbox_area_ratio": target["area_ratio"],
        "lock_score": target["lock_score"],
        "position_similarity": target["position_similarity"],
    }


def run_navigation(env, model, experiment_key, run_idx, save_frames=True):
    experiment_config = EXPERIMENTS[experiment_key]
    run_dir = os.path.join(
        OUT_DIR,
        f"experiment_{experiment_key}_{experiment_config['name']}",
        f"run_{run_idx:02d}",
    )
    reset_output_dir(run_dir)
    log_path = os.path.join(run_dir, "trajectory.csv")

    obs = env.reset()
    obs = set_start_state(env)
    start_time = time.perf_counter()

    nav = {
        "state": "SEARCH",
        "left_count": 0,
        "right_count": 0,
        "turn_cooldown": 0,
        "last_turn": None,
    }
    locked_target = None
    lost_frames = 0
    previous_target_box = None
    previous_turn_action = None
    target_switch_count = 0
    left_right_switch_count = 0
    collision_count = 0
    target_lost_frames = 0
    final_bbox_area_ratio = ""
    success_stop = False
    total_steps = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()

        for step in range(MAX_STEPS):
            rgb = obs["rgb"]
            bgr = rgb[:, :, [2, 1, 0]].copy()
            height, width = bgr.shape[:2]

            result = model(bgr, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]
            candidates = find_target_candidates(result, width, height)
            target, locked_target, lost_frames = choose_target(
                candidates,
                locked_target,
                lost_frames,
                width,
                height,
                experiment_config,
            )
            target_visible = target is not None and target.get("visible", False)
            target_locked = target is not None

            previous_target_box, target_switch_count = update_target_switch_count(
                target,
                previous_target_box,
                width,
                height,
                target_switch_count,
            )

            action, action_name = choose_action(
                target,
                target_visible,
                width,
                nav,
                experiment_config,
            )
            previous_turn_action, left_right_switch_count = update_turn_switch_count(
                action_name,
                previous_turn_action,
                left_right_switch_count,
            )

            image_path = ""
            if save_frames:
                annotated = result.plot()
                draw_locked_target(annotated, target)
                draw_status(annotated, step, nav, action_name, target, lost_frames)
                image_path = os.path.join(run_dir, f"frame_{step:03d}.jpg")
                cv2.imwrite(image_path, annotated)

            collision = False
            if action is not None:
                obs = env.step(action)
                collision = bool(env.sim.previous_step_collided)
            if collision:
                collision_count += 1

            target_lost_frames += int(target_locked and not target_visible)
            target_data = target_values(target)
            if target_data["bbox_area_ratio"] != "":
                final_bbox_area_ratio = target_data["bbox_area_ratio"]

            agent_position = env.sim.get_agent_state().position
            row = {
                "step": step,
                "state": nav["state"],
                "action": action_name,
                "target_detected": target_visible,
                "target_locked": target_locked,
                "lost_frames": lost_frames,
                "collision": collision,
                "agent_x": float(agent_position[0]),
                "agent_y": float(agent_position[1]),
                "agent_z": float(agent_position[2]),
                "target_switch_count": target_switch_count,
                "left_right_switch_count": left_right_switch_count,
                "candidate_count": len(candidates),
                "left_count": nav["left_count"],
                "right_count": nav["right_count"],
                "turn_cooldown": nav["turn_cooldown"],
                "last_turn": nav["last_turn"] or "",
                "image": image_path,
            }
            row.update(target_data)
            writer.writerow(row)

            total_steps = step + 1
            print(
                f"exp={experiment_key} run={run_idx:02d} "
                f"step={step:03d} state={nav['state']} action={action_name} "
                f"detected={target_visible} collision={collision} "
                f"lost={lost_frames} switches={left_right_switch_count}"
            )

            if nav["state"] == "STOP":
                success_stop = True

            if env.episode_over:
                print(f"exp={experiment_key} run={run_idx:02d} episode over")
                break

    runtime_sec = time.perf_counter() - start_time
    return {
        "experiment": experiment_key,
        "run": run_idx,
        "success_stop": success_stop,
        "total_steps": total_steps,
        "collision_count": collision_count,
        "left_right_switch_count": left_right_switch_count,
        "target_switch_count": target_switch_count,
        "target_lost_frames": target_lost_frames,
        "final_bbox_area_ratio": final_bbox_area_ratio,
        "runtime_sec": runtime_sec,
        "trajectory_csv": log_path,
    }


def write_summary(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "experiment_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def write_aggregate(rows):
    aggregate_path = os.path.join(OUT_DIR, "experiment_aggregate.csv")
    by_experiment = {}

    for row in rows:
        by_experiment.setdefault(row["experiment"], []).append(row)

    fields = [
        "experiment",
        "runs",
        "success_rate",
        "avg_total_steps",
        "avg_collision_count",
        "avg_left_right_switch_count",
        "avg_target_switch_count",
        "avg_target_lost_frames",
        "avg_final_bbox_area_ratio",
        "avg_runtime_sec",
    ]

    with open(aggregate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for experiment, experiment_rows in sorted(by_experiment.items()):
            final_areas = [
                float(row["final_bbox_area_ratio"])
                for row in experiment_rows
                if row["final_bbox_area_ratio"] != ""
            ]
            writer.writerow({
                "experiment": experiment,
                "runs": len(experiment_rows),
                "success_rate": sum(
                    bool(row["success_stop"]) for row in experiment_rows
                ) / len(experiment_rows),
                "avg_total_steps": np.mean([
                    row["total_steps"] for row in experiment_rows
                ]),
                "avg_collision_count": np.mean([
                    row["collision_count"] for row in experiment_rows
                ]),
                "avg_left_right_switch_count": np.mean([
                    row["left_right_switch_count"] for row in experiment_rows
                ]),
                "avg_target_switch_count": np.mean([
                    row["target_switch_count"] for row in experiment_rows
                ]),
                "avg_target_lost_frames": np.mean([
                    row["target_lost_frames"] for row in experiment_rows
                ]),
                "avg_final_bbox_area_ratio": np.mean(final_areas)
                if final_areas else "",
                "avg_runtime_sec": np.mean([
                    row["runtime_sec"] for row in experiment_rows
                ]),
            })

    return aggregate_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "C", "all"],
        default="C",
        help="A=original, B=state/deadzone/cooldown, C=B+target lock",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum navigation steps per run.",
    )
    parser.add_argument(
        "--stop-area-threshold",
        type=float,
        default=STOP_AREA_THRESHOLD,
        help="Stop when bbox_area_ratio reaches this value.",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Skip frame jpg output and only write CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUT_DIR,
        help="Directory for trajectory CSVs, summaries, frames, and videos.",
    )
    return parser.parse_args()


def main():
    global OUT_DIR, MAX_STEPS, STOP_AREA_THRESHOLD

    args = parse_args()
    OUT_DIR = args.output_dir
    MAX_STEPS = args.max_steps
    STOP_AREA_THRESHOLD = args.stop_area_threshold
    os.makedirs(OUT_DIR, exist_ok=True)

    experiment_keys = ["A", "B", "C"] if args.experiment == "all" else [args.experiment]
    model = YOLO("yolov8n.pt")
    env = build_env()
    summary_rows = []

    try:
        for experiment_key in experiment_keys:
            for run_idx in range(args.runs):
                summary_rows.append(
                    run_navigation(
                        env,
                        model,
                        experiment_key,
                        run_idx,
                        save_frames=not args.no_save_frames,
                    )
                )
    finally:
        env.close()

    summary_path = write_summary(summary_rows)
    aggregate_path = write_aggregate(summary_rows)
    print(f"saved summary to {summary_path}")
    print(f"saved aggregate to {aggregate_path}")


if __name__ == "__main__":
    main()
