"""Legacy Qwen experiment that drives Habitat-Sim without Habitat-Lab tasks."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import habitat_sim

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


VALID_ACTIONS = {
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
}

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_DIR / "outputs" / "qwen_habitat_sim" / "qwen_navigation.mp4"
)

TRAJECTORY_FIELDS = [
    "step",
    "action",
    "reason",
    "raw_model_output",
    "recent_actions",
    "pre_x",
    "pre_y",
    "pre_z",
    "pre_qx",
    "pre_qy",
    "pre_qz",
    "pre_qw",
    "pre_yaw_rad",
    "post_x",
    "post_y",
    "post_z",
    "post_qx",
    "post_qy",
    "post_qz",
    "post_qw",
    "post_yaw_rad",
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_distance_3d",
    "delta_distance_xz",
    "delta_yaw_rad",
    "forward_progress_ratio",
    "collision_estimate",
    "video_frame",
]


def create_simulator(scene_path: str, width: int, height: int):
    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = scene_path
    simulator_config.enable_physics = False

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_spec.resolution = [height, width]
    sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_spec.hfov = 90

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor_spec]

    agent_config.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=15.0),
        ),
    }

    configuration = habitat_sim.Configuration(
        simulator_config,
        [agent_config],
    )

    sim = habitat_sim.Simulator(configuration)

    # 将智能体放到导航网格上的随机可行走点
    if sim.pathfinder.is_loaded:
        start_position = sim.pathfinder.get_random_navigable_point()
        agent = sim.initialize_agent(0)

        state = agent.get_state()
        state.position = start_position
        agent.set_state(state)

        print("Start position:", start_position)
    else:
        print(
            "Warning: navmesh is not loaded; using the scene default position."
        )

    return sim


def load_model(model_name: str):
    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Qwen navigation requires torch and transformers. "
            "Install them in this environment before running the VLM policy."
        ) from exc

    print("Loading model:", model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    model.eval()
    print("Model loaded.")

    return model, processor


def rgb_to_pil(observation):
    rgb = observation["rgb"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return Image.fromarray(rgb.astype(np.uint8))


def quaternion_xyzw(rotation):
    return (
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
        float(rotation.w),
    )


def yaw_from_quaternion(rotation):
    qx, qy, qz, qw = quaternion_xyzw(rotation)
    return math.atan2(
        2.0 * (qw * qy + qx * qz),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def agent_pose(agent):
    state = agent.get_state()
    position = np.asarray(state.position, dtype=np.float64)
    qx, qy, qz, qw = quaternion_xyzw(state.rotation)
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
        "yaw_rad": yaw_from_quaternion(state.rotation),
    }


def angle_delta(after, before):
    return math.atan2(
        math.sin(after - before),
        math.cos(after - before),
    )


def extract_reason(response):
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        reason = payload.get("reason")
        if reason is not None:
            return str(reason).strip()

    match = re.search(r"\breason\b\s*[:=]\s*(.+)", response, flags=re.I)
    if match:
        return match.group(1).strip().strip("\"'")

    return ""


def trajectory_row(
    step,
    action,
    reason,
    raw_output,
    recent_actions,
    pre_pose,
    post_pose,
    collision_epsilon,
):
    delta_x = post_pose["x"] - pre_pose["x"]
    delta_y = post_pose["y"] - pre_pose["y"]
    delta_z = post_pose["z"] - pre_pose["z"]
    delta_distance_3d = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    delta_distance_xz = math.sqrt(delta_x**2 + delta_z**2)
    delta_yaw = angle_delta(post_pose["yaw_rad"], pre_pose["yaw_rad"])
    forward_progress_ratio = ""
    if action == "MOVE_FORWARD":
        forward_progress_ratio = delta_distance_xz / 0.25
    collision_estimate = (
        action == "MOVE_FORWARD" and delta_distance_xz <= collision_epsilon
    )

    return {
        "step": step,
        "action": action,
        "reason": reason,
        "raw_model_output": raw_output,
        "recent_actions": recent_actions,
        "pre_x": pre_pose["x"],
        "pre_y": pre_pose["y"],
        "pre_z": pre_pose["z"],
        "pre_qx": pre_pose["qx"],
        "pre_qy": pre_pose["qy"],
        "pre_qz": pre_pose["qz"],
        "pre_qw": pre_pose["qw"],
        "pre_yaw_rad": pre_pose["yaw_rad"],
        "post_x": post_pose["x"],
        "post_y": post_pose["y"],
        "post_z": post_pose["z"],
        "post_qx": post_pose["qx"],
        "post_qy": post_pose["qy"],
        "post_qz": post_pose["qz"],
        "post_qw": post_pose["qw"],
        "post_yaw_rad": post_pose["yaw_rad"],
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_z": delta_z,
        "delta_distance_3d": delta_distance_3d,
        "delta_distance_xz": delta_distance_xz,
        "delta_yaw_rad": delta_yaw,
        "forward_progress_ratio": forward_progress_ratio,
        "collision_estimate": collision_estimate,
        "video_frame": step,
    }


def decide_action(model, processor, image, goal, history, max_new_tokens):
    if process_vision_info is None:
        raise RuntimeError(
            "qwen-vl-utils is required to run Qwen vision inference. "
            "Install qwen-vl-utils in this environment."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to run Qwen vision inference."
        ) from exc

    recent_actions = ", ".join(history[-6:]) if history else "NONE"

    instruction = f"""
You control a mobile robot inside an indoor simulator.

Goal:
{goal}

Available actions:
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT

Recent actions:
{recent_actions}

Inspect the image and choose the next action.

Rules:
- Return exactly one JSON object with keys "action" and "reason".
- "action" must be exactly one available action.
- "reason" must be a short visual/navigation reason.
- Prefer MOVE_FORWARD when the path ahead is open.
- Turn when the path ahead is blocked or a turn clearly helps the goal.
- Avoid repeatedly alternating TURN_LEFT and TURN_RIGHT.
- STOP is unavailable during this exploration run.
- For this exploration task, continue moving unless movement is impossible.

Example:
{{"action": "MOVE_FORWARD", "reason": "The hallway ahead appears open."}}
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # device_map="auto" 时不能简单假定 model.device，
    # 输入放到首个模型参数所在的设备。
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    response_for_action = response.upper()

    match = re.search(
        r"\b(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)\b",
        response_for_action,
    )

    if match:
        return match.group(1), response, extract_reason(response)

    print("Invalid model output, fallback to MOVE_FORWARD:", repr(response))
    return "MOVE_FORWARD", response, "fallback after invalid model output"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        required=True,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--goal",
        default="Explore the indoor environment and move through open passages.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Maximum generated tokens for the action JSON.",
    )
    parser.add_argument(
        "--trajectory-output",
        default=None,
        help="CSV path for per-step trajectory logs. Defaults next to --output.",
    )
    parser.add_argument(
        "--collision-epsilon",
        type=float,
        default=1e-3,
        help=(
            "MOVE_FORWARD with less horizontal displacement than this is marked "
            "as a collision/blockage estimate."
        ),
    )

    args = parser.parse_args()

    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if args.trajectory_output is None:
        trajectory_path = output_path.with_name(
            f"{output_path.stem}_trajectory.csv"
        )
    else:
        trajectory_path = Path(args.trajectory_output).expanduser().resolve()

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    sim = create_simulator(
        str(scene_path),
        args.width,
        args.height,
    )
    agent = sim.get_agent(0)

    model, processor = load_model(args.model)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (args.width, args.height),
    )

    history = []

    try:
        observation = sim.get_sensor_observations()

        with open(trajectory_path, "w", newline="") as trajectory_file:
            trajectory_writer = csv.DictWriter(
                trajectory_file,
                fieldnames=TRAJECTORY_FIELDS,
            )
            trajectory_writer.writeheader()

            for step in range(args.max_steps):
                image = rgb_to_pil(observation)
                recent_actions = ", ".join(history[-6:]) if history else "NONE"

                action, raw_output, reason = decide_action(
                    model,
                    processor,
                    image,
                    args.goal,
                    history,
                    args.max_new_tokens,
                )

                # 防止模型刚开始就过早停止
                if action == "STOP" and step < 8:
                    print("Early STOP rejected; fallback to MOVE_FORWARD")
                    action = "MOVE_FORWARD"
                    reason = "early STOP rejected; fallback to MOVE_FORWARD"

                history.append(action)

                frame = cv2.cvtColor(
                    np.asarray(image),
                    cv2.COLOR_RGB2BGR,
                )

                cv2.putText(
                    frame,
                    f"Step {step}: {action}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                writer.write(frame)

                pre_pose = agent_pose(agent)

                if action == "STOP":
                    post_pose = pre_pose
                    print("Navigation stopped by model.")
                    trajectory_writer.writerow(
                        trajectory_row(
                            step,
                            action,
                            reason,
                            raw_output,
                            recent_actions,
                            pre_pose,
                            post_pose,
                            args.collision_epsilon,
                        )
                    )
                    break

                observation = sim.step(VALID_ACTIONS[action])
                post_pose = agent_pose(agent)
                row = trajectory_row(
                    step,
                    action,
                    reason,
                    raw_output,
                    recent_actions,
                    pre_pose,
                    post_pose,
                    args.collision_epsilon,
                )
                trajectory_writer.writerow(row)

                print(
                    f"[{step:03d}] "
                    f"action={action:<12} "
                    f"move_xz={row['delta_distance_xz']:.4f} "
                    f"dyaw={row['delta_yaw_rad']:.3f} "
                    f"collision_est={row['collision_estimate']} "
                    f"reason={reason!r} "
                    f"raw={raw_output!r}"
                )

    finally:
        writer.release()
        sim.close()

    print("\nFinished.")
    print("Actions:", history)
    print("Video:", output_path)
    print("Trajectory:", trajectory_path)


if __name__ == "__main__":
    main()
