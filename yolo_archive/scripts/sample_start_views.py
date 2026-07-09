import os
import csv
import cv2
import habitat
import numpy as np
from habitat.config import read_write
from ultralytics import YOLO

OUT_DIR = "outputs/start_views"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")

with read_write(config):
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 640
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 480
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = 70

env = habitat.Env(config=config)
sim = env.sim

log_path = f"{OUT_DIR}/start_views.csv"

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "position_x", "position_y", "position_z", "rotation_yaw", "detections", "image"])

    for idx in range(30):
        env.reset()

        # 随机找一个可导航点
        position = sim.sample_navigable_point()

        # 随机朝向
        yaw = np.random.uniform(-np.pi, np.pi)
        rotation = [0, np.sin(yaw / 2), 0, np.cos(yaw / 2)]

        agent_state = sim.get_agent_state()
        agent_state.position = position
        agent_state.rotation = rotation
        sim.set_agent_state(agent_state.position, agent_state.rotation)

        obs = sim.get_sensor_observations()
        rgb = obs["rgb"]
        bgr = rgb[:, :, [2, 1, 0]].copy()

        result = model(bgr, imgsz=960, conf=0.15, verbose=False)[0]
        annotated = result.plot()

        detections = []
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append(f"{name}:{conf:.2f}")

        image_path = f"{OUT_DIR}/view_{idx:03d}.jpg"
        cv2.imwrite(image_path, annotated)

        writer.writerow([
            idx,
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(yaw),
            ";".join(detections),
            image_path,
        ])

        print(f"{idx:03d}: pos={position}, yaw={yaw:.2f}, detections={detections}")

env.close()
print(f"saved to {OUT_DIR}")
