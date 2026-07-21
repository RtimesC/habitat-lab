"""Legacy rule-based PointNav smoke example."""

import os
import csv
import cv2
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

OUT_DIR = "outputs/pointnav_auto"
os.makedirs(OUT_DIR, exist_ok=True)

def save_rgb(obs, step):
    rgb_bgr = obs["rgb"][:, :, [2, 1, 0]]
    path = f"{OUT_DIR}/frame_{step:03d}.png"
    cv2.imwrite(path, rgb_bgr)
    return path

env = habitat.Env(
    config=habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
)

obs = env.reset()
log_path = f"{OUT_DIR}/trajectory.csv"

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "distance", "angle", "action", "image"])

    for step in range(100):
        distance = float(obs["pointgoal_with_gps_compass"][0])
        angle = float(obs["pointgoal_with_gps_compass"][1])
        image_path = save_rgb(obs, step)

        if distance < 0.3:
            action = HabitatSimActions.stop
            action_name = "stop"
        elif angle > 0.2:
            action = HabitatSimActions.turn_left
            action_name = "turn_left"
        elif angle < -0.2:
            action = HabitatSimActions.turn_right
            action_name = "turn_right"
        else:
            action = HabitatSimActions.move_forward
            action_name = "move_forward"

        writer.writerow([step, distance, angle, action_name, image_path])
        print(f"step={step:03d}, distance={distance:.3f}, angle={angle:.3f}, action={action_name}")

        obs = env.step(action)

        if env.episode_over:
            print("episode over")
            break

env.close()
print(f"saved trajectory to {log_path}")
