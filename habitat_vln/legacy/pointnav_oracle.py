"""Legacy shortest-path Oracle PointNav example."""

import csv
import os

import cv2

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

OUT_DIR = "outputs/pointnav_oracle"
os.makedirs(OUT_DIR, exist_ok=True)


def save_rgb(obs, step, action_name, distance, angle):
    rgb = obs["rgb"][:, :, [2, 1, 0]].copy()
    cv2.putText(
        rgb,
        f"step={step} action={action_name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        rgb,
        f"distance={distance:.2f} angle={angle:.2f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    path = f"{OUT_DIR}/frame_{step:03d}.png"
    cv2.imwrite(path, rgb)
    return path


env = habitat.Env(
    config=habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
)

obs = env.reset()

follower = ShortestPathFollower(
    sim=env.sim,
    goal_radius=0.2,
    return_one_hot=False,
)

log_path = f"{OUT_DIR}/trajectory.csv"

with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "distance", "angle", "action", "image"])

    for step in range(200):
        distance = float(obs["pointgoal_with_gps_compass"][0])
        angle = float(obs["pointgoal_with_gps_compass"][1])

        goal_position = env.current_episode.goals[0].position
        action = follower.get_next_action(goal_position)

        if action is None:
            action = HabitatSimActions.stop

        action_name = str(action)
        if action == HabitatSimActions.stop:
            action_name = "stop"
        elif action == HabitatSimActions.move_forward:
            action_name = "move_forward"
        elif action == HabitatSimActions.turn_left:
            action_name = "turn_left"
        elif action == HabitatSimActions.turn_right:
            action_name = "turn_right"

        image_path = save_rgb(obs, step, action_name, distance, angle)
        writer.writerow([step, distance, angle, action_name, image_path])

        print(
            f"step={step:03d}, distance={distance:.3f}, angle={angle:.3f}, action={action_name}"
        )

        obs = env.step(action)

        if env.episode_over:
            print("episode over")
            break

env.close()
print(f"saved trajectory to {log_path}")
