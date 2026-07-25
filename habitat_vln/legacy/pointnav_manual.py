"""Legacy keyboard-controlled PointNav example."""

import cv2

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


env = habitat.Env(
    config=habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
)

print("Environment creation successful")
observations = env.reset()

while not env.episode_over:
    goal = observations["pointgoal_with_gps_compass"]
    print(f"distance={float(goal[0]):.3f}, angle={float(goal[1]):.3f}")

    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    key = cv2.waitKey(0)

    if key == ord(FORWARD_KEY):
        action = HabitatSimActions.move_forward
    elif key == ord(LEFT_KEY):
        action = HabitatSimActions.turn_left
    elif key == ord(RIGHT_KEY):
        action = HabitatSimActions.turn_right
    elif key == ord(FINISH):
        action = HabitatSimActions.stop
    else:
        print("Use w/a/d/f")
        continue

    observations = env.step(action)

env.close()
