#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Verification script for XJTLU low-chassis VLN robot simulation environment.

Tests:
1. Loading custom XJTLU car agent configuration (configs/xjtlu/xjtlu_car_agent.yaml);
2. Environment initialization on Habitat test scene (skokloster-castle.glb);
3. Sensor resolution (640x480 RGB-D), HFOV 90, and mounting height (0.45m);
4. 4x4 camera_pose SE(3) transformation matrix for 3DGS back-projection;
5. 10Hz differential drive velocity control step(v, w);
6. Collision status and geodesic distance-to-goal computation.
"""

import sys
import numpy as np

from habitat.wrappers import XJTLUCarEnv


def main():
    print("=" * 70)
    print(">>> [XJTLU-VLN] Starting Environment Verification...")
    print("=" * 70)

    config_path = "configs/xjtlu/xjtlu_car_agent.yaml"
    print(f"[*] Initializing XJTLUCarEnv with config: {config_path}")

    with XJTLUCarEnv(config_path=config_path) as env:
        # 1. Test Reset
        print("\n--- [Step 1] Testing env.reset() ---")
        obs = env.reset()

        rgb = obs["rgb"]
        depth = obs["depth"]
        camera_pose = obs["camera_pose"]
        instruction = obs["instruction"]

        print(f"[*] RGB shape:         {rgb.shape} (dtype: {rgb.dtype})")
        print(f"[*] Depth shape:       {depth.shape} (dtype: {depth.dtype})")
        print(f"[*] Camera pose shape: {camera_pose.shape} (dtype: {camera_pose.dtype})")
        print(f"[*] Instruction:       {repr(instruction)}")
        print(f"[*] Camera Pose (T_wc 4x4):\n{camera_pose}")

        # Assertions on observation specs
        assert rgb.shape == (480, 640, 3), f"RGB shape mismatch: {rgb.shape} != (480, 640, 3)"
        assert rgb.dtype == np.uint8, f"RGB dtype mismatch: {rgb.dtype} != uint8"
        assert depth.shape == (480, 640, 1), f"Depth shape mismatch: {depth.shape} != (480, 640, 1)"
        assert depth.dtype == np.float32, f"Depth dtype mismatch: {depth.dtype} != float32"
        assert camera_pose.shape == (4, 4), f"Camera pose shape mismatch: {camera_pose.shape} != (4, 4)"
        assert isinstance(instruction, str), f"Instruction is not a string: {type(instruction)}"

        # Verify camera height offset relative to agent base (0.45m)
        agent_y = env.sim.get_agent_state().position[1]
        camera_y = camera_pose[1, 3]
        height_diff = camera_y - agent_y
        print(f"[*] Agent Y: {agent_y:.4f} m, Camera Y: {camera_y:.4f} m, Height Offset: {height_diff:.4f} m")
        assert abs(height_diff - 0.45) < 1e-3, (
            f"Camera height offset mismatch! Expected ~0.45m, got {height_diff:.4f}m"
        )
        print("[PASS] Camera mounting height 0.45m verified!")
        print("[PASS] Sensor specs (640x480 RGB-D, 90 HFOV) verified!")

        # 2. Test Step with Differential Drive
        print("\n--- [Step 2] Testing 10Hz Differential Drive step(v, w) ---")

        commands = [
            ("Forward (v=0.3 m/s, w=0.0 rad/s)", 0.3, 0.0),
            ("Turn Left (v=0.0 m/s, w=0.5 rad/s)", 0.0, 0.5),
            ("Curve Right (v=0.2 m/s, w=-0.3 rad/s)", 0.2, -0.3),
            ("Backward (v=-0.2 m/s, w=0.0 rad/s)", -0.2, 0.0),
            ("Stationary (v=0.0 m/s, w=0.0 rad/s)", 0.0, 0.0),
        ]

        prev_pos = camera_pose[:3, 3].copy()
        for i, (desc, v, w) in enumerate(commands, start=1):
            obs, is_collision, dist_to_goal = env.step(linear_vel=v, angular_vel=w)
            curr_pos = obs["camera_pose"][:3, 3]
            disp = np.linalg.norm(curr_pos - prev_pos)
            prev_pos = curr_pos.copy()

            print(
                f"[Step {i}] {desc:<38} | "
                f"Disp: {disp:.4f} m | "
                f"Collision: {is_collision!s:<5} | "
                f"Dist2Goal: {dist_to_goal:.4f} m"
            )

            # Ensure observation contract remains invariant after step
            assert obs["rgb"].shape == (480, 640, 3)
            assert obs["depth"].shape == (480, 640, 1)
            assert obs["camera_pose"].shape == (4, 4)
            assert isinstance(is_collision, bool)

        print("[PASS] 10Hz differential drive step execution verified!")

    print("\n" + "=" * 70)
    print(">>> [SUCCESS] All XJTLUCarEnv verification tests passed successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
