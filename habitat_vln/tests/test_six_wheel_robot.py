"""Fast checks for the render-only six-wheel vehicle embodiment."""

import habitat_sim
import numpy as np

from habitat_vln.envs.six_wheel_robot import (
    SixWheelRobotConfig,
    SixWheelRobotVisual,
)


TEST_SCENE = "data/versioned_data/habitat_test_scenes/apartment_1.glb"


def _simulator():
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = TEST_SCENE
    return habitat_sim.Simulator(
        habitat_sim.Configuration(config, [habitat_sim.agent.AgentConfiguration()])
    )


def test_six_wheel_body_is_render_only_and_tracks_agent_pose():
    """The vehicle has six wheels and follows the default navigation agent."""
    sim = _simulator()
    try:
        visual = SixWheelRobotVisual(SixWheelRobotConfig())
        visual.sync(sim)

        assert len(visual.object_ids) == 11
        chassis = visual.objects[0]
        assert not chassis.collidable
        assert chassis.motion_type == habitat_sim.physics.MotionType.KINEMATIC
        initial_position = np.array(chassis.translation)

        state = sim.get_agent(0).get_state()
        state.position = state.position + np.array([0.5, 0.0, 0.0])
        sim.get_agent(0).set_state(state, reset_sensors=True)
        visual.sync(sim)

        moved_position = np.array(chassis.translation)
        assert np.allclose(moved_position - initial_position, [0.5, 0.0, 0.0])
    finally:
        sim.close()
