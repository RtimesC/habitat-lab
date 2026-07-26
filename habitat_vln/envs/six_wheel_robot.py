"""A lightweight visible six-wheel body for the standard Habitat agent.

The PointNav/VLN agent remains responsible for navigation and collision checks.
This module adds a render-only vehicle body that follows that agent, so existing
navigation policies can be tested with a camera mounted at a realistic height.
"""

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import habitat_sim
import magnum as mn
from habitat_sim.utils.common import quat_to_magnum


ROBOT_BODY_NONE = "none"
ROBOT_BODY_SIX_WHEEL = "six_wheel"
ROBOT_VIEW_UUID = "robot_view"


@dataclass(frozen=True)
class SixWheelRobotConfig:
    """Physical dimensions used for the render-only six-wheel robot body."""

    camera_height: float = 0.62
    camera_forward_offset: float = -0.24
    debug_view: bool = False
    debug_view_width: int = 640
    debug_view_height: int = 480
    debug_view_hfov: int = 90


@dataclass(frozen=True)
class _RobotPart:
    """One primitive geometry and its pose relative to the navigation agent."""

    name: str
    template_handle: str
    scale: Tuple[float, float, float]
    local_position: Tuple[float, float, float]
    local_rotation: mn.Quaternion


def configure_six_wheel_sensors(sensors: Any, config: SixWheelRobotConfig) -> None:
    """Mount the normal RGB-D sensors on the vehicle and add an optional rear view."""
    camera_position = [0.0, config.camera_height, config.camera_forward_offset]
    for sensor_name in ("rgb_sensor", "depth_sensor"):
        sensor = sensors.get(sensor_name)
        if sensor is not None:
            sensor.position = list(camera_position)
            sensor.orientation = [0.0, 0.0, 0.0]

    if config.debug_view:
        sensors["robot_view_sensor"] = {
            "type": "HabitatSimRGBSensor",
            "uuid": ROBOT_VIEW_UUID,
            "width": config.debug_view_width,
            "height": config.debug_view_height,
            "hfov": config.debug_view_hfov,
            # The agent faces -Z. This camera sits at +Z behind the car and
            # keeps the default -Z view direction towards it from above.
            "position": [0.0, 1.25, 1.60],
            "orientation": [-0.35, 0.0, 0.0],
            "sensor_subtype": "PINHOLE",
            "noise_model": "None",
            "noise_model_kwargs": {},
        }


class SixWheelRobotVisual:
    """Create and synchronize a render-only six-wheel vehicle body."""

    _CUBE = "cubeSolid"
    _CYLINDER = (
        "cylinderSolid_rings_1_segments_12_halfLen_1_"
        "useTexCoords_false_useTangents_false_capEnds_true"
    )

    def __init__(self, config: SixWheelRobotConfig) -> None:
        self.config = config
        self._object_ids: List[int] = []
        self._objects: List[Any] = []
        wheel_rotation = mn.Quaternion.rotation(
            mn.Deg(90.0), mn.Vector3.z_axis()
        )
        identity = mn.Quaternion()
        self._parts: Sequence[_RobotPart] = (
            _RobotPart(
                "chassis",
                self._CUBE,
                (0.54, 0.24, 0.78),
                (0.0, 0.27, 0.0),
                identity,
            ),
            _RobotPart(
                "equipment_box",
                self._CUBE,
                (0.38, 0.12, 0.42),
                (0.0, 0.45, 0.04),
                identity,
            ),
            _RobotPart(
                "front_bumper",
                self._CUBE,
                (0.58, 0.10, 0.08),
                (0.0, 0.22, -0.43),
                identity,
            ),
            _RobotPart(
                "camera_mast",
                self._CUBE,
                (0.06, 0.24, 0.06),
                (0.0, 0.52, config.camera_forward_offset),
                identity,
            ),
            _RobotPart(
                "camera_housing",
                self._CUBE,
                (0.20, 0.08, 0.10),
                (0.0, config.camera_height, config.camera_forward_offset),
                identity,
            ),
            *self._wheel_parts(wheel_rotation),
        )

    @property
    def object_ids(self) -> Tuple[int, ...]:
        """Return the Habitat-Sim object identifiers used by this vehicle."""
        return tuple(self._object_ids)

    @property
    def objects(self) -> Tuple[Any, ...]:
        """Return the managed primitives, primarily for visual-system checks."""
        return tuple(self._objects)

    @staticmethod
    def _wheel_parts(rotation: mn.Quaternion) -> Sequence[_RobotPart]:
        """Build the left/right front, middle, and rear wheel specifications."""
        parts = []
        for side_name, x_position in (("left", -0.33), ("right", 0.33)):
            for axle_name, z_position in (
                ("front", -0.27),
                ("middle", 0.0),
                ("rear", 0.27),
            ):
                parts.append(
                    _RobotPart(
                        f"wheel_{side_name}_{axle_name}",
                        SixWheelRobotVisual._CYLINDER,
                        (0.13, 0.03, 0.13),
                        (x_position, 0.13, z_position),
                        rotation,
                    )
                )
        return parts

    def sync(self, sim: Any) -> None:
        """Create the body if needed, then align every part with the agent pose."""
        if not self._objects_exist(sim):
            self._objects = self._create_objects(sim)
            self._object_ids = [obj.object_id for obj in self._objects]

        state = self._agent_state(sim)
        agent_rotation = quat_to_magnum(state.rotation)
        for obj, part in zip(self._objects, self._parts):
            if not obj.is_alive:
                # A scene change can invalidate the old objects between the
                # existence check and this update. The next reset will rebuild.
                self._objects = []
                self._object_ids = []
                return
            local_position = mn.Vector3(*part.local_position)
            obj.translation = state.position + agent_rotation.transform_vector(
                local_position
            )
            obj.rotation = agent_rotation * part.local_rotation

    def close(self, sim: Any) -> None:
        """Release Python references before the wrapped Habitat environment closes."""
        # Habitat-Sim owns the primitives and removes them when its environment
        # closes. Keeping the managed references until this point is required in
        # 0.3.3; otherwise Python may release a visual object prematurely.
        del sim
        self._objects = []
        self._object_ids = []

    def _objects_exist(self, sim: Any) -> bool:
        del sim
        return len(self._objects) == len(self._parts) and all(
            obj.is_alive for obj in self._objects
        )

    @staticmethod
    def _agent_state(sim: Any) -> Any:
        """Read an agent state from either Habitat-Lab or Habitat-Sim directly."""
        get_agent_state = getattr(sim, "get_agent_state", None)
        if callable(get_agent_state):
            return get_agent_state()
        return sim.get_agent(0).get_state()

    def _create_objects(self, sim: Any) -> List[Any]:
        template_manager = sim.get_object_template_manager()
        object_manager = sim.get_rigid_object_manager()
        object_ids = []
        for index, part in enumerate(self._parts):
            template = template_manager.get_template_by_handle(
                part.template_handle
            )
            template.scale = mn.Vector3(*part.scale)
            template_id = template_manager.register_template(
                template,
                f"habitat_vln_six_wheel_{part.name}_{index}",
            )
            obj = object_manager.add_object_by_template_id(template_id)
            # The visible body must not introduce a second, inconsistent
            # collision model. Navigation still uses Habitat's agent collision.
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.collidable = False
            object_ids.append(obj)
        return object_ids


class SixWheelRobotEnvironment:
    """Delegate a Habitat environment while synchronizing its vehicle visual."""

    def __init__(self, environment: Any, config: SixWheelRobotConfig) -> None:
        self._environment = environment
        self._visual = SixWheelRobotVisual(config)
        self._debug_view = config.debug_view

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset Habitat, place the vehicle body, and refresh its debug camera."""
        observations = self._environment.reset(*args, **kwargs)
        self._visual.sync(self._environment.sim)
        self._refresh_robot_view(observations)
        return observations

    def step(self, action: Any) -> Any:
        """Execute one navigation action, then update the visible vehicle pose."""
        observations = self._environment.step(action)
        self._visual.sync(self._environment.sim)
        self._refresh_robot_view(observations)
        return observations

    def close(self) -> None:
        """Release vehicle primitives and then close the wrapped environment."""
        try:
            self._visual.close(self._environment.sim)
        finally:
            self._environment.close()

    def _refresh_robot_view(self, observations: Any) -> None:
        if not self._debug_view:
            return
        rendered = self._environment.sim.get_sensor_observations().get(
            ROBOT_VIEW_UUID
        )
        if rendered is not None:
            observations[ROBOT_VIEW_UUID] = rendered[:, :, :3]

    def __getattr__(self, name: str) -> Any:
        """Expose the normal Habitat Env API without duplicating its surface."""
        return getattr(self._environment, name)
