"""Reusable controller for the final Unitree G1 overhand throw.

This controller preserves the exact waypoints and balance parameters from
pitch_overhand_V1.py, but separates the controller logic from the MuJoCo
viewer and simulation loop.

It can be reused from:
- a visual demo script;
- a headless baseline evaluator;
- a future Gymnasium/PPO environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import mujoco
import numpy as np


WAYPOINTS: tuple[tuple[float, dict[str, float]], ...] = (
    (0.00, {}),

    # 1. Preparation
    (
        0.70,
        {
            "right_shoulder_pitch_joint": 0.30,
            "right_shoulder_roll_joint": -0.40,
            "right_elbow_joint": 0.50,
            "left_shoulder_pitch_joint": 0.30,
            "left_shoulder_roll_joint": 0.40,
            "left_elbow_joint": 0.50,
        },
    ),

    # 2. Raise the throwing arm
    (
        1.50,
        {
            "right_shoulder_pitch_joint": -1.80,
            "right_shoulder_roll_joint": -0.55,
            "right_elbow_joint": -0.30,
            "waist_yaw_joint": -0.15,
            "left_shoulder_pitch_joint": -0.30,
            "left_shoulder_roll_joint": 0.30,
            "left_elbow_joint": 0.60,
        },
    ),

    # 3. Cock the arm behind the head
    (
        2.10,
        {
            "right_shoulder_pitch_joint": -2.90,
            "right_shoulder_roll_joint": -0.55,
            "right_elbow_joint": -0.95,
            "waist_yaw_joint": -0.30,
            "waist_pitch_joint": -0.38,
            "left_shoulder_pitch_joint": -0.80,
            "left_shoulder_roll_joint": 0.25,
            "left_elbow_joint": 0.40,
        },
    ),

    # 4. Whip phase
    (
        2.35,
        {
            "right_shoulder_pitch_joint": -1.10,
            "right_shoulder_roll_joint": 0.10,
            "right_elbow_joint": 1.35,
            "waist_yaw_joint": 0.65,
            "waist_pitch_joint": 0.35,
            "left_shoulder_pitch_joint": 0.60,
            "left_shoulder_roll_joint": 0.35,
            "left_elbow_joint": 1.20,
        },
    ),

    # 5. Follow-through
    (
        3.05,
        {
            "right_shoulder_pitch_joint": -0.30,
            "right_shoulder_roll_joint": 0.10,
            "right_elbow_joint": 0.60,
            "waist_yaw_joint": 0.55,
            "waist_pitch_joint": 0.45,
            "left_shoulder_pitch_joint": 0.40,
            "left_shoulder_roll_joint": 0.35,
            "left_elbow_joint": 1.00,
        },
    ),

    # 6. Recovery
    (
        3.95,
        {
            "left_hip_pitch_joint": -0.10,
            "right_hip_pitch_joint": -0.10,
            "left_knee_joint": 0.20,
            "right_knee_joint": 0.20,
        },
    ),
)


@dataclass(frozen=True)
class ControllerEvent:
    """Information produced during a controller update."""

    released_this_step: bool
    released: bool
    release_time: float | None
    release_speed: float | None


class OverhandPitchController:
    """Reusable deterministic controller for the G1 overhand throw."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        release_time: float = 2.30,
        release_mode: str = "fixed",
        minimum_external_release_time: float = 1.80,
        release_threshold: float = 0.35,
        com_offset: float = 0.075,
        k_pos: float = 1.5,
        k_vel: float = 0.3,
        waypoints: Sequence[
            tuple[float, Mapping[str, float]]
        ] = WAYPOINTS,
        residual_joint_names: Sequence[str] = (),
    ) -> None:
        if release_mode not in {"fixed", "external"}:
            raise ValueError(
                "release_mode must be 'fixed' or 'external'."
            )

        self.model = model
        self.data = data

        self.release_time_target = float(release_time)
        self.release_mode = release_mode

        self.minimum_external_release_time = float(
            minimum_external_release_time
        )
        self.release_threshold = float(release_threshold)

        self.com_offset = float(com_offset)
        self.k_pos = float(k_pos)
        self.k_vel = float(k_vel)

        self.waypoints = tuple(
            (float(time_value), dict(pose))
            for time_value, pose in waypoints
        )

        self.residual_joint_names = tuple(
            residual_joint_names
        )

        # Obtain all actuator names and create:
        # actuator name -> actuator index.
        self.actuator_names = [
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            for actuator_id in range(model.nu)
        ]

        self.actuator_index = {
            name: actuator_id
            for actuator_id, name in enumerate(
                self.actuator_names
            )
            if name is not None
        }

        # Required model elements.
        self.grip_id = self._required_id(
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "grip",
        )

        self.ball_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "ball",
        )

        self.ball_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "ball_geom",
        )

        self.left_ankle_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "left_ankle_roll_link",
        )

        self.right_ankle_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "right_ankle_roll_link",
        )

        required_actuators = {
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        }

        missing = sorted(
            required_actuators
            - self.actuator_index.keys()
        )

        if missing:
            raise RuntimeError(
                "Missing required actuators: "
                + ", ".join(missing)
            )

        missing_residual = [
            name
            for name in self.residual_joint_names
            if name not in self.actuator_index
        ]

        if missing_residual:
            raise RuntimeError(
                "Unknown residual actuators: "
                + ", ".join(missing_residual)
            )

        # Orange while held.
        self.held_color = np.array(
            [0.85, 0.35, 0.15, 1.0],
            dtype=float,
        )

        # Green after release.
        self.released_color = np.array(
            [0.20, 0.80, 0.30, 1.0],
            dtype=float,
        )

        self.released = False
        self.release_time: float | None = None
        self.release_speed: float | None = None

    def _required_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
        """Find a required MuJoCo object by name."""

        object_id = mujoco.mj_name2id(
            self.model,
            object_type,
            name,
        )

        if object_id < 0:
            raise RuntimeError(
                f"Required MuJoCo object not found: "
                f"{name!r}"
            )

        return int(object_id)

    @staticmethod
    def smooth(
        start: float,
        end: float,
        progress: float,
    ) -> float:
        """Smoothly interpolate between two joint values."""

        progress = min(
            max(float(progress), 0.0),
            1.0,
        )

        # Smoothstep interpolation.
        progress = (
            progress
            * progress
            * (3.0 - 2.0 * progress)
        )

        return start + (end - start) * progress

    def pose_at(
        self,
        time_seconds: float,
    ) -> dict[str, float]:
        """Return the interpolated scripted pose."""

        if time_seconds <= self.waypoints[0][0]:
            return dict(self.waypoints[0][1])

        for (t0, pose0), (t1, pose1) in zip(
            self.waypoints,
            self.waypoints[1:],
        ):
            if t0 <= time_seconds < t1:
                progress = (
                    time_seconds - t0
                ) / (t1 - t0)

                joint_names = set(pose0) | set(pose1)

                return {
                    joint_name: self.smooth(
                        pose0.get(joint_name, 0.0),
                        pose1.get(joint_name, 0.0),
                        progress,
                    )
                    for joint_name in joint_names
                }

        return dict(self.waypoints[-1][1])

    def reset(
        self,
        *,
        reset_simulation: bool = True,
    ) -> None:
        """Reset the simulation and reactivate the grasp."""

        if reset_simulation:
            mujoco.mj_resetData(
                self.model,
                self.data,
            )

        self.released = False
        self.release_time = None
        self.release_speed = None

        # Activate the ideal hand-ball constraint.
        self.data.eq_active[self.grip_id] = 1

        # Restore the held-ball colour.
        self.model.geom_rgba[
            self.ball_geom_id
        ] = self.held_color

        # Recalculate positions and constraints.
        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def _apply_balance(
        self,
        control: np.ndarray,
    ) -> None:
        """Correct the ankles to preserve balance."""

        # Complete robot centre of mass.
        center_of_mass = self.data.subtree_com[0]

        if self.model.nbody <= 1:
            raise RuntimeError(
                "Expected a floating-base G1 model "
                "with body index 1."
            )

        # Preserve the same velocity calculation
        # used by the original final script.
        center_of_mass_velocity = (
            self.data.cvel[1, 3:6]
        )

        # Centre between both ankle bodies.
        support_center = 0.5 * (
            self.data.xpos[self.left_ankle_id]
            + self.data.xpos[self.right_ankle_id]
        )

        correction_x = (
            self.k_pos
            * (
                center_of_mass[0]
                - (
                    support_center[0]
                    + self.com_offset
                )
            )
            + self.k_vel
            * center_of_mass_velocity[0]
        )

        correction_y = (
            self.k_pos
            * (
                center_of_mass[1]
                - support_center[1]
            )
            + self.k_vel
            * center_of_mass_velocity[1]
        )

        for side in ("left", "right"):
            pitch_actuator = (
                f"{side}_ankle_pitch_joint"
            )
            roll_actuator = (
                f"{side}_ankle_roll_joint"
            )

            control[
                self.actuator_index[
                    pitch_actuator
                ]
            ] += correction_x

            control[
                self.actuator_index[
                    roll_actuator
                ]
            ] -= correction_y

    def _apply_residual_action(
        self,
        control: np.ndarray,
        residual_action:
            Mapping[str, float]
            | np.ndarray
            | None,
    ) -> None:
        """Add optional PPO residual corrections."""

        if residual_action is None:
            return

        # Dictionary mode:
        # {"right_elbow_joint": 0.05, ...}
        if isinstance(residual_action, Mapping):
            for actuator_name, value in (
                residual_action.items()
            ):
                actuator_id = (
                    self.actuator_index.get(
                        actuator_name
                    )
                )

                if actuator_id is None:
                    raise KeyError(
                        "Unknown residual actuator: "
                        f"{actuator_name!r}"
                    )

                control[actuator_id] += float(value)

            return

        # Vector mode, useful for PPO.
        residual_array = np.asarray(
            residual_action,
            dtype=float,
        ).reshape(-1)

        if len(residual_array) != len(
            self.residual_joint_names
        ):
            raise ValueError(
                "Residual vector length does not "
                "match residual_joint_names."
            )

        for actuator_name, value in zip(
            self.residual_joint_names,
            residual_array,
        ):
            control[
                self.actuator_index[
                    actuator_name
                ]
            ] += float(value)

    def compute_control(
        self,
        residual_action:
            Mapping[str, float]
            | np.ndarray
            | None = None,
    ) -> np.ndarray:
        """Compute commands without advancing physics."""

        control = np.zeros(
            self.model.nu,
            dtype=float,
        )

        current_pose = self.pose_at(
            float(self.data.time)
        )

        for actuator_name, target in (
            current_pose.items()
        ):
            actuator_id = (
                self.actuator_index.get(
                    actuator_name
                )
            )

            # Ignore optional joints that do not exist
            # rather than stopping the complete simulation.
            if actuator_id is not None:
                control[actuator_id] = float(target)

        self._apply_residual_action(
            control,
            residual_action,
        )

        self._apply_balance(control)

        return control

    def _should_release(
        self,
        release_command: float | None,
    ) -> bool:
        """Determine whether the ball must be released."""

        if self.released:
            return False

        current_time = float(self.data.time)

        if self.release_mode == "fixed":
            return (
                current_time
                >= self.release_time_target
            )

        # External mode for future PPO integration.
        if (
            current_time
            < self.minimum_external_release_time
        ):
            return False

        if release_command is None:
            return False

        return (
            float(release_command)
            >= self.release_threshold
        )

    def maybe_release(
        self,
        release_command: float | None = None,
    ) -> bool:
        """Release the ball once."""

        if not self._should_release(
            release_command
        ):
            return False

        # Record speed immediately before release.
        self.release_speed = float(
            np.linalg.norm(
                self.data.cvel[
                    self.ball_body_id,
                    3:6,
                ]
            )
        )

        self.release_time = float(
            self.data.time
        )

        # Disable the equality constraint named grip.
        self.data.eq_active[self.grip_id] = 0

        # Change the ball colour to green.
        self.model.geom_rgba[
            self.ball_geom_id
        ] = self.released_color

        self.released = True

        return True

    def before_step(
        self,
        *,
        residual_action:
            Mapping[str, float]
            | np.ndarray
            | None = None,
        release_command: float | None = None,
    ) -> ControllerEvent:
        """Apply controls immediately before mj_step."""

        self.data.ctrl[:] = self.compute_control(
            residual_action=residual_action
        )

        released_this_step = self.maybe_release(
            release_command=release_command
        )

        return ControllerEvent(
            released_this_step=(
                released_this_step
            ),
            released=self.released,
            release_time=self.release_time,
            release_speed=self.release_speed,
        )

    def get_info(
        self,
    ) -> dict[str, bool | float | None]:
        """Return information for logs or Gymnasium."""

        return {
            "released": self.released,
            "release_time": self.release_time,
            "release_speed": self.release_speed,
        }