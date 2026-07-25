"""Deterministic two-robot catch controller for ``world_catch.xml``.

The controller contains no viewer loop and does not call ``mj_step``.  It can
therefore be reused by:

* ``Simulations/catch_game_FINAL_VERSION.py`` for the visual demonstration;
* a headless evaluator;
* a future Gymnasium/PPO environment.

The deterministic motion is the baseline.  A learning environment can add
small residual joint commands through ``residual_actions`` and can optionally
learn the release instant through ``release_commands``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, Sequence

import mujoco
import numpy as np


Pose = Mapping[str, float]
ResidualValue = Mapping[str, float] | Sequence[float] | np.ndarray
ResidualActions = Mapping[int, ResidualValue]
ReleaseCommands = Mapping[int, float]


class CatchState(IntEnum):
    """State machine used by the deterministic exchange."""

    ROBOT_1_THROWING = 0
    BALL_TO_ROBOT_2 = 1
    ROBOT_2_THROWING = 2
    BALL_TO_ROBOT_1 = 3
    EXCHANGE_COMPLETE = 4


REST_POSE: dict[str, float] = {
    "left_hip_pitch_joint": -0.10,
    "right_hip_pitch_joint": -0.10,
    "left_knee_joint": 0.20,
    "right_knee_joint": 0.20,
}


CATCH_POSE: dict[str, float] = {
    "right_shoulder_pitch_joint": -1.45,
    "right_shoulder_roll_joint": -0.25,
    "right_elbow_joint": 0.20,
    "right_wrist_roll_joint": 1.20,
    "right_wrist_yaw_joint": -1.00,
    "left_shoulder_pitch_joint": -0.60,
    "left_shoulder_roll_joint": 0.35,
    "left_elbow_joint": 0.90,
}


# Kept separate so each receiving pose can be tuned independently later.
CATCH_POSE_BY_ROBOT: dict[int, dict[str, float]] = {
    1: dict(CATCH_POSE),
    2: dict(CATCH_POSE),
}


RECOIL_POSE: dict[str, float] = {
    "right_shoulder_pitch_joint": +0.55,
    "right_elbow_joint": +0.40,
    "waist_pitch_joint": -0.18,
    "left_hip_pitch_joint": -0.08,
    "right_hip_pitch_joint": -0.08,
    "left_knee_joint": +0.16,
    "right_knee_joint": +0.16,
}


ABSORB_POSE: dict[str, float] = {
    "waist_pitch_joint": -0.15,
    "left_hip_pitch_joint": -0.55,
    "right_hip_pitch_joint": -0.55,
    "left_knee_joint": +1.00,
    "right_knee_joint": +1.00,
    "left_ankle_pitch_joint": -0.45,
    "right_ankle_pitch_joint": -0.45,
}


THROW_SEQUENCE: tuple[tuple[float, dict[str, float]], ...] = (
    (0.00, dict(REST_POSE)),
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
    (
        3.05,
        {
            "right_shoulder_pitch_joint": -0.30,
            "right_shoulder_roll_joint": -0.10,
            "right_elbow_joint": 0.60,
            "waist_yaw_joint": 0.55,
            "waist_pitch_joint": 0.45,
            "left_shoulder_pitch_joint": 0.40,
            "left_shoulder_roll_joint": 0.35,
            "left_elbow_joint": 1.00,
            "left_hip_pitch_joint": 0.10,
            "left_knee_joint": 0.20,
        },
    ),
    (3.95, dict(REST_POSE)),
)


DEFAULT_RESIDUAL_JOINTS: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
)


@dataclass(frozen=True)
class CatchControllerConfig:
    """Controller and state-machine parameters."""

    release_time: float = 2.26
    release_mode: str = "fixed"  # ``fixed`` or ``external``.
    minimum_external_release_time: float = 1.80
    release_threshold: float = 0.35

    catch_radius: float = 0.15
    immediate_catch_radius: float = 0.03
    raise_time: float = 0.50
    hold_time: float = 0.80
    lower_time: float = 0.70
    settle_time: float = 0.80
    pause_time: float = 0.40
    timeout: float = 14.0

    recoil_rise: float = 0.12
    recoil_decay: float = 0.45
    # The supplied working script used 1.0, so that value is preserved.
    # Reduce it to approximately 0.45 if a trained residual destabilises a catch.
    absorb_gain: float = 1.00

    com_offset: float = 0.075
    balance_k_pos: float = 1.5
    balance_k_vel: float = 0.3

    residual_scale: float = 1.0
    residual_limit: float = 0.30

    auto_reset_cycle: bool = True
    auto_reset_timeout: bool = True

    held_color: tuple[float, float, float, float] = (0.85, 0.35, 0.15, 1.0)
    released_color: tuple[float, float, float, float] = (0.20, 0.80, 0.30, 1.0)

    def __post_init__(self) -> None:
        if self.release_mode not in {"fixed", "external"}:
            raise ValueError("release_mode must be 'fixed' or 'external'.")
        positive = {
            "release_time": self.release_time,
            "catch_radius": self.catch_radius,
            "raise_time": self.raise_time,
            "hold_time": self.hold_time,
            "lower_time": self.lower_time,
            "settle_time": self.settle_time,
            "timeout": self.timeout,
            "recoil_rise": self.recoil_rise,
            "recoil_decay": self.recoil_decay,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError("These values must be positive: " + ", ".join(invalid))
        if self.residual_limit < 0.0:
            raise ValueError("residual_limit cannot be negative.")


@dataclass(frozen=True)
class CatchControllerEvent:
    """Information generated during one controller update."""

    state: CatchState
    released_robot: int | None = None
    caught_robot: int | None = None
    release_speed: float | None = None
    exchange_complete: bool = False
    reset_performed: bool = False
    timeout_reset: bool = False
    manual_reset_detected: bool = False
    fallen_robots: tuple[int, ...] = field(default_factory=tuple)


class TwoRobotCatchController:
    """Reusable deterministic baseline for the two Unitree G1 robots."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        config: CatchControllerConfig | None = None,
        residual_joint_names: Sequence[str] = DEFAULT_RESIDUAL_JOINTS,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or CatchControllerConfig()
        self.residual_joint_names = tuple(residual_joint_names)

        self.actuator_index = {
            name: actuator_id
            for actuator_id in range(model.nu)
            if (
                name := mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_id,
                )
            )
            is not None
        }

        self.ball_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = self._required_id(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ball_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos_address = int(model.jnt_qposadr[self.ball_joint_id])
        self.ball_dof_address = int(model.jnt_dofadr[self.ball_joint_id])

        self.grip_id = {
            robot: self._required_id(mujoco.mjtObj.mjOBJ_EQUALITY, f"grip{robot}")
            for robot in (1, 2)
        }
        self.hand_id = {
            robot: self._required_id(
                mujoco.mjtObj.mjOBJ_BODY,
                f"r{robot}_right_wrist_yaw_link",
            )
            for robot in (1, 2)
        }
        self.pelvis_id = {
            robot: self._required_id(mujoco.mjtObj.mjOBJ_BODY, f"r{robot}_pelvis")
            for robot in (1, 2)
        }
        self.torso_id = {
            robot: self._required_id(mujoco.mjtObj.mjOBJ_BODY, f"r{robot}_torso_link")
            for robot in (1, 2)
        }
        self.ankle_ids = {
            robot: (
                self._required_id(
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"r{robot}_left_ankle_roll_link",
                ),
                self._required_id(
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"r{robot}_right_ankle_roll_link",
                ),
            )
            for robot in (1, 2)
        }

        self._validate_actuators()
        self._catch_throw_sequence = self._build_catch_throw_sequence()

        self.state = CatchState.ROBOT_1_THROWING
        self.cycle_start_time = 0.0
        self.flight_start = {1: 0.0, 2: 0.0}
        self.catch_time = {1: 0.0, 2: 0.0}
        self.previous_catch_distance = float("inf")
        self.last_time = float(data.time)
        self.completed_exchanges = 0

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"Required MuJoCo object not found: {name!r}")
        return int(object_id)

    def _validate_actuators(self) -> None:
        required_suffixes = set(DEFAULT_RESIDUAL_JOINTS) | {
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        }
        missing = [
            f"r{robot}_{suffix}"
            for robot in (1, 2)
            for suffix in sorted(required_suffixes)
            if f"r{robot}_{suffix}" not in self.actuator_index
        ]
        if missing:
            raise RuntimeError("Missing required actuators: " + ", ".join(missing))

        unknown_residuals = [
            suffix
            for suffix in self.residual_joint_names
            if any(
                f"r{robot}_{suffix}" not in self.actuator_index
                for robot in (1, 2)
            )
        ]
        if unknown_residuals:
            raise RuntimeError(
                "Unknown residual joint suffixes: " + ", ".join(unknown_residuals)
            )

    def _build_catch_throw_sequence(
        self,
    ) -> tuple[tuple[float, dict[str, float]], ...]:
        return (
            (0.0, dict(CATCH_POSE)),
            (self.config.settle_time, dict(CATCH_POSE)),
            *tuple(
                (time_value + self.config.settle_time, dict(pose))
                for time_value, pose in THROW_SEQUENCE[1:]
            ),
        )

    @staticmethod
    def _smooth(start: float, end: float, progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        progress = progress * progress * (3.0 - 2.0 * progress)
        return start + (end - start) * progress

    @classmethod
    def _mix(cls, pose_0: Pose, pose_1: Pose, progress: float) -> dict[str, float]:
        joint_names = set(pose_0) | set(pose_1)
        return {
            joint_name: cls._smooth(
                float(pose_0.get(joint_name, 0.0)),
                float(pose_1.get(joint_name, 0.0)),
                progress,
            )
            for joint_name in joint_names
        }

    @classmethod
    def _pose_at(
        cls,
        time_seconds: float,
        sequence: Sequence[tuple[float, Pose]],
    ) -> dict[str, float]:
        if time_seconds <= sequence[0][0]:
            return dict(sequence[0][1])

        for (time_0, pose_0), (time_1, pose_1) in zip(sequence, sequence[1:]):
            if time_0 <= time_seconds < time_1:
                progress = (time_seconds - time_0) / (time_1 - time_0)
                return cls._mix(pose_0, pose_1, progress)

        return dict(sequence[-1][1])

    def _add_recoil(self, pose: Pose, elapsed: float) -> dict[str, float]:
        if elapsed < 0.0:
            return dict(pose)

        if elapsed < self.config.recoil_rise:
            amplitude = elapsed / self.config.recoil_rise
        else:
            amplitude = np.exp(
                -(elapsed - self.config.recoil_rise) / self.config.recoil_decay
            )

        output = dict(pose)
        for joint_name, value in RECOIL_POSE.items():
            output[joint_name] = output.get(joint_name, 0.0) + value * amplitude
        for joint_name, value in ABSORB_POSE.items():
            output[joint_name] = (
                output.get(joint_name, 0.0)
                + value * self.config.absorb_gain * amplitude
            )
        return output

    def reset(self, *, reset_simulation: bool = True) -> None:
        """Restore the initial held-ball state."""

        if reset_simulation:
            mujoco.mj_resetData(self.model, self.data)

        # Put both robots directly in the same resting pose used at the end of
        # the sequence.  This avoids a discontinuity on the first control step.
        for robot in (1, 2):
            for joint_suffix, value in REST_POSE.items():
                joint_id = self._required_id(
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"r{robot}_{joint_suffix}",
                )
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qpos_address] = float(value)

        self.data.eq_active[self.grip_id[1]] = 1
        self.data.eq_active[self.grip_id[2]] = 0
        self.model.geom_rgba[self.ball_geom_id] = np.asarray(
            self.config.held_color,
            dtype=float,
        )

        mujoco.mj_forward(self.model, self.data)
        self._snap_ball_to_hand(1)

        self.state = CatchState.ROBOT_1_THROWING
        self.cycle_start_time = float(self.data.time)
        self.flight_start = {1: 0.0, 2: 0.0}
        self.catch_time = {1: 0.0, 2: 0.0}
        self.previous_catch_distance = float("inf")
        self.last_time = float(self.data.time)

    def _apply_pose(self, control: np.ndarray, robot: int, pose: Pose) -> None:
        for joint_suffix, value in pose.items():
            actuator_name = f"r{robot}_{joint_suffix}"
            actuator_id = self.actuator_index.get(actuator_name)
            if actuator_id is not None:
                control[actuator_id] = float(value)

    def _robot_pose(self, robot: int) -> dict[str, float]:
        time_now = float(self.data.time)

        if robot == 1:
            if self.state <= CatchState.ROBOT_2_THROWING:
                return self._pose_at(
                    time_now - self.cycle_start_time,
                    THROW_SEQUENCE,
                )
            if self.state == CatchState.BALL_TO_ROBOT_1:
                return self._mix(
                    REST_POSE,
                    CATCH_POSE_BY_ROBOT[1],
                    (time_now - self.flight_start[2]) / self.config.raise_time,
                )

            elapsed = time_now - self.catch_time[1]
            if elapsed < self.config.hold_time:
                return self._add_recoil(CATCH_POSE_BY_ROBOT[1], elapsed)
            return self._mix(
                CATCH_POSE_BY_ROBOT[1],
                REST_POSE,
                (elapsed - self.config.hold_time) / self.config.lower_time,
            )

        if robot == 2:
            if self.state == CatchState.ROBOT_1_THROWING:
                return dict(REST_POSE)
            if self.state == CatchState.BALL_TO_ROBOT_2:
                return self._mix(
                    REST_POSE,
                    CATCH_POSE_BY_ROBOT[2],
                    (time_now - self.flight_start[1]) / self.config.raise_time,
                )

            elapsed = time_now - self.catch_time[2]
            scripted_pose = self._pose_at(elapsed, self._catch_throw_sequence)
            return self._add_recoil(scripted_pose, elapsed)

        raise ValueError("robot must be 1 or 2.")

    def _apply_balance(self, control: np.ndarray, robot: int) -> None:
        center_of_mass = self.data.subtree_com[self.pelvis_id[robot]]
        center_of_mass_velocity = self.data.cvel[self.pelvis_id[robot], 3:6]
        left_ankle, right_ankle = self.ankle_ids[robot]
        support_center = 0.5 * (
            self.data.xpos[left_ankle] + self.data.xpos[right_ankle]
        )

        facing_sign = 1.0 if robot == 1 else -1.0
        correction_x = (
            self.config.balance_k_pos
            * (
                center_of_mass[0]
                - (support_center[0] + self.config.com_offset * facing_sign)
            )
            + self.config.balance_k_vel * center_of_mass_velocity[0]
        )
        correction_y = (
            self.config.balance_k_pos
            * (center_of_mass[1] - support_center[1])
            + self.config.balance_k_vel * center_of_mass_velocity[1]
        )

        for side in ("left", "right"):
            pitch_id = self.actuator_index[f"r{robot}_{side}_ankle_pitch_joint"]
            roll_id = self.actuator_index[f"r{robot}_{side}_ankle_roll_joint"]
            control[pitch_id] += correction_x * facing_sign
            control[roll_id] -= correction_y * facing_sign

    def _normalise_residual_mapping(
        self,
        robot: int,
        residual: Mapping[str, float],
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        prefix = f"r{robot}_"
        for supplied_name, supplied_value in residual.items():
            suffix = supplied_name[len(prefix) :] if supplied_name.startswith(prefix) else supplied_name
            if suffix not in self.residual_joint_names:
                raise KeyError(
                    f"Residual joint {supplied_name!r} is not enabled. "
                    f"Enabled suffixes: {self.residual_joint_names}"
                )
            output[suffix] = float(supplied_value)
        return output

    def _apply_residual(
        self,
        control: np.ndarray,
        robot: int,
        residual: ResidualValue | None,
    ) -> None:
        if residual is None:
            return

        if isinstance(residual, Mapping):
            residual_by_joint = self._normalise_residual_mapping(robot, residual)
        else:
            residual_array = np.asarray(residual, dtype=float).reshape(-1)
            if residual_array.size != len(self.residual_joint_names):
                raise ValueError(
                    f"Robot {robot} residual vector has length {residual_array.size}; "
                    f"expected {len(self.residual_joint_names)}."
                )
            residual_by_joint = dict(zip(self.residual_joint_names, residual_array))

        limit = self.config.residual_limit
        for joint_suffix, value in residual_by_joint.items():
            clipped_value = float(np.clip(value, -limit, limit)) if limit > 0.0 else 0.0
            actuator_id = self.actuator_index[f"r{robot}_{joint_suffix}"]
            control[actuator_id] += clipped_value * self.config.residual_scale

    def _clip_control_to_joint_ranges(self, control: np.ndarray) -> None:
        """Keep position targets inside the corresponding joint limits."""

        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            if bool(self.model.jnt_limited[joint_id]):
                lower, upper = self.model.jnt_range[joint_id]
                control[actuator_id] = np.clip(control[actuator_id], lower, upper)

    def compute_control(
        self,
        residual_actions: ResidualActions | None = None,
    ) -> np.ndarray:
        """Compute the complete two-robot position command."""

        control = np.zeros(self.model.nu, dtype=float)
        residual_actions = residual_actions or {}

        for robot in (1, 2):
            self._apply_pose(control, robot, self._robot_pose(robot))
            self._apply_residual(control, robot, residual_actions.get(robot))
            self._apply_balance(control, robot)

        self._clip_control_to_joint_ranges(control)
        return control

    def _snap_ball_to_hand(self, robot: int) -> None:
        """Match the ball pose and velocity to the selected weld exactly."""

        hand_rotation = self.data.xmat[self.hand_id[robot]].reshape(3, 3)
        relative_position = self.model.eq_data[self.grip_id[robot], 3:6]
        self.data.qpos[self.ball_qpos_address : self.ball_qpos_address + 3] = (
            self.data.xpos[self.hand_id[robot]]
            - hand_rotation @ relative_position
        )

        hand_quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(hand_quaternion, self.data.xmat[self.hand_id[robot]])
        ball_quaternion = np.zeros(4, dtype=float)
        mujoco.mju_mulQuat(
            ball_quaternion,
            hand_quaternion,
            self.model.eq_data[self.grip_id[robot], 6:10].copy(),
        )
        self.data.qpos[
            self.ball_qpos_address + 3 : self.ball_qpos_address + 7
        ] = ball_quaternion
        self.data.qvel[
            self.ball_dof_address : self.ball_dof_address + 6
        ] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def palm_position(self, robot: int) -> np.ndarray:
        hand_rotation = self.data.xmat[self.hand_id[robot]].reshape(3, 3)
        relative_position = self.model.eq_data[self.grip_id[robot], 3:6]
        return (
            self.data.xpos[self.hand_id[robot]]
            - hand_rotation @ relative_position
        ).copy()

    def distance_to_hand(self, robot: int) -> float:
        return float(
            np.linalg.norm(
                self.data.xpos[self.ball_body_id] - self.palm_position(robot)
            )
        )

    def ball_speed(self) -> float:
        return float(
            np.linalg.norm(self.data.cvel[self.ball_body_id, 3:6])
        )

    def _should_catch(self, robot: int) -> bool:
        distance = self.distance_to_hand(robot)
        inside_window = distance < self.config.catch_radius
        closest_point_reached = (
            distance > self.previous_catch_distance
            or distance < self.config.immediate_catch_radius
        )
        return bool(inside_window and closest_point_reached)

    def _should_release(
        self,
        robot: int,
        local_throw_time: float,
        release_commands: ReleaseCommands | None,
    ) -> bool:
        if self.config.release_mode == "fixed":
            return local_throw_time >= self.config.release_time

        if local_throw_time < self.config.minimum_external_release_time:
            return False
        if release_commands is None or robot not in release_commands:
            return False
        return float(release_commands[robot]) >= self.config.release_threshold

    def _release_ball(self, robot: int) -> float:
        speed = self.ball_speed()
        self.data.eq_active[self.grip_id[robot]] = 0
        self.model.geom_rgba[self.ball_geom_id] = np.asarray(
            self.config.released_color,
            dtype=float,
        )
        return speed

    def _catch_ball(self, robot: int) -> None:
        other_robot = 2 if robot == 1 else 1
        self._snap_ball_to_hand(robot)
        self.data.eq_active[self.grip_id[other_robot]] = 0
        self.data.eq_active[self.grip_id[robot]] = 1
        self.model.geom_rgba[self.ball_geom_id] = np.asarray(
            self.config.held_color,
            dtype=float,
        )

    def has_fallen(self, robot: int) -> bool:
        pelvis_low = self.data.xpos[self.pelvis_id[robot], 2] < 0.55
        torso_up_z = self.data.xmat[self.torso_id[robot]].reshape(3, 3)[2, 2]
        torso_tilted = torso_up_z < 0.65
        return bool(pelvis_low or torso_tilted)

    def fallen_robots(self) -> tuple[int, ...]:
        return tuple(robot for robot in (1, 2) if self.has_fallen(robot))

    def _transition_state_machine(
        self,
        release_commands: ReleaseCommands | None,
    ) -> CatchControllerEvent:
        now = float(self.data.time)
        released_robot: int | None = None
        caught_robot: int | None = None
        release_speed: float | None = None
        exchange_complete = False
        reset_performed = False

        if self.state == CatchState.ROBOT_1_THROWING:
            local_throw_time = now - self.cycle_start_time
            if self._should_release(1, local_throw_time, release_commands):
                release_speed = self._release_ball(1)
                self.flight_start[1] = now
                self.previous_catch_distance = float("inf")
                self.state = CatchState.BALL_TO_ROBOT_2
                released_robot = 1

        elif self.state == CatchState.BALL_TO_ROBOT_2:
            if self._should_catch(2):
                self._catch_ball(2)
                self.catch_time[2] = now
                self.previous_catch_distance = float("inf")
                self.state = CatchState.ROBOT_2_THROWING
                caught_robot = 2

        elif self.state == CatchState.ROBOT_2_THROWING:
            local_throw_time = now - self.catch_time[2] - self.config.settle_time
            if self._should_release(2, local_throw_time, release_commands):
                release_speed = self._release_ball(2)
                self.flight_start[2] = now
                self.previous_catch_distance = float("inf")
                self.state = CatchState.BALL_TO_ROBOT_1
                released_robot = 2

        elif self.state == CatchState.BALL_TO_ROBOT_1:
            if self._should_catch(1):
                self._catch_ball(1)
                self.catch_time[1] = now
                self.previous_catch_distance = float("inf")
                self.state = CatchState.EXCHANGE_COMPLETE
                self.completed_exchanges += 1
                caught_robot = 1
                exchange_complete = True

        elif self.state == CatchState.EXCHANGE_COMPLETE:
            elapsed = now - self.catch_time[1]
            cycle_finished = elapsed >= (
                self.config.hold_time
                + self.config.lower_time
                + self.config.pause_time
            )
            if cycle_finished and self.config.auto_reset_cycle:
                self.reset(reset_simulation=True)
                reset_performed = True

        # Save the current distance for closest-approach catch detection.
        if self.state == CatchState.BALL_TO_ROBOT_2:
            self.previous_catch_distance = self.distance_to_hand(2)
        elif self.state == CatchState.BALL_TO_ROBOT_1:
            self.previous_catch_distance = self.distance_to_hand(1)
        else:
            self.previous_catch_distance = float("inf")

        return CatchControllerEvent(
            state=self.state,
            released_robot=released_robot,
            caught_robot=caught_robot,
            release_speed=release_speed,
            exchange_complete=exchange_complete,
            reset_performed=reset_performed,
            fallen_robots=self.fallen_robots(),
        )

    def before_step(
        self,
        *,
        residual_actions: ResidualActions | None = None,
        release_commands: ReleaseCommands | None = None,
    ) -> CatchControllerEvent:
        """Update controls and game state immediately before ``mj_step``."""

        now = float(self.data.time)
        manual_reset = now + 1e-12 < self.last_time
        timeout = (
            now - self.cycle_start_time > self.config.timeout
            and self.state != CatchState.EXCHANGE_COMPLETE
        )

        if manual_reset:
            self.reset(reset_simulation=False)
        elif timeout and self.config.auto_reset_timeout:
            self.reset(reset_simulation=True)

        self.data.ctrl[:] = self.compute_control(residual_actions)
        event = self._transition_state_machine(release_commands)

        if manual_reset or timeout:
            event = CatchControllerEvent(
                state=self.state,
                released_robot=event.released_robot,
                caught_robot=event.caught_robot,
                release_speed=event.release_speed,
                exchange_complete=event.exchange_complete,
                reset_performed=True,
                timeout_reset=bool(timeout),
                manual_reset_detected=bool(manual_reset),
                fallen_robots=event.fallen_robots,
            )

        self.last_time = float(self.data.time)
        return event

    def get_training_info(self) -> dict[str, object]:
        """Return compact state information useful for logging or Gymnasium."""

        return {
            "state": int(self.state),
            "state_name": self.state.name,
            "ball_position": self.data.xpos[self.ball_body_id].copy(),
            "ball_velocity": self.data.cvel[self.ball_body_id, 3:6].copy(),
            "ball_speed": self.ball_speed(),
            "distance_to_robot_1_hand": self.distance_to_hand(1),
            "distance_to_robot_2_hand": self.distance_to_hand(2),
            "robot_1_fallen": self.has_fallen(1),
            "robot_2_fallen": self.has_fallen(2),
            "completed_exchanges": self.completed_exchanges,
        }


__all__ = [
    "ABSORB_POSE",
    "CATCH_POSE",
    "CATCH_POSE_BY_ROBOT",
    "CatchControllerConfig",
    "CatchControllerEvent",
    "CatchState",
    "DEFAULT_RESIDUAL_JOINTS",
    "RECOIL_POSE",
    "REST_POSE",
    "THROW_SEQUENCE",
    "TwoRobotCatchController",
]