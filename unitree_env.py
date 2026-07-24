i"""Gymnasium environment for the Unitree G1 overhand throw.

The environment reuses OverhandPitchController instead of replacing the
working scripted movement.

Available modes:

1. release_only:
   The controller performs the complete scripted throw.
   PPO only decides when to release the ball.

2. residual:
   PPO applies small corrections to five joints and also decides when
   to release the ball.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from overhand_pitch_controller import OverhandPitchController


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "Model"
    / "g1_ball.xml"
)


class UnitreeDropEnv(gym.Env):
    """Gymnasium environment for the Unitree G1 ball-throwing task."""

    metadata = {
        "render_modes": []
    }

    # Joints that PPO may observe and optionally correct.
    ACTIVE_JOINTS = (
        "waist_yaw_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_elbow_joint",
    )

    # Maximum residual correction for each controlled joint.
    #
    # final command =
    # scripted controller command + PPO correction
    RESIDUAL_SCALES = np.array(
        [
            0.10,  # waist yaw
            0.08,  # waist pitch
            0.12,  # shoulder pitch
            0.10,  # shoulder roll
            0.12,  # elbow
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        mode: str = "release_only",
        target_x: float = 3.0,
        target_y: float = 0.0,
        target_radius: float = 0.50,
        episode_time: float = 5.0,
        frame_skip: int = 10,
        minimum_release_time: float = 1.80,
        release_threshold: float = 0.35,
    ) -> None:
        super().__init__()

        if mode not in {
            "release_only",
            "residual",
        }:
            raise ValueError(
                "mode must be 'release_only' or 'residual'."
            )

        if frame_skip < 1:
            raise ValueError(
                "frame_skip must be at least 1."
            )

        if episode_time <= 0:
            raise ValueError(
                "episode_time must be positive."
            )

        if target_radius <= 0:
            raise ValueError(
                "target_radius must be positive."
            )

        self.mode = mode
        self.episode_time = float(episode_time)
        self.frame_skip = int(frame_skip)

        # -------------------------------------------------------------
        # LOAD MUJOCO MODEL
        # -------------------------------------------------------------

        selected_model_path = (
            model_path
            if model_path is not None
            else DEFAULT_MODEL_PATH
        )

        self.model_path = (
            Path(selected_model_path)
            .expanduser()
            .resolve()
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                "MuJoCo model not found: "
                f"{self.model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )

        self.data = mujoco.MjData(
            self.model
        )

        # -------------------------------------------------------------
        # REQUIRED MODEL ELEMENTS
        # -------------------------------------------------------------

        self.ball_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "ball",
        )

        self.ball_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "ball_geom",
        )

        self.base_body_id = (
            self._find_base_body()
        )

        self.ball_radius = float(
            self.model.geom_size[
                self.ball_geom_id,
                0,
            ]
        )

        # Find all plane geometries.
        # Normally the ground is a plane attached to the world body.
        self.floor_geom_ids = {
            geom_id
            for geom_id in range(
                self.model.ngeom
            )
            if int(
                self.model.geom_type[
                    geom_id
                ]
            )
            == int(
                mujoco.mjtGeom.mjGEOM_PLANE
            )
        }

        # -------------------------------------------------------------
        # TARGET
        # -------------------------------------------------------------

        self.target_xy = np.array(
            [
                target_x,
                target_y,
            ],
            dtype=np.float64,
        )

        self.target_pos = np.array(
            [
                target_x,
                target_y,
                self.ball_radius,
            ],
            dtype=np.float64,
        )

        self.target_radius = float(
            target_radius
        )

        # -------------------------------------------------------------
        # JOINT ADDRESSES
        # -------------------------------------------------------------

        self.joint_ids = {
            joint_name: self._required_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            for joint_name in self.ACTIVE_JOINTS
        }

        self.joint_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[
                    self.joint_ids[joint_name]
                ]
                for joint_name
                in self.ACTIVE_JOINTS
            ],
            dtype=np.int32,
        )

        self.joint_dof_addresses = np.array(
            [
                self.model.jnt_dofadr[
                    self.joint_ids[joint_name]
                ]
                for joint_name
                in self.ACTIVE_JOINTS
            ],
            dtype=np.int32,
        )

        # -------------------------------------------------------------
        # REUSABLE SCRIPTED CONTROLLER
        # -------------------------------------------------------------

        residual_joint_names = (
            self.ACTIVE_JOINTS
            if self.mode == "residual"
            else ()
        )

        self.controller = (
            OverhandPitchController(
                self.model,
                self.data,
                release_mode="external",
                minimum_external_release_time=(
                    minimum_release_time
                ),
                release_threshold=(
                    release_threshold
                ),
                residual_joint_names=(
                    residual_joint_names
                ),
            )
        )

        # -------------------------------------------------------------
        # ACTION SPACE
        # -------------------------------------------------------------

        if self.mode == "release_only":
            # One action:
            # action[0] = release signal
            action_size = 1
        else:
            # Six actions:
            # action[0:5] = residual corrections
            # action[5]   = release signal
            action_size = 6

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_size,),
            dtype=np.float32,
        )

        # -------------------------------------------------------------
        # OBSERVATION SPACE
        # -------------------------------------------------------------
        #
        # 5 joint positions
        # 5 joint velocities
        # 3 ball-to-target coordinates
        # 3 ball linear velocities
        # 3 base-up-vector components
        # 1 normalized episode time
        # 1 released flag
        #
        # Total = 21 values
        # -------------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(21,),
            dtype=np.float32,
        )

        self.control_dt = (
            float(
                self.model.opt.timestep
            )
            * self.frame_skip
        )

        # -------------------------------------------------------------
        # EPISODE STATE
        # -------------------------------------------------------------

        self.impact_detected = False

        self.impact_position = np.full(
            3,
            np.nan,
            dtype=np.float64,
        )

        self.impact_error = np.nan
        self.success = False
        self.fell = False

        self.previous_action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )

    # -----------------------------------------------------------------
    # MODEL UTILITIES
    # -----------------------------------------------------------------

    def _required_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
        """Return the ID of a required MuJoCo object."""

        object_id = mujoco.mj_name2id(
            self.model,
            object_type,
            name,
        )

        if object_id < 0:
            raise RuntimeError(
                "Required MuJoCo object "
                f"not found: {name!r}"
            )

        return int(object_id)

    def _find_base_body(self) -> int:
        """Find the main torso or pelvis body."""

        possible_names = (
            "torso_link",
            "pelvis",
            "trunk",
            "base_link",
        )

        for body_name in possible_names:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )

            if body_id >= 0:
                return int(body_id)

        # Fallback for the current floating-base G1 model.
        if self.model.nbody <= 1:
            raise RuntimeError(
                "The model does not contain "
                "a moving robot body."
            )

        return 1

    # -----------------------------------------------------------------
    # GYMNASIUM RESET
    # -----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        super().reset(seed=seed)

        # Reset MuJoCo and reactivate the grip.
        self.controller.reset(
            reset_simulation=True
        )

        self.impact_detected = False

        self.impact_position[:] = np.nan
        self.impact_error = np.nan

        self.success = False
        self.fell = False

        self.previous_action[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # -----------------------------------------------------------------
    # GYMNASIUM STEP
    # -----------------------------------------------------------------

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(
            self.action_space.shape
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # -------------------------------------------------------------
        # INTERPRET PPO ACTION
        # -------------------------------------------------------------

        if self.mode == "release_only":
            residual_action = None
            release_command = float(
                action[0]
            )
        else:
            residual_action = (
                action[:5].astype(
                    np.float64
                )
                * self.RESIDUAL_SCALES
            )

            release_command = float(
                action[5]
            )

        released_this_control_step = False
        impacted_this_control_step = False

        # -------------------------------------------------------------
        # ADVANCE PHYSICS
        # -------------------------------------------------------------

        for _ in range(
            self.frame_skip
        ):
            controller_event = (
                self.controller.before_step(
                    residual_action=(
                        residual_action
                    ),
                    release_command=(
                        release_command
                    ),
                )
            )

            if (
                controller_event
                .released_this_step
            ):
                released_this_control_step = True

            mujoco.mj_step(
                self.model,
                self.data,
            )

            # Check robot stability first.
            self.fell = (
                self._robot_fell()
            )

            # Detect first ball-floor contact.
            if (
                self.controller.released
                and not self.impact_detected
                and self._detect_floor_impact()
            ):
                impacted_this_control_step = True

            if (
                self.fell
                or self.impact_detected
            ):
                break

        # -------------------------------------------------------------
        # REWARD
        # -------------------------------------------------------------

        reward = self._compute_reward(
            action=action,
            released_this_step=(
                released_this_control_step
            ),
            impacted_this_step=(
                impacted_this_control_step
            ),
        )

        # -------------------------------------------------------------
        # TERMINATION
        # -------------------------------------------------------------

        terminated = bool(
            self.fell
            or self.impact_detected
        )

        truncated = bool(
            not terminated
            and self.data.time
            >= self.episode_time
        )

        # Penalize policies that never release.
        if (
            truncated
            and not self.controller.released
        ):
            reward -= 20.0

        self.previous_action = (
            action.copy()
        )

        observation = self._get_obs()
        info = self._get_info()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # -----------------------------------------------------------------
    # IMPACT DETECTION
    # -----------------------------------------------------------------

    def _detect_floor_impact(
        self,
    ) -> bool:
        """Detect and record the first ball-floor contact."""

        for contact_index in range(
            self.data.ncon
        ):
            contact = self.data.contact[
                contact_index
            ]

            if (
                contact.geom1
                == self.ball_geom_id
            ):
                other_geom_id = int(
                    contact.geom2
                )

            elif (
                contact.geom2
                == self.ball_geom_id
            ):
                other_geom_id = int(
                    contact.geom1
                )

            else:
                continue

            is_floor = (
                other_geom_id
                in self.floor_geom_ids
            )

            if not is_floor:
                continue

            self.impact_detected = True

            self.impact_position = (
                np.array(
                    contact.pos,
                    dtype=np.float64,
                )
            )

            self.impact_error = float(
                np.linalg.norm(
                    self.impact_position[:2]
                    - self.target_xy
                )
            )

            self.success = bool(
                self.impact_error
                <= self.target_radius
                and not self.fell
            )

            return True

        # Fallback in case the ground geometry is not a plane.
        ball_z = float(
            self.data.geom_xpos[
                self.ball_geom_id,
                2,
            ]
        )

        if (
            ball_z
            <= self.ball_radius + 0.005
        ):
            self.impact_detected = True

            self.impact_position = (
                self.data.geom_xpos[
                    self.ball_geom_id
                ].copy()
            )

            self.impact_error = float(
                np.linalg.norm(
                    self.impact_position[:2]
                    - self.target_xy
                )
            )

            self.success = bool(
                self.impact_error
                <= self.target_radius
                and not self.fell
            )

            return True

        return False

    # -----------------------------------------------------------------
    # ROBOT STABILITY
    # -----------------------------------------------------------------

    def _base_up_vector(
        self,
    ) -> np.ndarray:
        """Return the vertical axis of the robot base in world coordinates."""

        rotation_matrix = (
            self.data.xmat[
                self.base_body_id
            ].reshape(3, 3)
        )

        return rotation_matrix[
            :,
            2,
        ].copy()

    def _robot_fell(
        self,
    ) -> bool:
        """Detect whether the robot has fallen."""

        base_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        base_up_z = float(
            self._base_up_vector()[2]
        )

        return bool(
            base_height < 0.45
            or base_up_z < 0.40
        )

    # -----------------------------------------------------------------
    # BALL VELOCITY
    # -----------------------------------------------------------------

    def _ball_linear_velocity(
        self,
    ) -> np.ndarray:
        """Return ball linear velocity in world coordinates."""

        velocity = np.zeros(
            6,
            dtype=np.float64,
        )

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ball_body_id,
            velocity,
            0,
        )

        # MuJoCo order:
        # angular velocity first,
        # linear velocity second.
        return velocity[3:6]

    # -----------------------------------------------------------------
    # REWARD
    # -----------------------------------------------------------------

    def _compute_reward(
        self,
        *,
        action: np.ndarray,
        released_this_step: bool,
        impacted_this_step: bool,
    ) -> float:
        reward = 0.0

        # Small time cost.
        reward -= 0.002

        # Reward upright posture.
        base_up_z = float(
            self._base_up_vector()[2]
        )

        reward += (
            0.01
            * max(base_up_z, 0.0)
        )

        # In residual mode, keep PPO corrections small and smooth.
        if self.mode == "residual":
            residual_penalty = float(
                np.mean(
                    np.square(
                        action[:5]
                    )
                )
            )

            action_rate_penalty = float(
                np.mean(
                    np.square(
                        action
                        - self.previous_action
                    )
                )
            )

            reward -= (
                0.01
                * residual_penalty
            )

            reward -= (
                0.002
                * action_rate_penalty
            )

        # Small reward for a valid release.
        if released_this_step:
            release_speed = (
                self.controller.release_speed
                if self.controller.release_speed
                is not None
                else 0.0
            )

            reward += 0.5

            # Small speed contribution.
            # Accuracy remains more important than velocity.
            reward += (
                0.05
                * min(
                    release_speed,
                    10.0,
                )
            )

        # Main reward: landing accuracy.
        if impacted_this_step:
            accuracy = float(
                np.exp(
                    -0.5
                    * (
                        self.impact_error
                        / 0.35
                    )
                    ** 2
                )
            )

            reward += (
                25.0
                * accuracy
            )

            if self.success:
                reward += 25.0

        # Strong fall penalty.
        if self.fell:
            reward -= 50.0

        return reward

    # -----------------------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------------------

    def _get_obs(
        self,
    ) -> np.ndarray:
        joint_positions = (
            self.data.qpos[
                self.joint_qpos_addresses
            ]
            / np.pi
        )

        joint_velocities = (
            self.data.qvel[
                self.joint_dof_addresses
            ]
            / 10.0
        )

        ball_position = (
            self.data.geom_xpos[
                self.ball_geom_id
            ]
        )

        ball_to_target = (
            ball_position
            - self.target_pos
        ) / 3.0

        ball_velocity = (
            self._ball_linear_velocity()
            / 10.0
        )

        base_up_vector = (
            self._base_up_vector()
        )

        normalized_time = np.array(
            [
                self.data.time
                / self.episode_time
            ],
            dtype=np.float64,
        )

        released_flag = np.array(
            [
                float(
                    self.controller.released
                )
            ],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                joint_positions,
                joint_velocities,
                ball_to_target,
                ball_velocity,
                base_up_vector,
                normalized_time,
                released_flag,
            ]
        )

        observation = np.clip(
            observation,
            -10.0,
            10.0,
        )

        return observation.astype(
            np.float32
        )

    # -----------------------------------------------------------------
    # INFO DICTIONARY
    # -----------------------------------------------------------------

    def _get_info(
        self,
    ) -> dict[str, Any]:
        current_ball_xy = (
            self.data.geom_xpos[
                self.ball_geom_id,
                :2,
            ]
        )

        current_target_error = float(
            np.linalg.norm(
                current_ball_xy
                - self.target_xy
            )
        )

        return {
            "mode": self.mode,

            "released": (
                self.controller.released
            ),

            "release_time": (
                self.controller.release_time
                if self.controller.release_time
                is not None
                else np.nan
            ),

            "release_speed": (
                self.controller.release_speed
                if self.controller.release_speed
                is not None
                else np.nan
            ),

            "impact_detected": (
                self.impact_detected
            ),

            "impact_x": float(
                self.impact_position[0]
            ),

            "impact_y": float(
                self.impact_position[1]
            ),

            "impact_error": float(
                self.impact_error
            ),

            "current_target_error": (
                current_target_error
            ),

            "success": self.success,
            "fell": self.fell,

            "simulation_time": float(
                self.data.time
            ),
        }

    def close(
        self,
    ) -> None:
        """Gymnasium compatibility method."""

        return None