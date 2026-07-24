"""Safe Gymnasium environment for the Unitree G1 overhand throw.

The environment reuses the team's OverhandPitchController.

Available modes
---------------
release_only:
    The deterministic controller executes the complete movement.
    PPO only decides when to release the ball.

residual:
    PPO adds very small corrections to five selected joints and also
    decides when to release the ball.

The residual corrections are intentionally small so that a zero action
reproduces the existing stable baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from overhand_pitch_controller import OverhandPitchController


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "Model"
    / "g1_ball.xml"
)


class UnitreeDropEnv(gym.Env):
    """Safe RL environment for the Unitree G1 throwing task."""

    metadata = {"render_modes": []}

    CONTROLLED_JOINTS = (
        "waist_yaw_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_elbow_joint",
    )

    # Maximum correction added to the deterministic controller.
    #
    # These values are deliberately much smaller than the previous ±0.5 rad.
    RESIDUAL_SCALES = np.array(
        [
            0.030,  # waist yaw
            0.025,  # waist pitch
            0.040,  # right shoulder pitch
            0.030,  # right shoulder roll
            0.040,  # right elbow
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
        frame_skip: int = 5,
        minimum_release_time: float = 2.10,
        maximum_release_time: float = 2.45,
        reference_release_time: float = 2.30,
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

        if minimum_release_time >= maximum_release_time:
            raise ValueError(
                "minimum_release_time must be lower than "
                "maximum_release_time."
            )

        if target_radius <= 0:
            raise ValueError(
                "target_radius must be positive."
            )

        self.mode = mode
        self.episode_time = float(episode_time)
        self.frame_skip = int(frame_skip)

        self.minimum_release_time = float(
            minimum_release_time
        )

        self.maximum_release_time = float(
            maximum_release_time
        )

        self.reference_release_time = float(
            reference_release_time
        )

        # ---------------------------------------------------------
        # Load the exact model used by the current project.
        # ---------------------------------------------------------

        selected_model_path = (
            Path(model_path)
            if model_path is not None
            else DEFAULT_MODEL_PATH
        )

        self.model_path = (
            selected_model_path
            .expanduser()
            .resolve()
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )

        self.data = mujoco.MjData(
            self.model
        )

        self.control_dt = (
            float(self.model.opt.timestep)
            * self.frame_skip
        )

        # ---------------------------------------------------------
        # Required model objects.
        # ---------------------------------------------------------

        self.ball_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "ball",
        )

        self.ball_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "ball_geom",
        )

        self.base_body_id = self._find_base_body()

        self.ball_radius = float(
            self.model.geom_size[
                self.ball_geom_id,
                0,
            ]
        )

        # Detect plane geoms, normally the floor.
        self.floor_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_type[geom_id])
            == int(mujoco.mjtGeom.mjGEOM_PLANE)
        }

        # ---------------------------------------------------------
        # Target.
        # ---------------------------------------------------------

        self.target_xy = np.array(
            [
                target_x,
                target_y,
            ],
            dtype=np.float64,
        )

        self.target_position = np.array(
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

        # ---------------------------------------------------------
        # Joint addresses.
        # ---------------------------------------------------------

        self.joint_ids = {
            joint_name: self._required_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            for joint_name in self.CONTROLLED_JOINTS
        }

        self.joint_qpos_addresses = np.array(
            [
                self.model.jnt_qposadr[
                    self.joint_ids[joint_name]
                ]
                for joint_name
                in self.CONTROLLED_JOINTS
            ],
            dtype=np.int32,
        )

        self.joint_dof_addresses = np.array(
            [
                self.model.jnt_dofadr[
                    self.joint_ids[joint_name]
                ]
                for joint_name
                in self.CONTROLLED_JOINTS
            ],
            dtype=np.int32,
        )

        # ---------------------------------------------------------
        # Reuse the controller currently working in your model.
        # ---------------------------------------------------------

        residual_joint_names = (
            self.CONTROLLED_JOINTS
            if self.mode == "residual"
            else ()
        )

        self.controller = OverhandPitchController(
            self.model,
            self.data,
            release_mode="external",
            minimum_external_release_time=(
                self.minimum_release_time
            ),
            release_threshold=release_threshold,
            residual_joint_names=residual_joint_names,
        )

        # ---------------------------------------------------------
        # Action space.
        # ---------------------------------------------------------

        if self.mode == "release_only":
            # action[0] = release signal
            action_size = 1
        else:
            # action[0:5] = residual joint corrections
            # action[5]   = release signal
            action_size = 6

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_size,),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Observation space: 29 values.
        #
        # 5 reference-relative joint positions
        # 5 joint velocities
        # 3 ball-to-target coordinates
        # 3 ball linear velocities
        # 3 base-up-vector components
        # 1 relative base-height value
        # 1 normalized time
        # 1 released flag
        # 6 previous-action values
        # 1 planar target distance
        # ---------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(29,),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # Episode state.
        # ---------------------------------------------------------

        self.nominal_base_height = 0.0

        self.impact_detected = False

        self.impact_position = np.full(
            3,
            np.nan,
            dtype=np.float64,
        )

        self.impact_error = np.nan

        self.success = False
        self.unsafe_posture = False
        self.forced_release = False

        # Always retain six values, even in release_only mode.
        self.previous_action = np.zeros(
            6,
            dtype=np.float32,
        )

    # =============================================================
    # Model utilities
    # =============================================================

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
                f"Required MuJoCo object not found: {name!r}"
            )

        return int(object_id)

    def _find_base_body(self) -> int:
        """Find the torso or pelvis body used for safety metrics."""

        candidate_names = (
            "torso_link",
            "pelvis",
            "trunk",
            "base_link",
        )

        for body_name in candidate_names:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )

            if body_id >= 0:
                return int(body_id)

        # The existing controller also assumes body 1 is the moving base.
        if self.model.nbody <= 1:
            raise RuntimeError(
                "The model does not contain a moving robot body."
            )

        return 1

    # =============================================================
    # Gymnasium API
    # =============================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        """Reset MuJoCo, the controller and all episode metrics."""

        super().reset(seed=seed)

        self.controller.reset(
            reset_simulation=True
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.nominal_base_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        self.impact_detected = False

        self.impact_position[:] = np.nan
        self.impact_error = np.nan

        self.success = False
        self.unsafe_posture = False
        self.forced_release = False

        self.previous_action[:] = 0.0

        return (
            self._get_observation(),
            self._get_info(),
        )

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
        """Apply one PPO action and advance several physics steps."""

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.action_space.shape)

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # Convert to a fixed six-value representation.
        complete_action = np.zeros(
            6,
            dtype=np.float32,
        )

        if self.mode == "release_only":
            residual_action = None
            complete_action[5] = action[0]
        else:
            residual_action = (
                action[:5].astype(np.float64)
                * self.RESIDUAL_SCALES
            )

            complete_action[:] = action

        released_this_control_step = False
        impacted_this_control_step = False

        # Update the reference trajectory at every MuJoCo step.
        for _ in range(self.frame_skip):
            current_time = float(
                self.data.time
            )

            release_signal = float(
                complete_action[5]
            )

            force_release_candidate = False

            # Prevent an unrealistically early release.
            if (
                current_time
                < self.minimum_release_time
            ):
                release_signal = -1.0

            # Prevent the policy from never releasing.
            elif (
                current_time
                >= self.maximum_release_time
                and not self.controller.released
            ):
                release_signal = 1.0
                force_release_candidate = True

            controller_event = (
                self.controller.before_step(
                    residual_action=residual_action,
                    release_command=release_signal,
                )
            )

            if controller_event.released_this_step:
                released_this_control_step = True

                if force_release_candidate:
                    self.forced_release = True

            self._clip_actuator_controls()

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.unsafe_posture = (
                self._is_posture_unsafe()
            )

            if (
                self.controller.released
                and not self.impact_detected
                and self._detect_floor_impact()
            ):
                impacted_this_control_step = True

            if (
                self.unsafe_posture
                or self.impact_detected
            ):
                break

        reward = self._compute_reward(
            complete_action=complete_action,
            released_this_step=(
                released_this_control_step
            ),
            impacted_this_step=(
                impacted_this_control_step
            ),
        )

        terminated = bool(
            self.unsafe_posture
            or self.impact_detected
        )

        truncated = bool(
            not terminated
            and self.data.time
            >= self.episode_time
        )

        if (
            truncated
            and not self.controller.released
        ):
            reward -= 20.0

        self.previous_action = (
            complete_action.copy()
        )

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    # =============================================================
    # Control safety
    # =============================================================

    def _clip_actuator_controls(self) -> None:
        """Respect the actuator control ranges defined in MJCF."""

        control_limited = np.asarray(
            self.model.actuator_ctrllimited,
            dtype=bool,
        )

        if not np.any(control_limited):
            return

        minimum_control = (
            self.model.actuator_ctrlrange[
                :,
                0,
            ]
        )

        maximum_control = (
            self.model.actuator_ctrlrange[
                :,
                1,
            ]
        )

        self.data.ctrl[
            control_limited
        ] = np.clip(
            self.data.ctrl[
                control_limited
            ],
            minimum_control[
                control_limited
            ],
            maximum_control[
                control_limited
            ],
        )

    def _base_up_vector(self) -> np.ndarray:
        """Return the local vertical axis of the robot base."""

        rotation_matrix = (
            self.data.xmat[
                self.base_body_id
            ].reshape(3, 3)
        )

        return rotation_matrix[
            :,
            2,
        ].copy()

    def _is_posture_unsafe(self) -> bool:
        """Terminate before the robot completes a severe fall."""

        base_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        base_up_z = float(
            self._base_up_vector()[2]
        )

        height_loss = (
            self.nominal_base_height
            - base_height
        )

        return bool(
            base_up_z < 0.60
            or height_loss > 0.22
        )

    # =============================================================
    # Impact and velocity
    # =============================================================

    def _detect_floor_impact(self) -> bool:
        """Record the first contact between the ball and floor."""

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

            if (
                self.floor_geom_ids
                and other_geom_id
                not in self.floor_geom_ids
            ):
                continue

            self._record_impact(
                np.asarray(
                    contact.pos,
                    dtype=np.float64,
                )
            )

            return True

        # Fallback for a floor not defined as a plane.
        ball_position = (
            self.data.geom_xpos[
                self.ball_geom_id
            ]
        )

        if (
            ball_position[2]
            <= self.ball_radius + 0.004
        ):
            self._record_impact(
                ball_position.copy()
            )

            return True

        return False

    def _record_impact(
        self,
        position: np.ndarray,
    ) -> None:
        """Store the first landing position and accuracy."""

        self.impact_detected = True
        self.impact_position = position.copy()

        self.impact_error = float(
            np.linalg.norm(
                self.impact_position[:2]
                - self.target_xy
            )
        )

        self.success = bool(
            self.impact_error
            <= self.target_radius
            and not self.unsafe_posture
        )

    def _ball_linear_velocity(
        self,
    ) -> np.ndarray:
        """Return ball linear velocity in world coordinates."""

        object_velocity = np.zeros(
            6,
            dtype=np.float64,
        )

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ball_body_id,
            object_velocity,
            0,
        )

        # MuJoCo returns angular velocity followed by linear velocity.
        return object_velocity[3:6]

    # =============================================================
    # Reward
    # =============================================================

    def _compute_reward(
        self,
        *,
        complete_action: np.ndarray,
        released_this_step: bool,
        impacted_this_step: bool,
    ) -> float:
        """Compute stability, tracking and task rewards."""

        reward = -0.001

        base_up_z = float(
            self._base_up_vector()[2]
        )

        base_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        height_error = (
            base_height
            - self.nominal_base_height
        )

        # Continuous stability reward.
        upright_reward = max(
            base_up_z,
            0.0,
        )

        height_reward = float(
            np.exp(
                -20.0
                * height_error
                * height_error
            )
        )

        reward += 0.020 * upright_reward
        reward += 0.010 * height_reward

        # Track the deterministic reference movement.
        reference_pose = (
            self.controller.pose_at(
                float(self.data.time)
            )
        )

        current_joint_positions = (
            self.data.qpos[
                self.joint_qpos_addresses
            ]
        )

        reference_positions = np.array(
            [
                reference_pose.get(
                    joint_name,
                    0.0,
                )
                for joint_name
                in self.CONTROLLED_JOINTS
            ],
            dtype=np.float64,
        )

        tracking_error = float(
            np.mean(
                np.square(
                    current_joint_positions
                    - reference_positions
                )
            )
        )

        tracking_reward = float(
            np.exp(
                -5.0
                * tracking_error
            )
        )

        reward += 0.020 * tracking_reward

        # Penalize only the five residual-control values.
        if self.mode == "residual":
            residual_penalty = float(
                np.mean(
                    np.square(
                        complete_action[:5]
                    )
                )
            )

            action_rate_penalty = float(
                np.mean(
                    np.square(
                        complete_action[:5]
                        - self.previous_action[:5]
                    )
                )
            )

            reward -= (
                0.005
                * residual_penalty
            )

            reward -= (
                0.002
                * action_rate_penalty
            )

        # Reward a valid release without rewarding excessive speed.
        if released_this_step:
            release_time = float(
                self.controller.release_time
            )

            timing_error = (
                release_time
                - self.reference_release_time
            )

            timing_reward = float(
                np.exp(
                    -0.5
                    * (
                        timing_error / 0.08
                    )
                    ** 2
                )
            )

            reward += 0.50
            reward += 1.50 * timing_reward

            # Waiting until forced release is valid but undesirable.
            if self.forced_release:
                reward -= 4.0

        # Landing accuracy is the dominant task reward.
        if impacted_this_step:
            accuracy_reward = float(
                np.exp(
                    -0.5
                    * (
                        self.impact_error / 0.35
                    )
                    ** 2
                )
            )

            reward += (
                50.0
                * accuracy_reward
            )

            if self.success:
                reward += 50.0

        if self.unsafe_posture:
            reward -= 100.0

        return float(reward)

    # =============================================================
    # Observation
    # =============================================================

    def _get_observation(self) -> np.ndarray:
        """Build a normalized observation vector."""

        reference_pose = (
            self.controller.pose_at(
                float(self.data.time)
            )
        )

        reference_positions = np.array(
            [
                reference_pose.get(
                    joint_name,
                    0.0,
                )
                for joint_name
                in self.CONTROLLED_JOINTS
            ],
            dtype=np.float64,
        )

        joint_position_errors = (
            self.data.qpos[
                self.joint_qpos_addresses
            ]
            - reference_positions
        ) / 0.50

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
            - self.target_position
        ) / 3.0

        ball_velocity = (
            self._ball_linear_velocity()
            / 10.0
        )

        base_up_vector = (
            self._base_up_vector()
        )

        base_height = float(
            self.data.xpos[
                self.base_body_id,
                2,
            ]
        )

        relative_base_height = np.array(
            [
                (
                    base_height
                    - self.nominal_base_height
                )
                / 0.30
            ],
            dtype=np.float64,
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

        planar_target_distance = np.array(
            [
                np.linalg.norm(
                    ball_position[:2]
                    - self.target_xy
                )
                / 3.0
            ],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                joint_position_errors,
                joint_velocities,
                ball_to_target,
                ball_velocity,
                base_up_vector,
                relative_base_height,
                normalized_time,
                released_flag,
                self.previous_action,
                planar_target_distance,
            ]
        )

        return np.clip(
            observation,
            -10.0,
            10.0,
        ).astype(np.float32)

    # =============================================================
    # Information
    # =============================================================

    def _get_info(
        self,
    ) -> dict[str, Any]:
        """Return episode metrics for evaluation and logging."""

        return {
            "mode": self.mode,

            "released": (
                self.controller.released
            ),

            "forced_release": (
                self.forced_release
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

            "success": self.success,

            "unsafe_posture": (
                self.unsafe_posture
            ),

            "fell": (
                self.unsafe_posture
            ),

            "simulation_time": float(
                self.data.time
            ),
        }

    def close(self) -> None:
        """Gymnasium compatibility method."""

        return None