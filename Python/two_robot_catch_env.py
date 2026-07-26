"""Two-robot PPO environment with clean right-hand catch enforcement.

This version keeps the original deterministic physics parameters. It adds only
the robot-1 receiving-arm correction, a smaller right-hand catch window, and an
explicit penalty when the returning ball touches robot 1's left arm.

This environment is designed for:

    Model/world_catch.xml
    Python/catch_controller_FINAL_VERSION.py

The deterministic controller supplies a stable reference motion. PPO can be
trained in two modes:

release_only
    The reference controller moves both robots. PPO only decides the release
    instant for robot 1 and robot 2.

full_exchange
    PPO decides both release instants and adds small residual corrections to
    the throwing/receiving arm of the currently active robot.

The ideal grasp remains automatic: when the ball reaches the catch window,
the controller activates grip1 or grip2. Therefore PPO learns to move the hand
into the correct place, not finger-level grasp mechanics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from catch_controller_clean_right_hand import (
    CatchControllerConfig,
    CatchState,
    TwoRobotCatchController,
)


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Model" / "world_catch.xml"


class TwoRobotCatchCleanRightHandEnv(gym.Env):
    """Residual-RL environment for a complete two-robot ball exchange."""

    metadata = {"render_modes": []}

    CONTROLLED_JOINTS = (
        "waist_yaw_joint",
        "waist_pitch_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    # Maximum PPO correction, in radians, around the deterministic reference.
    RESIDUAL_SCALES = np.array(
        [
            0.030,  # waist yaw
            0.025,  # waist pitch
            0.050,  # shoulder pitch
            0.040,  # shoulder roll
            0.035,  # shoulder yaw
            0.050,  # elbow
            0.025,  # wrist roll
            0.025,  # wrist pitch
            0.025,  # wrist yaw
        ],
        dtype=np.float64,
    )

    JOINTS_PER_ROBOT = len(CONTROLLED_JOINTS)
    MAX_ACTION_SIZE = 2 * JOINTS_PER_ROBOT + 2
    OBSERVATION_SIZE = 87

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        mode: str = "release_only",
        episode_time: float = 11.0,
        frame_skip: int = 5,
        minimum_release_time: float = 2.05,
        maximum_release_time: float = 2.45,
        reference_release_time: float = 2.26,
        release_threshold: float = 0.35,
        catch_radius: float = 0.15,
        maximum_flight_time: float = 2.50,
        throw_velocity_noise: float = 0.0,
        robot2_start_probability: float = 0.50,
        robot2_release_bonus: float = 30.0,
        wrong_arm_contact_penalty: float = 120.0,
        terminate_on_wrong_arm_contact: bool = True,
    ) -> None:
        super().__init__()

        if mode not in {"release_only", "full_exchange"}:
            raise ValueError(
                "mode must be 'release_only' or 'full_exchange'."
            )
        if episode_time <= 0.0:
            raise ValueError("episode_time must be positive.")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1.")
        if minimum_release_time >= maximum_release_time:
            raise ValueError(
                "minimum_release_time must be lower than maximum_release_time."
            )
        if maximum_flight_time <= 0.0:
            raise ValueError("maximum_flight_time must be positive.")
        if throw_velocity_noise < 0.0:
            raise ValueError("throw_velocity_noise cannot be negative.")
        if not 0.0 <= robot2_start_probability <= 1.0:
            raise ValueError(
                "robot2_start_probability must be between 0.0 and 1.0."
            )
        if robot2_release_bonus < 0.0:
            raise ValueError("robot2_release_bonus cannot be negative.")
        if wrong_arm_contact_penalty < 0.0:
            raise ValueError(
                "wrong_arm_contact_penalty cannot be negative."
            )

        self.mode = mode
        self.episode_time = float(episode_time)
        self.frame_skip = int(frame_skip)
        self.minimum_release_time = float(minimum_release_time)
        self.maximum_release_time = float(maximum_release_time)
        self.reference_release_time = float(reference_release_time)
        self.maximum_flight_time = float(maximum_flight_time)
        self.throw_velocity_noise = float(throw_velocity_noise)
        self.robot2_start_probability = float(
            robot2_start_probability
        )
        self.robot2_release_bonus = float(robot2_release_bonus)
        self.wrong_arm_contact_penalty = float(
            wrong_arm_contact_penalty
        )
        self.terminate_on_wrong_arm_contact = bool(
            terminate_on_wrong_arm_contact
        )

        selected_model_path = (
            Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        )
        self.model_path = selected_model_path.expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(self.model.opt.timestep) * self.frame_skip

        self.ball_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "ball",
        )
        self.ball_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "ball_geom",
        )
        self.ball_radius = float(self.model.geom_size[self.ball_geom_id, 0])
        # Collect all collision geometries belonging to robot 1's left arm.
        # Body names are used instead of geom names because many MuJoCo geoms
        # are unnamed while their parent bodies remain reliably named.
        self.robot_1_left_arm_geom_ids: set[int] = set()
        self.robot_1_left_arm_geom_bodies: dict[int, str] = {}
        left_arm_tokens = (
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_hand",
            "left_palm",
        )

        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name is None:
                continue

            if (
                body_name.startswith("r1_")
                and any(token in body_name for token in left_arm_tokens)
            ):
                self.robot_1_left_arm_geom_ids.add(int(geom_id))
                self.robot_1_left_arm_geom_bodies[int(geom_id)] = body_name

        if not self.robot_1_left_arm_geom_ids:
            print(
                "Warning: no robot-1 left-arm collision geometries were "
                "found. Wrong-arm contact detection is disabled."
            )

        self.pelvis_ids = {
            robot: self._required_id(
                mujoco.mjtObj.mjOBJ_BODY,
                f"r{robot}_pelvis",
            )
            for robot in (1, 2)
        }
        self.torso_ids = {
            robot: self._required_id(
                mujoco.mjtObj.mjOBJ_BODY,
                f"r{robot}_torso_link",
            )
            for robot in (1, 2)
        }

        self.floor_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_type[geom_id])
            == int(mujoco.mjtGeom.mjGEOM_PLANE)
        }

        self.joint_qpos_addresses = np.zeros(
            (2, self.JOINTS_PER_ROBOT),
            dtype=np.int32,
        )
        self.joint_dof_addresses = np.zeros_like(
            self.joint_qpos_addresses
        )
        self.actuator_ids = np.zeros_like(
            self.joint_qpos_addresses
        )

        for robot in (1, 2):
            for index, suffix in enumerate(self.CONTROLLED_JOINTS):
                full_name = f"r{robot}_{suffix}"

                joint_id = self._required_id(
                    mujoco.mjtObj.mjOBJ_JOINT,
                    full_name,
                )
                actuator_id = self._required_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    full_name,
                )

                self.joint_qpos_addresses[robot - 1, index] = int(
                    self.model.jnt_qposadr[joint_id]
                )
                self.joint_dof_addresses[robot - 1, index] = int(
                    self.model.jnt_dofadr[joint_id]
                )
                self.actuator_ids[robot - 1, index] = actuator_id

        controller_config = CatchControllerConfig(
            release_time=self.reference_release_time,
            release_mode="external",
            minimum_external_release_time=self.minimum_release_time,
            release_threshold=float(release_threshold),
            catch_radius=float(catch_radius),
            timeout=self.episode_time + 1.0,
            residual_scale=1.0,
            residual_limit=0.30,
            auto_reset_cycle=False,
            auto_reset_timeout=False,
        )

        self.controller = TwoRobotCatchController(
            self.model,
            self.data,
            config=controller_config,
            residual_joint_names=self.CONTROLLED_JOINTS,
        )

        # Both modes deliberately use the same action and observation spaces.
        # This allows a release_only policy to be loaded and continued in
        # full_exchange mode without changing the PPO network dimensions.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.MAX_ACTION_SIZE,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(
            self.MAX_ACTION_SIZE,
            dtype=np.float32,
        )
        self.nominal_pelvis_heights = np.zeros(2, dtype=np.float64)

        self.release_times = {1: np.nan, 2: np.nan}
        self.release_speeds = {1: np.nan, 2: np.nan}
        self.forced_release = {1: False, 2: False}
        self.caught = {1: False, 2: False}
        self.catch_events = {1: False, 2: False}
        self.minimum_hand_distance = {1: np.inf, 2: np.inf}
        self.starting_robot = 1
        self.wrong_arm_contact = False
        self.wrong_arm_contact_body = ""
        self.wrong_arm_contact_geom_id = -1

        self.success = False
        self.miss_reason = ""
        self.fallen = ()
        self.episode_steps = 0
        self._previous_flight_distance = np.nan

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _required_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
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

    def _ball_linear_velocity(self) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ball_body_id,
            velocity,
            0,
        )
        return velocity[3:6].copy()

    def _base_up_vector(self, robot: int) -> np.ndarray:
        rotation = self.data.xmat[
            self.torso_ids[robot]
        ].reshape(3, 3)
        return rotation[:, 2].copy()

    def _phase_elapsed(self) -> float:
        now = float(self.data.time)
        state = self.controller.state

        if state == CatchState.ROBOT_1_THROWING:
            return now - self.controller.cycle_start_time
        if state == CatchState.BALL_TO_ROBOT_2:
            return now - self.controller.flight_start[1]
        if state == CatchState.ROBOT_2_THROWING:
            return (
                now
                - self.controller.catch_time[2]
                - self.controller.config.settle_time
            )
        if state == CatchState.BALL_TO_ROBOT_1:
            return now - self.controller.flight_start[2]
        if state == CatchState.EXCHANGE_COMPLETE:
            return now - self.controller.catch_time[1]
        return 0.0

    def _current_thrower(self) -> int | None:
        if self.controller.state == CatchState.ROBOT_1_THROWING:
            return 1
        if self.controller.state == CatchState.ROBOT_2_THROWING:
            return 2
        return None

    def _current_receiver(self) -> int | None:
        if self.controller.state == CatchState.BALL_TO_ROBOT_2:
            return 2
        if self.controller.state == CatchState.BALL_TO_ROBOT_1:
            return 1
        return None

    def _active_residual_robot(self) -> int | None:
        """Return the robot whose residual action currently affects physics."""

        state = self.controller.state
        if state in {
            CatchState.ROBOT_1_THROWING,
            CatchState.BALL_TO_ROBOT_1,
        }:
            return 1
        if state in {
            CatchState.ROBOT_2_THROWING,
            CatchState.BALL_TO_ROBOT_2,
        }:
            return 2
        return None

    def _ball_robot_1_left_arm_contact(
        self,
    ) -> tuple[bool, int, str]:
        """Return whether the ball touches robot 1's left arm.

        Returns:
            ``(contact_detected, geom_id, parent_body_name)``.
        """

        if not self.robot_1_left_arm_geom_ids:
            return False, -1, ""

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)

            if geom_1 == self.ball_geom_id:
                other_geom = geom_2
            elif geom_2 == self.ball_geom_id:
                other_geom = geom_1
            else:
                continue

            if other_geom in self.robot_1_left_arm_geom_ids:
                return (
                    True,
                    other_geom,
                    self.robot_1_left_arm_geom_bodies.get(
                        other_geom,
                        "unknown",
                    ),
                )

        return False, -1, ""

    def _mark_wrong_arm_contact_if_present(self) -> bool:
        """Record an invalid return catch through robot 1's left arm.

        The check is active only while the ball is travelling from robot 2
        to robot 1. Once detected, the episode is marked as failed.
        """

        if self.controller.state != CatchState.BALL_TO_ROBOT_1:
            return False

        touched, geom_id, body_name = (
            self._ball_robot_1_left_arm_contact()
        )
        if not touched:
            return False

        self.wrong_arm_contact = True
        self.wrong_arm_contact_geom_id = int(geom_id)
        self.wrong_arm_contact_body = str(body_name)
        self.success = False
        self.miss_reason = "ball_hit_robot_1_left_arm"
        return True

    def _ball_has_floor_contact(self) -> bool:
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]

            if int(contact.geom1) == self.ball_geom_id:
                other = int(contact.geom2)
            elif int(contact.geom2) == self.ball_geom_id:
                other = int(contact.geom1)
            else:
                continue

            if not self.floor_geom_ids or other in self.floor_geom_ids:
                return True

        ball_z = float(self.data.xpos[self.ball_body_id, 2])
        return bool(ball_z <= self.ball_radius + 0.004)

    def _detect_miss(self) -> str:
        receiver = self._current_receiver()
        if receiver is None:
            return ""

        if self._ball_has_floor_contact():
            return "ball_hit_floor"

        throwing_robot = 1 if receiver == 2 else 2
        flight_elapsed = (
            float(self.data.time)
            - self.controller.flight_start[throwing_robot]
        )

        if flight_elapsed > self.maximum_flight_time:
            return "flight_timeout"

        ball_x = float(self.data.xpos[self.ball_body_id, 0])
        palm_x = float(self.controller.palm_position(receiver)[0])

        if receiver == 2 and ball_x > palm_x + 0.35:
            return "ball_passed_robot_2"
        if receiver == 1 and ball_x < palm_x - 0.35:
            return "ball_passed_robot_1"

        return ""

    # ------------------------------------------------------------------
    # Action conversion
    # ------------------------------------------------------------------

    def _canonical_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(self.action_space.shape)
        action = np.clip(action, -1.0, 1.0)

        return action.copy()

    def _residual_actions(
        self,
        canonical: np.ndarray,
    ) -> dict[int, np.ndarray] | None:
        if self.mode != "full_exchange":
            return None

        robot_1 = (
            canonical[: self.JOINTS_PER_ROBOT].astype(np.float64)
            * self.RESIDUAL_SCALES
        )
        robot_2 = (
            canonical[
                self.JOINTS_PER_ROBOT : 2 * self.JOINTS_PER_ROBOT
            ].astype(np.float64)
            * self.RESIDUAL_SCALES
        )

        # Only the robot currently throwing or receiving receives residual
        # commands. The idle robot stays on the stable reference trajectory.
        state = self.controller.state

        if state in {
            CatchState.ROBOT_1_THROWING,
            CatchState.BALL_TO_ROBOT_1,
        }:
            return {1: robot_1}

        if state in {
            CatchState.ROBOT_2_THROWING,
            CatchState.BALL_TO_ROBOT_2,
        }:
            return {2: robot_2}

        return None

    def _release_commands(
        self,
        canonical: np.ndarray,
    ) -> tuple[dict[int, float], int | None]:
        commands = {1: -1.0, 2: -1.0}
        thrower = self._current_thrower()

        if thrower is None:
            return commands, None

        signal = float(
            canonical[-2] if thrower == 1 else canonical[-1]
        )
        local_time = self._phase_elapsed()
        forced_robot: int | None = None

        if local_time < self.minimum_release_time:
            signal = -1.0
        elif local_time >= self.maximum_release_time:
            signal = 1.0
            forced_robot = thrower

        commands[thrower] = signal
        return commands, forced_robot

    def baseline_action(self) -> np.ndarray:
        """Return a zero-residual action with reference release timing."""

        action = np.zeros(
            self.action_space.shape,
            dtype=np.float32,
        )
        signal = 1.0 if self._phase_elapsed() >= self.reference_release_time else -1.0

        thrower = self._current_thrower()

        action[-2:] = -1.0
        if thrower == 1:
            action[-2] = signal
        elif thrower == 2:
            action[-1] = signal

        return action

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        requested_start = options.get("start_robot")

        if requested_start is None:
            self.starting_robot = (
                2
                if self.np_random.random()
                < self.robot2_start_probability
                else 1
            )
        else:
            try:
                self.starting_robot = int(requested_start)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "options['start_robot'] must be 1 or 2."
                ) from exc
            if self.starting_robot not in (1, 2):
                raise ValueError(
                    "options['start_robot'] must be 1 or 2."
                )

        self.controller.reset(
            reset_simulation=True,
            starting_robot=self.starting_robot,
        )
        self.controller.completed_exchanges = 0
        mujoco.mj_forward(self.model, self.data)

        self.nominal_pelvis_heights = np.array(
            [
                self.data.xpos[self.pelvis_ids[1], 2],
                self.data.xpos[self.pelvis_ids[2], 2],
            ],
            dtype=np.float64,
        )

        self.previous_action[:] = 0.0
        self.release_times = {1: np.nan, 2: np.nan}
        self.release_speeds = {1: np.nan, 2: np.nan}
        self.forced_release = {1: False, 2: False}

        # ``caught`` represents possession/progress in the observation.
        # Robot 2 already possesses the ball in a robot-2-start episode.
        self.caught = {
            1: False,
            2: bool(self.starting_robot == 2),
        }

        # ``catch_events`` records only catches that physically occurred
        # during the current episode, keeping metrics unbiased.
        self.catch_events = {1: False, 2: False}

        self.minimum_hand_distance = {1: np.inf, 2: np.inf}
        self.wrong_arm_contact = False
        self.wrong_arm_contact_body = ""
        self.wrong_arm_contact_geom_id = -1
        self.success = False
        self.miss_reason = ""
        self.fallen = ()
        self.episode_steps = 0
        self._previous_flight_distance = np.nan

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        canonical = self._canonical_action(action)
        active_residual_robot = self._active_residual_robot()

        release_event_robot: int | None = None
        catch_event_robot: int | None = None
        exchange_completed = False
        catch_progress = 0.0

        for _ in range(self.frame_skip):
            # Contacts in ``data.ncon`` were generated by the previous physics
            # step. Check them before the controller can activate grip1.
            if self._mark_wrong_arm_contact_if_present():
                if self.terminate_on_wrong_arm_contact:
                    break

            receiver_before = self._current_receiver()
            if receiver_before is not None:
                distance_before = self.controller.distance_to_hand(
                    receiver_before
                )
                self.minimum_hand_distance[receiver_before] = min(
                    self.minimum_hand_distance[receiver_before],
                    distance_before,
                )

                if np.isfinite(self._previous_flight_distance):
                    catch_progress += float(
                        np.clip(
                            self._previous_flight_distance - distance_before,
                            -0.05,
                            0.05,
                        )
                    )
                self._previous_flight_distance = distance_before
            else:
                self._previous_flight_distance = np.nan

            residual_actions = self._residual_actions(canonical)
            release_commands, forced_candidate = self._release_commands(
                canonical
            )

            thrower_before = self._current_thrower()
            local_throw_time = self._phase_elapsed()

            event = self.controller.before_step(
                residual_actions=residual_actions,
                release_commands=release_commands,
            )

            if event.released_robot is not None:
                robot = int(event.released_robot)
                release_event_robot = robot
                self.release_times[robot] = float(local_throw_time)
                self.release_speeds[robot] = float(
                    0.0 if event.release_speed is None else event.release_speed
                )

                if forced_candidate == robot:
                    self.forced_release[robot] = True

                if self.throw_velocity_noise > 0.0:
                    noise = self.np_random.uniform(
                        -self.throw_velocity_noise,
                        self.throw_velocity_noise,
                        size=3,
                    )
                    address = self.controller.ball_dof_address
                    self.data.qvel[address : address + 3] += noise

            if event.caught_robot is not None:
                robot = int(event.caught_robot)
                catch_event_robot = robot
                self.caught[robot] = True
                self.catch_events[robot] = True
                self._previous_flight_distance = np.nan

            if event.exchange_complete:
                exchange_completed = True
                self.success = True

            mujoco.mj_step(self.model, self.data)

            # Check contacts generated by the physics step that has just run.
            if (
                not self.success
                and self._mark_wrong_arm_contact_if_present()
                and self.terminate_on_wrong_arm_contact
            ):
                break

            self.fallen = self.controller.fallen_robots()
            if self.fallen:
                break

            if not self.success:
                self.miss_reason = self._detect_miss()
                if self.miss_reason:
                    break

        reward = self._compute_reward(
            canonical=canonical,
            release_event_robot=release_event_robot,
            catch_event_robot=catch_event_robot,
            exchange_completed=exchange_completed,
            catch_progress=catch_progress,
            active_residual_robot=active_residual_robot,
        )

        self.episode_steps += 1
        self.previous_action = canonical.copy()

        terminated = bool(
            self.success
            or self.fallen
            or self.miss_reason
        )
        truncated = bool(
            not terminated
            and self.data.time >= self.episode_time
        )

        if truncated:
            reward -= 40.0
            if not self.miss_reason:
                self.miss_reason = "episode_timeout"

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(),
        )

    # ------------------------------------------------------------------
    # Reward and observations
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        *,
        canonical: np.ndarray,
        release_event_robot: int | None,
        catch_event_robot: int | None,
        exchange_completed: bool,
        catch_progress: float,
        active_residual_robot: int | None,
    ) -> float:
        reward = -0.002

        upright_values = []
        height_values = []

        for robot in (1, 2):
            upright_values.append(
                max(float(self._base_up_vector(robot)[2]), 0.0)
            )

            current_height = float(
                self.data.xpos[self.pelvis_ids[robot], 2]
            )
            height_error = (
                current_height
                - self.nominal_pelvis_heights[robot - 1]
            )
            height_values.append(
                float(np.exp(-20.0 * height_error * height_error))
            )

        reward += 0.015 * float(np.mean(upright_values))
        reward += 0.010 * float(np.mean(height_values))

        reference_control = self.controller.compute_control(
            residual_actions=None
        )
        current_joint_positions = self.data.qpos[
            self.joint_qpos_addresses
        ]
        reference_positions = np.stack(
            [
                reference_control[self.actuator_ids[0]],
                reference_control[self.actuator_ids[1]],
            ]
        )
        tracking_error = float(
            np.mean(
                np.square(
                    current_joint_positions - reference_positions
                )
            )
        )
        reward += 0.015 * float(
            np.exp(-5.0 * tracking_error)
        )

        reward += 8.0 * catch_progress

        # Penalise only the residual block that actually affected physics.
        # Previously both 9-action blocks were penalised at every step, which
        # pushed robot 2's outputs toward zero even during robot-1 episodes.
        if active_residual_robot is not None:
            start = (
                (active_residual_robot - 1)
                * self.JOINTS_PER_ROBOT
            )
            stop = start + self.JOINTS_PER_ROBOT
            residual_values = canonical[start:stop]
            previous_residuals = self.previous_action[start:stop]

            residual_penalty_weight = (
                0.004 if self.mode == "full_exchange" else 0.001
            )
            rate_penalty_weight = (
                0.002 if self.mode == "full_exchange" else 0.0005
            )

            reward -= residual_penalty_weight * float(
                np.mean(np.square(residual_values))
            )
            reward -= rate_penalty_weight * float(
                np.mean(
                    np.square(
                        residual_values - previous_residuals
                    )
                )
            )

        if release_event_robot is not None:
            release_time = float(
                self.release_times[release_event_robot]
            )
            timing_error = (
                release_time - self.reference_release_time
            )
            timing_reward = float(
                np.exp(
                    -0.5
                    * (timing_error / 0.08) ** 2
                )
            )
            reward += 2.0 + 4.0 * timing_reward

            if self.forced_release[release_event_robot]:
                reward -= 8.0

            # Give the under-trained return throw an explicit milestone.
            # This is temporary curriculum shaping; the final objective still
            # remains robot 1 catching the return and completing the exchange.
            if release_event_robot == 2:
                reward += self.robot2_release_bonus

        if catch_event_robot == 2:
            reward += 60.0
        elif catch_event_robot == 1:
            reward += 90.0

        if exchange_completed:
            reward += 100.0

        if self.miss_reason == "ball_hit_robot_1_left_arm":
            reward -= self.wrong_arm_contact_penalty
        elif self.miss_reason:
            reward -= 80.0

        if self.fallen:
            reward -= 150.0

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        reference_control = self.controller.compute_control(
            residual_actions=None
        )
        reference_positions = np.stack(
            [
                reference_control[self.actuator_ids[0]],
                reference_control[self.actuator_ids[1]],
            ]
        )

        joint_position_errors = (
            self.data.qpos[self.joint_qpos_addresses]
            - reference_positions
        ) / 0.50
        joint_velocities = (
            self.data.qvel[self.joint_dof_addresses]
            / 10.0
        )

        ball_position = (
            self.data.xpos[self.ball_body_id].copy()
            / np.array([6.0, 3.0, 3.0], dtype=np.float64)
        )
        ball_velocity = self._ball_linear_velocity() / 10.0

        ball_world_position = self.data.xpos[
            self.ball_body_id
        ]
        hand_relative = np.concatenate(
            [
                (
                    ball_world_position
                    - self.controller.palm_position(1)
                )
                / 3.0,
                (
                    ball_world_position
                    - self.controller.palm_position(2)
                )
                / 3.0,
            ]
        )

        up_vectors = np.concatenate(
            [
                self._base_up_vector(1),
                self._base_up_vector(2),
            ]
        )

        height_deltas = np.array(
            [
                (
                    self.data.xpos[self.pelvis_ids[1], 2]
                    - self.nominal_pelvis_heights[0]
                )
                / 0.30,
                (
                    self.data.xpos[self.pelvis_ids[2], 2]
                    - self.nominal_pelvis_heights[1]
                )
                / 0.30,
            ],
            dtype=np.float64,
        )

        phase_one_hot = np.zeros(5, dtype=np.float64)
        phase_one_hot[int(self.controller.state)] = 1.0

        times = np.array(
            [
                float(self.data.time) / self.episode_time,
                self._phase_elapsed() / 4.0,
            ],
            dtype=np.float64,
        )

        event_flags = np.array(
            [
                float(np.isfinite(self.release_times[1])),
                float(self.caught[2]),
                float(np.isfinite(self.release_times[2])),
                float(self.caught[1]),
            ],
            dtype=np.float64,
        )

        observation = np.concatenate(
            [
                joint_position_errors.reshape(-1),
                joint_velocities.reshape(-1),
                ball_position,
                ball_velocity,
                hand_relative,
                up_vectors,
                height_deltas,
                phase_one_hot,
                times,
                event_flags,
                self.previous_action,
            ]
        )

        if observation.shape != (self.OBSERVATION_SIZE,):
            raise RuntimeError(
                "Observation construction error: "
                f"got {observation.shape}, "
                f"expected {(self.OBSERVATION_SIZE,)}."
            )

        return np.clip(
            observation,
            -20.0,
            20.0,
        ).astype(np.float32)

    def _safe_metric(self, value: float) -> float:
        return float(value) if np.isfinite(value) else float("nan")

    def _get_info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "robot2_start_probability": float(
                self.robot2_start_probability
            ),
            "state": int(self.controller.state),
            "state_name": self.controller.state.name,
            "success": bool(self.success),
            "completed_exchange": bool(self.success),
            "starting_robot": int(self.starting_robot),
            "started_with_robot_2": bool(
                self.starting_robot == 2
            ),
            "caught_by_robot_2": bool(
                self.catch_events[2]
            ),
            "caught_by_robot_1": bool(
                self.catch_events[1]
            ),
            "released_by_robot_1": bool(
                np.isfinite(self.release_times[1])
            ),
            "released_by_robot_2": bool(
                np.isfinite(self.release_times[2])
            ),
            "robot_2_return_success": bool(
                np.isfinite(self.release_times[2])
                and self.catch_events[1]
            ),
            "release_time_robot_1": self._safe_metric(
                self.release_times[1]
            ),
            "release_time_robot_2": self._safe_metric(
                self.release_times[2]
            ),
            "release_speed_robot_1": self._safe_metric(
                self.release_speeds[1]
            ),
            "release_speed_robot_2": self._safe_metric(
                self.release_speeds[2]
            ),
            "forced_release_robot_1": bool(
                self.forced_release[1]
            ),
            "forced_release_robot_2": bool(
                self.forced_release[2]
            ),
            "minimum_distance_robot_2": self._safe_metric(
                self.minimum_hand_distance[2]
            ),
            "minimum_distance_robot_1": self._safe_metric(
                self.minimum_hand_distance[1]
            ),
            "wrong_arm_contact": bool(self.wrong_arm_contact),
            "wrong_arm_contact_body": self.wrong_arm_contact_body,
            "wrong_arm_contact_geom_id": int(
                self.wrong_arm_contact_geom_id
            ),
            "wrong_arm_contact_penalty": float(
                self.wrong_arm_contact_penalty
            ),
            "miss_reason": self.miss_reason,
            "fallen_robots": tuple(self.fallen),
            "simulation_time": float(self.data.time),
            "episode_steps": int(self.episode_steps),
        }

    def close(self) -> None:
        return None


# Compatibility alias for existing training and test utilities.
TwoRobotCatchEnv = TwoRobotCatchCleanRightHandEnv
