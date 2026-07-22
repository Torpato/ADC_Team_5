from __future__ import annotations

import argparse

import mujoco
import numpy as np


MJCF_MODEL = r"""
<mujoco model="guided_humanoid_squat">

  <compiler angle="radian"/>

  <option
      timestep="0.002"
      gravity="0 0 -9.81"
      integrator="implicitfast"/>

  <default>
    <joint
        damping="5"
        armature="0.02"
        limited="true"/>

    <geom
        friction="0.9 0.005 0.0001"
        condim="3"
        density="700"/>

    <position
        kp="0"
        ctrllimited="true"/>
  </default>

  <asset>
    <material name="floor_material"
              rgba="0.75 0.75 0.75 1"/>

    <material name="pelvis_material"
              rgba="0.25 0.35 0.80 1"/>

    <material name="left_material"
              rgba="0.25 0.70 0.35 1"/>

    <material name="right_material"
              rgba="0.85 0.35 0.25 1"/>

    <material name="torso_material"
              rgba="0.35 0.45 0.85 1"/>
  </asset>

  <worldbody>

    <light
        pos="0 0 3"
        dir="0 0 -1"
        diffuse="1 1 1"/>

    <geom
        name="floor"
        type="plane"
        size="4 4 0.1"
        material="floor_material"/>

    <!--
    The pelvis is constrained to move only vertically.

    This is intentionally artificial: it creates a guided squat exercise
    without requiring a complete humanoid balance controller.
    -->
    <body name="pelvis" pos="0 0 0.80">

      <joint
          name="pelvis_vertical"
          type="slide"
          axis="0 0 1"
          range="-0.15 0.02"
          damping="80"
          armature="0.1"/>

      <geom
          name="pelvis_geom"
          type="box"
          size="0.13 0.10 0.08"
          material="pelvis_material"/>

      <body name="torso" pos="0 0 0.24">

        <geom
            name="torso_geom"
            type="box"
            size="0.14 0.09 0.20"
            material="torso_material"/>

        <body name="head" pos="0 0 0.31">
          <geom
              name="head_geom"
              type="sphere"
              size="0.09"
              material="torso_material"/>
        </body>

        <body name="left_arm" pos="0 0.18 0.13">
          <geom
              type="capsule"
              fromto="0 0 0 0 0 -0.36"
              size="0.035"
              material="left_material"/>
        </body>

        <body name="right_arm" pos="0 -0.18 0.13">
          <geom
              type="capsule"
              fromto="0 0 0 0 0 -0.36"
              size="0.035"
              material="right_material"/>
        </body>

      </body>

      <!-- LEFT LEG -->
      <body name="left_thigh" pos="0 0.09 -0.08">

        <joint
            name="left_hip_yaw"
            type="hinge"
            axis="0 0 1"
            range="-0.7 0.7"/>

        <joint
            name="left_hip_roll"
            type="hinge"
            axis="1 0 0"
            range="-0.5 0.5"/>

        <joint
            name="left_hip_pitch"
            type="hinge"
            axis="0 1 0"
            range="-1.2 0.8"/>

        <geom
            name="left_thigh_geom"
            type="capsule"
            fromto="0 0 0 0 0 -0.34"
            size="0.055"
            material="left_material"/>

        <body name="left_shin" pos="0 0 -0.34">

          <joint
              name="left_knee"
              type="hinge"
              axis="0 1 0"
              range="0 2.1"/>

          <geom
              name="left_shin_geom"
              type="capsule"
              fromto="0 0 0 0 0 -0.34"
              size="0.045"
              material="left_material"/>

          <body name="left_foot" pos="0 0 -0.34">

            <joint
                name="left_ankle_pitch"
                type="hinge"
                axis="0 1 0"
                range="-0.9 0.9"/>

            <joint
                name="left_ankle_roll"
                type="hinge"
                axis="1 0 0"
                range="-0.5 0.5"/>

            <geom
                name="left_foot_geom"
                type="box"
                pos="0.06 0 -0.025"
                size="0.12 0.055 0.025"
                material="left_material"/>

          </body>
        </body>
      </body>

      <!-- RIGHT LEG -->
      <body name="right_thigh" pos="0 -0.09 -0.08">

        <joint
            name="right_hip_yaw"
            type="hinge"
            axis="0 0 1"
            range="-0.7 0.7"/>

        <joint
            name="right_hip_roll"
            type="hinge"
            axis="1 0 0"
            range="-0.5 0.5"/>

        <joint
            name="right_hip_pitch"
            type="hinge"
            axis="0 1 0"
            range="-1.2 0.8"/>

        <geom
            name="right_thigh_geom"
            type="capsule"
            fromto="0 0 0 0 0 -0.34"
            size="0.055"
            material="right_material"/>

        <body name="right_shin" pos="0 0 -0.34">

          <joint
              name="right_knee"
              type="hinge"
              axis="0 1 0"
              range="0 2.1"/>

          <geom
              name="right_shin_geom"
              type="capsule"
              fromto="0 0 0 0 0 -0.34"
              size="0.045"
              material="right_material"/>

          <body name="right_foot" pos="0 0 -0.34">

            <joint
                name="right_ankle_pitch"
                type="hinge"
                axis="0 1 0"
                range="-0.9 0.9"/>

            <joint
                name="right_ankle_roll"
                type="hinge"
                axis="1 0 0"
                range="-0.5 0.5"/>

            <geom
                name="right_foot_geom"
                type="box"
                pos="0.06 0 -0.025"
                size="0.12 0.055 0.025"
                material="right_material"/>

          </body>
        </body>
      </body>

    </body>
  </worldbody>

  <actuator>

    <position
        name="pelvis_height_actuator"
        joint="pelvis_vertical"
        kp="2000"
        ctrlrange="-0.15 0.02"/>

    <position
        name="left_hip_yaw_actuator"
        joint="left_hip_yaw"
        ctrlrange="-0.7 0.7"/>

    <position
        name="left_hip_roll_actuator"
        joint="left_hip_roll"
        ctrlrange="-0.5 0.5"/>

    <position
        name="left_hip_pitch_actuator"
        joint="left_hip_pitch"
        kp="250"
        ctrlrange="-1.2 0.8"/>

    <position
        name="left_knee_actuator"
        joint="left_knee"
        kp="300"
        ctrlrange="0 2.1"/>

    <position
        name="left_ankle_pitch_actuator"
        joint="left_ankle_pitch"
        kp="250"
        ctrlrange="-0.9 0.9"/>

    <position
        name="left_ankle_roll_actuator"
        joint="left_ankle_roll"
        ctrlrange="-0.5 0.5"/>

    <position
        name="right_hip_yaw_actuator"
        joint="right_hip_yaw"
        ctrlrange="-0.7 0.7"/>

    <position
        name="right_hip_roll_actuator"
        joint="right_hip_roll"
        ctrlrange="-0.5 0.5"/>

    <position
        name="right_hip_pitch_actuator"
        joint="right_hip_pitch"
        kp="250"
        ctrlrange="-1.2 0.8"/>

    <position
        name="right_knee_actuator"
        joint="right_knee"
        kp="300"
        ctrlrange="0 2.1"/>

    <position
        name="right_ankle_pitch_actuator"
        joint="right_ankle_pitch"
        kp="250"
        ctrlrange="-0.9 0.9"/>

    <position
        name="right_ankle_roll_actuator"
        joint="right_ankle_roll"
        ctrlrange="-0.5 0.5"/>

  </actuator>

</mujoco>
"""


def smoothstep(value: float) -> float:
    """Cubic interpolation with zero slope at both endpoints."""
    value = float(np.clip(value, 0.0, 1.0))
    return 3.0 * value**2 - 2.0 * value**3


def interpolate(
    start: np.ndarray,
    goal: np.ndarray,
    progress: float,
) -> np.ndarray:
    blend = smoothstep(progress)
    return start + blend * (goal - start)


def get_reference(
    simulation_time: float,
    standing: np.ndarray,
    crouching: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Eight-second repeating cycle:

    0–1 s: standing
    1–3 s: move down
    3–4 s: crouching
    4–6 s: move up
    6–8 s: standing
    """
    cycle_duration = 8.0
    cycle_time = simulation_time % cycle_duration

    if cycle_time < 1.0:
        return standing.copy(), "standing"

    if cycle_time < 3.0:
        progress = (cycle_time - 1.0) / 2.0
        return (
            interpolate(standing, crouching, progress),
            "moving down",
        )

    if cycle_time < 4.0:
        return crouching.copy(), "crouching"

    if cycle_time < 6.0:
        progress = (cycle_time - 4.0) / 2.0
        return (
            interpolate(crouching, standing, progress),
            "moving up",
        )

    return standing.copy(), "standing"


def create_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(MJCF_MODEL)
    data = mujoco.MjData(model)

    # Actuator order:
    # 0  pelvis vertical
    # 1  left hip yaw
    # 2  left hip roll
    # 3  left hip pitch
    # 4  left knee
    # 5  left ankle pitch
    # 6  left ankle roll
    # 7  right hip yaw
    # 8  right hip roll
    # 9  right hip pitch
    # 10 right knee
    # 11 right ankle pitch
    # 12 right ankle roll

    standing = np.zeros(model.nu)

    crouching = np.array(
        [
            -0.068,  # pelvis moves downward

            0.0,     # left hip yaw
            0.0,     # left hip roll
            -0.45,   # left hip pitch
            0.90,    # left knee
            -0.45,   # left ankle pitch
            0.0,     # left ankle roll

            0.0,     # right hip yaw
            0.0,     # right hip roll
            -0.45,   # right hip pitch
            0.90,    # right knee
            -0.45,   # right ankle pitch
            0.0,     # right ankle roll
        ],
        dtype=float,
    )

    data.qpos[:] = standing
    data.ctrl[:] = standing

    mujoco.mj_forward(model, data)

    return model, data


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
) -> None:
    standing = np.zeros(model.nu)

    crouching = np.array(
        [
            -0.068,
            0.0,
            0.0,
            -0.45,
            0.90,
            -0.45,
            0.0,
            0.0,
            0.0,
            -0.45,
            0.90,
            -0.45,
            0.0,
        ],
        dtype=float,
    )

    next_report_time = 0.0

    while data.time < duration:
        reference, phase = get_reference(
            data.time,
            standing,
            crouching,
        )

        data.ctrl[:] = reference
        mujoco.mj_step(model, data)

        if data.time >= next_report_time:
            pelvis_height = data.xpos[
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "pelvis",
                ),
                2,
            ]

            print(
                f"t={data.time:5.2f} s  "
                f"phase={phase:12s}  "
                f"pelvis_z={pelvis_height:.3f} m  "
                f"knee={data.qpos[4]:.3f} rad"
            )

            next_report_time += 0.25

    print(f"\nSimulation completed at t={data.time:.2f} s")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    from mujoco import viewer

    standing = np.zeros(model.nu)

    crouching = np.array(
        [
            -0.068,
            0.0,
            0.0,
            -0.45,
            0.90,
            -0.45,
            0.0,
            0.0,
            0.0,
            -0.45,
            0.90,
            -0.45,
            0.0,
        ],
        dtype=float,
    )

    last_phase = ""

    def control_callback(
        callback_model: mujoco.MjModel,
        callback_data: mujoco.MjData,
    ) -> None:
        nonlocal last_phase

        reference, phase = get_reference(
            callback_data.time,
            standing,
            crouching,
        )

        callback_data.ctrl[:] = reference

        if phase != last_phase:
            print(
                f"t={callback_data.time:5.2f} s: {phase}"
            )
            last_phase = phase

    mujoco.set_mjcb_control(control_callback)

    print("Opening MuJoCo viewer.")
    print("Press Space if the simulation is paused.")
    print("Close the viewer window to finish.")

    try:
        viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the viewer.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Headless simulation duration.",
    )

    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    model, data = create_model()

    if args.headless:
        run_headless(
            model,
            data,
            duration=args.duration,
        )
    else:
        run_viewer(model, data)


if __name__ == "__main__":
    main()