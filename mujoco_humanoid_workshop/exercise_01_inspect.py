from __future__ import annotations

import argparse
import subprocess
import sys

import mujoco

from common import MODEL_PATH, load_model, reset_pose


def safe_name(name: str | None, fallback: str) -> str:
    """Return a printable object name even if the MJCF object is unnamed."""
    return name if name is not None else fallback


def print_model_summary(model: mujoco.MjModel) -> None:
    print("\nMODEL SUMMARY")
    print(f"Bodies:               {model.nbody}")
    print(f"Joints:               {model.njnt}")
    print(f"Position coordinates: {model.nq}")
    print(f"Velocity coordinates: {model.nv}")
    print(f"Actuators:            {model.nu}")
    print(f"Sensors:              {model.nsensor}")
    print(f"Equality constraints: {model.neq}")

    print("\nJOINT TABLE")
    print("id  name                     qpos address  dof address  range")
    for joint_id in range(model.njnt):
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        name = safe_name(raw_name, f"<joint_{joint_id}>")
        lower, upper = model.jnt_range[joint_id]

        print(
            f"{joint_id:2d}  {name:24s} "
            f"{int(model.jnt_qposadr[joint_id]):12d} "
            f"{int(model.jnt_dofadr[joint_id]):11d} "
            f"[{float(lower): .2f}, {float(upper): .2f}]"
        )

    print("\nACTUATORS")
    for actuator_id in range(model.nu):
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        name = safe_name(raw_name, f"<actuator_{actuator_id}>")
        print(f"{actuator_id:2d}: {name}")

    print("\nSENSORS")
    for sensor_id in range(model.nsensor):
        raw_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
        )
        name = safe_name(raw_name, f"<sensor_{sensor_id}>")

        print(
            f"{sensor_id:2d}: {name:24s} "
            f"dimension={int(model.sensor_dim[sensor_id])} "
            f"address={int(model.sensor_adr[sensor_id])}"
        )

    print("\n Joints")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        print(joint_id, name)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
) -> None:
    if duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    while data.time < duration:
        mujoco.mj_step(model, data)

    print(f"\nSimulation completed at t={data.time:.2f} s")


def launch_standalone_viewer() -> None:
    """Launch MuJoCo's viewer in a separate process.

    This intentionally avoids viewer.launch_passive(), whose additional
    viewer thread can expose GLFW/OpenGL problems as a native segmentation
    fault on some systems.
    """
    command = [
        sys.executable,
        "-m",
        "mujoco.viewer",
        f"--mjcf={MODEL_PATH}",
    ]

    print("\nOpening the standalone MuJoCo viewer.")
    print("Close the viewer window to return to the terminal.")

    completed = subprocess.run(command, check=False)

    if completed.returncode != 0:
        raise RuntimeError(
            "The standalone viewer exited abnormally with return code "
            f"{completed.returncode}. The model and headless physics may "
            "still be valid; this points to the local graphics/viewer stack."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics without opening a graphical viewer.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Headless simulation duration in seconds.",
    )
    args = parser.parse_args()

    print("Loading model...", flush=True)
    model, data = load_model()

    print("Resetting standing pose...", flush=True)
    reset_pose(model, data, keyframe="stand", floating=False)

    print_model_summary(model)

    if args.headless:
        run_headless(model, data, args.duration)
        return

    launch_standalone_viewer()


if __name__ == "__main__":
    main()
