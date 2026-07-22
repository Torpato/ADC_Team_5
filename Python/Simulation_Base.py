from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from pathlib import Path

import mujoco

from Logging import JointCsvLogger


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
    

'''
def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
) -> None:
    if duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    while data.time < duration:
        mujoco.mj_step(model, data)

    print(f"\nSimulation completed at t={data.time:.2f} s")'''


def launch_standalone_viewer(model_path: str) -> None:
    """Launch MuJoCo's standalone viewer in a separate process."""
    command = [
        sys.executable,
        "-m",
        "mujoco.viewer",
        f"--mjcf={model_path}",
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
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the MJCF/XML model.",
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the standalone MuJoCo viewer while the simulation runs.",
    )

    parser.add_argument(
        "--log-output",
        type=str,
        default="Joint_Log_Episode_1.csv",
        help="Base CSV file path for joint data logging.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log joint data once every N simulation steps.",
    )

    args = parser.parse_args()

    print("Loading model...", flush=True)
    model_path = args.model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print("Resetting standing pose...", flush=True)
    # reset_pose(model, data, keyframe="stand", floating=False)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print_model_summary(model)

    viewer_thread = None
    if args.viewer:
        print("Launching standalone MuJoCo viewer in the background...", flush=True)
        viewer_thread = threading.Thread(
            target=launch_standalone_viewer,
            args=(args.model,),
            daemon=True,
        )
        viewer_thread.start()

    if args.log_output:
        logger = JointCsvLogger(Path(args.log_output), log_every=args.log_every)
        try:
            while data.time < args.duration:
                mujoco.mj_step(model, data)
                logger.log_step(data)
        finally:
            logger.close()

    if args.viewer and viewer_thread is not None:
        viewer_thread.join()


if __name__ == "__main__":
    main()
