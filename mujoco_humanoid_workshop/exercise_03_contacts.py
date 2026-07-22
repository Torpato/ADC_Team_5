from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "simple_humanoid.xml"
)


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise ValueError(
            f"Object '{name}' was not found in the MuJoCo model."
        )

    return object_id


def read_sensor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_name: str,
) -> np.ndarray:
    sensor_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        sensor_name,
    )

    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])

    return data.sensordata[
        address:address + dimension
    ].copy()


def set_geom_friction(
    model: mujoco.MjModel,
    geom_name: str,
    sliding_friction: float,
) -> None:
    geom_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    model.geom_friction[geom_id] = np.array(
        [
            sliding_friction,
            0.005,
            0.0001,
        ],
        dtype=float,
    )


def configure_friction(
    model: mujoco.MjModel,
    sliding_friction: float,
) -> None:
    if sliding_friction < 0:
        raise ValueError(
            "The friction coefficient must be non-negative."
        )

    for geom_name in (
        "floor",
        "left_foot_geom",
        "right_foot_geom",
    ):
        set_geom_friction(
            model,
            geom_name,
            sliding_friction,
        )


def reset_standing_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    keyframe_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "stand",
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        keyframe_id,
    )

    # Release the pelvis from the world.
    weld_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        "pelvis_weld",
    )

    if weld_id >= 0:
        data.eq_active[weld_id] = 0

    data.xfrc_applied[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def torso_tilt_degrees(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    torso_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "torso",
    )

    rotation = data.xmat[torso_id].reshape(3, 3)

    torso_up_z = float(
        np.clip(
            rotation[2, 2],
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(torso_up_z)
        )
    )


class ContactExperiment:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        friction: float,
        lateral_force: float,
        duration: float,
        push_start: float,
        push_duration: float,
        report_period: float,
    ) -> None:
        self.model = model
        self.data = data

        self.friction = friction
        self.lateral_force = lateral_force
        self.duration = duration
        self.push_start = push_start
        self.push_duration = push_duration
        self.report_period = report_period

        self.torso_id = get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso",
        )

        self.pelvis_id = get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "pelvis",
        )

        self.left_foot_id = get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_foot",
        )

        self.right_foot_id = get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_foot",
        )

        self.initial_left_position = (
            data.xpos[self.left_foot_id].copy()
        )

        self.initial_right_position = (
            data.xpos[self.right_foot_id].copy()
        )

        self.next_report_time = 0.0

        self.maximum_left_force = 0.0
        self.maximum_right_force = 0.0
        self.maximum_tilt = 0.0

        self.minimum_pelvis_height = float(
            data.xpos[self.pelvis_id, 2]
        )

        self.experiment_finished = False
        self.result_printed = False

    def apply_force(self) -> None:
        self.data.xfrc_applied[:] = 0.0

        push_active = (
            self.push_start
            <= self.data.time
            < self.push_start + self.push_duration
        )

        if push_active:
            self.data.xfrc_applied[
                self.torso_id,
                1,
            ] = self.lateral_force

    def update_measurements(self) -> None:
        if self.data.time > self.duration:
            return

        left_force = float(
            read_sensor(
                self.model,
                self.data,
                "left_foot_force",
            )[0]
        )

        right_force = float(
            read_sensor(
                self.model,
                self.data,
                "right_foot_force",
            )[0]
        )

        tilt = torso_tilt_degrees(
            self.model,
            self.data,
        )

        pelvis_height = float(
            self.data.xpos[self.pelvis_id, 2]
        )

        self.maximum_left_force = max(
            self.maximum_left_force,
            left_force,
        )

        self.maximum_right_force = max(
            self.maximum_right_force,
            right_force,
        )

        self.maximum_tilt = max(
            self.maximum_tilt,
            tilt,
        )

        self.minimum_pelvis_height = min(
            self.minimum_pelvis_height,
            pelvis_height,
        )

        if self.data.time >= self.next_report_time:
            centre_of_mass = read_sensor(
                self.model,
                self.data,
                "humanoid_com",
            )

            print(
                f"{self.data.time:5.2f} s  "
                f"{self.data.ncon:8d}   "
                f"{left_force:9.2f} N  "
                f"{right_force:10.2f} N  "
                f"{centre_of_mass[0]:8.3f}  "
                f"{centre_of_mass[1]:8.3f}  "
                f"{pelvis_height:8.3f}  "
                f"{tilt:6.2f} deg"
            )

            self.next_report_time += (
                self.report_period
            )

    def simulation_callback(
        self,
        callback_model: mujoco.MjModel,
        callback_data: mujoco.MjData,
    ) -> None:
        del callback_model
        del callback_data

        if self.data.time < self.duration:
            self.apply_force()
            self.update_measurements()
        else:
            self.data.xfrc_applied[:] = 0.0

            if not self.experiment_finished:
                self.experiment_finished = True

                print(
                    "\nExperiment completed. "
                    "Close the viewer window to finish."
                )

    def print_header(self) -> None:
        print(
            "\nCONTACT AND FRICTION EXPERIMENT"
        )

        print(
            f"Friction coefficient: {self.friction:.3f}"
        )

        print(
            f"Lateral force:        "
            f"{self.lateral_force:.1f} N"
        )

        print(
            f"Push interval:        "
            f"{self.push_start:.2f}-"
            f"{self.push_start + self.push_duration:.2f} s"
        )

        print(
            "\n"
            "time     contacts   left force   right force  "
            "CoM x      CoM y      pelvis z   tilt"
        )

    def print_result(self) -> None:
        if self.result_printed:
            return

        self.result_printed = True

        final_left_position = (
            self.data.xpos[self.left_foot_id].copy()
        )

        final_right_position = (
            self.data.xpos[self.right_foot_id].copy()
        )

        left_displacement = float(
            np.linalg.norm(
                final_left_position[:2]
                - self.initial_left_position[:2]
            )
        )

        right_displacement = float(
            np.linalg.norm(
                final_right_position[:2]
                - self.initial_right_position[:2]
            )
        )

        final_left_force = float(
            read_sensor(
                self.model,
                self.data,
                "left_foot_force",
            )[0]
        )

        final_right_force = float(
            read_sensor(
                self.model,
                self.data,
                "right_foot_force",
            )[0]
        )

        final_centre_of_mass = read_sensor(
            self.model,
            self.data,
            "humanoid_com",
        )

        print("\nRESULT")

        print(
            f"Simulation time:          "
            f"{min(self.data.time, self.duration):.3f} s"
        )

        print(
            f"Friction coefficient:     "
            f"{self.friction:.3f}"
        )

        print(
            f"Push impulse:             "
            f"{self.lateral_force * self.push_duration:.3f} N s"
        )

        print(
            f"Maximum left-foot force:  "
            f"{self.maximum_left_force:.2f} N"
        )

        print(
            f"Maximum right-foot force: "
            f"{self.maximum_right_force:.2f} N"
        )

        print(
            f"Final left-foot force:    "
            f"{final_left_force:.2f} N"
        )

        print(
            f"Final right-foot force:   "
            f"{final_right_force:.2f} N"
        )

        print(
            f"Left-foot displacement:   "
            f"{left_displacement:.4f} m"
        )

        print(
            f"Right-foot displacement:  "
            f"{right_displacement:.4f} m"
        )

        print(
            f"Maximum torso tilt:       "
            f"{self.maximum_tilt:.2f} degrees"
        )

        print(
            f"Minimum pelvis height:    "
            f"{self.minimum_pelvis_height:.3f} m"
        )

        print(
            f"Final centre of mass:     "
            f"[{final_centre_of_mass[0]:.3f}, "
            f"{final_centre_of_mass[1]:.3f}, "
            f"{final_centre_of_mass[2]:.3f}]"
        )


def run_headless(
    experiment: ContactExperiment,
) -> None:
    while experiment.data.time < experiment.duration:
        experiment.apply_force()

        mujoco.mj_step(
            experiment.model,
            experiment.data,
        )

        experiment.update_measurements()

    experiment.data.xfrc_applied[:] = 0.0
    experiment.print_result()


def run_managed_viewer(
    experiment: ContactExperiment,
) -> None:
    from mujoco import viewer

    print("\nOpening the MuJoCo viewer.")
    print("Press Space if the simulation is paused.")
    print(
        "Open the Rendering panel to enable "
        "Contact Point and Contact Force visualization."
    )
    print("Close the viewer window after the experiment.")

    mujoco.set_mjcb_control(
        experiment.simulation_callback
    )

    try:
        viewer.launch(
            experiment.model,
            experiment.data,
        )
    finally:
        mujoco.set_mjcb_control(None)

    experiment.print_result()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--friction",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--lateral-force",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--push-start",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--push-duration",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--report-period",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--headless",
        action="store_true",
    )

    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError(
            "--duration must be greater than zero."
        )

    if args.push_start < 0:
        raise ValueError(
            "--push-start must be non-negative."
        )

    if args.push_duration <= 0:
        raise ValueError(
            "--push-duration must be greater than zero."
        )

    if args.report_period <= 0:
        raise ValueError(
            "--report-period must be greater than zero."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MuJoCo model not found: {MODEL_PATH}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    configure_friction(
        model,
        args.friction,
    )

    reset_standing_pose(
        model,
        data,
    )

    experiment = ContactExperiment(
        model=model,
        data=data,
        friction=args.friction,
        lateral_force=args.lateral_force,
        duration=args.duration,
        push_start=args.push_start,
        push_duration=args.push_duration,
        report_period=args.report_period,
    )

    experiment.print_header()

    if args.headless:
        run_headless(experiment)
    else:
        run_managed_viewer(experiment)


if __name__ == "__main__":
    main()