from __future__ import annotations

import argparse

import mujoco

from common import (
    has_fallen,
    load_model,
    object_id,
    reset_pose,
    set_contact_friction,
    torso_tilt_degrees,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--friction",
        type=float,
        default=0.9,
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
        "--headless",
        action="store_true",
    )

    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError(
            "--duration must be greater than zero."
        )

    if args.friction < 0:
        raise ValueError(
            "--friction must be non-negative."
        )

    if args.push_start < 0:
        raise ValueError(
            "--push-start must be non-negative."
        )

    if args.push_duration <= 0:
        raise ValueError(
            "--push-duration must be greater than zero."
        )

    model, data = load_model()

    set_contact_friction(
        model,
        args.friction,
    )

    reset_pose(
        model,
        data,
        keyframe="stand",
        floating=True,
    )

    torso_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "torso",
    )

    pelvis_id = object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )

    maximum_tilt = 0.0

    minimum_pelvis_height = float(
        data.xpos[pelvis_id, 2]
    )

    fall_time: float | None = None
    experiment_finished = False
    completion_message_printed = False

    def update_measurements(
        current_data: mujoco.MjData,
    ) -> None:
        nonlocal maximum_tilt
        nonlocal minimum_pelvis_height
        nonlocal fall_time

        current_tilt = torso_tilt_degrees(
            model,
            current_data,
        )

        current_pelvis_height = float(
            current_data.xpos[pelvis_id, 2]
        )

        maximum_tilt = max(
            maximum_tilt,
            current_tilt,
        )

        minimum_pelvis_height = min(
            minimum_pelvis_height,
            current_pelvis_height,
        )

        if (
            fall_time is None
            and has_fallen(model, current_data)
        ):
            fall_time = float(
                current_data.time
            )

    def apply_push(
        current_data: mujoco.MjData,
    ) -> None:
        current_data.xfrc_applied[:] = 0.0

        push_active = (
            args.push_start
            <= current_data.time
            < args.push_start + args.push_duration
        )

        if push_active:
            current_data.xfrc_applied[
                torso_id,
                0,
            ] = args.force

    def run_headless() -> None:
        nonlocal experiment_finished

        while data.time < args.duration:
            apply_push(data)

            mujoco.mj_step(
                model,
                data,
            )

            update_measurements(data)

            if fall_time is not None:
                break

        data.xfrc_applied[:] = 0.0
        experiment_finished = True

    def run_managed_viewer() -> None:
        nonlocal experiment_finished
        nonlocal completion_message_printed

        from mujoco import viewer

        def control_callback(
            callback_model: mujoco.MjModel,
            callback_data: mujoco.MjData,
        ) -> None:
            nonlocal experiment_finished
            nonlocal completion_message_printed

            del callback_model

            update_measurements(
                callback_data
            )

            if callback_data.time < args.duration:
                apply_push(
                    callback_data
                )
            else:
                callback_data.xfrc_applied[:] = 0.0
                experiment_finished = True

            if fall_time is not None:
                callback_data.xfrc_applied[:] = 0.0
                experiment_finished = True

            if (
                experiment_finished
                and not completion_message_printed
            ):
                completion_message_printed = True

                print(
                    "\nExperiment completed."
                )

                print(
                    "Close the viewer window "
                    "to print the final result."
                )

        print(
            "\nOpening the MuJoCo managed viewer."
        )

        print(
            "Press Space if the simulation is paused."
        )

        print(
            "Contact visualization can be enabled "
            "from the viewer's Rendering panel."
        )

        print(
            "Close the viewer window after the test."
        )

        mujoco.set_mjcb_control(
            control_callback
        )

        try:
            viewer.launch(
                model,
                data,
            )
        finally:
            mujoco.set_mjcb_control(
                None
            )

            data.xfrc_applied[:] = 0.0

    if args.headless:
        run_headless()
    else:
        run_managed_viewer()

    update_measurements(data)

    print(
        "\nPUSH-TEST RESULT"
    )

    print(
        f"Applied force:          "
        f"{args.force:.1f} N"
    )

    print(
        f"Push impulse:           "
        f"{args.force * args.push_duration:.2f} N s"
    )

    print(
        f"Friction coefficient:   "
        f"{args.friction:.3f}"
    )

    print(
        f"Maximum torso tilt:     "
        f"{maximum_tilt:.2f} deg"
    )

    print(
        f"Minimum pelvis height:  "
        f"{minimum_pelvis_height:.3f} m"
    )

    print(
        f"Fallen:                 "
        f"{'yes' if fall_time is not None else 'no'}"
    )

    if fall_time is not None:
        print(
            f"Detected fall at:       "
            f"{fall_time:.3f} s"
        )


if __name__ == "__main__":
    main()