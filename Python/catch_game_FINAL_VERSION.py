from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

# Standard repository location first; the script directory is also accepted
# so both downloaded files can be tested together before being moved.
for candidate in (PROJECT_ROOT / "Python", SCRIPT_PATH.parent):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from catch_controller_FINAL_VERSION import (
        CatchControllerConfig,
        CatchControllerEvent,
        TwoRobotCatchController,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import Python/catch_controller_FINAL_VERSION.py. "
        "Place the controller in the project's Python directory."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final two-robot MuJoCo catch demonstration."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "Model" / "world_catch.xml",
        help="Path to world_catch.xml.",
    )
    parser.add_argument(
        "--camera",
        default="demo",
        help="Fixed MuJoCo camera name (demo, lado, alto, r1 or r2).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of complete exchanges in headless mode; 0 means unlimited.",
    )
    parser.add_argument(
        "--release-mode",
        choices=("fixed", "external"),
        default="fixed",
        help="Use fixed baseline release timing or external release commands.",
    )
    parser.add_argument(
        "--release-time",
        type=float,
        default=2.26,
        help="Local release time for each deterministic throw.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Do not sleep to match real time.",
    )
    return parser.parse_args()


def load_simulation(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    return model, data


def sleep_to_realtime(step_started: float, timestep: float) -> None:
    remaining = timestep - (time.perf_counter() - step_started)
    if remaining > 0.0:
        time.sleep(remaining)


def print_event(event: CatchControllerEvent) -> None:
    if event.released_robot is not None:
        speed = 0.0 if event.release_speed is None else event.release_speed
        if event.released_robot == 1:
            print(f"Robot 1 throws at {speed:.2f} m/s")
        else:
            print(f"Robot 2 returns the ball at {speed:.2f} m/s")

    if event.caught_robot == 2:
        print("Robot 2 catches the ball")
    elif event.caught_robot == 1:
        print("Robot 1 catches the ball -- complete exchange\n")

    if event.timeout_reset:
        print("Safety reset: the exchange exceeded the timeout.")
    elif event.manual_reset_detected:
        print("Viewer reset detected: controller state restored.")

    if event.fallen_robots:
        robots = ", ".join(str(robot) for robot in event.fallen_robots)
        print(f"Warning: fall condition detected for robot(s): {robots}")


def set_fixed_camera(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    camera_name: str,
) -> None:
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )
    if camera_id < 0:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(model.ncam)
        ]
        raise ValueError(
            f"Unknown camera {camera_name!r}. Available cameras: {available}"
        )

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def run_visual(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: TwoRobotCatchController,
    *,
    camera_name: str,
    realtime: bool,
) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_fixed_camera(viewer, model, camera_name)

        while viewer.is_running():
            step_started = time.perf_counter()

            # This deterministic demonstration supplies no residual actions.
            # A Gymnasium environment can pass one vector per robot here.
            event = controller.before_step(
                residual_actions=None,
                release_commands=None,
            )
            print_event(event)

            mujoco.mj_step(model, data)
            viewer.sync()

            if realtime:
                sleep_to_realtime(step_started, model.opt.timestep)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: TwoRobotCatchController,
    *,
    cycles: int,
    realtime: bool,
) -> None:
    initial_count = controller.completed_exchanges
    target_count = None if cycles == 0 else initial_count + cycles

    while target_count is None or controller.completed_exchanges < target_count:
        step_started = time.perf_counter()

        event = controller.before_step(
            residual_actions=None,
            release_commands=None,
        )
        print_event(event)
        mujoco.mj_step(model, data)

        if realtime:
            sleep_to_realtime(step_started, model.opt.timestep)

    info = controller.get_training_info()
    print(
        "Headless baseline finished: "
        f"{info['completed_exchanges']} complete exchange(s)."
    )


def main() -> None:
    args = parse_arguments()
    if args.cycles < 0:
        raise ValueError("--cycles must be zero or a positive integer.")

    model, data = load_simulation(args.model)
    config = CatchControllerConfig(
        release_time=args.release_time,
        release_mode=args.release_mode,
        # The visual demo repeats automatically.  A future RL environment
        # should normally set both auto-reset flags to False and terminate the
        # episode itself after a success, timeout or fall.
        auto_reset_cycle=True,
        auto_reset_timeout=True,
    )
    controller = TwoRobotCatchController(model, data, config=config)
    controller.reset()

    if args.release_mode == "external":
        raise ValueError(
            "The demo runner does not generate external release commands. "
            "Use --release-mode fixed here; external mode is intended for the "
            "future Gymnasium/PPO environment."
        )

    if args.headless:
        run_headless(
            model,
            data,
            controller,
            cycles=args.cycles,
            realtime=not args.no_realtime,
        )
    else:
        run_visual(
            model,
            data,
            controller,
            camera_name=args.camera,
            realtime=not args.no_realtime,
        )


if __name__ == "__main__":
    main()