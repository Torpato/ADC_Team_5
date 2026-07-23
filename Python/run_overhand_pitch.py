
from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from overhand_pitch_controller import (
    OverhandPitchController,
)


# Find the project root automatically.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "Model"
    / "g1_ball.xml"
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the final reusable G1 "
            "overhand throw controller."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to Model/g1_ball.xml.",
    )

    parser.add_argument(
        "--release-time",
        type=float,
        default=2.30,
        help=(
            "Fixed ball-release time "
            "in simulation seconds."
        ),
    )

    parser.add_argument(
        "--reset-time",
        type=float,
        default=5.00,
        help=(
            "Episode duration before "
            "resetting the robot."
        ),
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without opening "
            "the MuJoCo viewer."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "Number of episodes "
            "in headless mode."
        ),
    )

    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help=(
            "Do not sleep to maintain "
            "real-time speed."
        ),
    )

    return parser.parse_args()


def load_model(
    model_path: Path,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the MuJoCo model and create MjData."""

    resolved_path = (
        model_path
        .expanduser()
        .resolve()
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "MuJoCo model not found: "
            f"{resolved_path}"
        )

    model = mujoco.MjModel.from_xml_path(
        str(resolved_path)
    )

    data = mujoco.MjData(model)

    return model, data


def report_release(
    episode: int,
    release_time: float | None,
    release_speed: float | None,
) -> None:
    """Print the release information."""

    if (
        release_time is None
        or release_speed is None
    ):
        return

    print(
        f"Episode {episode}: "
        f"released at {release_time:.3f} s "
        f"with speed {release_speed:.2f} m/s"
    )


def run_with_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: OverhandPitchController,
    *,
    reset_time: float,
    realtime: bool,
) -> None:
    """Run continuously using the MuJoCo viewer."""

    episode = 1

    controller.reset()

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while viewer.is_running():
            step_start = time.perf_counter()

            # Start another throw.
            if data.time >= reset_time:
                episode += 1
                controller.reset()

            # Calculate the scripted movement and
            # release the ball when required.
            event = controller.before_step()

            if event.released_this_step:
                report_release(
                    episode,
                    event.release_time,
                    event.release_speed,
                )

            # Advance MuJoCo physics.
            mujoco.mj_step(
                model,
                data,
            )

            # Update the viewer.
            viewer.sync()

            # Keep approximately real-time speed.
            if realtime:
                elapsed = (
                    time.perf_counter()
                    - step_start
                )

                remaining = (
                    model.opt.timestep
                    - elapsed
                )

                if remaining > 0:
                    time.sleep(remaining)


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: OverhandPitchController,
    *,
    reset_time: float,
    episodes: int,
) -> None:
    """Run without opening the viewer."""

    if episodes < 1:
        raise ValueError(
            "--episodes must be at least 1."
        )

    for episode in range(
        1,
        episodes + 1,
    ):
        controller.reset()

        while data.time < reset_time:
            event = controller.before_step()

            if event.released_this_step:
                report_release(
                    episode,
                    event.release_time,
                    event.release_speed,
                )

            mujoco.mj_step(
                model,
                data,
            )

        if not controller.released:
            print(
                f"Episode {episode}: "
                "warning — ball was not released."
            )


def main() -> None:
    """Program entry point."""

    args = parse_arguments()

    if args.release_time <= 0:
        raise ValueError(
            "--release-time must be positive."
        )

    if args.reset_time <= args.release_time:
        raise ValueError(
            "--reset-time must be greater "
            "than --release-time."
        )

    model, data = load_model(
        args.model
    )

    controller = OverhandPitchController(
        model,
        data,
        release_time=args.release_time,
        release_mode="fixed",
    )

    print(
        "Model:",
        args.model.expanduser().resolve(),
    )

    print(
        f"Release time: "
        f"{args.release_time:.3f} s | "
        f"Reset time: "
        f"{args.reset_time:.3f} s"
    )

    if args.headless:
        run_headless(
            model,
            data,
            controller,
            reset_time=args.reset_time,
            episodes=args.episodes,
        )
    else:
        run_with_viewer(
            model,
            data,
            controller,
            reset_time=args.reset_time,
            realtime=not args.no_realtime,
        )


if __name__ == "__main__":
    main()