
from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from unitree_env import UnitreeDropEnv


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize a trained PPO "
            "throwing policy."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "release_only",
            "residual",
        ),
        default="release_only",
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--target-x",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--target-y",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--target-radius",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def find_policy(
    *,
    selected_policy: Path | None,
    mode: str,
    run_name: str | None,
) -> Path:
    """Find the best or final trained model."""

    if selected_policy is not None:
        policy_path = (
            selected_policy
            .expanduser()
            .resolve()
        )

        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Policy not found: {policy_path}"
            )

        return policy_path

    selected_run_name = (
        run_name
        if run_name
        else f"ppo_safe_{mode}"
    )

    best_model = (
        PROJECT_ROOT
        / "runs"
        / selected_run_name
        / "best"
        / "best_model.zip"
    )

    final_model = (
        PROJECT_ROOT
        / "runs"
        / selected_run_name
        / "final_model.zip"
    )

    if best_model.is_file():
        return best_model.resolve()

    if final_model.is_file():
        return final_model.resolve()

    raise FileNotFoundError(
        "No trained model was found.\n"
        f"Expected:\n"
        f"  {best_model}\n"
        f"or:\n"
        f"  {final_model}"
    )


def format_metric(
    value: float,
    decimals: int = 3,
) -> str:
    """Format metrics that may contain NaN."""

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def main() -> None:
    args = parse_arguments()

    policy_path = find_policy(
        selected_policy=args.policy,
        mode=args.mode,
        run_name=args.run_name,
    )

    environment = UnitreeDropEnv(
        mode=args.mode,
        target_x=args.target_x,
        target_y=args.target_y,
        target_radius=args.target_radius,
        frame_skip=5,
        minimum_release_time=2.10,
        maximum_release_time=2.45,
        reference_release_time=2.30,
    )

    print("Loading policy:")
    print(policy_path)

    model = PPO.load(
        str(policy_path),
        env=environment,
    )

    observation, _ = (
        environment.reset()
    )

    episode_number = 1
    episode_reward = 0.0

    print("Opening the MuJoCo viewer...")

    with mujoco.viewer.launch_passive(
        environment.model,
        environment.data,
    ) as viewer:

        while viewer.is_running():
            step_start = (
                time.perf_counter()
            )

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(
                action
            )

            episode_reward += reward

            viewer.sync()

            elapsed = (
                time.perf_counter()
                - step_start
            )

            remaining = (
                environment.control_dt
                - elapsed
            )

            if remaining > 0:
                time.sleep(
                    remaining
                )

            if terminated or truncated:
                print(
                    f"\nEpisode {episode_number}"
                )

                print(
                    "  Reward:",
                    format_metric(
                        episode_reward,
                        2,
                    ),
                )

                print(
                    "  Success:",
                    info["success"],
                )

                print(
                    "  Released:",
                    info["released"],
                )

                print(
                    "  Forced release:",
                    info["forced_release"],
                )

                print(
                    "  Release time:",
                    format_metric(
                        info["release_time"]
                    ),
                    "s",
                )

                print(
                    "  Release speed:",
                    format_metric(
                        info["release_speed"]
                    ),
                    "m/s",
                )

                print(
                    "  Impact error:",
                    format_metric(
                        info["impact_error"]
                    ),
                    "m",
                )

                print(
                    "  Unsafe posture:",
                    info["unsafe_posture"],
                )

                time.sleep(
                    max(
                        args.pause,
                        0.0,
                    )
                )

                observation, _ = (
                    environment.reset()
                )

                episode_number += 1
                episode_reward = 0.0

    environment.close()


if __name__ == "__main__":
    main()