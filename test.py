"""Visualize a trained PPO policy with MuJoCo.

On macOS:

    mjpython Python/test.py

Use the final model:

    mjpython Python/test.py \
        --policy runs/ppo_release_only/final_model.zip

Residual policy:

    mjpython Python/test.py \
        --mode residual \
        --policy runs/ppo_residual/best/best_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
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
        "--policy",
        type=Path,
        default=None,
        help=(
            "Path to the PPO .zip file. "
            "When omitted, the script first "
            "tries best_model.zip and then "
            "final_model.zip."
        ),
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
        help=(
            "Pause between episodes "
            "in seconds."
        ),
    )

    return parser.parse_args()


def find_policy(
    selected_policy: Path | None,
    mode: str,
) -> Path:
    """Find the policy file that should be loaded."""

    if selected_policy is not None:
        policy_path = (
            selected_policy
            .expanduser()
            .resolve()
        )

        if not policy_path.is_file():
            raise FileNotFoundError(
                "Trained policy not found: "
                f"{policy_path}"
            )

        return policy_path

    run_name = f"ppo_{mode}"

    best_policy = (
        PROJECT_ROOT
        / "runs"
        / run_name
        / "best"
        / "best_model.zip"
    )

    final_policy = (
        PROJECT_ROOT
        / "runs"
        / run_name
        / "final_model.zip"
    )

    if best_policy.is_file():
        return best_policy.resolve()

    if final_policy.is_file():
        return final_policy.resolve()

    raise FileNotFoundError(
        "No trained policy was found.\n"
        f"Expected one of:\n"
        f"  {best_policy}\n"
        f"  {final_policy}"
    )


def main() -> None:
    args = parse_arguments()

    policy_path = find_policy(
        args.policy,
        args.mode,
    )

    environment = UnitreeDropEnv(
        mode=args.mode,
        target_x=args.target_x,
        target_y=args.target_y,
        target_radius=args.target_radius,
    )

    print(
        "Loading policy:",
        policy_path,
    )

    model = PPO.load(
        str(policy_path),
        env=environment,
    )

    observation, _ = (
        environment.reset()
    )

    episode = 1
    episode_reward = 0.0

    print(
        "Opening MuJoCo viewer..."
    )

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

            # Keep playback close to real time.
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

            if (
                terminated
                or truncated
            ):
                print(
                    f"Episode {episode}:"
                )

                print(
                    f"  Reward: "
                    f"{episode_reward:.2f}"
                )

                print(
                    f"  Success: "
                    f"{info['success']}"
                )

                print(
                    f"  Released: "
                    f"{info['released']}"
                )

                print(
                    f"  Release time: "
                    f"{info['release_time']:.3f} s"
                )

                print(
                    f"  Release speed: "
                    f"{info['release_speed']:.3f} m/s"
                )

                print(
                    f"  Impact error: "
                    f"{info['impact_error']:.3f} m"
                )

                print(
                    f"  Robot fell: "
                    f"{info['fell']}"
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

                episode += 1
                episode_reward = 0.0

    environment.close()


if __name__ == "__main__":
    main()