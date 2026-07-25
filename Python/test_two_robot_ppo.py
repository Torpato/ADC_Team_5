"""Visualize a trained two-robot PPO policy in MuJoCo.

On macOS run this file with mjpython:

    mjpython Python/test_two_robot_ppo.py --mode release_only
    mjpython Python/test_two_robot_ppo.py --mode full_exchange
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from two_robot_catch_env import TwoRobotCatchEnv


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained two-robot PPO policy."
    )
    parser.add_argument(
        "--mode",
        choices=("release_only", "full_exchange"),
        default="release_only",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "Model" / "world_catch.xml",
    )
    parser.add_argument("--camera", default="demo")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--pause", type=float, default=0.75)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic rather than deterministic actions.",
    )
    return parser.parse_args()


def find_policy(args: argparse.Namespace) -> Path:
    if args.policy is not None:
        policy_path = args.policy.expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Policy not found: {policy_path}"
            )
        return policy_path

    run_name = (
        args.run_name
        if args.run_name
        else f"ppo_two_robot_{args.mode}"
    )
    run_directory = PROJECT_ROOT / "runs" / run_name

    candidates = (
        run_directory / "best" / "best_model.zip",
        run_directory / "final_model.zip",
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No trained policy found. Expected one of:\n"
        + "\n".join(f"  {path}" for path in candidates)
    )


def set_camera(
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
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                index,
            )
            for index in range(model.ncam)
        ]
        raise ValueError(
            f"Unknown camera {camera_name!r}. "
            f"Available cameras: {available}"
        )

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def format_value(value: object, decimals: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not np.isfinite(numeric):
        return "N/A"

    return f"{numeric:.{decimals}f}"


def print_episode(
    episode: int,
    reward: float,
    info: dict[str, object],
) -> None:
    print(f"\nEpisode {episode}")
    print("  Reward:", format_value(reward, 2))
    print("  Complete exchange:", info["success"])
    print("  Robot 2 caught:", info["caught_by_robot_2"])
    print("  Robot 1 caught return:", info["caught_by_robot_1"])
    print(
        "  Robot 1 release:",
        format_value(info["release_time_robot_1"]),
        "s |",
        format_value(info["release_speed_robot_1"]),
        "m/s",
    )
    print(
        "  Robot 2 release:",
        format_value(info["release_time_robot_2"]),
        "s |",
        format_value(info["release_speed_robot_2"]),
        "m/s",
    )
    print(
        "  Minimum distance to robot 2:",
        format_value(info["minimum_distance_robot_2"]),
        "m",
    )
    print(
        "  Minimum distance to robot 1:",
        format_value(info["minimum_distance_robot_1"]),
        "m",
    )
    print("  Miss reason:", info["miss_reason"] or "none")
    print("  Fallen robots:", info["fallen_robots"] or "none")


def main() -> None:
    args = parse_arguments()
    policy_path = find_policy(args)

    environment = TwoRobotCatchEnv(
        model_path=args.model,
        mode=args.mode,
        frame_skip=5,
        throw_velocity_noise=0.0,
    )

    print("Loading policy:")
    print(policy_path)

    model = PPO.load(
        str(policy_path),
        env=environment,
    )
    observation, _ = environment.reset()

    episode = 1
    episode_reward = 0.0

    print("Opening MuJoCo viewer...")

    with mujoco.viewer.launch_passive(
        environment.model,
        environment.data,
    ) as viewer:
        set_camera(
            viewer,
            environment.model,
            args.camera,
        )

        while viewer.is_running():
            step_started = time.perf_counter()

            action, _ = model.predict(
                observation,
                deterministic=not args.stochastic,
            )
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            episode_reward += float(reward)
            viewer.sync()

            elapsed = time.perf_counter() - step_started
            remaining = environment.control_dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

            if terminated or truncated:
                print_episode(
                    episode,
                    episode_reward,
                    info,
                )

                if (
                    args.episodes > 0
                    and episode >= args.episodes
                ):
                    break

                time.sleep(max(args.pause, 0.0))
                observation, _ = environment.reset()
                episode += 1
                episode_reward = 0.0

    environment.close()


if __name__ == "__main__":
    main()
