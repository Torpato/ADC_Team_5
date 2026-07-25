"""Visualize a trained two-robot PPO policy in MuJoCo."""

from __future__ import annotations

import argparse
import json
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
        default="full_exchange",
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
        "--start-robot",
        choices=("random", "1", "2"),
        default="random",
        help="Force every visual episode to start with robot 1 or 2.",
    )
    parser.add_argument(
        "--robot2-start-probability",
        type=float,
        default=None,
        help=(
            "Override the probability stored in configuration.json. "
            "Used only when --start-robot=random."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
    )
    return parser.parse_args()


def run_directory_for(args: argparse.Namespace) -> Path:
    run_name = (
        args.run_name
        if args.run_name
        else f"ppo_two_robot_{args.mode}"
    )
    return PROJECT_ROOT / "runs" / run_name


def find_policy(args: argparse.Namespace) -> Path:
    if args.policy is not None:
        policy_path = args.policy.expanduser().resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Policy not found: {policy_path}"
            )
        return policy_path

    run_directory = run_directory_for(args)
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


def load_configuration(args: argparse.Namespace) -> dict[str, object]:
    configuration_path = (
        run_directory_for(args) / "configuration.json"
    )
    if not configuration_path.is_file():
        return {}
    return json.loads(configuration_path.read_text(encoding="utf-8"))


def reset_environment(
    environment: TwoRobotCatchEnv,
    start_robot: str,
) -> tuple[np.ndarray, dict[str, object]]:
    options = None
    if start_robot in {"1", "2"}:
        options = {"start_robot": int(start_robot)}
    return environment.reset(options=options)


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
    print("  Started with robot:", info["starting_robot"])
    print("  Reward:", format_value(reward, 2))
    print("  Complete exchange:", info["success"])
    print("  Robot 2 caught during episode:", info["caught_by_robot_2"])
    print("  Robot 2 released:", info["released_by_robot_2"])
    print("  Robot 1 caught return:", info["caught_by_robot_1"])
    print("  Robot 2 return success:", info["robot_2_return_success"])
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
    print("  Miss reason:", info["miss_reason"] or "none")
    print("  Fallen robots:", info["fallen_robots"] or "none")


def main() -> None:
    args = parse_arguments()
    policy_path = find_policy(args)
    configuration = load_configuration(args)

    probability = (
        args.robot2_start_probability
        if args.robot2_start_probability is not None
        else float(
            configuration.get(
                "eval_robot2_start_probability",
                configuration.get(
                    "robot2_start_probability",
                    0.50,
                ),
            )
        )
    )

    environment = TwoRobotCatchEnv(
        model_path=args.model,
        mode=args.mode,
        frame_skip=int(configuration.get("frame_skip", 5)),
        minimum_release_time=float(
            configuration.get("minimum_release_time", 2.05)
        ),
        maximum_release_time=float(
            configuration.get("maximum_release_time", 2.45)
        ),
        reference_release_time=float(
            configuration.get("reference_release_time", 2.26)
        ),
        throw_velocity_noise=0.0,
        robot2_start_probability=probability,
        robot2_release_bonus=float(
            configuration.get("robot2_release_bonus", 30.0)
        ),
    )

    print("Loading policy:")
    print(policy_path)
    print("Robot-2 random-start probability:", probability)

    model = PPO.load(str(policy_path), env=environment)
    observation, _ = reset_environment(
        environment,
        args.start_robot,
    )

    episode = 1
    episode_reward = 0.0

    with mujoco.viewer.launch_passive(
        environment.model,
        environment.data,
    ) as viewer:
        set_camera(viewer, environment.model, args.camera)

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

            remaining = (
                environment.control_dt
                - (time.perf_counter() - step_started)
            )
            if remaining > 0.0:
                time.sleep(remaining)

            if terminated or truncated:
                print_episode(episode, episode_reward, info)

                if args.episodes > 0 and episode >= args.episodes:
                    break

                time.sleep(max(args.pause, 0.0))
                observation, _ = reset_environment(
                    environment,
                    args.start_robot,
                )
                episode += 1
                episode_reward = 0.0

    environment.close()


if __name__ == "__main__":
    main()
