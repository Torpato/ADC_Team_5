"""Headless smoke test for the two-robot Gymnasium environment.

This test does not require a trained PPO model. It checks:

1. Gymnasium API compatibility.
2. Model/controller loading.
3. A scripted zero-residual exchange using the reference release time.

Run:
    python Python/smoke_test_two_robot_env.py --mode release_only
    python Python/smoke_test_two_robot_env.py --mode full_exchange
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from two_robot_catch_env import TwoRobotCatchEnv


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("release_only", "full_exchange"),
        default="release_only",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "Model" / "world_catch.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    environment = TwoRobotCatchEnv(
        model_path=args.model,
        mode=args.mode,
        frame_skip=5,
        throw_velocity_noise=0.0,
    )

    print("Checking Gymnasium API...")
    check_env(environment, warn=True)
    print("Gymnasium check passed.")

    successes = 0

    for episode in range(1, args.episodes + 1):
        observation, _ = environment.reset()
        del observation

        terminated = False
        truncated = False
        total_reward = 0.0
        info = {}

        while not (terminated or truncated):
            action = environment.baseline_action()
            (
                _,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)
            total_reward += float(reward)

        successes += int(bool(info["success"]))

        print(
            f"Episode {episode}: "
            f"success={info['success']} | "
            f"catch2={info['caught_by_robot_2']} | "
            f"catch1={info['caught_by_robot_1']} | "
            f"reward={total_reward:.2f} | "
            f"miss={info['miss_reason'] or 'none'} | "
            f"falls={info['fallen_robots'] or 'none'}"
        )

    environment.close()

    print(
        f"\nBaseline success rate: "
        f"{successes}/{args.episodes}"
    )

    if successes == 0:
        raise SystemExit(
            "The environment loads, but the deterministic baseline "
            "did not complete an exchange. Run the visual baseline and "
            "retune release/catch parameters before PPO training."
        )


if __name__ == "__main__":
    main()
