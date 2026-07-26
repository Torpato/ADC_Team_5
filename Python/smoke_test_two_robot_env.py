"""Headless smoke test for both curriculum starting states."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from two_robot_catch_env import (
    TwoRobotCatchEnv,
)


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("release_only", "full_exchange"),
        default="full_exchange",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per selected starting robot.",
    )
    parser.add_argument(
        "--start-robot",
        choices=("both", "1", "2", "random"),
        default="both",
    )
    parser.add_argument(
        "--robot2-start-probability",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--wrong-arm-contact-penalty",
        type=float,
        default=120.0,
    )
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
        robot2_start_probability=args.robot2_start_probability,
        wrong_arm_contact_penalty=args.wrong_arm_contact_penalty,
        terminate_on_wrong_arm_contact=True,
    )

    print("Checking Gymnasium API...")
    check_env(environment, warn=True)
    print("Gymnasium check passed.")

    if args.start_robot == "both":
        starts: list[int | None] = [1, 2]
    elif args.start_robot == "random":
        starts = [None]
    else:
        starts = [int(args.start_robot)]

    total_successes = 0
    total_episodes = 0

    for requested_start in starts:
        successes = 0
        label = (
            "random"
            if requested_start is None
            else str(requested_start)
        )
        print(f"\nStarting robot: {label}")

        for episode in range(1, args.episodes + 1):
            options = (
                None
                if requested_start is None
                else {"start_robot": requested_start}
            )
            _, reset_info = environment.reset(options=options)
            actual_start = int(reset_info["starting_robot"])

            terminated = False
            truncated = False
            total_reward = 0.0
            info: dict[str, object] = reset_info

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

            success = int(bool(info["success"]))
            successes += success
            total_successes += success
            total_episodes += 1

            print(
                f"Episode {episode}: "
                f"start={actual_start} | "
                f"success={info['success']} | "
                f"released2={info['released_by_robot_2']} | "
                f"return={info['robot_2_return_success']} | "
                f"wrong_arm={info['wrong_arm_contact']} | "
                f"wrong_body={info['wrong_arm_contact_body'] or 'none'} | "
                f"reward={total_reward:.2f} | "
                f"miss={info['miss_reason'] or 'none'}"
            )

        print(
            f"Success rate for start {label}: "
            f"{successes}/{args.episodes}"
        )

    environment.close()
    print(
        f"\nOverall baseline success rate: "
        f"{total_successes}/{total_episodes}"
    )

    if total_successes == 0:
        raise SystemExit(
            "The environment loads, but neither starting state "
            "completed its target exchange."
        )


if __name__ == "__main__":
    main()
