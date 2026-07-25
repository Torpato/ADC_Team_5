"""Train PPO for the complete two-robot catch exchange.

Examples
--------
Quick validation:
    python Python/train_two_robot_ppo.py \
        --mode release_only \
        --timesteps 20000 \
        --run-name catch_release_smoke

Release timing:
    python Python/train_two_robot_ppo.py \
        --mode release_only \
        --timesteps 300000

Continue the release policy and enable throwing/receiving corrections:
    python Python/train_two_robot_ppo.py \
        --mode full_exchange \
        --timesteps 1000000 \
        --throw-noise 0.02 \
        --resume runs/ppo_two_robot_release_only/best/best_model.zip
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from two_robot_catch_env import TwoRobotCatchEnv


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


class ExchangeMetricsCallback(BaseCallback):
    """Log rolling exchange and catch rates to TensorBoard."""

    def __init__(self, window_size: int = 100) -> None:
        super().__init__()
        self.successes: deque[float] = deque(maxlen=window_size)
        self.catch_2: deque[float] = deque(maxlen=window_size)
        self.catch_1: deque[float] = deque(maxlen=window_size)
        self.falls: deque[float] = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.successes.append(float(info.get("success", False)))
            self.catch_2.append(
                float(info.get("caught_by_robot_2", False))
            )
            self.catch_1.append(
                float(info.get("caught_by_robot_1", False))
            )
            self.falls.append(
                float(bool(info.get("fallen_robots", ())))
            )

        if self.successes:
            self.logger.record(
                "exchange/success_rate_100",
                float(np.mean(self.successes)),
            )
            self.logger.record(
                "exchange/robot_2_catch_rate_100",
                float(np.mean(self.catch_2)),
            )
            self.logger.record(
                "exchange/robot_1_catch_rate_100",
                float(np.mean(self.catch_1)),
            )
            self.logger.record(
                "exchange/fall_rate_100",
                float(np.mean(self.falls)),
            )

        return True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO for the two-robot catch exchange."
    )
    parser.add_argument(
        "--mode",
        choices=("release_only", "full_exchange"),
        default="release_only",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "Model" / "world_catch.xml",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument(
        "--throw-noise",
        type=float,
        default=0.0,
        help="Uniform release-velocity disturbance in m/s.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Continue from a PPO .zip. Both modes use identical spaces, "
            "so release_only can be continued as full_exchange."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
    )
    return parser.parse_args()


def create_environment(
    args: argparse.Namespace,
    *,
    evaluation: bool,
) -> TwoRobotCatchEnv:
    return TwoRobotCatchEnv(
        model_path=args.model,
        mode=args.mode,
        frame_skip=args.frame_skip,
        minimum_release_time=2.05,
        maximum_release_time=2.45,
        reference_release_time=2.26,
        throw_velocity_noise=(
            0.0 if evaluation else args.throw_noise
        ),
    )


def main() -> None:
    args = parse_arguments()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")

    run_name = (
        args.run_name
        if args.run_name
        else f"ppo_two_robot_{args.mode}"
    )
    run_directory = PROJECT_ROOT / "runs" / run_name
    checkpoint_directory = run_directory / "checkpoints"
    best_directory = run_directory / "best"
    evaluation_directory = run_directory / "evaluation"
    tensorboard_directory = run_directory / "tensorboard"

    for directory in (
        run_directory,
        checkpoint_directory,
        best_directory,
        evaluation_directory,
        tensorboard_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not args.skip_env_check:
        print("Checking the Gymnasium environment...")
        environment_for_check = create_environment(
            args,
            evaluation=True,
        )
        check_env(environment_for_check, warn=True)
        environment_for_check.close()
        print("Environment check passed.")

    monitor_info = (
        "success",
        "caught_by_robot_2",
        "caught_by_robot_1",
        "forced_release_robot_1",
        "forced_release_robot_2",
    )

    training_environment = Monitor(
        create_environment(args, evaluation=False),
        filename=str(run_directory / "training_monitor.csv"),
        info_keywords=monitor_info,
    )
    evaluation_environment = Monitor(
        create_environment(args, evaluation=True),
        info_keywords=monitor_info,
    )

    evaluation_frequency = max(
        5_000,
        min(50_000, args.timesteps // 10),
    )
    checkpoint_frequency = max(
        10_000,
        min(100_000, args.timesteps // 5),
    )

    callbacks = CallbackList(
        [
            ExchangeMetricsCallback(window_size=100),
            CheckpointCallback(
                save_freq=checkpoint_frequency,
                save_path=str(checkpoint_directory),
                name_prefix="ppo_checkpoint",
            ),
            EvalCallback(
                evaluation_environment,
                best_model_save_path=str(best_directory),
                log_path=str(evaluation_directory),
                eval_freq=evaluation_frequency,
                n_eval_episodes=10,
                deterministic=True,
                render=False,
            ),
        ]
    )

    configuration = {
        "mode": args.mode,
        "timesteps": args.timesteps,
        "model": str(args.model.expanduser().resolve()),
        "seed": args.seed,
        "frame_skip": args.frame_skip,
        "throw_noise": args.throw_noise,
        "minimum_release_time": 2.05,
        "maximum_release_time": 2.45,
        "reference_release_time": 2.26,
        "controlled_joints": list(
            TwoRobotCatchEnv.CONTROLLED_JOINTS
        ),
        "residual_scales": (
            TwoRobotCatchEnv.RESIDUAL_SCALES.tolist()
        ),
        "action_size": int(
            training_environment.action_space.shape[0]
        ),
        "observation_size": int(
            training_environment.observation_space.shape[0]
        ),
    }
    (run_directory / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume model not found: {resume_path}"
            )

        print("Resuming PPO model:")
        print(resume_path)
        model = PPO.load(
            str(resume_path),
            env=training_environment,
            device=args.device,
            tensorboard_log=str(tensorboard_directory),
        )
        reset_num_timesteps = False
    else:
        if args.mode == "release_only":
            learning_rate = 2.5e-4
            n_steps = 2048
            entropy = 0.002
        else:
            learning_rate = 1.0e-4
            n_steps = 4096
            entropy = 0.003

        model = PPO(
            policy="MlpPolicy",
            env=training_environment,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=256,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.15,
            ent_coef=entropy,
            vf_coef=0.50,
            max_grad_norm=0.40,
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 256],
                    "vf": [256, 256],
                }
            },
            tensorboard_log=str(tensorboard_directory),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        reset_num_timesteps = True

    print("\nStarting PPO training")
    print("  Mode:", args.mode)
    print("  Timesteps:", args.timesteps)
    print("  Action size:", training_environment.action_space.shape)
    print(
        "  Observation size:",
        training_environment.observation_space.shape,
    )
    print("  Run directory:", run_directory)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_model_path = run_directory / "final_model"
    model.save(str(final_model_path))

    training_environment.close()
    evaluation_environment.close()

    print("\nTraining completed.")
    print("Final model:", final_model_path.with_suffix(".zip"))
    print("Best model:", best_directory / "best_model.zip")


if __name__ == "__main__":
    main()
