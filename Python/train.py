"""Train PPO for the current Unitree G1 model.

Recommended order:

1. Release timing only:
   python Python/train.py --mode release_only --timesteps 10000

2. Longer release training:
   python Python/train.py --mode release_only --timesteps 300000

3. Safe residual training:
   python Python/train.py --mode residual --timesteps 500000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import (
    check_env,
)
from stable_baselines3.common.monitor import (
    Monitor,
)

from unitree_env import UnitreeDropEnv


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a safe PPO policy for "
            "the Unitree G1 throw."
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
        "--timesteps",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--run-name",
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
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def create_environment(
    args: argparse.Namespace,
) -> UnitreeDropEnv:
    """Create one independent environment."""

    return UnitreeDropEnv(
        mode=args.mode,
        target_x=args.target_x,
        target_y=args.target_y,
        target_radius=args.target_radius,
        frame_skip=5,
        minimum_release_time=2.10,
        maximum_release_time=2.45,
        reference_release_time=2.30,
    )


def main() -> None:
    args = parse_arguments()

    if args.timesteps <= 0:
        raise ValueError(
            "--timesteps must be positive."
        )

    run_name = (
        args.run_name
        if args.run_name
        else f"ppo_safe_{args.mode}"
    )

    run_directory = (
        PROJECT_ROOT
        / "runs"
        / run_name
    )

    checkpoint_directory = (
        run_directory
        / "checkpoints"
    )

    best_model_directory = (
        run_directory
        / "best"
    )

    evaluation_directory = (
        run_directory
        / "evaluation"
    )

    tensorboard_directory = (
        run_directory
        / "tensorboard"
    )

    for directory in (
        run_directory,
        checkpoint_directory,
        best_model_directory,
        evaluation_directory,
        tensorboard_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Checking the Gymnasium environment...")

    environment_for_check = (
        create_environment(args)
    )

    check_env(
        environment_for_check,
        warn=True,
    )

    environment_for_check.close()

    print("Environment check passed.")

    training_environment = Monitor(
        create_environment(args),
        filename=str(
            run_directory
            / "training_monitor.csv"
        ),
    )

    evaluation_environment = Monitor(
        create_environment(args)
    )

    evaluation_frequency = max(
        2_000,
        min(
            20_000,
            args.timesteps // 5,
        ),
    )

    checkpoint_frequency = max(
        5_000,
        min(
            50_000,
            args.timesteps // 4,
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(
            checkpoint_directory
        ),
        name_prefix="ppo_checkpoint",
    )

    evaluation_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path=str(
            best_model_directory
        ),
        log_path=str(
            evaluation_directory
        ),
        eval_freq=evaluation_frequency,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    configuration = {
        "mode": args.mode,
        "timesteps": args.timesteps,
        "target_x": args.target_x,
        "target_y": args.target_y,
        "target_radius": args.target_radius,
        "seed": args.seed,
        "minimum_release_time": 2.10,
        "maximum_release_time": 2.45,
        "reference_release_time": 2.30,
        "residual_scales": (
            UnitreeDropEnv
            .RESIDUAL_SCALES
            .tolist()
        ),
    }

    (
        run_directory
        / "configuration.json"
    ).write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Creating the PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=training_environment,

        # Conservative learning parameters.
        learning_rate=1e-4,

        n_steps=1024,
        batch_size=128,
        n_epochs=8,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.15,

        # Low exploration prevents unnecessarily erratic actions.
        ent_coef=0.001,

        vf_coef=0.50,
        max_grad_norm=0.30,

        policy_kwargs={
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            }
        },

        tensorboard_log=str(
            tensorboard_directory
        ),

        verbose=1,
        seed=args.seed,
        device="auto",
    )

    print("Starting training...")
    print(f"Mode: {args.mode}")
    print(f"Timesteps: {args.timesteps}")
    print(
        f"Target: "
        f"({args.target_x}, {args.target_y})"
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[
            checkpoint_callback,
            evaluation_callback,
        ],
        progress_bar=True,
    )

    final_model_path = (
        run_directory
        / "final_model"
    )

    model.save(
        str(final_model_path)
    )

    training_environment.close()
    evaluation_environment.close()

    print("Training completed.")

    print(
        "Final model:",
        final_model_path.with_suffix(".zip"),
    )

    print(
        "Best model:",
        best_model_directory
        / "best_model.zip",
    )


if __name__ == "__main__":
    main()