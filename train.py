"""Train PPO using the current Unitree G1 model and controller.

Quick smoke test:

    python Python/train.py --timesteps 10000

Release-timing training:

    python Python/train.py --timesteps 300000

Residual training:

    python Python/train.py \
        --mode residual \
        --run-name ppo_residual \
        --timesteps 1000000
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
            "Train PPO for the Unitree G1 "
            "overhand throw."
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "release_only",
            "residual",
        ),
        default="release_only",
        help=(
            "release_only lets PPO learn only "
            "the release timing. residual also "
            "allows small joint corrections."
        ),
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=300_000,
        help="Total PPO training steps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
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
        "--run-name",
        default=None,
        help=(
            "Folder name inside runs/. "
            "Generated automatically when omitted."
        ),
    )

    return parser.parse_args()


def create_environment(
    *,
    mode: str,
    target_x: float,
    target_y: float,
    target_radius: float,
) -> UnitreeDropEnv:
    """Create one environment instance."""

    return UnitreeDropEnv(
        mode=mode,
        target_x=target_x,
        target_y=target_y,
        target_radius=target_radius,
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
        else f"ppo_{args.mode}"
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

    tensorboard_directory = (
        run_directory
        / "tensorboard"
    )

    evaluation_directory = (
        run_directory
        / "evaluation"
    )

    for directory in (
        run_directory,
        checkpoint_directory,
        best_model_directory,
        tensorboard_directory,
        evaluation_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------
    # VALIDATE ENVIRONMENT
    # -------------------------------------------------------------

    print(
        "Checking the custom "
        "Gymnasium environment..."
    )

    environment_for_check = (
        create_environment(
            mode=args.mode,
            target_x=args.target_x,
            target_y=args.target_y,
            target_radius=args.target_radius,
        )
    )

    check_env(
        environment_for_check,
        warn=True,
    )

    environment_for_check.close()

    print(
        "Environment check passed."
    )

    # -------------------------------------------------------------
    # TRAINING ENVIRONMENT
    # -------------------------------------------------------------

    training_environment = Monitor(
        create_environment(
            mode=args.mode,
            target_x=args.target_x,
            target_y=args.target_y,
            target_radius=args.target_radius,
        ),
        filename=str(
            run_directory
            / "training_monitor.csv"
        ),
    )

    # -------------------------------------------------------------
    # EVALUATION ENVIRONMENT
    # -------------------------------------------------------------

    evaluation_environment = Monitor(
        create_environment(
            mode=args.mode,
            target_x=args.target_x,
            target_y=args.target_y,
            target_radius=args.target_radius,
        )
    )

    # Adapt frequencies to short and long runs.
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
            args.timesteps // 5,
        ),
    )

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=checkpoint_frequency,
            save_path=str(
                checkpoint_directory
            ),
            name_prefix="ppo_checkpoint",
        )
    )

    evaluation_callback = (
        EvalCallback(
            evaluation_environment,
            best_model_save_path=str(
                best_model_directory
            ),
            log_path=str(
                evaluation_directory
            ),
            eval_freq=(
                evaluation_frequency
            ),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )
    )

    # -------------------------------------------------------------
    # SAVE CONFIGURATION
    # -------------------------------------------------------------

    configuration = {
        "mode": args.mode,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "target_x": args.target_x,
        "target_y": args.target_y,
        "target_radius": (
            args.target_radius
        ),
        "evaluation_frequency": (
            evaluation_frequency
        ),
        "checkpoint_frequency": (
            checkpoint_frequency
        ),
    }

    configuration_path = (
        run_directory
        / "configuration.json"
    )

    configuration_path.write_text(
        json.dumps(
            configuration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # CREATE PPO
    # -------------------------------------------------------------

    print(
        "Creating the PPO model..."
    )

    model = PPO(
        policy="MlpPolicy",
        env=training_environment,

        learning_rate=3e-4,

        n_steps=2048,
        batch_size=256,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,

        clip_range=0.20,

        ent_coef=0.01,
        vf_coef=0.50,

        max_grad_norm=0.50,

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

    # -------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------

    print(
        "Starting training:"
    )

    print(
        f"  Mode: {args.mode}"
    )

    print(
        f"  Timesteps: "
        f"{args.timesteps}"
    )

    print(
        f"  Target: "
        f"({args.target_x}, "
        f"{args.target_y})"
    )

    print(
        f"  Target radius: "
        f"{args.target_radius} m"
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[
            checkpoint_callback,
            evaluation_callback,
        ],
        progress_bar=True,
    )

    # -------------------------------------------------------------
    # SAVE FINAL POLICY
    # -------------------------------------------------------------

    final_model_path = (
        run_directory
        / "final_model"
    )

    model.save(
        str(final_model_path)
    )

    training_environment.close()
    evaluation_environment.close()

    print(
        "Training completed."
    )

    print(
        "Final model:",
        final_model_path.with_suffix(
            ".zip"
        ),
    )

    print(
        "Best model:",
        best_model_directory
        / "best_model.zip",
    )


if __name__ == "__main__":
    main()