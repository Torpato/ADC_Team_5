"""Train PPO for the complete two-robot catch exchange.

Examples
--------
Quick validation:
    python Python/train_two_robot_ppo_clean_right_hand.py \
        --mode release_only \
        --timesteps 20000 \
        --run-name catch_release_smoke

Release timing:
    python Python/train_two_robot_ppo_clean_right_hand.py \
        --mode release_only \
        --timesteps 300000

Continue the release policy and enable throwing/receiving corrections:
    python Python/train_two_robot_ppo_clean_right_hand.py \
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
from stable_baselines3.common.utils import get_schedule_fn

from two_robot_catch_env import (
    TwoRobotCatchEnv,
)


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


class ExchangeMetricsCallback(BaseCallback):
    """Log overall and start-conditioned exchange metrics."""

    def __init__(self, window_size: int = 100) -> None:
        super().__init__()
        self.successes: deque[float] = deque(maxlen=window_size)
        self.catch_2: deque[float] = deque(maxlen=window_size)
        self.catch_1: deque[float] = deque(maxlen=window_size)
        self.release_2: deque[float] = deque(maxlen=window_size)
        self.return_success: deque[float] = deque(maxlen=window_size)
        self.falls: deque[float] = deque(maxlen=window_size)
        self.start_2: deque[float] = deque(maxlen=window_size)
        self.wrong_arm_contacts: deque[float] = deque(
            maxlen=window_size
        )
        self.robot1_start_success: deque[float] = deque(
            maxlen=window_size
        )
        self.robot2_start_success: deque[float] = deque(
            maxlen=window_size
        )

    def _record_mean(
        self,
        name: str,
        values: deque[float],
    ) -> None:
        if values:
            self.logger.record(name, float(np.mean(values)))

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            success = float(info.get("success", False))
            starting_robot = int(info.get("starting_robot", 1))

            self.successes.append(success)
            self.catch_2.append(
                float(info.get("caught_by_robot_2", False))
            )
            self.catch_1.append(
                float(info.get("caught_by_robot_1", False))
            )
            self.release_2.append(
                float(info.get("released_by_robot_2", False))
            )
            self.return_success.append(
                float(info.get("robot_2_return_success", False))
            )
            self.falls.append(
                float(bool(info.get("fallen_robots", ())))
            )
            self.start_2.append(float(starting_robot == 2))
            self.wrong_arm_contacts.append(
                float(info.get("wrong_arm_contact", False))
            )

            if starting_robot == 1:
                self.robot1_start_success.append(success)
            else:
                self.robot2_start_success.append(success)

        self._record_mean(
            "exchange/success_rate_100",
            self.successes,
        )
        self._record_mean(
            "exchange/robot_2_catch_rate_100",
            self.catch_2,
        )
        self._record_mean(
            "exchange/robot_1_catch_rate_100",
            self.catch_1,
        )
        self._record_mean(
            "exchange/robot_2_release_rate_100",
            self.release_2,
        )
        self._record_mean(
            "exchange/robot_2_return_success_rate_100",
            self.return_success,
        )
        self._record_mean(
            "exchange/fall_rate_100",
            self.falls,
        )
        self._record_mean(
            "exchange/wrong_arm_contact_rate_100",
            self.wrong_arm_contacts,
        )
        self._record_mean(
            "curriculum/robot_2_start_fraction_100",
            self.start_2,
        )
        self._record_mean(
            "curriculum/robot_1_start_success_100",
            self.robot1_start_success,
        )
        self._record_mean(
            "curriculum/robot_2_start_success_100",
            self.robot2_start_success,
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
    parser.add_argument(
        "--robot2-start-probability",
        type=float,
        default=0.50,
        help=(
            "Probability that a training episode starts with the ball "
            "already held by robot 2. Use 0.70 for the robot-2 focus stage."
        ),
    )
    parser.add_argument(
        "--eval-robot2-start-probability",
        type=float,
        default=0.50,
        help=(
            "Robot-2 start probability used only for evaluation. Keep "
            "this at 0.50 to measure both directions fairly."
        ),
    )
    parser.add_argument(
        "--robot2-release-bonus",
        type=float,
        default=30.0,
        help="Temporary curriculum reward when robot 2 releases the ball.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Override PPO learning rate. For fine-tuning an existing "
            "robot-1 policy, 5e-5 is recommended."
        ),
    )
    parser.add_argument(
        "--wrong-arm-contact-penalty",
        type=float,
        default=120.0,
        help=(
            "Penalty applied when the return throw touches robot 1's "
            "left arm before a valid right-hand catch."
        ),
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "Delete an existing run directory before training. Disabled "
            "by default to protect previous results."
        ),
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
        robot2_start_probability=(
            args.eval_robot2_start_probability
            if evaluation
            else args.robot2_start_probability
        ),
        robot2_release_bonus=args.robot2_release_bonus,
        wrong_arm_contact_penalty=args.wrong_arm_contact_penalty,
        terminate_on_wrong_arm_contact=True,
    )


def main() -> None:
    args = parse_arguments()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    for name, probability in (
        (
            "--robot2-start-probability",
            args.robot2_start_probability,
        ),
        (
            "--eval-robot2-start-probability",
            args.eval_robot2_start_probability,
        ),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if args.robot2_release_bonus < 0.0:
        raise ValueError("--robot2-release-bonus cannot be negative.")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.wrong_arm_contact_penalty < 0.0:
        raise ValueError(
            "--wrong-arm-contact-penalty cannot be negative."
        )

    run_name = (
        args.run_name
        if args.run_name
        else f"ppo_two_robot_clean_right_hand_{args.mode}"
    )
    run_directory = PROJECT_ROOT / "runs" / run_name

    if run_directory.exists() and any(run_directory.iterdir()):
        if not args.overwrite_run:
            raise FileExistsError(
                f"Run directory already exists and will not be overwritten: "
                f"{run_directory}\nChoose a new --run-name, or use "
                "--overwrite-run only when deletion is intentional."
            )

        import shutil
        shutil.rmtree(run_directory)

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
        "starting_robot",
        "started_with_robot_2",
        "released_by_robot_1",
        "released_by_robot_2",
        "robot_2_return_success",
        "wrong_arm_contact",
        "wrong_arm_contact_penalty",
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
        "robot2_start_probability": (
            args.robot2_start_probability
        ),
        "eval_robot2_start_probability": (
            args.eval_robot2_start_probability
        ),
        "robot2_release_bonus": args.robot2_release_bonus,
        "wrong_arm_contact_penalty": (
            args.wrong_arm_contact_penalty
        ),
        "terminate_on_wrong_arm_contact": True,
        "resume_model": (
            str(args.resume.expanduser().resolve())
            if args.resume is not None
            else None
        ),
        "learning_rate_override": args.learning_rate,
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

        if args.learning_rate is not None:
            model.learning_rate = float(args.learning_rate)
            model.lr_schedule = get_schedule_fn(
                float(args.learning_rate)
            )
            for parameter_group in (
                model.policy.optimizer.param_groups
            ):
                parameter_group["lr"] = float(
                    args.learning_rate
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

        if args.learning_rate is not None:
            learning_rate = float(args.learning_rate)

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
    print(
        "  Training robot-2 start probability:",
        args.robot2_start_probability,
    )
    print(
        "  Evaluation robot-2 start probability:",
        args.eval_robot2_start_probability,
    )
    print(
        "  Wrong-arm contact penalty:",
        args.wrong_arm_contact_penalty,
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
