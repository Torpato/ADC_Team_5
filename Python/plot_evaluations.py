"""Plot Stable-Baselines3 evaluation data without overwriting old runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PYTHON_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIRECTORY.parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="ppo_two_robot_full_exchange",
    )
    parser.add_argument("--window", type=int, default=5)
    return parser.parse_args()


def centred_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()

    output = np.empty_like(values, dtype=float)
    radius = window // 2

    for index in range(values.size):
        start = max(0, index - radius)
        stop = min(values.size, index + radius + 1)
        output[index] = float(np.mean(values[start:stop]))

    return output


def main() -> None:
    args = parse_arguments()
    run_directory = PROJECT_ROOT / "runs" / args.run_name
    evaluation_path = (
        run_directory / "evaluation" / "evaluations.npz"
    )

    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)

    data = np.load(evaluation_path)
    timesteps = data["timesteps"]
    results = data["results"]
    episode_lengths = data["ep_lengths"]

    # EvalCallback stores one column per evaluation episode.  The previous
    # plotting script incorrectly treated column 2 as a precomputed mean.
    reward_mean = np.mean(results, axis=1)
    reward_std = np.std(results, axis=1)
    length_mean = np.mean(episode_lengths, axis=1)

    smooth_reward = centred_mean(
        reward_mean,
        max(args.window, 1),
    )

    figure, (reward_axis, length_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    reward_axis.plot(
        timesteps,
        reward_mean,
        label="Mean evaluation reward",
    )
    reward_axis.fill_between(
        timesteps,
        reward_mean - reward_std,
        reward_mean + reward_std,
        alpha=0.20,
        label="±1 standard deviation",
    )
    reward_axis.plot(
        timesteps,
        smooth_reward,
        linestyle=":",
        linewidth=2.5,
        label=f"Centred mean (window={args.window})",
    )
    reward_axis.set_ylabel("Reward")
    reward_axis.set_title(
        f"Evaluation metrics — {args.run_name}"
    )
    reward_axis.grid(True, alpha=0.3)
    reward_axis.legend()

    length_axis.plot(
        timesteps,
        length_mean,
        label="Mean episode length",
    )
    length_axis.set_xlabel("Timesteps")
    length_axis.set_ylabel("Episode length")
    length_axis.grid(True, alpha=0.3)
    length_axis.legend()

    figure.tight_layout()

    output_path = run_directory / "evaluation_plot.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Plot saved to:", output_path)
    plt.show()


if __name__ == "__main__":
    main()
