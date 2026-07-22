from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import mujoco

_LOG_PATTERN = re.compile(r"^Joint_Log_Episode_(\d+)\.csv$", re.IGNORECASE)


def _next_episode_path(output_path: Path) -> Path:
    logs_dir = Path(__file__).resolve().parent.parent / "Logs"
    if output_path.parent in (Path("."), Path("")):
        directory = logs_dir
    else:
        directory = output_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    existing_episodes: list[int] = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".csv":
            match = _LOG_PATTERN.match(path.name)
            if match:
                existing_episodes.append(int(match.group(1)))

    if output_path.exists():
        existing_episodes.append(0)

    next_episode = max(existing_episodes, default=0) + 1
    return directory / f"Joint_Log_Episode_{next_episode}.csv"


class JointCsvLogger:
    def __init__(self, output_path: Path, log_every: int = 10) -> None:
        if log_every <= 0:
            raise ValueError("log_every must be greater than zero.")
        self.output_path = _next_episode_path(output_path)
        self.log_every = log_every
        self._csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._write_header()
        self._step = 0

    def _write_header(self) -> None:
        self._writer.writerow(["step", "time", "qpos", "qvel"])

    # CSV columns are written in this order:
    #   step  - simulation step number at which this row was logged
    #   time  - simulation time after the step
    #   qpos  - full qpos array for all coordinates, encoded as JSON
    #   qvel  - full qvel array for all velocities, encoded as JSON
    #   Left Hip (Pitch, Roll, Yaw), Left Knee (Pitch), Left Ankle (Pitch, Roll), 
    #   Right Hip (Pitch, Roll, Yaw), Right Knee (Pitch), Right Ankle (Pitch, Roll), 
    #   Waist (Yaw, Roll, Pitch), 
    #   Left Shoulder (Pitch, Roll, Yaw), Left Elbow (Pitch), Left Wrist (Roll, Pitch, Yaw), 
    #   Right Shoulder (Pitch, Roll, Yaw), Right Elbow (Pitch), Right Wrist (Roll, Pitch, Yaw)
    def should_log(self) -> bool:
        return self._step % self.log_every == 0

    def log_step(self, data: mujoco.MjData) -> None:
        self._step += 1
        if not self.should_log():
            return

        qpos_values = data.qpos.tolist()
        qvel_values = data.qvel.tolist()
        self._writer.writerow(
            [
                self._step,
                float(data.time),
                json.dumps(qpos_values, ensure_ascii=False),
                json.dumps(qvel_values, ensure_ascii=False),
            ]
        )

    def close(self) -> None:
        self._csv_file.close()


def create_logger(output_path: Path | None = None, log_every: int = 10) -> JointCsvLogger:
    if output_path is None:
        output_path = Path.cwd() / "Joint_Log_Episode_1.csv"
    return JointCsvLogger(output_path, log_every=log_every)
