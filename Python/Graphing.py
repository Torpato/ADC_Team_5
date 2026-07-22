from __future__ import annotations

import csv
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

LOGS_DIR = Path(__file__).resolve().parent.parent / "Logs"
LOG_FILE_PATTERN = re.compile(r".*\.csv$", re.IGNORECASE)

JOINT_DEFINITIONS = {
    "Left Hip": {
        "indices": [7, 8, 9],
        "components": ["Pitch", "Roll", "Yaw"],
    },
    "Left Knee": {
        "indices": [10],
        "components": ["Pitch"],
    },
    "Left Ankle": {
        "indices": [11, 12],
        "components": ["Pitch", "Roll"],
    },
    "Right Hip": {
        "indices": [13, 14, 15],
        "components": ["Pitch", "Roll", "Yaw"],
    },
    "Right Knee": {
        "indices": [16],
        "components": ["Pitch"],
    },
    "Right Ankle": {
        "indices": [17, 18],
        "components": ["Pitch", "Roll"],
    },
    "Waist": {
        "indices": [19, 20, 21],
        "components": ["Yaw", "Roll", "Pitch"],
    },
    "Left Shoulder": {
        "indices": [22, 23, 24],
        "components": ["Pitch", "Roll", "Yaw"],
    },
    "Left Elbow": {
        "indices": [25],
        "components": ["Pitch"],
    },
    "Left Wrist": {
        "indices": [26, 27, 28],
        "components": ["Roll", "Pitch", "Yaw"],
    },
    "Right Shoulder": {
        "indices": [29, 30, 31],
        "components": ["Pitch", "Roll", "Yaw"],
    },
    "Right Elbow": {
        "indices": [32],
        "components": ["Pitch"],
    },
    "Right Wrist": {
        "indices": [33, 34, 35],
        "components": ["Roll", "Pitch", "Yaw"],
    },
}

ALL_COMPONENTS = ["Pitch", "Yaw", "Roll"]


class LogPlotter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Joint Log Viewer")
        self.geometry("1000x700")

        self.log_files = self._find_log_files()
        self.selected_log = tk.StringVar()
        self.selected_joint = tk.StringVar()
        self.selected_data_type = tk.StringVar(value="qpos")
        self.component_states = {component: tk.BooleanVar(value=True) for component in ALL_COMPONENTS}
        self.current_data: dict[str, list] = {"time": [], "qpos": [], "qvel": []}

        self._build_widgets()
        self._draw_plot()

    def _find_log_files(self) -> list[str]:
        if not LOGS_DIR.exists():
            return []
        files = [path.name for path in sorted(LOGS_DIR.iterdir()) if path.is_file() and LOG_FILE_PATTERN.match(path.name)]
        return files

    def _build_widgets(self) -> None:
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=12, pady=8)

        ttk.Label(controls_frame, text="Select log:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        log_combo = ttk.Combobox(controls_frame, values=self.log_files, textvariable=self.selected_log, state="readonly")
        log_combo.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        log_combo.bind("<<ComboboxSelected>>", lambda event: self._on_selection_change())

        ttk.Label(controls_frame, text="Select joint:").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        joint_combo = ttk.Combobox(
            controls_frame,
            values=list(JOINT_DEFINITIONS.keys()),
            textvariable=self.selected_joint,
            state="readonly",
        )
        joint_combo.grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        joint_combo.bind("<<ComboboxSelected>>", lambda event: self._on_selection_change())

        # Data type selector: choose Position (`qpos`) or Velocity (`qvel`)
        ttk.Label(controls_frame, text="Data:").grid(row=0, column=4, padx=4, pady=4, sticky="w")
        rb_pos = ttk.Radiobutton(
            controls_frame, text="Position", variable=self.selected_data_type, value="qpos", command=self._on_selection_change
        )
        rb_vel = ttk.Radiobutton(
            controls_frame, text="Velocity", variable=self.selected_data_type, value="qvel", command=self._on_selection_change
        )
        rb_pos.grid(row=0, column=5, padx=2, pady=4)
        rb_vel.grid(row=0, column=6, padx=2, pady=4)

        for idx, component in enumerate(ALL_COMPONENTS, start=7):
            button = ttk.Checkbutton(
                controls_frame,
                text=component,
                variable=self.component_states[component],
                command=self._on_selection_change,
            )
            button.grid(row=0, column=idx, padx=4, pady=4)

        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(3, weight=1)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _on_selection_change(self) -> None:
        if not self.selected_log.get() or not self.selected_joint.get():
            return
        self._load_selected_log()
        self._draw_plot()

    def _load_selected_log(self) -> None:
        log_path = LOGS_DIR / self.selected_log.get()
        if not log_path.exists():
            self.current_data = {"time": [], "qpos": []}
            return

        times: list[float] = []
        qpos_rows: list[list[float]] = []
        qvel_rows: list[list[float]] = []
        with log_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    time_val = float(row["time"])
                    qpos_values = json.loads(row["qpos"])
                    qvel_values = json.loads(row.get("qvel", "[]"))
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
                times.append(time_val)
                qpos_rows.append(qpos_values)
                qvel_rows.append(qvel_values)

        self.current_data = {"time": times, "qpos": qpos_rows, "qvel": qvel_rows}

    def _draw_plot(self) -> None:
        self.ax.clear()
        if not self.current_data["time"] or not self.selected_joint.get():
            self.ax.set_title("Select a log and a joint to display data.")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Value")
            self.canvas.draw()
            return

        joint_name = self.selected_joint.get()
        joint_info = JOINT_DEFINITIONS[joint_name]
        channel_names = joint_info["components"]
        channel_indices = joint_info["indices"]
        active_components = [comp for comp in channel_names if self.component_states.get(comp, tk.BooleanVar()).get()]

        times = self.current_data["time"]
        data_type = self.selected_data_type.get()
        data_rows = self.current_data.get(data_type, [])

        plotted = False
        for comp_name, comp_index in zip(channel_names, channel_indices):
            if comp_name not in active_components:
                continue
            # Safely extract values from the chosen data rows (qpos or qvel)
            values = [row[comp_index] if comp_index < len(row) else None for row in data_rows]
            # Keep time alignment for non-missing values
            times_filtered = [t for t, v in zip(times, values) if v is not None]
            values_filtered = [v for v in values if v is not None]
            if not values_filtered:
                continue
            self.ax.plot(times_filtered, values_filtered, label=comp_name)
            plotted = True

        if not plotted:
            self.ax.text(0.5, 0.5, "No active channels selected.", ha="center", va="center", transform=self.ax.transAxes)

        self.ax.set_title(f"{joint_name} values over time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Joint value")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    app = LogPlotter()
    app.mainloop()
