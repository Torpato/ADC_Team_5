from __future__ import annotations

import ast
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from Mapper import get_simulation_path


def list_simulations() -> list[Path]:
    sim_root = get_simulation_path("")
    if not sim_root.exists():
        raise FileNotFoundError(f"Simulations folder not found: {sim_root}")

    return sorted(
        path
        for path in sim_root.iterdir()
        if path.is_file() and path.suffix == ".py" and path.name != "__init__.py"
    )


def extract_script_arguments(script_path: Path) -> list[dict[str, str]]:
    if not script_path.exists():
        return []

    try:
        source = script_path.read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError:
        return []

    parser_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        value = node.value
        if not isinstance(value, ast.Call):
            continue

        func = value.func
        if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
            if isinstance(func.value, ast.Name) and func.value.id == "argparse":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        parser_names.add(target.id)
        elif isinstance(func, ast.Name) and func.id == "ArgumentParser":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    parser_names.add(target.id)

    arguments: list[dict[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue

        target = func.value
        if not isinstance(target, ast.Name) or target.id not in parser_names:
            continue

        flags: list[str] = []
        name: str | None = None
        default: str | None = None

        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("-"):
                    flags.append(arg.value)
                elif name is None:
                    name = arg.value

        for keyword in call.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                name = str(keyword.value.value)
            elif keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                default = str(keyword.value.value)

        if name is None and flags:
            name = flags[-1].lstrip("-").replace("-", "_")

        if name is None:
            continue

        label = flags[0] if flags else name
        if default is not None:
            label = f"{label} (default={default})"

        arguments.append({
            "dest": name,
            "flag": flags[0] if flags else name,
            "label": label,
            "is_optional": bool(flags),
        })

    return arguments


def run_simulation(script_path: Path, args: list[str] | None = None) -> None:
    if not script_path.exists():
        messagebox.showerror("Run Simulation", f"Simulation file not found: {script_path}")
        return

    command = [sys.executable, str(script_path)]
    if args:
        command.extend(args)

    kwargs = {
        "cwd": script_path.parent.parent,
    }

    if sys.platform == "win32" and hasattr(subprocess, "CREATE_NEW_CONSOLE"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

    try:
        subprocess.Popen(command, **kwargs)
    except Exception as exc:
        messagebox.showerror("Run Simulation", f"Unable to launch simulation:\n{exc}")
    else:
        messagebox.showinfo("Run Simulation", f"Running {script_path.name} in a new process.")


class SimulationRunnerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ADC Simulation Launcher")
        self.geometry("680x420")
        self.resizable(False, False)

        self.simulations: list[Path] = []
        self.selected_script: Path | None = None
        self.argument_fields: dict[str, tuple[dict[str, str], tk.Entry]] = {}

        self.listbox = tk.Listbox(self, activestyle="dotbox", font=("Segoe UI", 10))
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.listbox.bind("<Double-Button-1>", self.on_double_click)

        scrollbar = tk.Scrollbar(self, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=scrollbar.set)

        self.arg_frame = tk.LabelFrame(self, text="Command-line Arguments")
        self.arg_frame.place(x=340, y=16, width=320, height=280)

        run_button = tk.Button(self, text="Run Selected Simulation", command=self.run_selected)
        refresh_button = tk.Button(self, text="Refresh List", command=self.refresh)
        quit_button = tk.Button(self, text="Quit", command=self.destroy)

        self.status_label = tk.Label(self, text="Select a simulation and enter argument values.", anchor="w")

        self.listbox.place(x=16, y=16, width=300, height=280)
        scrollbar.place(x=316, y=16, width=18, height=280)
        run_button.place(x=16, y=310, width=220, height=40)
        refresh_button.place(x=250, y=310, width=120, height=40)
        quit_button.place(x=382, y=310, width=120, height=40)
        self.status_label.place(x=16, y=360, width=644, height=24)

        self.refresh()

    def refresh(self) -> None:
        self.simulations = list_simulations()
        self.listbox.delete(0, tk.END)

        for path in self.simulations:
            self.listbox.insert(tk.END, path.name)

        self.selected_script = None
        self.clear_argument_fields()
        self.status_label.config(text=f"Loaded {len(self.simulations)} simulations.")

    def clear_argument_fields(self) -> None:
        for child in self.arg_frame.winfo_children():
            child.destroy()
        self.argument_fields.clear()

        label = tk.Label(self.arg_frame, text="No arguments detected.", anchor="w")
        label.place(x=12, y=12)

    def build_argument_fields(self, args: list[dict[str, str]]) -> None:
        self.clear_argument_fields()

        if not args:
            return

        for idx, arg in enumerate(args):
            label = tk.Label(self.arg_frame, text=arg["label"], anchor="w")
            entry = tk.Entry(self.arg_frame, width=30)
            label.place(x=12, y=12 + idx * 32)
            entry.place(x=12, y=32 + idx * 32, width=292)
            self.argument_fields[arg["dest"]] = (arg, entry)

    def on_select(self, event: tk.Event[tk.Widget]) -> None:
        selection = self.listbox.curselection()
        if not selection:
            self.selected_script = None
            self.status_label.config(text="No simulation selected.")
            self.clear_argument_fields()
            return

        self.selected_script = self.simulations[selection[0]]
        self.status_label.config(text=f"Selected: {self.selected_script.name}")

        args = extract_script_arguments(self.selected_script)
        self.build_argument_fields(args)
        if args:
            self.status_label.config(text=f"Selected: {self.selected_script.name}. Arguments available.")

    def on_double_click(self, event: tk.Event[tk.Widget]) -> None:
        self.run_selected()

    def build_command_args(self) -> list[str]:
        args: list[str] = []
        for _, (arg, entry) in self.argument_fields.items():
            value = entry.get().strip()
            if not value:
                continue
            if arg["is_optional"]:
                args.append(arg["flag"])
                args.append(value)
            else:
                args.append(value)
        return args

    def run_selected(self) -> None:
        if self.selected_script is None:
            messagebox.showwarning("Run Simulation", "Please select a simulation file first.")
            return

        args = self.build_command_args()
        self.status_label.config(text=f"Launching {self.selected_script.name}...")
        run_simulation(self.selected_script, args)


def main() -> None:
    app = SimulationRunnerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
