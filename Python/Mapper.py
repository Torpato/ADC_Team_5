from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "Model"
PYTHON_ROOT = PROJECT_ROOT / "Python"
SIMULATIONS_ROOT = PROJECT_ROOT / "Simulations"
MJC_WORKSHOP_ROOT = PROJECT_ROOT / "mujoco_humanoid_workshop"


def get_file_path(*relative_parts: str) -> Path:
    """Return a path relative to the repository root.

    Usage:
        from Python.Mapper import get_file_path
        model = get_file_path("Model", "g1_ball.xml")
    """
    return PROJECT_ROOT.joinpath(*relative_parts)


def get_model_path(model_filename: str) -> Path:
    """Return a path to a model file inside the Model directory."""
    return MODEL_ROOT / model_filename


def get_simulation_path(sim_filename: str) -> Path:
    """Return a path to a simulation file inside the Simulations directory."""
    return SIMULATIONS_ROOT / sim_filename


__all__ = [
    "PROJECT_ROOT",
    "MODEL_ROOT",
    "PYTHON_ROOT",
    "SIMULATIONS_ROOT",
    "MJC_WORKSHOP_ROOT",
    "get_file_path",
    "get_model_path",
    "get_simulation_path",
]
