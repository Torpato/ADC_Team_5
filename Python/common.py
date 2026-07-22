from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "simple_humanoid.xml"


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_index = mujoco.mj_name2id(model, object_type, name)
    if object_index < 0:
        raise ValueError(f"Object not found: {name}")
    return object_index


def reset_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    keyframe: str = "stand",
    floating: bool = False,
) -> None:
    key_id = object_id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    weld_id = object_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "pelvis_weld")
    data.eq_active[weld_id] = 0 if floating else 1

    mujoco.mj_forward(model, data)


def sensor_value(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_name: str,
) -> np.ndarray:
    sensor_id = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    start = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return data.sensordata[start : start + dimension].copy()


def set_contact_friction(model: mujoco.MjModel, coefficient: float) -> None:
    if coefficient < 0:
        raise ValueError("Friction coefficient must be non-negative.")

    for geom_name in ("floor", "left_foot_geom", "right_foot_geom"):
        geom_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        model.geom_friction[geom_id] = np.array(
            [coefficient, 0.005, 0.0001], dtype=float
        )


def sleep_to_realtime(step_started: float, timestep: float) -> None:
    remaining = timestep - (time.time() - step_started)
    if remaining > 0:
        time.sleep(remaining)


def torso_tilt_degrees(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    torso_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    rotation = data.xmat[torso_id].reshape(3, 3)
    vertical_component = float(np.clip(rotation[2, 2], -1.0, 1.0))
    return float(np.degrees(np.arccos(vertical_component)))


def has_fallen(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    pelvis_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_up_z = data.xmat[torso_id].reshape(3, 3)[2, 2]
    return bool(data.xpos[pelvis_id, 2] < 0.55 or torso_up_z < 0.65)
