import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Python"))

from Mapper import get_model_path

MODEL_PATH = get_model_path("g1_ball.xml")

# --- momentos do lancamento (segundos) ---
T_WIND = 1.00   # comeca a armar o braco
T_SWING = 1.80  # comeca o movimento para a frente
T_END = 2.20    # fim do movimento
T_RELEASE = 2.09  # instante em que a mao larga a bola
T_RESET = 6.00  # recomeca tudo

WIND_SHOULDER, WIND_ELBOW = 1.2, 1.4    # braco atras
THROW_SHOULDER, THROW_ELBOW = -1.6, 0.1  # braco a frente e em cima

m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
d = mujoco.MjData(m)

names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
A = {n: i for i, n in enumerate(names)}
GRIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
BALL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")


def smooth(a, b, s):
    """Interpola de a ate b, com arranque e paragem suaves."""
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def arm_targets(t):
    if t < T_WIND:
        return 0.0, 0.0
    if t < T_SWING:
        s = (t - T_WIND) / (T_SWING - T_WIND)
        return smooth(0, WIND_SHOULDER, s), smooth(0, WIND_ELBOW, s)
    if t < T_END:
        s = (t - T_SWING) / (T_END - T_SWING)
        return (smooth(WIND_SHOULDER, THROW_SHOULDER, s),
                smooth(WIND_ELBOW, THROW_ELBOW, s))
    return THROW_SHOULDER, THROW_ELBOW


def reset():
    mujoco.mj_resetData(m, d)
    d.eq_active[GRIP] = 1


reset()
released = False

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if d.time >= T_RESET:
            reset()
            released = False

        shoulder, elbow = arm_targets(d.time)
        d.ctrl[:] = 0.0
        d.ctrl[A["right_shoulder_pitch_joint"]] = shoulder
        d.ctrl[A["right_elbow_joint"]] = elbow

        if not released and d.time >= T_RELEASE:
            d.eq_active[GRIP] = 0          # a mao larga a bola
            released = True
            v = np.linalg.norm(d.cvel[BALL, 3:6])
            print(f"bola largada a {v:.2f} m/s")

        mujoco.mj_step(m, d)
        viewer.sync()

        # deixa a simulacao correr a velocidade real
        wait = m.opt.timestep - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)
