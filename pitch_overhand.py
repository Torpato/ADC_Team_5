"""
Run with:   python3 pitch_overhand.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "Model/g1_ball.xml"

T_RELEASE = 2.29   # instante da largada (muito sensivel: +-0.03 s muda tudo)
T_RESET = 5.00     # recomeca o lancamento

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
A = {n: i for i, n in enumerate(names)}
GRIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
BALL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
BALL_GEOM = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
L_ANKLE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
R_ANKLE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

# --- as poses-chave do lancamento -------------------------------------
WAYPOINTS = [
    # posicao neutra
    (0.00, dict()),

    # 1. preparacao
    (0.70, dict(right_shoulder_pitch_joint=0.30, right_shoulder_roll_joint=-0.40,
                right_elbow_joint=0.50,
                left_shoulder_pitch_joint=0.30, left_shoulder_roll_joint=0.40,
                left_elbow_joint=0.50)),

    # 2. levantar o braco pela frente ate ao alto
    (1.50, dict(right_shoulder_pitch_joint=-1.80, right_shoulder_roll_joint=-0.55,
                right_elbow_joint=-0.30,
                waist_yaw_joint=-0.20,
                left_shoulder_pitch_joint=-0.30, left_shoulder_roll_joint=0.30,
                left_elbow_joint=0.60)),

    # 3. armar: braco vertical, cotovelo dobrado, bola atras da cabeca,
    #    tronco enrolado e inclinado para tras
    (2.10, dict(right_shoulder_pitch_joint=-2.90, right_shoulder_roll_joint=-0.55,
                right_elbow_joint=-0.95,
                waist_yaw_joint=-0.40, waist_pitch_joint=-0.18,
                left_shoulder_pitch_joint=-0.80, left_shoulder_roll_joint=0.25,
                left_elbow_joint=0.40)),

    # 4. chicote: tronco flete e desenrola, cotovelo estende,
    #    a mao passa por cima do ombro
    (2.35, dict(right_shoulder_pitch_joint=-1.10, right_shoulder_roll_joint=0.10,
                right_elbow_joint=1.35,
                waist_yaw_joint=0.65, waist_pitch_joint=0.35,
                left_shoulder_pitch_joint=0.60, left_shoulder_roll_joint=0.35,
                left_elbow_joint=1.20)),

    # 5. acompanhamento: o braco cruza o corpo
    (3.05, dict(right_shoulder_pitch_joint=-0.30, right_shoulder_roll_joint=0.10,
                right_elbow_joint=0.60,
                waist_yaw_joint=0.55, waist_pitch_joint=0.45,
                left_shoulder_pitch_joint=0.40, left_shoulder_roll_joint=0.35,
                left_elbow_joint=1.00)),

    # 6. recuperacao: joelhos ligeiramente fletidos, tudo o resto neutro
    (3.95, dict(left_hip_pitch_joint=-0.10, right_hip_pitch_joint=-0.10,
                left_knee_joint=0.20, right_knee_joint=0.20)),
]

# equilibrio: o alvo do centro de massa fica um pouco A FRENTE dos
# tornozelos, porque o pe tem muito mais "dedos" do que calcanhar --
# o robo aguenta desequilibrios para a frente, mas quase nenhum para tras
COM_OFFSET = 0.035
K_POS, K_VEL = 1.5, 0.3


def smooth(a, b, s):
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def pose_at(t):
    if t <= WAYPOINTS[0][0]:
        return WAYPOINTS[0][1]
    for (t0, p0), (t1, p1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if t0 <= t < t1:
            s = (t - t0) / (t1 - t0)
            keys = set(p0) | set(p1)
            return {k: smooth(p0.get(k, 0.0), p1.get(k, 0.0), s) for k in keys}
    return WAYPOINTS[-1][1]


def balance(ctrl):
    """Corrige os tornozelos para manter o centro de massa sobre os pes."""
    com = d.subtree_com[0]
    vcom = d.cvel[1, 3:6]
    support = 0.5 * (d.xpos[L_ANKLE] + d.xpos[R_ANKLE])
    ax = K_POS * (com[0] - (support[0] + COM_OFFSET)) + K_VEL * vcom[0]
    ay = K_POS * (com[1] - support[1]) + K_VEL * vcom[1]
    for side in ("left", "right"):
        ctrl[A[f"{side}_ankle_pitch_joint"]] += ax
        ctrl[A[f"{side}_ankle_roll_joint"]] -= ay


def reset():
    mujoco.mj_resetData(m, d)
    d.eq_active[GRIP] = 1
    m.geom_rgba[BALL_GEOM] = [0.85, 0.35, 0.15, 1]


reset()
released = False

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        if d.time >= T_RESET:
            reset()
            released = False

        ctrl = np.zeros(m.nu)
        for joint, value in pose_at(d.time).items():
            if joint in A:
                ctrl[A[joint]] = value
        balance(ctrl)
        d.ctrl[:] = ctrl

        if not released and d.time >= T_RELEASE:
            d.eq_active[GRIP] = 0
            released = True
            m.geom_rgba[BALL_GEOM] = [0.2, 0.8, 0.3, 1]
            print(f"largada a {np.linalg.norm(d.cvel[BALL, 3:6]):.2f} m/s")

        mujoco.mj_step(m, d)
        viewer.sync()

        wait = m.opt.timestep - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)
