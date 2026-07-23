"""Dois G1 frente a frente a jogar a bola.

O robo 1 lanca -> o robo 2 apanha -> o robo 2 lanca de volta ->
o robo 1 apanha -> pequena pausa -> recomeca.

Correr com:   python3 catch_game.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "Model/world_catch.xml"

T_RELEASE = 2.26   # instante da largada, contado no relogio de cada robo
R_CATCH = 0.22     # a que distancia a mao "fecha" sobre a bola (m)
T_PAUSA = 0.50     # pausa depois do ultimo apanho, antes de recomecar
T_TIMEOUT = 14.0   # rede de seguranca: se algo correr mal, recomeca

ORANGE = [0.85, 0.35, 0.15, 1]   # bola na mao
GREEN = [0.20, 0.80, 0.30, 1]    # bola em voo

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

ACT = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
BALL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
BALL_GEOM = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
GRIP = {r: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip{r}") for r in (1, 2)}
HAND = {r: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{r}_right_wrist_yaw_link") for r in (1, 2)}
PELVIS = {r: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{r}_pelvis") for r in (1, 2)}
ANKLES = {r: (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{r}_left_ankle_roll_link"),
              mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{r}_right_ankle_roll_link"))
          for r in (1, 2)}

COM_OFFSET = 0.075
K_POS, K_VEL = 1.5, 0.3

# pose de espera: braco a frente, palma aberta para receber a bola
CATCH_POSE = dict(right_shoulder_pitch_joint=-1.45, right_shoulder_roll_joint=-0.15,
                  right_elbow_joint=0.35,
                  left_shoulder_pitch_joint=-0.60, left_shoulder_roll_joint=0.35,
                  left_elbow_joint=0.90)

# o lancamento por cima do ombro, igual ao do pitch_overhand
THROW = [
    (0.00, dict()),
    (0.70, dict(right_shoulder_pitch_joint=0.30, right_shoulder_roll_joint=-0.40,
                right_elbow_joint=0.50,
                left_shoulder_pitch_joint=0.30, left_shoulder_roll_joint=0.40,
                left_elbow_joint=0.50)),
    (1.50, dict(right_shoulder_pitch_joint=-1.80, right_shoulder_roll_joint=-0.55,
                right_elbow_joint=-0.30, waist_yaw_joint=-0.15,
                left_shoulder_pitch_joint=-0.30, left_shoulder_roll_joint=0.30,
                left_elbow_joint=0.60)),
    (2.10, dict(right_shoulder_pitch_joint=-2.90, right_shoulder_roll_joint=-0.55,
                right_elbow_joint=-0.95, waist_yaw_joint=-0.30, waist_pitch_joint=-0.38,
                left_shoulder_pitch_joint=-0.80, left_shoulder_roll_joint=0.25,
                left_elbow_joint=0.40)),
    (2.35, dict(right_shoulder_pitch_joint=-1.10, right_shoulder_roll_joint=0.10,
                right_elbow_joint=1.35, waist_yaw_joint=0.65, waist_pitch_joint=0.35,
                left_shoulder_pitch_joint=0.60, left_shoulder_roll_joint=0.35,
                left_elbow_joint=1.20)),
    (3.05, dict(right_shoulder_pitch_joint=-0.30, right_shoulder_roll_joint=-0.10,
                right_elbow_joint=0.60, waist_yaw_joint=0.55, waist_pitch_joint=0.45,
                left_shoulder_pitch_joint=0.40, left_shoulder_roll_joint=0.35,
                left_elbow_joint=1.00)),
    (3.95, dict(left_hip_pitch_joint=-0.10, right_hip_pitch_joint=-0.10,
                left_knee_joint=0.20, right_knee_joint=0.20)),
]

# quem lanca primeiro: lanca e depois levanta a luva para receber de volta
SEQ_THROW = THROW + [(4.60, CATCH_POSE)]
# quem apanha primeiro: espera de luva no ar e so depois lanca
SEQ_CATCH_THROW = [(0.00, CATCH_POSE)] + THROW[1:]


def smooth(a, b, s):
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def pose_at(t, seq):
    if t <= seq[0][0]:
        return seq[0][1]
    for (t0, p0), (t1, p1) in zip(seq, seq[1:]):
        if t0 <= t < t1:
            s = (t - t0) / (t1 - t0)
            keys = set(p0) | set(p1)
            return {k: smooth(p0.get(k, 0.0), p1.get(k, 0.0), s) for k in keys}
    return seq[-1][1]


def apply_pose(ctrl, r, pose):
    """Escreve uma pose nos atuadores do robo r (os nomes tem prefixo r1_/r2_)."""
    for joint, value in pose.items():
        name = f"r{r}_{joint}"
        if name in ACT:
            ctrl[ACT[name]] = value


def balance(ctrl, r):
    """Equilibrio pelos tornozelos. O robo 2 esta virado ao contrario,
    por isso o sinal das correcoes em x e y tem de ser invertido."""
    com = d.subtree_com[PELVIS[r]]
    vcom = d.cvel[PELVIS[r], 3:6]
    la, ra = ANKLES[r]
    support = 0.5 * (d.xpos[la] + d.xpos[ra])
    sgn = 1.0 if r == 1 else -1.0
    ax = K_POS * (com[0] - (support[0] + COM_OFFSET * sgn)) + K_VEL * vcom[0]
    ay = K_POS * (com[1] - support[1]) + K_VEL * vcom[1]
    for side in ("left", "right"):
        ctrl[ACT[f"r{r}_{side}_ankle_pitch_joint"]] += ax * sgn
        ctrl[ACT[f"r{r}_{side}_ankle_roll_joint"]] -= ay * sgn


def dist_to_hand(r):
    return np.linalg.norm(d.xpos[BALL] - d.xpos[HAND[r]])


def ball_speed():
    return np.linalg.norm(d.cvel[BALL, 3:6])


def reset():
    mujoco.mj_resetData(m, d)
    d.eq_active[GRIP[1]] = 1     # bola na mao do robo 1
    d.eq_active[GRIP[2]] = 0
    m.geom_rgba[BALL_GEOM] = ORANGE


# estados: 0 r1 lanca | 1 bola vai | 2 r2 lanca | 3 bola volta | 4 pausa
reset()
state = 0
t_catch = 0.0     # instante em que o robo 2 apanhou (relogio dele comeca ai)
t_done = 0.0      # instante do ultimo apanho
prev_time = d.time

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # recomeco automatico ou reset manual no viewer (o tempo anda para tras)
        if d.time < prev_time or d.time > T_TIMEOUT or (state == 4 and d.time - t_done > T_PAUSA):
            reset()
            state = 0
        prev_time = d.time

        ctrl = np.zeros(m.nu)

        # --- robo 1: faz sempre a sua sequencia (lanca, depois espera) ---
        apply_pose(ctrl, 1, pose_at(d.time, SEQ_THROW))

        # --- robo 2: espera de luva no ar ate apanhar, dai em diante lanca ---
        if state < 2:
            apply_pose(ctrl, 2, CATCH_POSE)
        else:
            apply_pose(ctrl, 2, pose_at(d.time - t_catch, SEQ_CATCH_THROW))

        balance(ctrl, 1)
        balance(ctrl, 2)
        d.ctrl[:] = ctrl

        # --- maquina de estados ---
        if state == 0 and d.time >= T_RELEASE:
            d.eq_active[GRIP[1]] = 0
            m.geom_rgba[BALL_GEOM] = GREEN
            state = 1
            print(f"robo 1 lanca a {ball_speed():.2f} m/s")

        elif state == 1 and dist_to_hand(2) < R_CATCH:
            d.eq_active[GRIP[2]] = 1
            m.geom_rgba[BALL_GEOM] = ORANGE
            t_catch = d.time
            state = 2
            print("robo 2 apanha")

        elif state == 2 and d.time - t_catch >= T_RELEASE:
            d.eq_active[GRIP[2]] = 0
            m.geom_rgba[BALL_GEOM] = GREEN
            state = 3
            print(f"robo 2 devolve a {ball_speed():.2f} m/s")

        elif state == 3 and dist_to_hand(1) < R_CATCH:
            d.eq_active[GRIP[1]] = 1
            m.geom_rgba[BALL_GEOM] = ORANGE
            t_done = d.time
            state = 4
            print("robo 1 apanha -- troca completa\n")

        mujoco.mj_step(m, d)
        viewer.sync()

        wait = m.opt.timestep - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)
