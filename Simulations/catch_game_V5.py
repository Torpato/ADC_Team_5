"""
Run with:   python3 simulations/catch_game_V5.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "Model/world_catch.xml"

T_RELEASE = 2.26   # instante da largada, no relogio de cada lancamento
R_CATCH = 0.15     # janela em que a mao pode fechar sobre a bola (m)
T_RAISE = 0.50     # tempo que o braco demora a subir para apanhar
T_HOLD = 0.80      # segura a bola apos o apanho (com o recuo a dissipar)
T_LOWER = 0.70     # tempo a baixar o braco de volta a pose de descanso
T_SETTLE = 0.80    # tempo a recompor-se depois de apanhar, antes de armar o braco
T_PAUSA = 0.40     # pequena pausa ja em descanso antes de recomecar
T_TIMEOUT = 14.0   # rede de seguranca

ORANGE = [0.85, 0.35, 0.15, 1]
GREEN = [0.20, 0.80, 0.30, 1]

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
BALL_Q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
BALL_V = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]

COM_OFFSET = 0.075
K_POS, K_VEL = 1.5, 0.3

# braco a frente, palma aberta, para receber
CATCH_POSE = dict(right_shoulder_pitch_joint=-1.45, right_shoulder_roll_joint=-0.25,
                  right_elbow_joint=0.20,
                  # o pulso roda para a palma ficar virada para a bola que vem
                  right_wrist_roll_joint=1.20, right_wrist_yaw_joint=-1.00,
                  left_shoulder_pitch_joint=-0.60, left_shoulder_roll_joint=0.35,
                  left_elbow_joint=0.90)

# o robo 1 recebe a bola um pouco mais para o lado (a devolucao vem de um
# ponto ligeiramente diferente), por isso abre mais o ombro
CATCH_POSE_R = {1: dict(CATCH_POSE, right_shoulder_roll_joint=0.00),
                2: dict(CATCH_POSE)}

# pose de descanso: bracos em baixo, joelhos ligeiramente fletidos
REST_POSE = dict(left_hip_pitch_joint=-0.10, right_hip_pitch_joint=-0.10,
                 left_knee_joint=0.20, right_knee_joint=0.20)

# recuo de impacto: quanto cada junta cede ao apanhar (somado a pose de apanho)
RECOIL = dict(right_shoulder_pitch_joint=+0.55,   # o braco cede para tras
              right_elbow_joint=+0.40,
              waist_pitch_joint=-0.18,            # o tronco inclina para tras
              left_hip_pitch_joint=-0.08, right_hip_pitch_joint=-0.08,
              left_knee_joint=+0.16, right_knee_joint=+0.16)
RECOIL_RISE, RECOIL_DECAY = 0.12, 0.45   # sobe em 0.12 s, dissipa em ~0.45 s

# absorcao de corpo inteiro ao receber: agacha e recua sobre os calcanhares.
# ABSORB_GAIN acima de ~0.5 faz o robo sentar-se no chao -- o controlador de
# equilibrio so tem autoridade nos tornozelos e nao aguenta mais do que isto.
ABSORB = dict(waist_pitch_joint=-0.30,
              left_hip_pitch_joint=-0.30, right_hip_pitch_joint=-0.30,
              left_knee_joint=+0.55, right_knee_joint=+0.55,
              left_ankle_pitch_joint=+0.20, right_ankle_pitch_joint=+0.20)
ABSORB_GAIN = 0.30

# o lancamento por cima do ombro
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
    # acompanhamento: a perna esquerda levanta para tras (ver LEG_LIFT abaixo)
    (3.05, dict(right_shoulder_pitch_joint=-0.30, right_shoulder_roll_joint=-0.10,
                right_elbow_joint=0.60, waist_yaw_joint=0.55, waist_pitch_joint=0.45,
                left_shoulder_pitch_joint=0.40, left_shoulder_roll_joint=0.35,
                left_elbow_joint=1.00,
                left_hip_pitch_joint=0.10, left_knee_joint=0.20)),
    (3.95, REST_POSE),
]
THROW[0] = (0.00, REST_POSE)   # o lancamento comeca e acaba na pose de descanso
# quem apanhou lanca a partir da pose de apanho
# quem apanhou espera T_SETTLE a recompor-se e so depois arma o braco
SEQ_CATCH_THROW = ([(0.00, CATCH_POSE), (T_SETTLE, CATCH_POSE)]
                   + [(t + T_SETTLE, p) for t, p in THROW[1:]])


def smooth(a, b, s):
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def mix(p0, p1, s):
    """Interpola entre duas poses (juntas ausentes valem 0)."""
    keys = set(p0) | set(p1)
    return {k: smooth(p0.get(k, 0.0), p1.get(k, 0.0), s) for k in keys}


def pose_at(t, seq):
    if t <= seq[0][0]:
        return seq[0][1]
    for (t0, p0), (t1, p1) in zip(seq, seq[1:]):
        if t0 <= t < t1:
            return mix(p0, p1, (t - t0) / (t1 - t0))
    return seq[-1][1]


def add_recoil(pose, dt):
    """Soma a reacao ao impacto: o braco cede, o tronco inclina para tras e
    o corpo agacha sobre os calcanhares, tudo a dissipar em ~0.5 s."""
    if dt < 0:
        return pose
    if dt < RECOIL_RISE:
        a = dt / RECOIL_RISE
    else:
        a = np.exp(-(dt - RECOIL_RISE) / RECOIL_DECAY)
    out = dict(pose)
    for j, amp in RECOIL.items():
        out[j] = out.get(j, 0.0) + amp * a
    for j, amp in ABSORB.items():
        out[j] = out.get(j, 0.0) + amp * ABSORB_GAIN * a
    return out


def apply_pose(ctrl, r, pose):
    for joint, value in pose.items():
        name = f"r{r}_{joint}"
        if name in ACT:
            ctrl[ACT[name]] = value


def balance(ctrl, r):
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


def snap_to_hand(r):
    """Poe a bola na posicao de equilibrio da soldadura e anula-lhe a
    velocidade, para nao haver o efeito de 'passar e voltar para tras'."""
    R = d.xmat[HAND[r]].reshape(3, 3)
    d.qpos[BALL_Q:BALL_Q + 3] = d.xpos[HAND[r]] - R @ m.eq_data[GRIP[r], 3:6]
    d.qvel[BALL_V:BALL_V + 6] = 0
    mujoco.mj_forward(m, d)


def palm_pos(r):
    """Ponto onde a soldadura vai segurar a bola, ou seja, o centro da palma."""
    R = d.xmat[HAND[r]].reshape(3, 3)
    return d.xpos[HAND[r]] - R @ m.eq_data[GRIP[r], 3:6]


def dist_to_hand(r):
    return np.linalg.norm(d.xpos[BALL] - palm_pos(r))


def should_catch(r, prev):
    """Fecha a mao no ponto de maior aproximacao a palma: dentro da janela
    R_CATCH e ja a afastar-se. Assim o salto da bola e o menor possivel."""
    dist = dist_to_hand(r)
    return dist < R_CATCH and (dist > prev or dist < 0.03), dist


def ball_speed():
    return np.linalg.norm(d.cvel[BALL, 3:6])


def reset():
    """Reposicao do estado inicial. As juntas ficam ja na pose de descanso,
    igual a pose em que os robos terminam o ciclo -- assim os bracos nao
    saltam; o unico ajuste visivel e a base voltar ao lugar (o robo 1
    desloca-se ~13 cm e roda ~17 graus durante o lancamento)."""
    mujoco.mj_resetData(m, d)
    for r in (1, 2):
        for j, v in REST_POSE.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r{r}_{j}")
            d.qpos[m.jnt_qposadr[jid]] = v
    d.eq_active[GRIP[1]] = 1
    d.eq_active[GRIP[2]] = 0
    m.geom_rgba[BALL_GEOM] = ORANGE
    mujoco.mj_forward(m, d)
    snap_to_hand(1)


# estados: 0 r1 lanca | 1 bola vai (r2 levanta o braco) | 2 r2 lanca
#          3 bola volta (r1 levanta o braco) | 4 segura, baixa o braco e recomeca
reset()
state = 0
t_cycle = 0.0     # inicio do ciclo atual (o relogio do lancamento do r1)
t_fly1 = 0.0      # instante em que r1 largou (r2 comeca a subir o braco)
t_catch2 = 0.0    # instante em que r2 apanhou (relogio do lancamento dele)
t_fly2 = 0.0      # instante em que r2 largou (r1 comeca a subir o braco)
t_catch1 = 0.0    # instante em que r1 apanhou de volta
prev_time = d.time
prev_dist = 9.9   # distancia da bola a palma no passo anterior

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # reset manual no viewer, ou rede de seguranca se um apanho falhar
        if d.time < prev_time or d.time - t_cycle > T_TIMEOUT:
            reset()
            state = 0
            t_cycle = d.time
        prev_time = d.time

        ctrl = np.zeros(m.nu)

        # --- robo 1 ---
        if state <= 2:
            # lanca e recolhe para a pose de descanso
            apply_pose(ctrl, 1, pose_at(d.time - t_cycle, THROW))
        elif state == 3:
            # a bola vem a caminho: sobe o braco de descanso para apanho
            apply_pose(ctrl, 1, mix(REST_POSE, CATCH_POSE_R[1], (d.time - t_fly2) / T_RAISE))
        else:
            # apanhou: absorve o impacto, segura, e baixa o braco ate descanso
            dt = d.time - t_catch1
            if dt < T_HOLD:
                apply_pose(ctrl, 1, add_recoil(CATCH_POSE_R[1], dt))
            else:
                apply_pose(ctrl, 1, mix(CATCH_POSE_R[1], REST_POSE, (dt - T_HOLD) / T_LOWER))

        # --- robo 2 ---
        if state == 0:
            # espera de bracos em baixo
            apply_pose(ctrl, 2, REST_POSE)
        elif state == 1:
            # a bola esta no ar: sobe o braco
            apply_pose(ctrl, 2, mix(REST_POSE, CATCH_POSE_R[2], (d.time - t_fly1) / T_RAISE))
        else:
            # apanhou (com recuo de impacto) e depois devolve
            pose = pose_at(d.time - t_catch2, SEQ_CATCH_THROW)
            apply_pose(ctrl, 2, add_recoil(pose, d.time - t_catch2))

        balance(ctrl, 1)
        balance(ctrl, 2)
        d.ctrl[:] = ctrl

        # --- maquina de estados ---
        if state == 0 and d.time - t_cycle >= T_RELEASE:
            d.eq_active[GRIP[1]] = 0
            m.geom_rgba[BALL_GEOM] = GREEN
            t_fly1 = d.time
            state = 1
            print(f"robo 1 lanca a {ball_speed():.2f} m/s")

        elif state == 1 and should_catch(2, prev_dist)[0]:
            snap_to_hand(2)
            d.eq_active[GRIP[2]] = 1
            m.geom_rgba[BALL_GEOM] = ORANGE
            t_catch2 = d.time
            state = 2
            print("robo 2 apanha")

        elif state == 2 and d.time - t_catch2 >= T_RELEASE + T_SETTLE:
            d.eq_active[GRIP[2]] = 0
            m.geom_rgba[BALL_GEOM] = GREEN
            t_fly2 = d.time
            state = 3
            print(f"robo 2 devolve a {ball_speed():.2f} m/s")

        elif state == 3 and should_catch(1, prev_dist)[0]:
            snap_to_hand(1)
            d.eq_active[GRIP[1]] = 1
            m.geom_rgba[BALL_GEOM] = ORANGE
            t_catch1 = d.time
            state = 4
            print("robo 1 apanha -- troca completa\n")

        elif state == 4 and d.time - t_catch1 >= T_HOLD + T_LOWER + T_PAUSA:
            # braco ja em baixo: reposicao quase invisivel e novo ciclo
            reset()                 # mj_resetData poe d.time a 0
            t_cycle = d.time
            prev_time = d.time      # senao o proximo frame parecia um reset manual
            state = 0

        prev_dist = dist_to_hand(2) if state == 1 else (dist_to_hand(1) if state == 3 else 9.9)

        mujoco.mj_step(m, d)
        viewer.sync()

        wait = m.opt.timestep - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)
