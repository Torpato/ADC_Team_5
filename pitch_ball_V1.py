"""
Run with:   python3 pitch_ball_V1.py
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL = "Model/g1_ball.xml"

T_RELEASE = 2.21   
T_RESET = 5.00   

COIL = -0.45       # Chest back rotation 
UNCOIL = 0.25      # Chest front rotation
WAIST_PITCH = 0.08  # Waist pitch in the starting position

# gains for balance
K_POS, K_VEL = 1.5, 0.3

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
A = {n: i for i, n in enumerate(names)}
GRIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grip")
BALL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
L_ANKLE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
R_ANKLE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

# Key poses for the throwing motion, with timestamps and joint angles.
WAYPOINTS = [
    # Neutral position
    (0.00, dict()),

    # Initial position: arms folded in front of the body
    (0.70, dict(right_shoulder_pitch_joint=0.50, right_shoulder_roll_joint=-0.40,
                right_elbow_joint=0.90,
                left_shoulder_pitch_joint=0.50, left_shoulder_roll_joint=0.40,
                left_elbow_joint=0.90)),

    # Chest starts to coil, arm rising
    (1.30, dict(waist_yaw_joint=COIL * 0.5, waist_pitch_joint=-0.10,
                right_shoulder_pitch_joint=0.95, right_shoulder_roll_joint=-0.55,
                right_elbow_joint=1.05,
                left_shoulder_pitch_joint=0.30, left_shoulder_roll_joint=0.20,
                left_elbow_joint=1.30)),

    # Maximum coil: arm extended behind, hand pointed at the target
    (1.95, dict(waist_yaw_joint=COIL, waist_pitch_joint=-0.15,
                right_shoulder_pitch_joint=1.35, right_shoulder_roll_joint=-0.55,
                right_elbow_joint=1.45,
                left_shoulder_pitch_joint=-0.85, left_elbow_joint=0.60)),

    # Uncoil everything at once and here is when the ball is released
    (2.35, dict(waist_yaw_joint=UNCOIL, waist_pitch_joint=WAIST_PITCH,
                right_shoulder_pitch_joint=-1.60, right_shoulder_roll_joint=-0.15,
                right_elbow_joint=0.10,
                left_shoulder_pitch_joint=0.85, left_shoulder_roll_joint=0.40,
                left_elbow_joint=1.45)),

    # Follow through
    (3.05, dict(waist_yaw_joint=UNCOIL + 0.30, waist_pitch_joint=WAIST_PITCH * 0.3,
                right_shoulder_pitch_joint=-0.45, right_elbow_joint=1.15,
                left_shoulder_pitch_joint=0.55, left_shoulder_roll_joint=0.35,
                left_elbow_joint=1.05)),
]


def smooth(a, b, s):
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def pose_at(t):
    """Interpolate between key poses."""
    if t <= WAYPOINTS[0][0]:
        return WAYPOINTS[0][1]
    for (t0, p0), (t1, p1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if t0 <= t < t1:
            s = (t - t0) / (t1 - t0)
            keys = set(p0) | set(p1)
            return {k: smooth(p0.get(k, 0.0), p1.get(k, 0.0), s) for k in keys}
    return WAYPOINTS[-1][1]


def balance(ctrl):
    """Compute ankle torques to keep the robot balanced."""
    com = d.subtree_com[0]
    vcom = d.cvel[1, 3:6]
    support = 0.5 * (d.xpos[L_ANKLE] + d.xpos[R_ANKLE])
    ax = K_POS * (com[0] - support[0]) + K_VEL * vcom[0]
    ay = K_POS * (com[1] - support[1]) + K_VEL * vcom[1]
    for side in ("left", "right"):
        ctrl[A[f"{side}_ankle_pitch_joint"]] += ax
        ctrl[A[f"{side}_ankle_roll_joint"]] -= ay


def reset():
    mujoco.mj_resetData(m, d)
    d.eq_active[GRIP] = 1
    m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")] = [0.85, 0.35, 0.15, 1]


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
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
            m.geom_rgba[gid] = [0.2, 0.8, 0.3, 1]
            print(f"largada a {np.linalg.norm(d.cvel[BALL, 3:6]):.2f} m/s")

        mujoco.mj_step(m, d)
        viewer.sync()

        wait = m.opt.timestep - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)
