# Humanoid Robotics in MuJoCo: Model, Actuation, Contact and Disturbance

This package is designed for a two-hour introductory workshop using only MuJoCo, Python and NumPy. It intentionally avoids reinforcement learning, Gymnasium and ROS.

## 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import mujoco; print(mujoco.__version__)"
```

## 2. Verify the model

```bash
python exercise_01_inspect.py
```

For a terminal-only check:

```bash
python exercise_01_inspect.py --headless
```

## 3. Exercise order

### Exercise 1: Inspect the model

```bash
python exercise_01_inspect.py
```

Tasks:

1. Find `worldbody`, `body`, `joint`, `geom`, `site`, `actuator`, `sensor`, `keyframe`, and `equality` in `models/simple_humanoid.xml`.
2. Explain why the free root contributes seven position coordinates but six velocity coordinates.
3. Find the qpos address of the left knee.
4. Find which actuator controls the right ankle pitch joint.
5. Change one visual material and reload the model.

### Exercise 2: Generate a scripted squat

```bash
python exercise_02_pose.py
```

Tasks:

1. Replace smoothstep interpolation with linear interpolation.
2. Compare the resulting motion.
3. Change the default actuator `kp` from 140 to 40, then to 300.
4. Change joint damping from 3 to 0.3, then to 10.
5. Record overshoot, oscillation and tracking delay qualitatively.

### Exercise 3: Measure contact and friction

```bash
python exercise_03_contacts.py --friction 0.9
python exercise_03_contacts.py --friction 0.3
python exercise_03_contacts.py --friction 0.05
```

Tasks:

1. Compare left and right touch-sensor readings before the lateral force.
2. Observe how the centre of mass moves.
3. Record foot displacement for each friction value.
4. Explain why visible contact and large normal force do not guarantee sufficient tangential grip.

### Exercise 4: Run a repeatable push test

```bash
python exercise_04_push_test.py --force 20
python exercise_04_push_test.py --force 30
python exercise_04_push_test.py --force 40
```

Fast terminal-only trials:

```bash
python exercise_04_push_test.py --force 30 --headless
```

Tasks:

1. Find the largest force that does not trigger the fall criterion in three seconds.
2. Repeat at friction coefficients 0.9, 0.3 and 0.05.
3. Compare force with impulse: impulse = force x push duration.
4. Modify the fall criterion and explain the consequences.

## 4. Important model feature

The model contains a weld equality constraint named `pelvis_weld`.

- Exercises 1 and 2 keep it active, creating a safe fixed-base model.
- Exercises 3 and 4 deactivate it, creating a floating-base humanoid.

This allows the same MJCF file to demonstrate both joint motion and whole-body dynamics.

## 5. Suggested 120-minute timetable

- 0-10 min: introduction and final demonstration
- 10-30 min: model inspection
- 30-55 min: posture actuation and interpolation
- 55-65 min: discussion and break
- 65-90 min: sensors, contact and friction
- 90-115 min: disturbance test and fall detection
- 115-120 min: Unitree G1 comparison and conclusions


## Segmentation-fault diagnostic

First test the physics without opening an OpenGL window:

```bash
python exercise_01_inspect.py --headless --duration 0.1
```

If that succeeds, test MuJoCo's standalone viewer directly:

```bash
python -m mujoco.viewer --mjcf=models/simple_humanoid.xml
```

If the standalone viewer also crashes, the problem is in the graphics/GLFW
installation rather than in the workshop model. Create a clean Python `venv`
and install the official PyPI package with `pip`; avoid mixing a pip MuJoCo
wheel with Conda-provided GLFW or OpenGL libraries.

For a Python-level crash trace, run:

```bash
python -X faulthandler exercise_01_inspect.py
```

On macOS, use MuJoCo's required launcher for passive-viewer scripts:

```bash
mjpython exercise_01_inspect.py
```
