# Two-Robot PPO: Robot 2 Focus Curriculum

This version keeps the existing 20-action PPO space, so a model trained with
the previous full-exchange environment can be resumed.

## Curriculum

- Training: 70% of episodes start with robot 2 holding the ball.
- Training: 30% of episodes start with robot 1 holding the ball.
- Evaluation: 50% robot 1 / 50% robot 2.
- The residual-action penalty is applied only to the robot whose actions
  currently affect the simulation.
- Robot 2 receives an explicit temporary release milestone reward.
- Starting-state metrics are recorded separately in TensorBoard.

## Files to replace

Copy all files from `Python/` into the project's `Python/` directory:

- catch_controller_FINAL_VERSION.py
- two_robot_catch_env.py
- train_two_robot_ppo.py
- test_two_robot_ppo.py
- smoke_test_two_robot_env.py
- plot_evaluations.py

## Safety

The training script refuses to use a non-empty run directory unless
`--overwrite-run` is explicitly provided. Use a new `--run-name` for every
stage.

## Validate both starting states

```bash
python Python/smoke_test_two_robot_env.py --mode full_exchange --start-robot both --episodes 5
```

## Robot-2 focus stage

Use the existing best model:

```bash
python -u Python/train_two_robot_ppo.py --mode full_exchange --timesteps 300000 --robot2-start-probability 0.70 --eval-robot2-start-probability 0.50 --robot2-release-bonus 30 --learning-rate 5e-5 --throw-noise 0.0 --resume runs/ppo_two_robot_full_exchange/best/best_model.zip --run-name ppo_two_robot_robot2_focus_70_v1
```

If `best_model.zip` does not exist, use the previous `final_model.zip`.

## Evaluate robot 2 directly

```bash
mjpython Python/test_two_robot_ppo.py --mode full_exchange --run-name ppo_two_robot_robot2_focus_70_v1 --start-robot 2 --camera lado --episodes 10
```

## Balanced 50/50 consolidation stage

```bash
python -u Python/train_two_robot_ppo.py --mode full_exchange --timesteps 300000 --robot2-start-probability 0.50 --eval-robot2-start-probability 0.50 --robot2-release-bonus 10 --learning-rate 3e-5 --throw-noise 0.0 --resume runs/ppo_two_robot_robot2_focus_70_v1/best/best_model.zip --run-name ppo_two_robot_balanced_50_v1
```
