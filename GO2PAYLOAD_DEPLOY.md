# Go2 payload policy handoff

The policy uses the stock course wireless runner and the standalone
[`heavy_weights.py`](go2_deploy/custom_implementation/heavy_weights.py)
implementation. The task name is `go2payload`. Its 48 actor inputs include a
2 Hz gait clock. The nominal hip, thigh, and calf angles are `0.0 / 0.96 / −1.36`
rad, and the action scales are `0.2 / 0.5 / 0.5` rad per policy unit.

## Staff controller settings for the unloaded session

| Setting | Value |
| --- | --- |
| Policy control rate | 50 Hz |
| Joint stiffness / damping (Kp / Kd) | 20 / 0.5 on all joints |
| Action smoothing (EMA) | Off |
| Forward / sideways / yaw limits | ±1.0 m/s / ±0.5 m/s / ±1.5 rad/s |

Apply these in the staff deployment configuration. The stock bundle can carry
the gains and joystick limits when they were set at export. The update rate and
EMA must still be set in the configuration used on the robot. Training uses 200 Hz
physics under the 50 Hz actor. The first hardware session is unloaded.

Follow the course's remote procedure in [DEPLOY.md](DEPLOY.md). With the bundle
installed, the staff operator starts the launcher with **L1 + R1**, waits for
recovery and standing, and presses **A** to enter custom control. This
submission uses the course's startup sequence. A separate ramp is optional
only if staff chooses to provide one in their runner.

## Retrain and package

```bash
uv run --frozen --extra gpu --no-dev python -m scripts.train \
  --task go2payload --device cuda:0 --headless --num_envs 4096 \
  --max_iterations 2000 --disable_wandb
```

Share `heavy_weights.py` and the full `logs/go2payload/<run>/` folder, including
the selected `model_*.pt` and `files/gym/envs/` training snapshot. Keep the
snapshot's task registration and both `go2payload.py` and `go2payload_config.py`.
