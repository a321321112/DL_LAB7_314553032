# Lab 7 Evaluation Results

This document records evaluation results for Task 1 to Task 3. Each result should include the evaluated model path, command, environment steps, per-seed rewards, mean reward, pass/fail status, and follow-up notes.

## Summary Table

| Task | Algorithm | Environment | Model | Env Steps | Mean Reward | Target | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Task 1 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum.pt` | 199,600 | -225.655 | > -150 | Not passed |
| Task 2 | PPO-Clip + GAE | `Pendulum-v1` | Pending | Pending | Pending | > -150 | Pending |
| Task 3 | PPO-Clip + GAE | `Walker2d-v5` | Pending | Pending | Pending | >= 2500 | Pending |

## Task 1: A2C on Pendulum-v1

### Evaluation Metadata

- Date: 2026-05-15 10:28:11 CST
- Algorithm: A2C
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum.pt`
- Training environment steps in checkpoint: `199600`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-225.655`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.440 | Yes |
| 1 | -0.422 | Yes |
| 2 | -263.759 | No |
| 3 | -722.371 | No |
| 4 | -591.889 | No |
| 5 | -124.545 | Yes |
| 6 | -0.468 | Yes |
| 7 | -127.672 | Yes |
| 8 | -136.909 | Yes |
| 9 | -279.650 | No |
| 10 | -500.780 | No |
| 11 | -299.269 | No |
| 12 | -271.357 | No |
| 13 | -254.237 | No |
| 14 | -268.615 | No |
| 15 | -126.301 | Yes |
| 16 | -2.455 | Yes |
| 17 | -275.195 | No |
| 18 | -134.769 | Yes |
| 19 | -1.996 | Yes |

### Standard Check

- Required average reward: `> -150`
- Current average reward: `-225.655`
- Gap to target: `75.655` reward points below target
- Seeds above `-150`: `10 / 20`
- Seeds below or equal to `-150`: `10 / 20`
- Full-score timing target for Task 1: reach average reward `> -150` within `200000` environment steps
- Current checkpoint step: `199600`
- Current conclusion: the model is close to the full-score step limit, but it **does not meet** the Task 1 performance standard yet because the 20-seed mean reward is below `-150`.

### Notes and Next Actions

- The policy performs well on some seeds but fails badly on several others, especially seeds `3`, `4`, and `10`; this suggests unstable policy behavior rather than complete failure.
- Continue training beyond `199600` steps and re-evaluate seeds `0` to `19`.
- If additional training does not improve the mean reward, try lower actor learning rate or entropy weight, for example:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 1000 \
  --actor-lr 5e-5 \
  --critic-lr 5e-4 \
  --entropy-weight 1e-3 \
  --model-path LAB7_314553032_task1_a2c_pendulum.pt
```

- Keep every future evaluation output in this file so the final report can reference the exact model, environment steps, command, and mean reward.
