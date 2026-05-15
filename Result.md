# Lab 7 Evaluation Results

This document records evaluation results for Task 1 to Task 3. Each result should include the evaluated model path, command, environment steps, per-seed rewards, mean reward, pass/fail status, and follow-up notes.

## Summary Table

| Task | Algorithm | Environment | Model | Env Steps | Mean Reward | Target | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Task 1 run 1 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum.pt` | 199,600 | -225.655 | > -150 | Not passed |
| Task 1 run 2 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_v2.pt` | 384,600 | -162.898 | > -150 | Not passed |
| Task 2 | PPO-Clip + GAE | `Pendulum-v1` | Pending | Pending | Pending | > -150 | Pending |
| Task 3 | PPO-Clip + GAE | `Walker2d-v5` | Pending | Pending | Pending | >= 2500 | Pending |

## Task 1: A2C on Pendulum-v1

### Run 1 Evaluation Metadata

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

### Run 2 Evaluation Metadata

- Date: 2026-05-15 10:56:41 CST
- Algorithm: A2C
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_v2.pt`
- Training environment steps in checkpoint: `384600`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-162.898`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 2 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_v2.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 2 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -131.483 | Yes |
| 1 | -0.516 | Yes |
| 2 | -129.240 | Yes |
| 3 | -258.760 | No |
| 4 | -261.886 | No |
| 5 | -129.757 | Yes |
| 6 | -0.564 | Yes |
| 7 | -128.626 | Yes |
| 8 | -133.658 | Yes |
| 9 | -271.509 | No |
| 10 | -352.201 | No |
| 11 | -269.188 | No |
| 12 | -136.166 | Yes |
| 13 | -249.935 | No |
| 14 | -266.871 | No |
| 15 | -127.849 | Yes |
| 16 | -2.549 | Yes |
| 17 | -272.641 | No |
| 18 | -132.464 | Yes |
| 19 | -2.095 | Yes |

### Run 2 Standard Check

- Required average reward: `> -150`
- Current average reward: `-162.898`
- Gap to target: `12.898` reward points below target
- Seeds above `-150`: `12 / 20`
- Seeds below or equal to `-150`: `8 / 20`
- Full-score timing target for Task 1: reach average reward `> -150` within `200000` environment steps
- Current checkpoint step: `384600`
- Current conclusion: the model is much closer than run 1, but it **still does not meet** the Task 1 performance standard because the 20-seed mean reward is below `-150`.

### Run 2 Comparison Against Run 1

| Metric | Run 1 | Run 2 | Change |
| --- | ---: | ---: | ---: |
| Env steps | 199,600 | 384,600 | +185,000 |
| Mean reward | -225.655 | -162.898 | +62.757 |
| Gap to target | 75.655 | 12.898 | Improved by 62.757 |
| Seeds above -150 | 10 / 20 | 12 / 20 | +2 seeds |

### Run 2 W&B Curve Notes

- The training return improves substantially after roughly `100k` to `150k` steps and remains much better than the first run.
- `critic loss` decreases and becomes mostly near zero late in training, which indicates better value fitting than run 1.
- `actor loss` also becomes smaller and more stable after roughly `150k` to `200k` steps.
- The policy still fails on several seeds, especially seeds `3`, `4`, `9`, `10`, `11`, `13`, `14`, and `17`, so robustness across initial states is still insufficient.

### Run 2 Next Actions

- Continue from the same conservative setting or rerun with slightly lower exploration to reduce bad-seed failures.
- Suggested next training run:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 5e-4 \
  --discount-factor 0.9 \
  --model-path LAB7_314553032_task1_a2c_pendulum_v3.pt \
  --wandb-run-name pendulum-a2c-lr3e5-ent5e4
```

- If run 3 still stays just below `-150`, consider a code-level improvement: save checkpoints by periodic 20-seed evaluation mean instead of single training episode `best_return`, because a single near-zero training episode does not guarantee robust evaluation across seeds.
