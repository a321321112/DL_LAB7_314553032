# Lab 7 Evaluation Results

This document records evaluation results for Task 1 to Task 3. Each result should include the evaluated model path, command, environment steps, per-seed rewards, mean reward, pass/fail status, and follow-up notes.

## Summary Table

| Task | Algorithm | Environment | Model | Env Steps | Mean Reward | Target | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Task 1 run 1 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum.pt` | 199,600 | -225.655 | > -150 | Not passed |
| Task 1 run 2 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_v2.pt` | 384,600 | -162.898 | > -150 | Not passed |
| Task 1 run 3 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_v3.pt` | 368,800 | -161.347 | > -150 | Not passed |
| Task 1 run 4 | A2C | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_v4.pt` | 550,400 | -168.578 | > -150 | Not passed |
| Task 1 run 5 | A2C + eval-best checkpoint | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_evalbest.pt` | 500,000 | -160.474 | > -150 | Not passed |
| Task 1 run 6 | A2C + n-step + eval-best checkpoint | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_evalbest_nstep.pt` | 480,000 | -161.588 | > -150 | Not passed |
| Task 1 run 7 | A2C + n-step + log-std control + eval-best checkpoint | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_evalbest_logstd.pt` | 500,000 | -164.090 | > -150 | Not passed |
| Task 1 run 8 | A2C + n-step + log-std control + longer training | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_evalbest_logstd_long.pt` | 520,000 | -154.011 | > -150 | Not passed |
| Task 1 run 9 | A2C + n-step + log-std control + untilpass training | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt` | 520,000 | -154.011 | > -150 | Not passed |
| Task 1 run 10 | A2C + n-step + log-std control + gamma 0.95 | `Pendulum-v1` | `LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt` | 720,000 | -144.271 | > -150 | Passed |
| Task 2 run 1 | PPO-Clip + GAE | `Pendulum-v1` | `LAB7_314553032_task2_ppo_pendulum.pt` | 80,000 | -170.936 | > -150 | Not passed |
| Task 2 run 2 | PPO-Clip + GAE tuned | `Pendulum-v1` | `LAB7_314553032_task2_ppo_pendulum_v2.pt` | 200,000 | -149.784 | > -150 | Passed |
| Task 3 run 1 - 1M | PPO-Clip + GAE | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1m.pt` | 1,001,472 | 1370.166 | >= 2500 | Not passed |
| Task 3 run 1 - 1.5M | PPO-Clip + GAE | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1p5m.pt` | 1,501,184 | 787.852 | >= 2500 | Not passed |
| Task 3 run 1 - 2M | PPO-Clip + GAE | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2m.pt` | 2,000,896 | 802.840 | >= 2500 | Not passed |
| Task 3 run 1 - 2.5M | PPO-Clip + GAE | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2p5m.pt` | 2,500,608 | 2033.436 | >= 2500 | Not passed |
| Task 3 run 1 - 3M | PPO-Clip + GAE | `Walker2d-v5` | `LAB7_314553032_task3_ppo_3m.pt` | 3,000,320 | 2083.043 | >= 2500 | Not passed |
| Task 3 run 1 - best | PPO-Clip + GAE | `Walker2d-v5` | `LAB7_314553032_task3_best.pt` | 1,400,000 | 1908.055 | >= 2500 | Not passed |
| Task 3 run 2 - 1M | PPO-Clip + GAE tuned | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 2439.417 | >= 2500 | Not passed |
| Task 3 run 2 - 1.5M | PPO-Clip + GAE tuned | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 2166.355 | >= 2500 | Not passed |
| Task 3 run 2 - 2M | PPO-Clip + GAE tuned | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1405.281 | >= 2500 | Not passed |
| Task 3 run 2 - 2.5M | PPO-Clip + GAE tuned | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 955.271 | >= 2500 | Not passed |
| Task 3 run 2 - 3M | PPO-Clip + GAE tuned | `Walker2d-v5` | `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 2020.433 | >= 2500 | Not passed |
| Task 3 run 2 - best | PPO-Clip + GAE tuned | `Walker2d-v5` | `LAB7_314553032_task3_best_v2.pt` | 1,700,000 | 2537.552 | >= 2500 | Passed |
| Task 3 run 3 - 1M | PPO-Clip + GAE faster early | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 1351.505 | >= 2500 | Not passed |
| Task 3 run 3 - 1.5M | PPO-Clip + GAE faster early | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 1489.832 | >= 2500 | Not passed |
| Task 3 run 3 - 2M | PPO-Clip + GAE faster early | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 2096.274 | >= 2500 | Not passed |
| Task 3 run 3 - 2.5M | PPO-Clip + GAE faster early | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 3025.007 | >= 2500 | Passed |
| Task 3 run 3 - 3M | PPO-Clip + GAE faster early | `Walker2d-v5` | `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 2643.701 | >= 2500 | Passed |
| Task 3 run 3 - best | PPO-Clip + GAE faster early | `Walker2d-v5` | `LAB7_314553032_task3_best_v3.pt` | 3,250,000 | 3662.311 | >= 2500 | Passed |
| Task 3 run 4 - 1M | PPO-Clip + GAE early stable | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 2366.176 | >= 2500 | Not passed |
| Task 3 run 4 - 1.5M | PPO-Clip + GAE early stable | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 1567.004 | >= 2500 | Not passed |
| Task 3 run 4 - 2M | PPO-Clip + GAE early stable | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1263.130 | >= 2500 | Not passed |
| Task 3 run 4 - 2.5M | PPO-Clip + GAE early stable | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 2969.790 | >= 2500 | Passed |
| Task 3 run 4 - 3M | PPO-Clip + GAE early stable | `Walker2d-v5` | `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 3440.120 | >= 2500 | Passed |
| Task 3 run 4 - best | PPO-Clip + GAE early stable | `Walker2d-v5` | `LAB7_314553032_task3_best_v4.pt` | 3,250,000 | 3503.896 | >= 2500 | Passed |
| Task 3 run 5 - 1M | PPO-Clip + GAE early robust | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 1816.296 | >= 2500 | Not passed |
| Task 3 run 5 - 1.5M | PPO-Clip + GAE early robust | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 1021.052 | >= 2500 | Not passed |
| Task 3 run 5 - 2M | PPO-Clip + GAE early robust | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1307.263 | >= 2500 | Not passed |
| Task 3 run 5 - 2.5M | PPO-Clip + GAE early robust | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 2247.786 | >= 2500 | Not passed |
| Task 3 run 5 - 3M | PPO-Clip + GAE early robust | `Walker2d-v5` | `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 2421.599 | >= 2500 | Not passed |
| Task 3 run 5 - best | PPO-Clip + GAE early robust | `Walker2d-v5` | `LAB7_314553032_task3_best_v5.pt` | 3,250,000 | 3174.461 | >= 2500 | Passed |
| Task 3 run 6 - 1M | PPO-Clip + GAE v4 plus exploration | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 2231.630 | >= 2500 | Not passed |
| Task 3 run 6 - 1.5M | PPO-Clip + GAE v4 plus exploration | `Walker2d-v5` | `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 2356.637 | >= 2500 | Not passed |
| Task 3 run 6 - 2M | PPO-Clip + GAE v4 plus exploration | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1352.002 | >= 2500 | Not passed |
| Task 3 run 6 - 2.5M | PPO-Clip + GAE v4 plus exploration | `Walker2d-v5` | `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 2761.016 | >= 2500 | Passed |
| Task 3 run 6 - 3M | PPO-Clip + GAE v4 plus exploration | `Walker2d-v5` | `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 3369.157 | >= 2500 | Passed |
| Task 3 run 6 - best | PPO-Clip + GAE v4 plus exploration | `Walker2d-v5` | `LAB7_314553032_task3_best_v6.pt` | 3,200,000 | 3465.944 | >= 2500 | Passed |

## Task 1: A2C on Pendulum-v1

### Training Settings Summary

| Run | Model | Episodes | Actor LR | Critic LR | Gamma | Entropy Weight | W&B Run | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Run 1 | `LAB7_314553032_task1_a2c_pendulum.pt` | 1000 inferred | 1e-4 inferred | 1e-3 inferred | 0.9 | 1e-2 inferred | Not recorded | Exact training command was not recorded; values match the script defaults used before tuning. |
| Run 2 | `LAB7_314553032_task1_a2c_pendulum_v2.pt` | 2000 recommended | 5e-5 | 5e-4 | 0.9 | 1e-3 | `pendulum-a2c-lr5e5-ent1e3` inferred | Conservative LR and lower entropy run used after run 1 instability. |
| Run 3 | `LAB7_314553032_task1_a2c_pendulum_v3.pt` | 2500 | 3e-5 | 3e-4 | 0.9 | 5e-4 | `pendulum-a2c-lr3e5-ent5e4` | User-provided command. |
| Run 4 | `LAB7_314553032_task1_a2c_pendulum_v4.pt` | 3000 | 1e-5 | 3e-4 | 0.9 | 1e-4 | `pendulum-a2c-lr1e5-critic3e4-ent1e4` | User-provided command; lower actor LR and entropy did not improve final 20-seed mean. |
| Run 5 | `LAB7_314553032_task1_a2c_pendulum_evalbest.pt` | 2500 | 3e-5 | 3e-4 | 0.9 | 5e-4 | `pendulum-a2c-evalbest-lr3e5-ent5e4` | Uses periodic 20-seed eval-best checkpointing every 20000 steps. |
| Run 6 | `LAB7_314553032_task1_a2c_pendulum_evalbest_nstep.pt` | 2500 | 3e-5 | 3e-4 | 0.9 | 5e-4 | `pendulum-a2c-nstep-evalbest` | Adds `--n-step 5`, `--reward-scale 10.0`, and default advantage normalization. |
| Run 7 | `LAB7_314553032_task1_a2c_pendulum_evalbest_logstd.pt` | 2500 | 3e-5 | 3e-4 | 0.9 | 1e-4 | `pendulum-a2c-logstd-evalbest` | Adds `--init-log-std -0.5`, `--min-log-std -2.0`, and `--max-log-std 0.5`. |
| Run 8 | `LAB7_314553032_task1_a2c_pendulum_evalbest_logstd_long.pt` | 4000 | 3e-5 | 3e-4 | 0.9 | 1e-4 | `pendulum-a2c-logstd-long` | Same as run 7, only longer training. |
| Run 9 | `LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt` | 10000 cap | 3e-5 | 3e-4 | 0.9 | 1e-4 | `pendulum-a2c-untilpass` | Same as run 8 with `--target-eval-mean -150`; target was not reached. |
| Run 10 | `LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt` | 10000 cap | 3e-5 | 3e-4 | 0.95 | 1e-4 | `pendulum-a2c-gamma095` | Same as run 9 except `--discount-factor 0.95`; passed Task 1 target. |

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

### Run 3 Training Command

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

### Run 3 Evaluation Metadata

- Date: 2026-05-15 11:30:06 CST
- Algorithm: A2C
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_v3.pt`
- Training environment steps in checkpoint: `368800`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-161.347`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 3 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_v3.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 3 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.641 | Yes |
| 1 | -0.462 | Yes |
| 2 | -126.503 | Yes |
| 3 | -244.723 | No |
| 4 | -280.890 | No |
| 5 | -124.037 | Yes |
| 6 | -0.508 | Yes |
| 7 | -127.850 | Yes |
| 8 | -131.682 | Yes |
| 9 | -269.912 | No |
| 10 | -356.260 | No |
| 11 | -256.740 | No |
| 12 | -133.746 | Yes |
| 13 | -247.316 | No |
| 14 | -264.187 | No |
| 15 | -126.119 | Yes |
| 16 | -2.494 | Yes |
| 17 | -269.432 | No |
| 18 | -131.316 | Yes |
| 19 | -2.117 | Yes |

### Run 3 Standard Check

- Required average reward: `> -150`
- Current average reward: `-161.347`
- Gap to target: `11.347` reward points below target
- Seeds above `-150`: `12 / 20`
- Seeds below or equal to `-150`: `8 / 20`
- Full-score timing target for Task 1: reach average reward `> -150` within `200000` environment steps
- Current checkpoint step: `368800`
- Current conclusion: run 3 is the best mean reward so far, but it **still does not meet** the Task 1 performance standard.

### Run 3 Comparison

| Metric | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Env steps | 199,600 | 384,600 | 368,800 |
| Mean reward | -225.655 | -162.898 | -161.347 |
| Gap to target | 75.655 | 12.898 | 11.347 |
| Seeds above -150 | 10 / 20 | 12 / 20 | 12 / 20 |

Run 3 improves only `+1.551` mean reward over run 2. The lower actor learning rate and entropy weight improved training smoothness, but did not materially fix the same bad-seed failure cases.

### Run 3 W&B Curve Notes

- The return curve improves earlier and then plateaus, with most late returns much better than the initial `-1500` region.
- The late-stage return is still noisy and occasionally drops far below the main cluster, so robustness is still the key issue.
- `critic loss` becomes mostly near zero after roughly `150k` to `200k` steps, so the critic is no longer the primary bottleneck.
- `actor loss` also stabilizes after roughly `150k` steps, which suggests the current hyperparameters are not causing obvious update explosions.
- `best_return` reaches near-zero early, but this is not enough for grading because the fixed 20-seed evaluation mean remains below `-150`.

### Run 3 Next Actions

The next improvement should focus on model selection and bad-seed robustness, not just longer training.

Recommended code-level improvement:

- Add periodic evaluation during training every fixed number of environment steps, for example every `10000` or `20000` steps.
- Evaluate seeds `0` to `19` with deterministic mean action.
- Save the best checkpoint by 20-seed mean reward, not by single training episode `best_return`.
- This directly matches the assignment grading protocol and avoids selecting a checkpoint that only performs well on one easy episode.

If continuing with hyperparameter-only tuning first, use a slightly more conservative v4:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 3000 \
  --actor-lr 1e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --model-path LAB7_314553032_task1_a2c_pendulum_v4.pt \
  --wandb-run-name pendulum-a2c-lr1e5-critic3e4-ent1e4
```

Expected effect of v4:

- Lower `actor-lr` should reduce policy movement between updates.
- Lower `entropy-weight` should reduce excessive exploration and may reduce late return drops.
- Keeping `critic-lr` at `3e-4` is reasonable because critic loss already stabilizes in run 3.

However, because run 2 and run 3 are very close, periodic evaluation checkpointing is likely more valuable than another long run with only small hyperparameter changes.

### Run 4 Training Command

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 3000 \
  --actor-lr 1e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --model-path LAB7_314553032_task1_a2c_pendulum_v4.pt \
  --wandb-run-name pendulum-a2c-lr1e5-critic3e4-ent1e4
```

### Run 4 Evaluation Metadata

- Date: 2026-05-15 14:42:37 CST
- Algorithm: A2C
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_v4.pt`
- Training environment steps in checkpoint: `550400`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-168.578`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 4 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_v4.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 4 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.186 | Yes |
| 1 | -0.380 | Yes |
| 2 | -128.604 | Yes |
| 3 | -249.277 | No |
| 4 | -409.412 | No |
| 5 | -123.769 | Yes |
| 6 | -0.427 | Yes |
| 7 | -127.448 | Yes |
| 8 | -132.751 | Yes |
| 9 | -269.283 | No |
| 10 | -355.013 | No |
| 11 | -265.714 | No |
| 12 | -135.709 | Yes |
| 13 | -247.731 | No |
| 14 | -264.321 | No |
| 15 | -126.219 | Yes |
| 16 | -2.424 | Yes |
| 17 | -269.073 | No |
| 18 | -131.857 | Yes |
| 19 | -1.960 | Yes |

### Run 4 Standard Check

- Required average reward: `> -150`
- Current average reward: `-168.578`
- Gap to target: `18.578` reward points below target
- Seeds above `-150`: `12 / 20`
- Seeds below or equal to `-150`: `8 / 20`
- Full-score timing target for Task 1: reach average reward `> -150` within `200000` environment steps
- Current checkpoint step: `550400`
- Current conclusion: run 4 does **not** meet the Task 1 performance standard and is worse than run 3 despite longer training.

### Run 4 Comparison

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | ---: | ---: | ---: | ---: |
| Env steps | 199,600 | 384,600 | 368,800 | 550,400 |
| Mean reward | -225.655 | -162.898 | -161.347 | -168.578 |
| Gap to target | 75.655 | 12.898 | 11.347 | 18.578 |
| Seeds above -150 | 10 / 20 | 12 / 20 | 12 / 20 | 12 / 20 |

Run 4 is `7.231` reward points worse than run 3. Lowering `actor-lr` to `1e-5` and `entropy-weight` to `1e-4` did not improve robustness across the same difficult seeds.

### Run 4 W&B Curve Notes

- The return curve improves and then plateaus, but the late-stage return remains in a similar band to run 3.
- `critic loss` becomes stable and near zero after the early phase, so critic fitting is not the main issue.
- `actor loss` is stable late in training, indicating that simply reducing actor LR further does not solve the bad-seed failures.
- The same difficult seed pattern remains: seeds `3`, `4`, `9`, `10`, `11`, `13`, `14`, and `17` fail the `-150` threshold.

### Run 4 Conclusion and Next Action

The results now strongly suggest that more hyperparameter-only training is unlikely to reliably pass Task 1. The training code should be changed so that checkpoint selection matches the grading metric.

Recommended code change:

- Add a separate deterministic evaluation environment.
- During training, every `eval_interval` environment steps, evaluate seeds `0` to `19`.
- Save `LAB7_314553032_task1_a2c_pendulum_evalbest.pt` only when the 20-seed mean reward improves.
- Log `eval/mean_reward`, `eval/best_mean_reward`, and optionally each seed reward to W&B.
- Keep the current training-episode `best_return` logging for diagnostics, but stop using it as the main model-selection criterion.

Recommended implementation parameters:

```txt
eval_interval = 10000 or 20000 environment steps
eval_seed_start = 0
eval_seed_end = 19
eval_model_path = LAB7_314553032_task1_a2c_pendulum_evalbest.pt
```

After this change, rerun the best current hyperparameter setting, preferably run 3 settings:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 5e-4 \
  --discount-factor 0.9 \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest.pt \
  --wandb-run-name pendulum-a2c-evalbest-lr3e5-ent5e4
```

### Run 5 Training Command

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 5e-4 \
  --discount-factor 0.9 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-evalbest-lr3e5-ent5e4
```

### Run 5 Evaluation Metadata

- Date: 2026-05-15 15:47:07 CST
- Algorithm: A2C with periodic eval-best checkpointing
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_evalbest.pt`
- Training environment steps in checkpoint: `500000`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-160.474`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 5 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 5 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.094 | Yes |
| 1 | -0.394 | Yes |
| 2 | -126.227 | Yes |
| 3 | -245.244 | No |
| 4 | -268.612 | No |
| 5 | -124.581 | Yes |
| 6 | -0.442 | Yes |
| 7 | -127.359 | Yes |
| 8 | -131.970 | Yes |
| 9 | -269.503 | No |
| 10 | -353.584 | No |
| 11 | -254.584 | No |
| 12 | -133.407 | Yes |
| 13 | -247.390 | No |
| 14 | -264.163 | No |
| 15 | -126.261 | Yes |
| 16 | -2.431 | Yes |
| 17 | -269.617 | No |
| 18 | -131.649 | Yes |
| 19 | -1.970 | Yes |

### Run 5 Standard Check

- Required average reward: `> -150`
- Current average reward: `-160.474`
- Gap to target: `10.474` reward points below target
- Seeds above `-150`: `12 / 20`
- Seeds below or equal to `-150`: `8 / 20`
- Current checkpoint step: `500000`
- Current conclusion: run 5 is the best result so far, but it **still does not meet** the Task 1 performance standard.

### Run 5 Comparison

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Env steps | 199,600 | 384,600 | 368,800 | 550,400 | 500,000 |
| Mean reward | -225.655 | -162.898 | -161.347 | -168.578 | -160.474 |
| Gap to target | 75.655 | 12.898 | 11.347 | 18.578 | 10.474 |
| Seeds above -150 | 10 / 20 | 12 / 20 | 12 / 20 | 12 / 20 | 12 / 20 |

Run 5 improves checkpoint selection and is the current best mean reward, but the same `8 / 20` difficult seeds remain below the threshold.

### Run 5 W&B Curve Notes

- `eval/mean_reward` improves quickly early in training and then plateaus around the `-180` to `-170` region, with the best checkpoint reaching `-160.474` by external evaluation.
- `eval/best_mean_reward` flattens after the main improvement period, which suggests the current A2C setup has mostly converged under these settings.
- Training `return` continues to show good individual episodes, but fixed-seed eval confirms that robust mean performance remains below the target.

### Run 5 Next Actions

The eval-best checkpointing change helped select the best available snapshot, but it did not solve the underlying bad-seed robustness issue. Further improvement likely requires changing the learning signal, not only training longer.

Recommended next code-level changes, in priority order:

1. Implemented: n-step returns for A2C, controlled by `--n-step`.
2. Implemented: advantage normalization before actor update, enabled by default.
3. Implemented: optional reward scaling, controlled by `--reward-scale`.
4. Implemented after run 6: explicit action standard deviation control with `--init-log-std`, `--min-log-std`, and `--max-log-std`.

Recommended next experiment after code changes:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 5e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_nstep.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_nstep.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-nstep-evalbest
```

### Run 6 Training Command

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 5e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_nstep.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_nstep.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-nstep-evalbest
```

### Run 6 Evaluation Metadata

- Date: 2026-05-15
- Algorithm: A2C with n-step rollout updates, reward scaling, advantage normalization, and periodic eval-best checkpointing
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_evalbest_nstep.pt`
- Training environment steps in checkpoint: `480000`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-161.588`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 6 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_nstep.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 6 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.826 | Yes |
| 1 | -0.434 | Yes |
| 2 | -126.203 | Yes |
| 3 | -253.576 | No |
| 4 | -270.215 | No |
| 5 | -126.368 | Yes |
| 6 | -0.479 | Yes |
| 7 | -128.249 | Yes |
| 8 | -131.157 | Yes |
| 9 | -268.629 | No |
| 10 | -351.910 | No |
| 11 | -267.454 | No |
| 12 | -133.671 | Yes |
| 13 | -246.983 | No |
| 14 | -264.012 | No |
| 15 | -126.849 | Yes |
| 16 | -2.491 | Yes |
| 17 | -269.275 | No |
| 18 | -130.963 | Yes |
| 19 | -2.005 | Yes |

### Run 6 Standard Check

- Required average reward: `> -150`
- Current average reward: `-161.588`
- Gap to target: `11.588` reward points below target
- Seeds above `-150`: `12 / 20`
- Seeds below or equal to `-150`: `8 / 20`
- Current checkpoint step: `480000`
- Current conclusion: run 6 does **not** meet the Task 1 performance standard and is slightly worse than run 5 by `1.114` reward points.

### Run 6 Comparison

| Metric | Run 5 | Run 6 |
| --- | ---: | ---: |
| Env steps in selected checkpoint | 500,000 | 480,000 |
| Mean reward | -160.474 | -161.588 |
| Gap to target | 10.474 | 11.588 |
| Seeds above -150 | 12 / 20 | 12 / 20 |

Run 6 learns faster early in the W&B eval curve, but it converges to nearly the same fixed-seed performance band as run 5. The difficult seeds remain `3`, `4`, `9`, `10`, `11`, `13`, `14`, and `17`, so n-step returns and reward scaling did not solve the main robustness issue.

### Why the Training Step Looks Smaller

There are two separate meanings of step in this run:

1. The checkpoint metadata `training_environment_step=480000` means the saved `evalbest_nstep` model was selected at environment step `480000`. The training run can continue after that, but the eval-best checkpoint only updates when the 20-seed mean reward improves. If later evaluations are worse, the saved checkpoint keeps the earlier step.
2. The W&B chart x-axis labeled `Step` is W&B's internal logging step, not necessarily the environment step. After switching to `--n-step 5`, actor/critic losses are logged once per rollout update instead of once per environment transition. That reduces the number of W&B log rows by roughly a factor of 5, so the orange curve can visually end around a smaller x-axis value even though the logged `step` metric still reaches about `500000` environment steps.

For reporting, use the explicit `step` metric or the checkpoint field `training_environment_step`, not W&B's internal x-axis step count.

### Run 6 Next Actions

- Do not continue tuning only `n-step` yet; run 6 shows faster early improvement but no better final robustness.
- Implemented after run 6: action distribution control with configurable Gaussian `log_std` initialization and clamp bounds.
- Next experiment should reduce exploration variance directly: `--init-log-std -0.5 --min-log-std -2.0 --max-log-std 0.5 --entropy-weight 1e-4`.
- If hard seeds still fail, compare W&B `action/log_std` and `action/clamped_log_std` against evaluation mean reward before changing learning rates again.

Recommended next experiment after action std control:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_logstd.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_logstd.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-logstd-evalbest
```

### Run 7 Training Command

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2500 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_logstd.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_logstd.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-logstd-evalbest
```

### Run 7 Evaluation Metadata

- Date: 2026-05-15 18:15:26 CST
- Algorithm: A2C with n-step rollout updates, reward scaling, advantage normalization, action log-std control, and periodic eval-best checkpointing
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_evalbest_logstd.pt`
- Training environment steps in checkpoint: `500000`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-164.090`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 7 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_logstd.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 7 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -133.447 | Yes |
| 1 | -4.103 | Yes |
| 2 | -132.674 | Yes |
| 3 | -259.822 | No |
| 4 | -395.935 | No |
| 5 | -125.930 | Yes |
| 6 | -4.247 | Yes |
| 7 | -130.793 | Yes |
| 8 | -135.995 | Yes |
| 9 | -268.176 | No |
| 10 | -347.726 | No |
| 11 | -273.042 | No |
| 12 | -140.042 | Yes |
| 13 | -249.107 | No |
| 14 | -135.369 | Yes |
| 15 | -129.163 | Yes |
| 16 | -6.286 | Yes |
| 17 | -269.466 | No |
| 18 | -135.054 | Yes |
| 19 | -5.423 | Yes |

### Run 7 Standard Check

- Required average reward: `> -150`
- Current average reward: `-164.090`
- Gap to target: `14.090` reward points below target
- Seeds above `-150`: `13 / 20`
- Seeds below or equal to `-150`: `7 / 20`
- Current checkpoint step: `500000`
- Current conclusion: run 7 does **not** meet the Task 1 performance standard. It improves the number of seeds above `-150` from `12 / 20` to `13 / 20`, but the mean reward is worse than run 5 and run 6 because several failed seeds are still large negative outliers.

### Run 7 Comparison

| Metric | Run 5 | Run 6 | Run 7 |
| --- | ---: | ---: | ---: |
| Env steps in selected checkpoint | 500,000 | 480,000 | 500,000 |
| Mean reward | -160.474 | -161.588 | -164.090 |
| Gap to target | 10.474 | 11.588 | 14.090 |
| Seeds above -150 | 12 / 20 | 12 / 20 | 13 / 20 |

Run 7's curve improves quickly early, then continues with small late improvements instead of completely flattening. Since the selected checkpoint is at `500000` steps, the eval-best model was still improving at the end of this run. This supports one controlled follow-up where only training length is increased.

### Run 7 Next Action

Use the same hyperparameters as run 7 and only increase `--num-episodes`. The goal is to test whether the late-stage eval-best improvement continues without mixing in another parameter change.

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 4000 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_logstd_long.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_logstd_long.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-logstd-long
```

### Run 8 Training Command

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 4000 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_logstd_long.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_logstd_long.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --wandb-run-name pendulum-a2c-logstd-long
```

### Run 8 Evaluation Metadata

- Date: 2026-05-15 18:43:22 CST
- Algorithm: A2C with n-step rollout updates, reward scaling, advantage normalization, action log-std control, and periodic eval-best checkpointing
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_evalbest_logstd_long.pt`
- Training environment steps in checkpoint: `520000`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-154.011`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 8 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_logstd_long.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 8 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.943 | Yes |
| 1 | -0.503 | Yes |
| 2 | -126.962 | Yes |
| 3 | -253.557 | No |
| 4 | -264.764 | No |
| 5 | -123.080 | Yes |
| 6 | -0.549 | Yes |
| 7 | -128.385 | Yes |
| 8 | -131.424 | Yes |
| 9 | -265.025 | No |
| 10 | -345.161 | No |
| 11 | -267.399 | No |
| 12 | -134.347 | Yes |
| 13 | -245.306 | No |
| 14 | -134.134 | Yes |
| 15 | -126.156 | Yes |
| 16 | -2.612 | Yes |
| 17 | -266.924 | No |
| 18 | -130.922 | Yes |
| 19 | -2.075 | Yes |

### Run 8 Standard Check

- Required average reward: `> -150`
- Current average reward: `-154.011`
- Gap to target: `4.011` reward points below target
- Seeds above `-150`: `13 / 20`
- Seeds below or equal to `-150`: `7 / 20`
- Current checkpoint step: `520000`
- Current conclusion: run 8 is the best result so far and is close to passing, but it still **does not** meet the Task 1 performance standard.

### Run 8 Comparison

| Metric | Run 5 | Run 6 | Run 7 | Run 8 |
| --- | ---: | ---: | ---: | ---: |
| Env steps in selected checkpoint | 500,000 | 480,000 | 500,000 | 520,000 |
| Mean reward | -160.474 | -161.588 | -164.090 | -154.011 |
| Gap to target | 10.474 | 11.588 | 14.090 | 4.011 |
| Seeds above -150 | 12 / 20 | 12 / 20 | 13 / 20 | 13 / 20 |

Longer training improved run 7 by `10.079` reward points and is now only `4.011` points below the target. The remaining failures are concentrated in seeds `3`, `4`, `9`, `10`, `11`, `13`, and `17`.

### Run 8 Next Action

The code now supports automatic early stopping by fixed-seed eval-best mean. Use a large episode cap and stop when `eval/best_mean_reward > -150`.

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 10000 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_untilpass.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-a2c-untilpass
```

### Run 9 Training Command

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 10000 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.9 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_untilpass.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-a2c-untilpass
```

### Run 9 Evaluation Metadata

- Date: 2026-05-16 09:03:55 CST
- Algorithm: A2C with n-step rollout updates, reward scaling, advantage normalization, action log-std control, periodic eval-best checkpointing, and target eval early stopping
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt`
- Training environment steps in selected checkpoint: `520000`
- Observed training curve reached: approximately `2000000` environment steps
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-154.011`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 9 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 9 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.943 | Yes |
| 1 | -0.503 | Yes |
| 2 | -126.962 | Yes |
| 3 | -253.557 | No |
| 4 | -264.764 | No |
| 5 | -123.080 | Yes |
| 6 | -0.549 | Yes |
| 7 | -128.385 | Yes |
| 8 | -131.424 | Yes |
| 9 | -265.025 | No |
| 10 | -345.161 | No |
| 11 | -267.399 | No |
| 12 | -134.347 | Yes |
| 13 | -245.306 | No |
| 14 | -134.134 | Yes |
| 15 | -126.156 | Yes |
| 16 | -2.612 | Yes |
| 17 | -266.924 | No |
| 18 | -130.922 | Yes |
| 19 | -2.075 | Yes |

### Run 9 Standard Check

- Required average reward: `> -150`
- Current average reward: `-154.011`
- Gap to target: `4.011` reward points below target
- Seeds above `-150`: `13 / 20`
- Seeds below or equal to `-150`: `7 / 20`
- Current selected checkpoint step: `520000`
- Current conclusion: run 9 does **not** meet the Task 1 performance standard. It is numerically identical to run 8, which means the longer untilpass run did not find a better eval-best model after step `520000`.

### Why Run 9 Did Not Reach the Target

- The W&B training `step` curve reaches about `2000000`, but `training_environment_step=520000` in the checkpoint. This means the best fixed-seed evaluation happened early, and no later periodic evaluation improved it.
- `eval/best_mean_reward` becomes flat around `-154`, so more training with the same setting has likely converged to a local optimum.
- The same hard seeds remain below the threshold: `3`, `4`, `9`, `10`, `11`, `13`, and `17`.
- The mean reward is only `4.011` points below target, but seed `10` at `-345.161` and several `-260` range failures dominate the average.
- Because 13 seeds already pass, the main issue is not global instability. The policy still lacks a robust swing-up behavior for a subset of initial states.

### Run 9 Next Action

Do not keep extending the same command. The next controlled experiment should improve long-horizon swing-up behavior. Change only the discount factor first, because `gamma=0.9` may be too short-sighted for hard Pendulum initial states.

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 10000 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.95 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_gamma095.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-a2c-gamma095
```

If `gamma=0.95` still plateaus below `-150`, the next code-level fix should address the action clipping mismatch: the environment receives a clipped action, while the policy gradient currently uses the log probability of the pre-clipped sampled action.

### Run 10 Training Command

Recorded before evaluation on 2026-05-16 09:43:04 CST.

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 10000 \
  --actor-lr 3e-5 \
  --critic-lr 3e-4 \
  --entropy-weight 1e-4 \
  --discount-factor 0.95 \
  --n-step 5 \
  --reward-scale 10.0 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_gamma095.pt \
  --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-a2c-gamma095
```

### Run 10 Evaluation Metadata

- Date: 2026-05-16 10:20:09 CST
- Algorithm: A2C with n-step rollout updates, reward scaling, advantage normalization, action log-std control, periodic eval-best checkpointing, target eval early stopping, and `gamma=0.95`
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt`
- Training environment steps in selected checkpoint: `720000`
- Evaluation device shown by run: `cpu`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-144.271`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Passed**

### Run 10 Evaluation Command

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 10 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -131.044 | Yes |
| 1 | -1.489 | Yes |
| 2 | -125.728 | Yes |
| 3 | -243.061 | No |
| 4 | -256.891 | No |
| 5 | -119.188 | Yes |
| 6 | -1.545 | Yes |
| 7 | -128.074 | Yes |
| 8 | -131.416 | Yes |
| 9 | -257.844 | No |
| 10 | -333.565 | No |
| 11 | -253.214 | No |
| 12 | -132.396 | Yes |
| 13 | -242.822 | No |
| 14 | -126.448 | Yes |
| 15 | -126.482 | Yes |
| 16 | -3.485 | Yes |
| 17 | -136.571 | Yes |
| 18 | -130.977 | Yes |
| 19 | -3.173 | Yes |

### Run 10 Standard Check

- Required average reward: `> -150`
- Current average reward: `-144.271`
- Margin above target: `5.729` reward points
- Seeds above `-150`: `14 / 20`
- Seeds below or equal to `-150`: `6 / 20`
- Current selected checkpoint step: `720000`
- Current conclusion: run 10 **meets** the Task 1 performance standard.

### Run 10 Comparison

| Metric | Run 8 | Run 9 | Run 10 |
| --- | ---: | ---: | ---: |
| Discount factor | 0.9 | 0.9 | 0.95 |
| Env steps in selected checkpoint | 520,000 | 520,000 | 720,000 |
| Mean reward | -154.011 | -154.011 | -144.271 |
| Gap or margin to target | -4.011 | -4.011 | +5.729 |
| Seeds above -150 | 13 / 20 | 13 / 20 | 14 / 20 |

Increasing `gamma` from `0.9` to `0.95` improved the 20-seed mean by `9.740` reward points and pushed the model over the Task 1 threshold. The result supports the earlier diagnosis that the remaining hard seeds needed a longer-horizon learning signal.

### Run 10 Artifact Note

Keep `LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt` as the Task 1 final model snapshot. This file should be included in the final homework submission package if the assignment asks for model snapshots. It should not be committed to git unless explicitly required, because model checkpoint files are binary artifacts and may be large.

## Task 2: PPO-Clip + GAE on Pendulum-v1

### Run 1 Training Command

```bash
python ppo_pendulum.py \
  --mode train \
  --num-episodes 1000 \
  --actor-lr 3e-4 \
  --critic-lr 1e-3 \
  --discount-factor 0.9 \
  --tau 0.95 \
  --entropy-weight 1e-3 \
  --epsilon 0.2 \
  --rollout-len 2048 \
  --update-epoch 10 \
  --batch-size 64 \
  --model-path LAB7_314553032_task2_ppo_pendulum_trainbest.pt \
  --eval-model-path LAB7_314553032_task2_ppo_pendulum.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-ppo-baseline
```

### Run 1 Evaluation Metadata

- Date: 2026-05-16 11:30:39 CST
- Algorithm: PPO-Clip with GAE
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task2_ppo_pendulum.pt`
- Training environment steps in selected checkpoint: `80000`
- Evaluation device shown by run: `cpu`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-170.936`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Not passed**

### Run 1 Evaluation Command

```bash
python ppo_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task2_ppo_pendulum.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 1 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -130.543 | Yes |
| 1 | -0.887 | Yes |
| 2 | -126.203 | Yes |
| 3 | -271.847 | No |
| 4 | -415.161 | No |
| 5 | -124.497 | Yes |
| 6 | -0.751 | Yes |
| 7 | -127.332 | Yes |
| 8 | -130.858 | Yes |
| 9 | -272.318 | No |
| 10 | -363.199 | No |
| 11 | -276.061 | No |
| 12 | -133.893 | Yes |
| 13 | -248.559 | No |
| 14 | -263.039 | No |
| 15 | -126.262 | Yes |
| 16 | -5.244 | Yes |
| 17 | -269.666 | No |
| 18 | -130.180 | Yes |
| 19 | -2.220 | Yes |

### Run 1 Standard Check

- Required average reward: `> -150`
- Current average reward: `-170.936`
- Gap to target: `20.936` reward points below target
- Seeds above `-150`: `12 / 20`
- Seeds below or equal to `-150`: `8 / 20`
- Current selected checkpoint step: `80000`
- Current conclusion: run 1 does **not** meet the Task 2 performance standard.

### Run 1 Curve Notes

- `action/log_std` drops quickly to the lower clamp around `-2`, so the policy becomes nearly deterministic too early.
- Training return remains highly unstable after the early learning phase, with many episodes still dropping near `-1500`.
- The eval-best checkpoint was selected at only `80000` environment steps, while the W&B step metric shows training continued much longer. This indicates later training did not improve the fixed-seed eval mean.
- The failed seeds are similar to Task 1 hard seeds, but seed `4` and seed `10` are worse than the passing A2C run. PPO did not yet improve robustness.

### Run 1 Next Action

Do not continue the same setting. The next run should keep exploration alive longer and make PPO updates less aggressive.

Recommended next training command:

```bash
python ppo_pendulum.py \
  --mode train \
  --num-episodes 1000 \
  --actor-lr 1e-4 \
  --critic-lr 5e-4 \
  --discount-factor 0.95 \
  --tau 0.95 \
  --entropy-weight 2e-3 \
  --epsilon 0.2 \
  --rollout-len 4096 \
  --update-epoch 5 \
  --batch-size 128 \
  --init-log-std -0.25 \
  --min-log-std -1.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task2_ppo_pendulum_trainbest_v2.pt \
  --eval-model-path LAB7_314553032_task2_ppo_pendulum_v2.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-ppo-stable-v2
```

Rationale:

- `actor-lr 1e-4` and `update-epoch 5` reduce policy update aggressiveness.
- `rollout-len 4096` and `batch-size 128` give PPO a more stable batch.
- `min-log-std -1.0` prevents exploration from collapsing to `std ~= 0.135`; the minimum becomes `std ~= 0.368`.
- `entropy-weight 2e-3` helps resist premature deterministic behavior.
- `discount-factor 0.95` follows the successful Task 1 diagnosis that hard seeds need a longer-horizon signal.

### Run 2 Training Command

```bash
python ppo_pendulum.py \
  --mode train \
  --num-episodes 1000 \
  --actor-lr 1e-4 \
  --critic-lr 5e-4 \
  --discount-factor 0.95 \
  --tau 0.95 \
  --entropy-weight 2e-3 \
  --epsilon 0.2 \
  --rollout-len 4096 \
  --update-epoch 5 \
  --batch-size 128 \
  --init-log-std -0.25 \
  --min-log-std -1.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task2_ppo_pendulum_trainbest_v2.pt \
  --eval-model-path LAB7_314553032_task2_ppo_pendulum_v2.pt \
  --eval-interval 20000 \
  --eval-seed-start 0 \
  --eval-seed-end 19 \
  --target-eval-mean -150 \
  --wandb-run-name pendulum-ppo-stable-v2
```

### Run 2 Evaluation Metadata

- Date: 2026-05-16 11:47:02 CST
- Algorithm: PPO-Clip with GAE
- Environment: `Pendulum-v1`
- Model snapshot: `LAB7_314553032_task2_ppo_pendulum_v2.pt`
- Training environment steps in selected checkpoint: `200000`
- Evaluation device shown by run: `cpu`
- Evaluation seeds: `0` to `19`
- Number of evaluation episodes: `20`
- Mean reward: `-149.784`
- Assignment target: average reward `> -150` over 20 evaluation episodes
- Result: **Passed**

### Run 2 Evaluation Command

```bash
python ppo_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task2_ppo_pendulum_v2.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

### Run 2 Per-Seed Rewards

| Seed | Reward | Above -150 |
| ---: | ---: | --- |
| 0 | -127.955 | Yes |
| 1 | -0.631 | Yes |
| 2 | -125.184 | Yes |
| 3 | -245.315 | No |
| 4 | -262.657 | No |
| 5 | -118.033 | Yes |
| 6 | -0.632 | Yes |
| 7 | -125.365 | Yes |
| 8 | -130.550 | Yes |
| 9 | -257.645 | No |
| 10 | -321.026 | No |
| 11 | -258.130 | No |
| 12 | -132.190 | Yes |
| 13 | -242.966 | No |
| 14 | -127.974 | Yes |
| 15 | -123.089 | Yes |
| 16 | -3.588 | Yes |
| 17 | -260.603 | No |
| 18 | -130.003 | Yes |
| 19 | -2.150 | Yes |

### Run 2 Standard Check

- Required average reward: `> -150`
- Current average reward: `-149.784`
- Margin above target: `0.216` reward points
- Seeds above `-150`: `13 / 20`
- Seeds below or equal to `-150`: `7 / 20`
- Current selected checkpoint step: `200000`
- Current conclusion: run 2 **meets** the Task 2 performance standard and reaches the threshold at the full-score step boundary.

### Run 2 Comparison

| Metric | Task 2 Run 1 | Task 2 Run 2 |
| --- | ---: | ---: |
| Actor LR | 3e-4 | 1e-4 |
| Critic LR | 1e-3 | 5e-4 |
| Gamma | 0.9 | 0.95 |
| Entropy weight | 1e-3 | 2e-3 |
| Rollout length | 2048 | 4096 |
| Update epochs | 10 | 5 |
| Batch size | 64 | 128 |
| Min log std | -2.0 | -1.0 |
| Env steps in selected checkpoint | 80,000 | 200,000 |
| Mean reward | -170.936 | -149.784 |
| Gap or margin to target | -20.936 | +0.216 |
| Seeds above -150 | 12 / 20 | 13 / 20 |

### Why Run 2 Was Much Smoother

- The policy did not collapse exploration as quickly. In run 1, `action/log_std` hit the `-2` lower clamp early. In run 2, `min-log-std=-1.0` kept the minimum standard deviation around `exp(-1) ~= 0.368`, so PPO continued exploring hard initial states longer.
- The PPO update became less aggressive. Reducing `actor-lr` from `3e-4` to `1e-4` and `update-epoch` from `10` to `5` reduced destructive policy jumps.
- The rollout batch became more stable. `rollout-len=4096` and `batch-size=128` gave each PPO update more diverse trajectories and less noisy gradient estimates.
- `gamma=0.95` gave the agent a longer-horizon learning signal, which helped the same hard seeds that limited Task 1 A2C.
- PPO's clipped objective lets the model reuse rollout data across mini-batch epochs while limiting policy-ratio changes. Once the hyperparameters stopped over-updating the policy, PPO became more sample-efficient than the tuned A2C run.

### Why the Environment Interaction Count Is Lower

Task 2 run 2 selected a passing checkpoint at `200000` environment steps, while the best Task 1 A2C checkpoint passed at `720000` steps. PPO needed fewer environment interactions because each rollout is reused for several mini-batch epochs. In other words, PPO performs more gradient updates per collected sample, but clipping prevents those repeated updates from moving too far away from the behavior policy that collected the data.

The W&B panel x-axis can be visually confusing because update logs are fewer than environment steps. For grading and report, use the explicit checkpoint field:

```txt
training_environment_step=200000
```

### Run 2 Artifact Note

Keep `LAB7_314553032_task2_ppo_pendulum_v2.pt` as the current Task 2 final model snapshot. It should be included in the final homework submission package if model snapshots are required. Do not commit it to git unless explicitly requested.

## Task 3: PPO-Clip with GAE on Walker2d-v5

### Run 1 Training Settings

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 1500 \
  --actor-lr 3e-4 \
  --critic-lr 3e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 1e-3 \
  --value-coef 0.5 \
  --epsilon 0.2 \
  --rollout-len 2048 \
  --update-epoch 10 \
  --batch-size 64 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --init-log-std -0.5 \
  --min-log-std -2.0 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task3_train_latest.pt \
  --eval-model-path LAB7_314553032_task3_best.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 50000 \
  --eval-seed-start 0 \
  --eval-seed-end 4 \
  --wandb-run-name walker-ppo-baseline
```

### Run 1 Evaluation Summary

Date recorded: 2026-05-16

| Checkpoint | Training Environment Step | Mean Reward | Target | Status |
| --- | ---: | ---: | ---: | --- |
| `LAB7_314553032_task3_ppo_1m.pt` | 1,001,472 | 1370.166 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_1p5m.pt` | 1,501,184 | 787.852 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2m.pt` | 2,000,896 | 802.840 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2p5m.pt` | 2,500,608 | 2033.436 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_3m.pt` | 3,000,320 | 2083.043 | >= 2500 | Not passed |
| `LAB7_314553032_task3_best.pt` | 1,400,000 | 1908.055 | >= 2500 | Not passed |

### Run 1 Evaluation Command

```bash
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_3m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_best.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

### Run 1 Per-Seed Rewards

| Seed | 1M | 1.5M | 2M | 2.5M | 3M | Best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2818.553 | 1338.742 | 693.303 | 1880.593 | 2443.953 | 1405.919 |
| 1 | 1416.853 | 165.334 | 1344.781 | 1571.603 | 1551.739 | 1554.372 |
| 2 | 1089.211 | 1330.106 | 1018.590 | 1641.866 | 1012.635 | 1140.146 |
| 3 | 1177.363 | 589.523 | 799.308 | 3223.045 | 2186.228 | 1226.185 |
| 4 | 1042.459 | 449.528 | 618.794 | 3221.958 | 1335.610 | 1095.819 |
| 5 | 1234.585 | 393.055 | 808.400 | 681.370 | 2268.548 | 1049.194 |
| 6 | 1406.597 | 514.097 | 1205.809 | 1999.117 | 2864.666 | 1791.870 |
| 7 | 1276.571 | 713.866 | 757.344 | 1378.675 | 1695.475 | 1021.103 |
| 8 | 1354.865 | 514.827 | 682.769 | 1315.799 | 2018.846 | 2768.787 |
| 9 | 1049.038 | 760.324 | 693.987 | 1192.777 | 1606.018 | 2691.751 |
| 10 | 1080.762 | 831.361 | 640.124 | 2058.860 | 1535.192 | 1373.520 |
| 11 | 1101.881 | 780.220 | 178.526 | 1509.727 | 1251.457 | 3148.549 |
| 12 | 1284.362 | 490.075 | 931.060 | 1741.834 | 2013.482 | 3171.998 |
| 13 | 1284.853 | 165.089 | 996.788 | 2333.781 | 2968.971 | 3407.035 |
| 14 | 1249.132 | 1980.578 | 727.069 | 2558.559 | 2617.116 | 3033.582 |
| 15 | 1598.638 | 1469.460 | 714.105 | 1915.871 | 2865.116 | 815.347 |
| 16 | 2069.278 | 711.868 | 985.897 | 3295.637 | 3316.481 | 1073.185 |
| 17 | 1331.445 | 476.308 | 645.108 | 2956.862 | 1615.597 | 1651.041 |
| 18 | 1460.407 | 762.593 | 713.407 | 3136.871 | 3152.880 | 2921.105 |
| 19 | 1076.474 | 1320.089 | 901.630 | 1053.913 | 1340.845 | 1820.588 |
| Mean | 1370.166 | 787.852 | 802.840 | 2033.436 | 2083.043 | 1908.055 |

### Run 1 Curve Diagnosis

- Training return increases and frequently reaches above 2500 after later training, so the policy can learn a walking behavior.
- The 20-seed evaluation mean does not reach 2500. The best fixed checkpoint is 3M with `2083.043`, still `416.957` below target.
- `action/log_std`, `action/log_std_min`, and `action/log_std_max` all hit the lower clamp `-2.0` very early. Entropy also collapses early. This means exploration becomes nearly deterministic too soon for Walker2d.
- `clip_fraction` stays high, roughly around `0.55` to `0.65` for much of training. This usually means many PPO updates are outside the clipping range, so the policy update is too aggressive for the collected rollout distribution.
- The train-time eval-best checkpoint was selected using seeds `0` to `4`, but the final 20-seed result of `LAB7_314553032_task3_best.pt` is only `1908.055`. This means 5-seed eval was not representative enough for selecting the submitted best snapshot.

### Run 1 Conclusion

Current Task 3 run is promising but not passed. The next run should prioritize stability and checkpoint selection quality over raw training return:

- use more eval seeds during training, preferably `0` to `9`;
- reduce PPO update aggressiveness;
- prevent exploration from collapsing to `log_std=-2.0` too early;
- keep fixed 3M snapshots for grading.

### Run 2 Training Settings

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 800 \
  --actor-lr 1e-4 \
  --critic-lr 3e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 3e-3 \
  --value-coef 0.5 \
  --epsilon 0.1 \
  --rollout-len 4096 \
  --update-epoch 5 \
  --batch-size 256 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --init-log-std -0.5 \
  --min-log-std -1.5 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task3_train_latest_v2.pt \
  --eval-model-path LAB7_314553032_task3_best_v2.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 100000 \
  --eval-seed-start 0 \
  --eval-seed-end 9 \
  --wandb-run-name walker-ppo-stable-v2
```

### Run 2 Evaluation Summary

Date recorded: 2026-05-16

| Checkpoint | Training Environment Step | Mean Reward | Target | Status |
| --- | ---: | ---: | ---: | --- |
| `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 2439.417 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 2166.355 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1405.281 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 955.271 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 2020.433 | >= 2500 | Not passed |
| `LAB7_314553032_task3_best_v2.pt` | 1,700,000 | 2537.552 | >= 2500 | **Passed** |

### Run 2 Evaluation Command

```bash
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_3m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_best_v2.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

### Run 2 Per-Seed Rewards

| Seed | 1M | 1.5M | 2M | 2.5M | 3M | Best v2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2411.908 | 2674.958 | 1306.645 | 894.438 | 2713.420 | 2640.454 |
| 1 | 2316.552 | 2687.454 | 1036.007 | 992.814 | 1554.247 | 2544.356 |
| 2 | 2930.882 | 1808.179 | 1365.959 | 913.937 | 1705.652 | 2491.398 |
| 3 | 2478.689 | 2101.804 | 1338.756 | 961.491 | 1482.401 | 2482.266 |
| 4 | 2408.352 | 2739.682 | 1304.574 | 979.996 | 2920.900 | 2503.973 |
| 5 | 2396.114 | 1892.332 | 1242.153 | 1007.464 | 2796.372 | 2473.379 |
| 6 | 2646.261 | 2151.019 | 1276.414 | 922.843 | 1297.314 | 2572.261 |
| 7 | 2359.963 | 2115.078 | 1287.798 | 912.076 | 3355.378 | 2502.200 |
| 8 | 2337.186 | 1873.658 | 1209.507 | 910.283 | 3285.746 | 2474.193 |
| 9 | 2448.391 | 2177.395 | 1287.752 | 916.531 | 1290.791 | 2510.019 |
| 10 | 2442.647 | 2077.540 | 1306.879 | 932.337 | 3012.168 | 2596.110 |
| 11 | 2442.286 | 2556.785 | 1456.574 | 1031.437 | 1361.801 | 2467.947 |
| 12 | 2353.220 | 1823.492 | 1339.277 | 961.058 | 2190.873 | 2513.318 |
| 13 | 2361.262 | 1868.155 | 1374.849 | 964.567 | 1444.793 | 2516.267 |
| 14 | 2542.288 | 2074.982 | 1224.326 | 929.839 | 1645.265 | 2674.430 |
| 15 | 2353.733 | 2255.079 | 2522.750 | 997.887 | 1464.189 | 2485.064 |
| 16 | 2293.603 | 1660.307 | 1337.298 | 964.505 | 1425.418 | 2624.149 |
| 17 | 2557.894 | 2699.748 | 1034.751 | 997.492 | 1438.135 | 2502.879 |
| 18 | 2355.721 | 1745.451 | 1155.417 | 964.874 | 1503.060 | 2589.568 |
| 19 | 2351.384 | 2344.005 | 2697.929 | 949.554 | 2520.744 | 2586.816 |
| Mean | 2439.417 | 2166.355 | 1405.281 | 955.271 | 2020.433 | 2537.552 |

### Run 2 Standard Check

- Required average reward: `>= 2500`
- Current best 20-seed mean reward: `2537.552`
- Margin above target: `37.552`
- Best checkpoint: `LAB7_314553032_task3_best_v2.pt`
- Best checkpoint training environment step: `1700000`
- Seeds at or above `2500`: `13 / 20`
- Seeds below `2500`: `7 / 20`
- Current conclusion: run 2 **meets** the Task 3 performance standard.

### Run 2 Curve Diagnosis

- Compared with run 1, `action/log_std` declines much more slowly and does not hit the `-2.0` collapse. The average log standard deviation is still around `-0.9` late in training, so the policy keeps enough stochasticity for Walker2d.
- Entropy stays high for far longer. This improves exploration and prevents the agent from committing too early to a brittle gait.
- `clip_fraction` falls from the run 1 range of roughly `0.55-0.65` to about `0.13-0.22`, meaning fewer samples are being clipped. The PPO update is therefore less aggressive and closer to the collected rollout distribution.
- The curve is slower at the beginning because the update is more conservative, but the later policy is much more robust across seeds.
- Using evaluation seeds `0` to `9` during training selected a checkpoint that generalizes better to the final `0` to `19` evaluation.

### Run 2 Conclusion

Task 3 is now passed by `LAB7_314553032_task3_best_v2.pt` with mean reward `2537.552` at `1700000` environment steps. For final submission, keep:

```txt
LAB7_314553032_task3_ppo_1m.pt
LAB7_314553032_task3_ppo_1p5m.pt
LAB7_314553032_task3_ppo_2m.pt
LAB7_314553032_task3_ppo_2p5m.pt
LAB7_314553032_task3_ppo_3m.pt
LAB7_314553032_task3_best_v2.pt
```

Before packaging, rename or copy `LAB7_314553032_task3_best_v2.pt` to the required submission name:

```txt
LAB7_314553032_task3_best.pt
```

### Run 3 Training Settings

Run 3 keeps the stable v2 recipe but slightly increases early policy improvement speed:

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 800 \
  --actor-lr 1.5e-4 \
  --critic-lr 3e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 3e-3 \
  --value-coef 0.5 \
  --epsilon 0.12 \
  --rollout-len 4096 \
  --update-epoch 5 \
  --batch-size 256 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --init-log-std -0.5 \
  --min-log-std -1.5 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task3_train_latest_v3.pt \
  --eval-model-path LAB7_314553032_task3_best_v3.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 50000 \
  --eval-seed-start 0 \
  --eval-seed-end 9 \
  --wandb-run-name walker-ppo-v3-faster-early
```

### Run 3 Evaluation Summary

Date recorded: 2026-05-16

| Checkpoint | Training Environment Step | Mean Reward | Target | Status |
| --- | ---: | ---: | ---: | --- |
| `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 1351.505 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 1489.832 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 2096.274 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 3025.007 | >= 2500 | **Passed** |
| `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 2643.701 | >= 2500 | **Passed** |
| `LAB7_314553032_task3_best_v3.pt` | 3,250,000 | 3662.311 | >= 2500 | **Passed** |

### Run 3 Evaluation Command

```bash
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_3m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_best_v3.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

### Run 3 Per-Seed Rewards

| Seed | 1M | 1.5M | 2M | 2.5M | 3M | Best v3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1290.695 | 1905.966 | 1485.480 | 3392.802 | 2634.615 | 3597.424 |
| 1 | 1305.807 | 1427.701 | 3267.287 | 3292.333 | 3506.680 | 3788.385 |
| 2 | 1216.285 | 2083.619 | 1616.946 | 3286.938 | 1366.959 | 3745.848 |
| 3 | 1386.031 | 1744.130 | 1640.494 | 3368.474 | 1465.434 | 3577.045 |
| 4 | 1284.787 | 1182.277 | 2879.988 | 3254.150 | 1495.366 | 3743.760 |
| 5 | 1386.807 | 1231.507 | 3151.769 | 3131.822 | 1276.794 | 3435.139 |
| 6 | 1511.068 | 1525.156 | 3431.682 | 2067.281 | 3395.611 | 3745.788 |
| 7 | 1222.347 | 1378.221 | 1628.690 | 3092.386 | 1792.483 | 3645.482 |
| 8 | 959.760 | 1610.840 | 1880.195 | 2843.585 | 3454.645 | 3581.310 |
| 9 | 1313.990 | 1439.484 | 1882.388 | 2616.546 | 3385.815 | 3642.751 |
| 10 | 1186.265 | 1595.911 | 1602.815 | 3217.869 | 3491.036 | 3603.424 |
| 11 | 1382.123 | 1182.841 | 1730.851 | 3260.425 | 3459.783 | 3776.546 |
| 12 | 1379.670 | 1181.891 | 2052.878 | 3373.599 | 3670.983 | 3535.018 |
| 13 | 1393.417 | 1277.466 | 1598.825 | 3398.524 | 2984.231 | 3764.545 |
| 14 | 1552.808 | 1325.541 | 1734.187 | 3448.263 | 1330.728 | 3852.281 |
| 15 | 1404.375 | 1621.333 | 2555.595 | 2938.208 | 3355.766 | 3822.981 |
| 16 | 1455.564 | 2155.380 | 1932.438 | 1725.072 | 3564.000 | 3302.595 |
| 17 | 1682.610 | 1292.611 | 2306.116 | 3443.946 | 2385.905 | 3635.475 |
| 18 | 1331.346 | 1621.239 | 1646.213 | 1854.567 | 1323.917 | 3815.840 |
| 19 | 1384.341 | 1013.527 | 1900.641 | 3493.340 | 3533.279 | 3634.581 |
| Mean | 1351.505 | 1489.832 | 2096.274 | 3025.007 | 2643.701 | 3662.311 |

### Run 3 Standard Check

- Required average reward: `>= 2500`
- Earliest passing fixed snapshot: `LAB7_314553032_task3_ppo_2p5m.pt`
- Earliest passing fixed snapshot mean reward: `3025.007`
- Earliest passing fixed snapshot training environment step: `2502656`
- Best checkpoint: `LAB7_314553032_task3_best_v3.pt`
- Best checkpoint mean reward: `3662.311`
- Best checkpoint training environment step: `3250000`
- Current conclusion: run 3 improves Task 3 from the run 2 best-only pass to a fixed-snapshot pass at 2.5M steps.

### Run 3 Curve Comparison Against Run 2

Run 3 is more aggressive than run 2:

| Setting | Run 2 stable | Run 3 faster early | Expected effect |
| --- | ---: | ---: | --- |
| Actor LR | 1e-4 | 1.5e-4 | Faster policy improvement, higher instability risk |
| Clip epsilon | 0.1 | 0.12 | Allows slightly larger PPO policy movement |
| Eval interval | 100000 | 50000 | Captures good checkpoints more often |
| Entropy weight | 3e-3 | 3e-3 | Same exploration pressure |
| Min log std | -1.5 | -1.5 | Same lower bound for stochasticity |
| Rollout length | 4096 | 4096 | Same rollout stability |
| Update epochs | 5 | 5 | Same conservative reuse of rollout data |
| Batch size | 256 | 256 | Same lower gradient noise |

Observed curve differences:

- Run 3 return grows later but reaches much higher final performance. The best checkpoint reaches `3662.311`, far above run 2's `2537.552`.
- Run 3 `action/log_std` and entropy decline faster than run 2. This means the policy becomes more deterministic earlier, which helped exploit a good gait after the agent discovered it.
- Run 3 `clip_fraction` rises higher than run 2 late in training, and critic loss spikes more strongly. This shows the increased actor LR and epsilon make updates more aggressive.
- The early fixed snapshots are worse than run 2: run 3 1M is `1351.505`, while run 2 1M was `2439.417`. The faster setting did not improve early sample efficiency; it improved later exploitation.
- Run 3 becomes clearly strong by 2.5M and remains above target at 3M, so it is better for final performance but not better for 1M/1.5M scoring.

### Run 3 Conclusion

Run 3 passes Task 3 at the 2.5M fixed snapshot and achieves the best overall model so far. For final submission scoring by fixed snapshot, use run 3 snapshots if the goal is at least a 2.5M pass. If the goal is to pass earlier, the next experiment should keep run 2's early stability but add only a small amount of extra early learning speed.

### Run 4 Training Settings

Run 4 moves back toward run 2 stability while keeping a small early-learning boost:

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 800 \
  --actor-lr 1.2e-4 \
  --critic-lr 3e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 2.5e-3 \
  --value-coef 0.5 \
  --epsilon 0.11 \
  --rollout-len 4096 \
  --update-epoch 5 \
  --batch-size 256 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --init-log-std -0.5 \
  --min-log-std -1.5 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task3_train_latest_v4.pt \
  --eval-model-path LAB7_314553032_task3_best_v4.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 50000 \
  --eval-seed-start 0 \
  --eval-seed-end 9 \
  --wandb-run-name walker-ppo-v4-early-stable
```

### Run 4 Evaluation Summary

Date recorded: 2026-05-17

| Checkpoint | Training Environment Step | Mean Reward | Target | Status |
| --- | ---: | ---: | ---: | --- |
| `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 2366.176 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 1567.004 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1263.130 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 2969.790 | >= 2500 | **Passed** |
| `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 3440.120 | >= 2500 | **Passed** |
| `LAB7_314553032_task3_best_v4.pt` | 3,250,000 | 3503.896 | >= 2500 | **Passed** |

### Run 4 Evaluation Command

```bash
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_3m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_best_v4.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

### Run 4 Per-Seed Rewards

| Seed | 1M | 1.5M | 2M | 2.5M | 3M | Best v4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2599.617 | 1419.403 | 1135.089 | 3602.126 | 1997.116 | 3482.639 |
| 1 | 2558.082 | 1463.808 | 1164.587 | 2957.193 | 3686.575 | 3616.744 |
| 2 | 2637.632 | 1365.438 | 1444.119 | 3191.416 | 3643.735 | 3620.039 |
| 3 | 2575.550 | 1950.452 | 1488.574 | 3538.228 | 3637.214 | 3508.346 |
| 4 | 1582.184 | 2307.900 | 1224.456 | 3469.809 | 3771.648 | 3595.096 |
| 5 | 2542.571 | 1274.877 | 1582.675 | 3325.695 | 3669.482 | 3760.473 |
| 6 | 2731.230 | 1278.741 | 1339.293 | 3063.712 | 3646.020 | 3563.269 |
| 7 | 2472.504 | 1282.928 | 1152.324 | 3126.359 | 3430.726 | 3689.960 |
| 8 | 2624.315 | 1281.910 | 1182.664 | 3314.424 | 3656.461 | 3719.312 |
| 9 | 2042.065 | 1351.066 | 1434.399 | 3473.775 | 3675.491 | 3783.621 |
| 10 | 1552.392 | 2311.893 | 1346.496 | 3311.533 | 3635.292 | 3543.247 |
| 11 | 2644.361 | 1462.515 | 1170.436 | 2383.118 | 3583.433 | 3575.370 |
| 12 | 2511.037 | 1813.731 | 1105.560 | 1641.558 | 3653.603 | 3589.753 |
| 13 | 2496.513 | 2042.563 | 1013.844 | 2853.162 | 3591.735 | 3563.096 |
| 14 | 2301.111 | 1460.139 | 1093.376 | 2573.005 | 3656.436 | 3725.068 |
| 15 | 2490.780 | 1285.000 | 1198.711 | 3410.715 | 3259.933 | 2576.486 |
| 16 | 2571.137 | 1268.973 | 1059.880 | 2776.778 | 1466.987 | 3661.065 |
| 17 | 1743.404 | 1327.680 | 1317.085 | 2304.705 | 3757.391 | 3647.072 |
| 18 | 2611.570 | 1753.594 | 1543.688 | 3457.303 | 3673.704 | 2346.649 |
| 19 | 2035.471 | 1637.467 | 1265.338 | 1621.190 | 3709.410 | 3510.609 |
| Mean | 2366.176 | 1567.004 | 1263.130 | 2969.790 | 3440.120 | 3503.896 |

### Run 4 Standard Check

- Required average reward: `>= 2500`
- Earliest passing fixed snapshot: `LAB7_314553032_task3_ppo_2p5m.pt`
- Earliest passing fixed snapshot mean reward: `2969.790`
- Earliest passing fixed snapshot training environment step: `2502656`
- Best checkpoint: `LAB7_314553032_task3_best_v4.pt`
- Best checkpoint mean reward: `3503.896`
- Best checkpoint training environment step: `3250000`
- Current conclusion: run 4 also passes at 2.5M, but does not improve the earliest passing step relative to run 3.

### Run 4 Curve Comparison

Compared with run 3, run 4 is slightly more conservative:

| Setting | Run 3 faster early | Run 4 early stable | Expected effect |
| --- | ---: | ---: | --- |
| Actor LR | 1.5e-4 | 1.2e-4 | Less aggressive policy movement |
| Clip epsilon | 0.12 | 0.11 | Slightly tighter PPO update |
| Entropy weight | 3e-3 | 2.5e-3 | Slightly less exploration pressure |
| Eval interval | 50000 | 50000 | Same checkpoint selection frequency |
| Min log std | -1.5 | -1.5 | Same stochasticity floor |

Observed curve differences:

- Run 4 keeps entropy and `action/log_std` between run 2 and run 3. It is less aggressive than run 3 but more aggressive than run 2.
- The 1M fixed snapshot improves over run 3 (`2366.176` vs. `1351.505`) but is still below run 2's `2439.417` and below the `2500` target.
- Run 4 still has a mid-training collapse at 1.5M to 2M before recovering strongly by 2.5M. This suggests the policy finds a useful gait early, loses it during later updates, and then relearns a stronger gait.
- Run 4 3M and best checkpoints are strong (`3440.120`, `3503.896`), but best v3 remains the best overall model so far (`3662.311`).

### Run 4 Conclusion

Run 4 does not improve the earliest passing fixed checkpoint; it still passes at 2.5M. The best signal for reaching 1M is still run 2's early behavior, especially its 1M score of `2439.417`. The next run should specifically target the weak seeds at 1M instead of increasing late-training exploitation.

### Run 5 Training Settings

Run 5 attempted to keep run 2's early stability while increasing update capacity:

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 800 \
  --actor-lr 1e-4 \
  --critic-lr 3e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 2e-3 \
  --value-coef 0.5 \
  --epsilon 0.1 \
  --rollout-len 4096 \
  --update-epoch 6 \
  --batch-size 256 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --init-log-std -0.5 \
  --min-log-std -1.5 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task3_train_latest_v5.pt \
  --eval-model-path LAB7_314553032_task3_best_v5.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 50000 \
  --eval-seed-start 0 \
  --eval-seed-end 9 \
  --wandb-run-name walker-ppo-v5-early-robust
```

### Run 5 Evaluation Summary

Date recorded: 2026-05-17

| Checkpoint | Training Environment Step | Mean Reward | Target | Status |
| --- | ---: | ---: | ---: | --- |
| `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 1816.296 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 1021.052 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1307.263 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 2247.786 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 2421.599 | >= 2500 | Not passed |
| `LAB7_314553032_task3_best_v5.pt` | 3,250,000 | 3174.461 | >= 2500 | **Passed** |

### Run 5 Evaluation Command

```bash
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_3m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_best_v5.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

### Run 5 Per-Seed Rewards

| Seed | 1M | 1.5M | 2M | 2.5M | 3M | Best v5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1227.093 | 987.301 | 1461.184 | 1886.048 | 2252.158 | 3490.829 |
| 1 | 2572.377 | 991.626 | 1059.243 | 2068.414 | 2719.144 | 2130.730 |
| 2 | 1000.995 | 1339.335 | 1051.223 | 3024.402 | 2857.867 | 3471.398 |
| 3 | 1352.880 | 881.921 | 1427.670 | 3394.264 | 3483.887 | 2300.384 |
| 4 | 1339.779 | 951.130 | 1040.037 | 3006.917 | 2430.225 | 1987.802 |
| 5 | 2636.655 | 1382.958 | 1421.641 | 1636.339 | 1847.990 | 3520.409 |
| 6 | 1005.965 | 893.310 | 1235.276 | 1687.496 | 2908.581 | 2813.747 |
| 7 | 2528.144 | 928.892 | 1299.391 | 2832.741 | 1828.339 | 3537.684 |
| 8 | 2554.303 | 943.689 | 1447.679 | 1889.794 | 1935.155 | 3407.983 |
| 9 | 1345.584 | 1277.717 | 1495.184 | 3373.906 | 1995.145 | 3499.605 |
| 10 | 1219.411 | 1192.554 | 1210.205 | 1832.512 | 1956.788 | 3535.935 |
| 11 | 1363.010 | 1243.532 | 1532.858 | 1639.485 | 1553.353 | 3476.023 |
| 12 | 2503.557 | 899.928 | 1427.856 | 1626.127 | 2126.904 | 3533.817 |
| 13 | 2683.303 | 914.609 | 1428.382 | 1626.538 | 3317.965 | 3506.153 |
| 14 | 2539.678 | 904.822 | 1387.549 | 1687.989 | 2197.939 | 3423.666 |
| 15 | 1360.679 | 899.932 | 1193.839 | 1619.440 | 2635.344 | 2033.950 |
| 16 | 1358.012 | 937.899 | 1294.118 | 3491.029 | 1356.969 | 3498.602 |
| 17 | 1330.556 | 904.877 | 1171.623 | 1654.573 | 2031.790 | 3431.444 |
| 18 | 2544.548 | 919.625 | 1097.180 | 2972.134 | 3503.033 | 3418.992 |
| 19 | 1859.396 | 1025.390 | 1463.118 | 2005.575 | 3493.400 | 3470.063 |
| Mean | 1816.296 | 1021.052 | 1307.263 | 2247.786 | 2421.599 | 3174.461 |

### Run 5 Standard Check

- Required average reward: `>= 2500`
- Fixed snapshots do not pass in run 5.
- Best checkpoint: `LAB7_314553032_task3_best_v5.pt`
- Best checkpoint mean reward: `3174.461`
- Best checkpoint training environment step: `3250000`
- Current conclusion: run 5 is worse than run 4 for fixed-step scoring and should not replace run 4 or run 3.

### Run 5 Curve Comparison

Compared with run 4, run 5 changes:

| Setting | Run 4 early stable | Run 5 early robust | Expected effect |
| --- | ---: | ---: | --- |
| Actor LR | 1.2e-4 | 1e-4 | More conservative actor updates |
| Entropy weight | 2.5e-3 | 2e-3 | Less exploration pressure |
| Update epoch | 5 | 6 | More optimization on each rollout |
| Clip epsilon | 0.11 | 0.1 | Tighter clipping |

Observed differences:

- Run 5 is worse at every fixed snapshot than run 4.
- The extra update epoch did not improve sample efficiency. It appears to over-optimize each rollout and makes the fixed snapshots less robust.
- Lower entropy did not help early convergence. It likely reduced exploration before the policy had a reliable gait.
- The best checkpoint still passes, but fixed-step scoring is worse, so run 5 is not useful for the submission goal.

### Run 5 Conclusion

The next direction should return to run 4's settings and avoid `update-epoch=6`. The useful signal from run 4 is that the 1M snapshot had many strong seeds but several weak seeds pulled the average below 2500. The next experiment should try to improve robustness across seeds, not increase optimization per rollout.

### Run 6 Training Settings

Run 6 returns to run 4's update shape, lowers critic learning rate, and adds slightly more initial exploration:

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 800 \
  --actor-lr 1.2e-4 \
  --critic-lr 2e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 3e-3 \
  --value-coef 0.5 \
  --epsilon 0.11 \
  --rollout-len 4096 \
  --update-epoch 5 \
  --batch-size 256 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --init-log-std -0.45 \
  --min-log-std -1.5 \
  --max-log-std 0.5 \
  --model-path LAB7_314553032_task3_train_latest_v6.pt \
  --eval-model-path LAB7_314553032_task3_best_v6.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 50000 \
  --eval-seed-start 0 \
  --eval-seed-end 9 \
  --wandb-run-name walker-ppo-v6-v4-plus-exploration
```

### Run 6 Evaluation Summary

Date recorded: 2026-05-17

| Checkpoint | Training Environment Step | Mean Reward | Target | Status |
| --- | ---: | ---: | ---: | --- |
| `LAB7_314553032_task3_ppo_1m.pt` | 1,003,520 | 2231.630 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_1p5m.pt` | 1,503,232 | 2356.637 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2m.pt` | 2,002,944 | 1352.002 | >= 2500 | Not passed |
| `LAB7_314553032_task3_ppo_2p5m.pt` | 2,502,656 | 2761.016 | >= 2500 | **Passed** |
| `LAB7_314553032_task3_ppo_3m.pt` | 3,002,368 | 3369.157 | >= 2500 | **Passed** |
| `LAB7_314553032_task3_best_v6.pt` | 3,200,000 | 3465.944 | >= 2500 | **Passed** |

### Run 6 Evaluation Command

```bash
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_1p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_2p5m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_ppo_3m.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
python ppo_walker.py --mode eval --model-path LAB7_314553032_task3_best_v6.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

### Run 6 Per-Seed Rewards

| Seed | 1M | 1.5M | 2M | 2.5M | 3M | Best v6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2279.831 | 2619.219 | 1332.207 | 2562.243 | 2314.631 | 3587.872 |
| 1 | 2322.039 | 2556.860 | 1577.671 | 3314.877 | 3527.541 | 3522.505 |
| 2 | 2276.334 | 2644.921 | 1271.408 | 3177.533 | 2181.024 | 3515.037 |
| 3 | 2262.898 | 2684.790 | 1358.257 | 3281.236 | 3590.384 | 3475.742 |
| 4 | 2265.726 | 2454.746 | 1044.891 | 3247.651 | 3649.150 | 3456.925 |
| 5 | 2229.583 | 2642.764 | 1448.474 | 2072.570 | 3593.344 | 3403.865 |
| 6 | 2328.937 | 1017.778 | 1351.291 | 3240.064 | 3378.932 | 3438.994 |
| 7 | 2303.292 | 2682.688 | 1544.199 | 3161.170 | 3482.390 | 3495.508 |
| 8 | 2241.825 | 1538.191 | 1076.584 | 3537.598 | 3578.607 | 3602.812 |
| 9 | 2255.318 | 1496.337 | 1274.154 | 3447.691 | 3594.879 | 3501.659 |
| 10 | 2288.917 | 1542.007 | 1246.981 | 2085.511 | 3488.321 | 3561.903 |
| 11 | 2252.488 | 2651.000 | 1433.240 | 1591.839 | 3433.256 | 3572.476 |
| 12 | 2261.677 | 2539.488 | 1308.468 | 2063.803 | 3617.202 | 3511.916 |
| 13 | 2223.785 | 2524.419 | 1582.866 | 2455.849 | 3394.406 | 3471.630 |
| 14 | 2279.387 | 2687.726 | 1288.550 | 2751.884 | 3189.518 | 3587.985 |
| 15 | 2299.244 | 2632.879 | 1859.698 | 3794.766 | 3515.436 | 2762.780 |
| 16 | 2283.855 | 2470.979 | 1291.668 | 2806.641 | 3484.612 | 3423.692 |
| 17 | 2260.069 | 2517.935 | 1053.562 | 1792.409 | 3496.178 | 3502.567 |
| 18 | 2253.269 | 2604.536 | 1358.093 | 2300.046 | 3372.829 | 3460.040 |
| 19 | 1464.126 | 2623.467 | 1337.782 | 2534.940 | 3500.493 | 3462.976 |
| Mean | 2231.630 | 2356.637 | 1352.002 | 2761.016 | 3369.157 | 3465.944 |

### Run 6 Standard Check

- Required average reward: `>= 2500`
- Earliest passing fixed snapshot: `LAB7_314553032_task3_ppo_2p5m.pt`
- Earliest passing fixed snapshot mean reward: `2761.016`
- Best checkpoint: `LAB7_314553032_task3_best_v6.pt`
- Best checkpoint mean reward: `3465.944`
- Current conclusion: run 6 keeps fixed 2.5M and later snapshots above target, but does not improve the earliest passing step.

### Run 6 Curve Comparison

Compared with run 4 and run 5:

- Run 6 1M rewards have much lower variance than previous runs. Most seeds are tightly around `2220-2330`; seed 19 is the main outlier at `1464.126`.
- The tighter 1M distribution suggests the direction is more robust, but the policy is underperforming by about `250-300` reward points on most seeds.
- The higher entropy and less aggressive critic make the early gait more consistent, but also less optimized.
- 1.5M is closer to target than v4/v5, but still has weak seeds `6`, `8`, `9`, and `10`.
- 2M collapses again, then 2.5M and 3M recover strongly. This recurring mid-training collapse suggests that checkpoint selection around 1M and 1.5M matters more than relying on monotonic training.

### Run 6 Conclusion

Run 6 did not improve fixed-step scoring, but it revealed a useful direction: reduce early variance first, then slightly increase early exploitation. The next run should preserve run 6's stability while making the policy a bit more deterministic and slightly faster before 1M.
