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
| Task 2 | PPO-Clip + GAE | `Pendulum-v1` | Pending | Pending | Pending | > -150 | Pending |
| Task 3 | PPO-Clip + GAE | `Walker2d-v5` | Pending | Pending | Pending | >= 2500 | Pending |

## Task 1: A2C on Pendulum-v1

### Training Settings Summary

| Run | Model | Episodes | Actor LR | Critic LR | Gamma | Entropy Weight | W&B Run | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Run 1 | `LAB7_314553032_task1_a2c_pendulum.pt` | 1000 inferred | 1e-4 inferred | 1e-3 inferred | 0.9 | 1e-2 inferred | Not recorded | Exact training command was not recorded; values match the script defaults used before tuning. |
| Run 2 | `LAB7_314553032_task1_a2c_pendulum_v2.pt` | 2000 recommended | 5e-5 | 5e-4 | 0.9 | 1e-3 | `pendulum-a2c-lr5e5-ent1e3` inferred | Conservative LR and lower entropy run used after run 1 instability. |
| Run 3 | `LAB7_314553032_task1_a2c_pendulum_v3.pt` | 2500 | 3e-5 | 3e-4 | 0.9 | 5e-4 | `pendulum-a2c-lr3e5-ent5e4` | User-provided command. |
| Run 4 | `LAB7_314553032_task1_a2c_pendulum_v4.pt` | 3000 | 1e-5 | 3e-4 | 0.9 | 1e-4 | `pendulum-a2c-lr1e5-critic3e4-ent1e4` | User-provided command; lower actor LR and entropy did not improve final 20-seed mean. |
| Run 5 | `LAB7_314553032_task1_a2c_pendulum_evalbest.pt` | 2500 | 3e-5 | 3e-4 | 0.9 | 5e-4 | `pendulum-a2c-evalbest-lr3e5-ent5e4` | Uses periodic 20-seed eval-best checkpointing every 20000 steps. |

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
4. Next possible change: add explicit action standard deviation control, such as `--init-log-std` and optional lower clamp, because some seeds appear to need more reliable torque selection rather than more random exploration.

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
