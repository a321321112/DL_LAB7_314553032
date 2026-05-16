# Task 2 PPO Pendulum Implementation Plan

This plan is for Task 2 only. The goal is to finish `PPO-Clip + GAE` on `Pendulum-v1`, produce a reproducible 20-seed evaluation result, and save the final snapshot as:

```txt
LAB7_314553032_task2_ppo_pendulum.pt
```

## Current State

- Task 1 A2C is complete and passed with:
  - `LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt`
  - mean reward `-144.271`
- Task 2 source template exists as:
  - `ppo_pendulum-3.py`
- Task 2 final code file should be:
  - `ppo_pendulum.py`
- Current Task 2 template still has TODOs in:
  - `Actor`
  - `Critic`
  - `compute_gae`
  - PPO clipped actor loss
  - critic loss
- Current Task 2 template also lacks the Task 1-quality workflow:
  - `--mode train|eval`
  - checkpoint save/load
  - 20-seed deterministic evaluation
  - eval-best checkpointing
  - optional W&B via `--no-wandb`
  - result metadata printout

## Target Standard

Task 2 uses `Pendulum-v1`.

Required evaluation:

```txt
seeds: 0 to 19
episodes: 20
mean reward target: > -150
```

Performance scoring favors reaching `> -150` within `200000` environment steps.

## Implementation Policy

- Complete one testable functional unit at a time.
- After each functional unit:
  - run `python3 -m py_compile ppo_pendulum.py`
  - commit and push
  - record progress in `LAB7_PROGRESS.md`
- Do not overwrite Task 1 code unless a shared utility is intentionally introduced.
- Do not commit `.pt` model files unless explicitly required.

## Phase 1: File Setup

Tasks:

- Rename or copy `ppo_pendulum-3.py` to `ppo_pendulum.py`.
- Keep `ppo_pendulum-3.py` only if needed as an untouched reference.
- Add or confirm `.gitignore` excludes generated videos, W&B folders, and `.pt` checkpoints.
- Make imports robust:
  - `wandb` should be optional, as in `a2c_pendulum.py`.
  - `Path`, `Optional`, and typing helpers should be available.

Validation:

```bash
python3 -m py_compile ppo_pendulum.py
```

Commit target:

```txt
Prepare Task 2 PPO file
```

## Phase 2: PPO Actor and Critic

Implement:

- `Actor`
  - two hidden layers, same basic shape as Task 1 unless PPO template requires otherwise
  - Gaussian mean output scaled to Pendulum action range `[-2, 2]`
  - trainable `log_std`
  - optional log-std bounds if useful
- `Critic`
  - two hidden layers
  - scalar value output

Important details:

- Training uses sampled actions.
- Evaluation uses deterministic mean action.
- Log probability must sum over action dimensions:

```python
log_prob = dist.log_prob(action).sum(dim=-1)
entropy = dist.entropy().sum(dim=-1)
```

Validation:

```bash
python3 -m py_compile ppo_pendulum.py
```

Commit target:

```txt
Implement PPO Pendulum networks
```

## Phase 3: Rollout Buffer and GAE

Implement rollout storage for:

- states
- actions
- rewards
- dones or masks
- values
- old log probabilities

Implement GAE:

```txt
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
adv_t = delta_t + gamma * lambda * (1 - done_t) * adv_{t+1}
return_t = adv_t + V(s_t)
```

Use:

- `gamma = 0.9` as the PDF default for Pendulum first.
- `tau` / `gae_lambda = 0.95` as the first serious PPO setting.
- advantage normalization before PPO update.

Validation:

```bash
python3 -m py_compile ppo_pendulum.py
```

Optional runtime smoke test after CLI exists:

```bash
python ppo_pendulum.py --mode train --num-episodes 2 --no-wandb
```

Commit target:

```txt
Implement PPO GAE rollout returns
```

## Phase 4: PPO-Clip Update

Implement mini-batch PPO update:

```txt
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
actor_loss = -mean(min(surr1, surr2)) - entropy_coef * entropy
critic_loss = mse(value, return)
```

Recommended first settings:

```txt
rollout_len = 2048
update_epoch = 10
batch_size = 64
epsilon = 0.2
actor_lr = 3e-4
critic_lr = 1e-3
entropy_weight = 1e-3
tau = 0.95
gamma = 0.9
```

Track W&B metrics:

- `step`
- `episode`
- `return`
- `best_return`
- `actor loss`
- `critic loss`
- `entropy`
- `approx_kl` if easy to compute
- `clip_fraction` if easy to compute

Validation:

```bash
python3 -m py_compile ppo_pendulum.py
```

Commit target:

```txt
Implement PPO clipped update
```

## Phase 5: Train/Eval CLI and Checkpoints

Add Task 1-style CLI:

```txt
--mode train|eval
--model-path
--eval-model-path
--eval-interval
--eval-seed-start
--eval-seed-end
--target-eval-mean
--seed-start
--seed-end
--eval-episodes
--render-video
--video-folder
--no-wandb
--save-dir
```

Add checkpoint metadata:

- actor state dict
- critic state dict
- total environment step
- score or eval mean reward
- seed
- selection metric
- eval seed range
- PPO hyperparameters that affect reproduction

Evaluation output must include:

```txt
seed=<seed> reward=<reward>
model_path=<path>
training_environment_step=<step>
mean_reward=<mean>
```

Validation:

```bash
python3 -m py_compile ppo_pendulum.py
```

Commit target:

```txt
Add PPO Pendulum train eval CLI
```

## Phase 6: First Training Run

Run:

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

Then evaluate:

```bash
python ppo_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task2_ppo_pendulum.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

Record results in `Result.md` under Task 2.

## Phase 7: If First Run Does Not Pass

Tune in this order, one controlled change at a time:

1. Increase rollout stability:

```txt
rollout_len = 4096
update_epoch = 10
batch_size = 128
```

2. Reduce update aggressiveness:

```txt
actor_lr = 1e-4
critic_lr = 5e-4
epsilon = 0.1
```

3. Adjust exploration:

```txt
entropy_weight = 5e-4 or 1e-4
```

4. If the same hard seeds fail:

```txt
discount_factor = 0.95
```

Expected behavior:

- PPO should usually be more stable than A2C.
- PPO should need fewer emergency code-level changes if GAE and clipping are correct.
- If PPO does not outperform A2C on Pendulum, first suspect implementation issues:
  - wrong `log_prob` shape
  - missing advantage normalization
  - using stochastic action during evaluation
  - GAE done masks wrong
  - old log probs not detached
  - too many update epochs for one rollout

## Phase 8: Report Requirements for Task 2

Add to report:

- PPO clipped objective explanation.
- GAE explanation.
- How rollout collection works.
- Why PPO can reuse a rollout for multiple epochs.
- W&B curve:
  - x-axis must be environment steps
  - include evaluation reward curve if available
- 20-seed evaluation screenshot.
- Comparison against Task 1 A2C:
  - steps to pass
  - final mean reward
  - curve stability
  - hard-seed behavior

## Task 2 Done Criteria

- [ ] `ppo_pendulum.py` exists.
- [ ] `python3 -m py_compile ppo_pendulum.py` passes.
- [ ] PPO actor and critic are implemented.
- [ ] GAE is implemented and uses done masks correctly.
- [ ] PPO clipped objective is implemented.
- [ ] Deterministic 20-seed eval works.
- [ ] Eval-best checkpoint is saved by seeds `0` to `19` mean reward.
- [ ] Final model path is `LAB7_314553032_task2_ppo_pendulum.pt`.
- [ ] Mean reward is `> -150`.
- [ ] `Result.md` records command, env step, per-seed rewards, mean reward, and pass/fail.
- [ ] W&B curve screenshot is ready for the report.
