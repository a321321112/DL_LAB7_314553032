# Task 3 PPO Walker2d Implementation Plan

This plan is for Task 3 only. The goal is to implement and tune `PPO-Clip + GAE` on `Walker2d-v5`, save the required fixed-step model snapshots, and produce a reproducible 20-seed evaluation result.

## Current State

- Task 1 A2C Pendulum is complete and passed:
  - `LAB7_314553032_task1_a2c_pendulum_evalbest_gamma095.pt`
  - mean reward `-144.271`
- Task 2 PPO Pendulum is complete and passed:
  - `LAB7_314553032_task2_ppo_pendulum_v2.pt`
  - mean reward `-149.784`
- Task 3 source template exists as:
  - `ppo_walker-3.py`
- Task 3 final code file should be:
  - `ppo_walker.py`
- The current Task 3 template still has TODOs in:
  - `Actor`
  - `Critic`
  - `compute_gae`
  - PPO clipped actor loss
  - critic loss
- The current Task 3 template also lacks the workflow already built for Task 1 and Task 2:
  - `--mode train|eval`
  - checkpoint save/load
  - deterministic 20-seed evaluation
  - eval-best checkpointing
  - fixed-step checkpointing at 1M, 1.5M, 2M, 2.5M, and 3M environment steps
  - optional W&B via `--no-wandb`
  - result metadata printout

## Target Standard

Task 3 uses `Walker2d-v5`.

Required evaluation:

```txt
seeds: 0 to 19
episodes: 20
mean reward target: >= 2500
```

Performance scoring favors reaching `2500` with fewer environment steps. The homework also requires fixed-step snapshots:

```txt
LAB7_314553032_task3_ppo_1m.pt
LAB7_314553032_task3_ppo_1p5m.pt
LAB7_314553032_task3_ppo_2m.pt
LAB7_314553032_task3_ppo_2p5m.pt
LAB7_314553032_task3_ppo_3m.pt
LAB7_314553032_task3_best.pt
```

## Implementation Policy

- Complete one testable functional unit at a time.
- After each functional unit:
  - run `python3 -m py_compile ppo_walker.py`
  - commit and push
  - record progress in `LAB7_PROGRESS.md`
- Reuse the proven Task 2 PPO implementation pattern where possible.
- Do not overwrite Task 1 or Task 2 code.
- Do not commit `.pt` model files unless explicitly requested.

## Phase 1: File Setup

Tasks:

- Rename or copy `ppo_walker-3.py` to `ppo_walker.py`.
- Keep `ppo_walker-3.py` as the untouched reference if useful.
- Make imports robust:
  - `wandb` should be optional.
  - `Path`, `Optional`, and typing helpers should be available.
- Confirm `.gitignore` excludes W&B folders, videos, and `.pt` checkpoints.
- Use the same CLI style as `ppo_pendulum.py`.

Validation:

```bash
python3 -m py_compile ppo_walker.py
```

Commit target:

```txt
Prepare Task 3 PPO Walker file
```

## Phase 2: PPO Actor and Critic

Implement:

- `Actor`
  - accepts Walker observation dimension from `env.observation_space.shape[0]`
  - outputs a Gaussian policy over all Walker action dimensions
  - scales or clamps actions to the environment action range
  - uses trainable `log_std`
  - supports log-std bounds to avoid premature deterministic behavior
- `Critic`
  - accepts Walker observation dimension
  - outputs scalar state value `V(s)`

Important details:

- Walker2d action space is multidimensional, so log probability and entropy must sum over the action dimension:

```python
log_prob = dist.log_prob(action).sum(dim=-1)
entropy = dist.entropy().sum(dim=-1)
```

- Training uses sampled actions.
- Evaluation uses deterministic mean actions.
- Store old log probabilities during rollout collection and detach them before PPO update.

Validation:

```bash
python3 -m py_compile ppo_walker.py
```

Commit target:

```txt
Implement PPO Walker networks
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

Walker2d settings:

```txt
gamma = 0.99
gae lambda = 0.95
```

Use advantage normalization before each PPO update.

Validation:

```bash
python3 -m py_compile ppo_walker.py
```

Commit target:

```txt
Implement PPO Walker GAE rollout returns
```

## Phase 4: PPO-Clip Update

Implement mini-batch PPO update:

```txt
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
actor_loss = -mean(min(surr1, surr2)) - entropy_coef * entropy
critic_loss = value_coef * mse(value, return)
```

Recommended first Walker2d settings:

```txt
rollout_len = 2048
update_epoch = 10
batch_size = 64 or 128
epsilon = 0.2
actor_lr = 3e-4
critic_lr = 3e-4
entropy_weight = 0.0 to 1e-3
value_coef = 0.5
max_grad_norm = 0.5
gamma = 0.99
tau = 0.95
```

Track W&B metrics:

- `step`
- `episode`
- `return`
- `best_return`
- `eval/mean_reward`
- `eval/best_mean_reward`
- `actor loss`
- `critic loss`
- `entropy`
- `action/log_std`
- `approx_kl`
- `clip_fraction`
- `explained_variance` if easy to compute

Validation:

```bash
python3 -m py_compile ppo_walker.py
```

Commit target:

```txt
Implement PPO Walker clipped update
```

## Phase 5: Train/Eval CLI and Checkpoints

Add CLI options:

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
--snapshot-steps
```

Checkpoint metadata should include:

- actor state dict
- critic state dict
- total environment step
- training episode count
- latest training return
- best training return
- best eval mean reward
- seed
- eval seed range
- PPO hyperparameters
- observation and action dimensions

Evaluation output must include:

```txt
seed=<seed> reward=<reward>
model_path=<path>
training_environment_step=<step>
mean_reward=<mean>
```

Fixed-step snapshot behavior:

- Save the first checkpoint whose `total_step >= snapshot_step`.
- Snapshot filenames:
  - `LAB7_314553032_task3_ppo_1m.pt`
  - `LAB7_314553032_task3_ppo_1p5m.pt`
  - `LAB7_314553032_task3_ppo_2m.pt`
  - `LAB7_314553032_task3_ppo_2p5m.pt`
  - `LAB7_314553032_task3_ppo_3m.pt`
- Save `LAB7_314553032_task3_best.pt` whenever evaluation mean improves.

Validation:

```bash
python3 -m py_compile ppo_walker.py
```

Commit target:

```txt
Add PPO Walker train eval CLI
```

## Phase 6: Smoke Test

Run a short local test before expensive training:

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 2 \
  --rollout-len 256 \
  --update-epoch 1 \
  --batch-size 64 \
  --eval-interval 512 \
  --eval-seed-start 0 \
  --eval-seed-end 1 \
  --eval-episodes 2 \
  --model-path LAB7_314553032_task3_smoke_train.pt \
  --eval-model-path LAB7_314553032_task3_smoke_best.pt \
  --no-wandb
```

Then evaluate:

```bash
python ppo_walker.py \
  --mode eval \
  --model-path LAB7_314553032_task3_smoke_best.pt \
  --seed-start 0 \
  --seed-end 1 \
  --eval-episodes 2 \
  --no-wandb
```

Expected smoke-test result:

- Code runs without shape errors.
- Checkpoint save/load works.
- Evaluation prints the required metadata.
- Reward does not need to be good.

## Phase 7: First Serious Training Run

Initial command:

```bash
python ppo_walker.py \
  --mode train \
  --num-episodes 3000 \
  --actor-lr 3e-4 \
  --critic-lr 3e-4 \
  --discount-factor 0.99 \
  --tau 0.95 \
  --entropy-weight 1e-3 \
  --epsilon 0.2 \
  --rollout-len 2048 \
  --update-epoch 10 \
  --batch-size 64 \
  --value-coef 0.5 \
  --max-grad-norm 0.5 \
  --model-path LAB7_314553032_task3_train_latest.pt \
  --eval-model-path LAB7_314553032_task3_best.pt \
  --snapshot-steps 1000000,1500000,2000000,2500000,3000000 \
  --eval-interval 50000 \
  --eval-seed-start 0 \
  --eval-seed-end 4 \
  --target-eval-mean 2500 \
  --wandb-run-name walker-ppo-baseline
```

Reason for using seeds `0` to `4` during training:

- Full 20-seed eval on Walker2d is expensive.
- Frequent 5-seed eval is enough to choose checkpoints during training.
- Final reporting still requires seeds `0` to `19`.

After training, evaluate the best checkpoint:

```bash
python ppo_walker.py \
  --mode eval \
  --model-path LAB7_314553032_task3_best.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

Record results in `Result.md` under Task 3.

## Phase 8: Tuning Strategy

Tune one controlled group at a time.

If reward does not rise above early-fall behavior:

```txt
Check action scaling, log_prob summation, deterministic eval, done masks, and advantage normalization first.
```

If policy updates are unstable:

```txt
actor_lr = 1e-4
critic_lr = 1e-4 or 3e-4
epsilon = 0.1
update_epoch = 5
batch_size = 128
```

If exploration collapses too early:

```txt
entropy_weight = 3e-3 or 1e-2
min_log_std = -2.0 or -1.5
```

If policy remains too random late in training:

```txt
entropy_weight = 0.0 or 1e-4
min_log_std = -3.0
```

If critic loss dominates or value estimate is noisy:

```txt
value_coef = 0.25
critic_lr = 1e-4
normalize advantages
consider reward normalization only if needed
```

If the curve improves but plateaus below 2500:

```txt
train to 3M steps
reduce actor_lr to 1e-4
try epsilon = 0.1
compare entropy_weight = 0, 1e-4, 1e-3
```

## Phase 9: Empirical Study for Report

The report should include controlled comparisons for:

- clipping parameter:
  - `epsilon = 0.1`
  - `epsilon = 0.2`
  - optionally `epsilon = 0.3`
- entropy coefficient:
  - `entropy_weight = 0`
  - `entropy_weight = 1e-4`
  - `entropy_weight = 1e-3`
  - optionally `entropy_weight = 1e-2`

For each comparison, record:

- environment steps
- mean evaluation score
- curve stability
- whether the policy falls early or walks forward
- entropy and log-std behavior
- approximate KL and clip fraction if available

## Phase 10: Final Evaluation and Report Assets

Run final 20-seed evaluation for:

- fixed-step snapshots that matter for the report
- `LAB7_314553032_task3_best.pt`

Evaluation command pattern:

```bash
python ppo_walker.py \
  --mode eval \
  --model-path <checkpoint.pt> \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

Record each result in `Result.md`.

Prepare report assets:

- W&B training curve with environment steps on x-axis
- W&B evaluation mean reward curve
- final 20-seed evaluation screenshot
- comparison table against Task 2 PPO Pendulum and any A2C Walker baseline if required
- discussion of clipping parameter and entropy coefficient effects
- demo video if required

## Task 3 Done Criteria

- [ ] `ppo_walker.py` exists.
- [ ] `python3 -m py_compile ppo_walker.py` passes.
- [ ] PPO actor and critic support multidimensional continuous actions.
- [ ] GAE is implemented and uses done masks correctly.
- [ ] PPO clipped objective is implemented.
- [ ] Advantage normalization is used.
- [ ] Deterministic 20-seed eval works.
- [ ] Eval-best checkpoint is saved by evaluation mean reward.
- [ ] Fixed-step snapshots are saved at 1M, 1.5M, 2M, 2.5M, and 3M steps.
- [ ] `LAB7_314553032_task3_best.pt` reaches mean reward `>= 2500`, or best attempted result is documented with next steps.
- [ ] `Result.md` records commands, env steps, per-seed rewards, mean reward, and pass/fail.
- [ ] W&B curves are ready for the report.
