# Lab 7 Progress Tracker

Started: 2026-05-15 09:33:10 CST

This file tracks implementation progress for Lab 7. Each completed functional unit should be committed and pushed before moving to the next unit.

## Commit Policy

- Commit and push after each testable functional unit.
- Keep each commit focused on one implementation milestone.
- Record the commit hash, validation command, and result in this file.

## Progress Checklist

| Status | Unit | Commit | Validation | Notes |
| --- | --- | --- | --- | --- |
| Done | Add progress tracker | `a9e1468` | Manual file check | Created this checklist before implementation work. |
| Done | Prepare Task 1 project files | `915f6d8` | `python3 -m py_compile a2c_pendulum.py` passed | Renamed Task 1 file, added requirements and ignore rules. |
| Done | Implement A2C networks | `7957769` | `python3 -m py_compile a2c_pendulum.py` passed; runtime smoke blocked | Completed Gaussian actor and value critic. Runtime test blocked because local Python has no `torch`. |
| Done | Implement A2C update step | `ccf54b0` | `python3 -m py_compile a2c_pendulum.py` passed; runtime smoke blocked | Completed TD target, actor loss, critic loss, entropy. Runtime test blocked because local Python has no `torch`. |
| Done | Add Task 1 train/eval CLI | `00981b6` | `python3 -m py_compile a2c_pendulum.py` passed; runtime smoke blocked | Added model save/load, `--mode train|eval`, seed range evaluation, and `--no-wandb`. |
| Done | Document Task 1 execution progress | `a550e1f` | Final progress review | Recorded final Task 1 commands and current environment limitations. |
| Done | Add Task 1 training guide | `48604e2` | Markdown review and `python3 -m py_compile a2c_pendulum.py` passed | Added environment setup, training, evaluation, video, W&B, troubleshooting, and report command guidance. |
| Done | Fix PyTorch 2.6 checkpoint loading | `396ea5b` | `python3 -m py_compile a2c_pendulum.py` passed | Load trusted full checkpoints with metadata and save metadata as Python native values. |
| Done | Add periodic eval-best checkpointing | `6094235` | `python3 -m py_compile a2c_pendulum.py` passed | Added training-time 20-seed evaluation, eval-best checkpoint path, and W&B eval mean logging. |
| Done | Document periodic eval-best checkpointing | `faac8f9` | Documentation review | Updated training guide and progress notes for eval-best workflow. |
| Done | Add n-step A2C update controls | `404bb71` | `python3 -m py_compile a2c_pendulum.py` passed | Added n-step rollout updates, reward scaling, and advantage normalization. |
| Done | Add action log std controls | `18bce46` | `python3 -m py_compile a2c_pendulum.py` passed | Added configurable Gaussian action std initialization and clamp bounds. |
| Done | Add eval target early stopping | `5ed7aa8` | `python3 -m py_compile a2c_pendulum.py` passed | Added `--target-eval-mean` to stop training once eval-best mean passes the target. |
| Done | Prepare Task 2 PPO file | `238cf19` | `python3 -m py_compile ppo_pendulum.py` passed | Renamed Task 2 template to final source path. |
| Done | Implement Task 2 PPO Pendulum | `02c7288` | `python3 -m py_compile ppo_pendulum.py` passed | Implemented PPO networks, GAE, clipped objective, train/eval CLI, and eval-best checkpointing. |
| Done | Add Task 3 implementation plan | Current commit | Markdown review | Renamed `Plan.md` to `Task2_Plan.md` and added `Task3_Plan.md` for Walker2d PPO work. |
| Done | Prepare Task 3 PPO Walker file | Current commit | `python3 -m py_compile ppo_walker.py` passed | Created `ppo_walker.py` from the proven PPO workflow and switched defaults to `Walker2d-v5`. |
| Done | Adapt PPO Walker action and checkpoint logic | Current commit | `python3 -m py_compile ppo_walker.py` passed | Added Walker action bounds, value coefficient, 256-wide networks, fixed-step snapshots, and richer checkpoint metadata. |
| Done | Fix Walker MuJoCo imageio dependency | Current commit | `python3 -m py_compile ppo_walker.py` passed | Added `imageio` packages to requirements and only requests `rgb_array` rendering when video output is enabled. |
| Done | Record Task 3 baseline result | Current commit | Result review | Recorded 1M, 1.5M, 2M, 2.5M, 3M, and best checkpoint 20-seed evaluations. |
| Done | Record passing Task 3 result | Current commit | Result review | Recorded stable v2 Task 3 evaluations; `LAB7_314553032_task3_best_v2.pt` passes with 20-seed mean reward `2537.552`. |
| Done | Record Task 3 faster-early result | Current commit | Result review | Recorded v3 Task 3 evaluations; fixed 2.5M snapshot passes with mean reward `3025.007`, and best v3 reaches `3662.311`. |
| Done | Record Task 3 early-stable result | Current commit | Result review | Recorded v4 Task 3 evaluations; fixed 2.5M snapshot passes with mean reward `2969.790`, and best v4 reaches `3503.896`. |
| Done | Record Task 3 early-robust result | Current commit | Result review | Recorded v5 Task 3 evaluations; fixed snapshots do not pass, while best v5 reaches `3174.461`. |
| Done | Record Task 3 v4 plus exploration result | Current commit | Result review | Recorded v6 Task 3 evaluations; 1M variance is lower but mean remains below target, while 2.5M and 3M pass. |
| Done | Record Task 3 early-exploit result | Current commit | Result review | Recorded v7 Task 3 evaluations; best v7 reaches `4003.744`, but early fixed snapshots regress while 2.5M and 3M pass. |
| Done | Record Task 3 frequent-updates result | Current commit | Result review | Recorded v8 Task 3 evaluations; fixed 1.5M snapshot reaches `3400.151`, moving the earliest fixed-snapshot pass earlier than previous runs. |
| Done | Record Task 3 earlier-discovery result | Current commit | Result review | Recorded complete v9 Task 3 evaluations; 1M improves to `1808.769` and 1.5M passes at `2517.718`, but run 8 remains the stronger submission candidate. |
| Done | Record Task 3 balanced-early result | Current commit | Result review | Recorded v10 Task 3 evaluations; fixed 1M snapshot passes at `2603.904`, and best v10 reaches `4079.485`. |

## Validation Log

- 2026-05-15 09:33:10 CST: Progress tracker created.
- 2026-05-15 09:33:10 CST: Commit `a9e1468` pushed to `origin/main`.
- 2026-05-15 09:34:00 CST: Started Task 1 project preparation. Local `python3` is 3.14.3 and does not have `torch` installed, so dependency-based smoke tests may need a virtual environment.
- 2026-05-15 09:35:00 CST: `python3 -m py_compile a2c_pendulum.py` passed.
- 2026-05-15 09:35:00 CST: Commit `915f6d8` pushed to `origin/main`.
- 2026-05-15 09:36:00 CST: Implemented A2C Actor/Critic networks. `python3 -m py_compile a2c_pendulum.py` passed. Forward runtime smoke test is blocked because available local Python installations do not have `torch`, `gymnasium`, or `wandb`.
- 2026-05-15 09:36:00 CST: Commit `7957769` pushed to `origin/main`.
- 2026-05-15 09:37:00 CST: Implemented A2C update step. `python3 -m py_compile a2c_pendulum.py` passed. Short train runtime smoke test is blocked because local Python has no `torch`.
- 2026-05-15 09:37:00 CST: Commit `ccf54b0` pushed to `origin/main`.
- 2026-05-15 09:38:00 CST: Added Task 1 train/eval CLI, checkpoint save/load, seed range evaluation, and optional W&B logging. `python3 -m py_compile a2c_pendulum.py` passed. Train/eval runtime smoke tests are blocked because local Python has no `torch`.
- 2026-05-15 09:38:00 CST: Commit `00981b6` pushed to `origin/main`.
- 2026-05-15 09:39:00 CST: Documented Task 1 execution commands and current limitations.
- 2026-05-15 09:39:00 CST: Commit `a550e1f` pushed to `origin/main`.
- 2026-05-15 09:45:00 CST: Added `LAB7_TRAINING_GUIDE.md` with Task 1 training and evaluation instructions. `python3 -m py_compile a2c_pendulum.py` passed.
- 2026-05-15 09:45:00 CST: Commit `48604e2` pushed to `origin/main`.
- 2026-05-15 10:00:00 CST: Fixed PyTorch 2.6+ checkpoint loading error caused by `torch.load(..., weights_only=True)` default and checkpoint metadata.
- 2026-05-15 10:00:00 CST: Commit `396ea5b` pushed to `origin/main`.
- 2026-05-15: Added periodic training-time evaluation every `--eval-interval` steps. The eval-best checkpoint is selected by deterministic seeds `--eval-seed-start` to `--eval-seed-end` mean reward and saved separately from the single-episode training best checkpoint.
- 2026-05-15: Commit `6094235` pushed to `origin/main`.
- 2026-05-15: Commit `faac8f9` pushed to `origin/main`.
- 2026-05-15 15:53:29 CST: Added n-step A2C update logic with `--n-step`, `--reward-scale`, and default advantage normalization. `python3 -m py_compile a2c_pendulum.py` passed.
- 2026-05-15 15:53:29 CST: Commit `404bb71` pushed to `origin/main`.
- 2026-05-15 17:43:02 CST: Added action distribution controls with `--init-log-std`, `--min-log-std`, and `--max-log-std`. `python3 -m py_compile a2c_pendulum.py` passed.
- 2026-05-15 17:43:02 CST: Commit `18bce46` pushed to `origin/main`.
- 2026-05-15 18:43:22 CST: Added `--target-eval-mean` so long training can stop automatically after fixed-seed eval-best mean passes the target. `python3 -m py_compile a2c_pendulum.py` passed.
- 2026-05-15 18:43:22 CST: Commit `5ed7aa8` pushed to `origin/main`.
- 2026-05-16 10:28:57 CST: Prepared Task 2 PPO source path as `ppo_pendulum.py`. `python3 -m py_compile ppo_pendulum.py` passed.
- 2026-05-16 10:28:57 CST: Commit `238cf19` pushed to `origin/main`.
- 2026-05-16 10:33:18 CST: Implemented Task 2 PPO Pendulum core training and evaluation workflow. `python3 -m py_compile ppo_pendulum.py` passed.
- 2026-05-16 10:33:18 CST: Commit `02c7288` pushed to `origin/main`.
- 2026-05-16 11:58:00 CST: Renamed Task 2 plan from `Plan.md` to `Task2_Plan.md` and added `Task3_Plan.md` with Walker2d PPO implementation, training, checkpoint, tuning, and report plan.
- 2026-05-16 12:25:00 CST: Created `ppo_walker.py` from the proven PPO Pendulum workflow, changed default environment and artifact paths for Task 3, and verified syntax with `python3 -m py_compile ppo_walker.py`.
- 2026-05-16 12:32:00 CST: Adapted PPO Walker implementation for multidimensional action bounds, added `--value-coef`, `--hidden-dim`, `--snapshot-steps`, fixed-step snapshot files, and metadata for environment/action dimensions. `python3 -m py_compile ppo_walker.py` passed. Local runtime `--help` is blocked because this Mac Python does not have `gymnasium`; run runtime smoke tests in the lab `.venv`.
- 2026-05-16 12:45:00 CST: Fixed Task 3 startup dependency issue from Gymnasium MuJoCo importing `imageio`. Added `imageio` and `imageio-ffmpeg` to `requirements.txt`, and changed `ppo_walker.py` to request `render_mode="rgb_array"` only when `--render-video` is used.
- 2026-05-16 13:10:00 CST: Recorded Task 3 run 1 evaluations. Best fixed snapshot is 3M with mean reward `2083.043`; train-time best checkpoint has 20-seed mean reward `1908.055`. Current run does not pass the `>=2500` target.
- 2026-05-16 13:35:00 CST: Recorded Task 3 run 2 stable PPO evaluations. `LAB7_314553032_task3_best_v2.pt` reaches 20-seed mean reward `2537.552` at `1700000` environment steps and passes the Task 3 target.
- 2026-05-16 14:05:00 CST: Recorded Task 3 run 3 faster-early PPO evaluations. Fixed `2.5M` snapshot reaches mean reward `3025.007`, fixed `3M` snapshot reaches `2643.701`, and `LAB7_314553032_task3_best_v3.pt` reaches `3662.311`.
- 2026-05-17 00:20:00 CST: Recorded Task 3 run 4 early-stable PPO evaluations. Fixed `2.5M` snapshot reaches mean reward `2969.790`, fixed `3M` snapshot reaches `3440.120`, and `LAB7_314553032_task3_best_v4.pt` reaches `3503.896`.
- 2026-05-17 01:00:00 CST: Recorded Task 3 run 5 early-robust PPO evaluations. Fixed snapshots did not pass; `LAB7_314553032_task3_best_v5.pt` reaches `3174.461`, but fixed-step results are worse than run 4.
- 2026-05-17 01:35:00 CST: Recorded Task 3 run 6 v4-plus-exploration evaluations. Fixed `1M` snapshot has lower variance but mean `2231.630`; fixed `2.5M` and `3M` snapshots pass with `2761.016` and `3369.157`.
- 2026-05-18 00:00:00 CST: Recorded Task 3 run 7 early-exploit PPO evaluations. Fixed `2.5M` and `3M` snapshots pass with `3649.007` and `3568.163`; `LAB7_314553032_task3_best_v7.pt` reaches `4003.744`, but fixed `1M` regresses to `977.724`.
- 2026-05-18 00:30:00 CST: Recorded Task 3 run 8 frequent-update PPO evaluations. Fixed `1.5M` snapshot reaches mean reward `3400.151`, fixed `2M` reaches `3683.006`, and `LAB7_314553032_task3_best_v8.pt` reaches `3919.465`.
- 2026-05-19 00:00:00 CST: Recorded complete Task 3 run 9 earlier-discovery PPO evaluations. Fixed `1M` improves to `1808.769`, fixed `1.5M` passes at `2517.718`, fixed `2M` reaches `3264.254`, fixed `3M` is `3583.417`, and `LAB7_314553032_task3_best_v9.pt` reaches `3784.047`.
- 2026-05-19 00:30:00 CST: Recorded Task 3 run 10 balanced-early PPO evaluations. Fixed `1M` snapshot passes with mean reward `2603.904`, fixed `3M` reaches `3991.904`, and `LAB7_314553032_task3_best_v10.pt` reaches `4079.485`.

## Current Task 1 Commands

Train without W&B:

```bash
python a2c_pendulum.py --mode train --num-episodes 1000 --no-wandb
```

Train with W&B:

```bash
python a2c_pendulum.py --mode train --num-episodes 10000 --actor-lr 3e-5 --critic-lr 3e-4 --entropy-weight 1e-4 --discount-factor 0.9 --n-step 5 --reward-scale 10.0 --init-log-std -0.5 --min-log-std -2.0 --max-log-std 0.5 --model-path LAB7_314553032_task1_a2c_pendulum_trainbest_untilpass.pt --eval-interval 20000 --eval-model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt --target-eval-mean -150 --wandb-run-name pendulum-a2c-untilpass
```

Evaluate the saved snapshot on seeds 0 to 19:

```bash
python a2c_pendulum.py --mode eval --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

Record one evaluation video, then run seed evaluation:

```bash
python a2c_pendulum.py --mode eval --model-path LAB7_314553032_task1_a2c_pendulum.pt --render-video --video-folder videos/a2c_pendulum --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

## Current Limitations

- Task 1 implementation is complete at code level, but this machine currently lacks a compatible installed runtime for `torch`, `gymnasium`, and `wandb`.
- Verified command so far: `python3 -m py_compile a2c_pendulum.py`.
- Runtime train/eval validation should be rerun after creating a Python environment with dependencies from `requirements.txt`.
