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
| Done | Add Task 1 training guide | Pending current commit | Markdown review and `python3 -m py_compile a2c_pendulum.py` passed | Added environment setup, training, evaluation, video, W&B, troubleshooting, and report command guidance. |

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

## Current Task 1 Commands

Train without W&B:

```bash
python a2c_pendulum.py --mode train --num-episodes 1000 --no-wandb
```

Train with W&B:

```bash
python a2c_pendulum.py --mode train --num-episodes 1000 --wandb-run-name pendulum-a2c-run
```

Evaluate the saved snapshot on seeds 0 to 19:

```bash
python a2c_pendulum.py --mode eval --model-path LAB7_314553032_task1_a2c_pendulum.pt --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

Record one evaluation video, then run seed evaluation:

```bash
python a2c_pendulum.py --mode eval --model-path LAB7_314553032_task1_a2c_pendulum.pt --render-video --video-folder videos/a2c_pendulum --seed-start 0 --seed-end 19 --eval-episodes 20 --no-wandb
```

## Current Limitations

- Task 1 implementation is complete at code level, but this machine currently lacks a compatible installed runtime for `torch`, `gymnasium`, and `wandb`.
- Verified command so far: `python3 -m py_compile a2c_pendulum.py`.
- Runtime train/eval validation should be rerun after creating a Python environment with dependencies from `requirements.txt`.
