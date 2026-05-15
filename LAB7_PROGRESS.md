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
| Done | Implement A2C networks | Pending current commit | `python3 -m py_compile a2c_pendulum.py` passed; runtime smoke blocked | Completed Gaussian actor and value critic. Runtime test blocked because local Python has no `torch`. |
| Pending | Implement A2C update step | Pending | Short train smoke test | Complete TD target, actor loss, critic loss, entropy. |
| Pending | Add Task 1 train/eval CLI | Pending | Train/eval CLI smoke tests | Add model save/load and seed range evaluation. |
| Pending | Document Task 1 execution progress | Pending | Final progress review | Record final commands and current limitations. |

## Validation Log

- 2026-05-15 09:33:10 CST: Progress tracker created.
- 2026-05-15 09:33:10 CST: Commit `a9e1468` pushed to `origin/main`.
- 2026-05-15 09:34:00 CST: Started Task 1 project preparation. Local `python3` is 3.14.3 and does not have `torch` installed, so dependency-based smoke tests may need a virtual environment.
- 2026-05-15 09:35:00 CST: `python3 -m py_compile a2c_pendulum.py` passed.
- 2026-05-15 09:35:00 CST: Commit `915f6d8` pushed to `origin/main`.
- 2026-05-15 09:36:00 CST: Implemented A2C Actor/Critic networks. `python3 -m py_compile a2c_pendulum.py` passed. Forward runtime smoke test is blocked because available local Python installations do not have `torch`, `gymnasium`, or `wandb`.
