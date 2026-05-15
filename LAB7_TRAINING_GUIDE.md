# Lab 7 Training Guide

這份文件說明如何訓練與評估目前已完成的 Task 1 A2C Pendulum 程式。它的目的不是解釋演算法原理，而是讓使用者能依照固定流程建立環境、啟動訓練、保存 snapshot、執行 evaluation，並整理作業 report 需要的結果。

目前本 repo 已完成:

- Task 1: `a2c_pendulum.py`
- Progress tracker: `LAB7_PROGRESS.md`
- Dependency list: `requirements.txt`

Task 2 與 Task 3 尚未完成，請不要用這份 guide 訓練 PPO 或 Walker2d。

## 1. 建立 Python 環境

建議使用 Python 3.11。不要使用本機預設的 Python 3.14，因為 PyTorch 與 Gymnasium 可能尚未完整支援。

建立 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果系統沒有 `python3.11`，可以先安裝 Python 3.11，再重跑上面的步驟。

確認套件:

```bash
python -c "import torch, gymnasium, numpy, wandb, tqdm; print('ok')"
```

確認程式語法:

```bash
python -m py_compile a2c_pendulum.py
```

## 2. 快速 Smoke Test

先用很少的 episodes 確認程式可以跑，不要一開始就跑完整訓練。

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 2 \
  --no-wandb
```

成功時應該會看到:

- device 顯示，例如 `cpu` 或 `cuda`
- 每個 episode 的 total reward
- `LAB7_314553032_task1_a2c_pendulum.pt` 被保存

如果 smoke test 失敗，先修程式或環境，不要繼續長時間訓練。

## 3. 正式訓練 Task 1

不使用 W&B:

```bash
python a2c_pendulum.py \
  --mode train \
  --num-episodes 1000 \
  --model-path LAB7_314553032_task1_a2c_pendulum.pt \
  --no-wandb
```

使用 W&B:

```bash
wandb login
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

訓練時會保存兩種 snapshot:

```txt
LAB7_314553032_task1_a2c_pendulum_trainbest_untilpass.pt   # 單一 training episode best_return，診斷用
LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt    # seeds 0-19 mean reward 最佳，評分優先用
```

`evalbest` 比 `trainbest` 更重要，因為作業評分也是看 20 episodes mean reward。

## 4. 建議訓練策略

Task 1 的評分重點是 Pendulum 20 episodes 平均分數達到 `>-150`，且越早達成越好。

建議流程:

1. 先跑 `2` episodes smoke test。
2. 再跑 `100` episodes，確認 reward 是否有逐漸改善。
3. 若 reward 有改善，再跑 `1000` episodes。
4. 訓練中使用 periodic evaluation，每 `20000` environment steps 自動評估 seeds `0` 到 `19`。
5. 優先使用 `LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt` 做最終 evaluation。
6. 若平均分數仍低於 `-150`，再調整 learning rate、entropy weight 或訓練 episodes。

可嘗試的超參數:

```bash
--actor-lr 3e-5
--critic-lr 3e-4
--discount-factor 0.9
--entropy-weight 1e-4
--n-step 5
--reward-scale 10.0
--init-log-std -0.5
--min-log-std -2.0
--max-log-std 0.5
```

若訓練不穩，可嘗試:

```bash
--actor-lr 1e-5
--critic-lr 3e-4
--entropy-weight 1e-4
--n-step 10
--reward-scale 10.0
--init-log-std -0.75
--min-log-std -2.5
--max-log-std 0.25
```

目前程式的 A2C 更新已改為 n-step rollout。`--n-step 5` 代表累積 5 個 transition 後再更新一次 actor/critic，比原本每一步 one-step TD 更能看見較長期的 reward。`--reward-scale 10.0` 只縮小訓練時 critic target 的數值，evaluation 輸出的原始 Pendulum reward 不會被改變。advantage normalization 預設開啟；只有做 ablation 時才建議加 `--disable-advantage-norm` 關閉。

目前程式也支援 Gaussian action distribution 控制。`--init-log-std -0.5` 會讓初始 action standard deviation 約為 `exp(-0.5) = 0.61`，比原本 `1.0` 更保守；`--min-log-std` 和 `--max-log-std` 會限制訓練過程中的探索範圍。若 hard seeds 仍失敗，優先觀察 W&B 的 `action/log_std` 與 `action/clamped_log_std`，判斷 policy 是否仍然太隨機或太早失去探索。

若想讓程式一直訓練到 fixed-seed eval-best mean reward 達標，可以加上 `--target-eval-mean -150`，並把 `--num-episodes` 設為較大的上限，例如 `10000`。程式會在 periodic evaluation 後檢查 `eval/best_mean_reward > -150`，達成後保存 eval-best checkpoint 並自動停止。

## 5. 評估 Snapshot

用 seeds `0` 到 `19` 評估 20 episodes:

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

輸出會包含:

- 每個 seed 的 reward
- model path
- training environment step
- mean reward

Report 截圖必須包含 evaluation 結果和 environment step。

## 6. 錄製 Evaluation Video

錄製影片並同時跑 evaluation:

```bash
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt \
  --render-video \
  --video-folder videos/a2c_pendulum \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

影片會輸出到:

```txt
videos/a2c_pendulum/
```

注意: `videos/` 已被 `.gitignore` 排除，不會進入 Git commit。正式繳交時影片需另外放到 ZIP root，依作業規定命名。

## 7. W&B 紀錄重點

若使用 W&B，請確認 run 中至少有:

- `return`
- `best_return`
- `actor loss`
- `critic loss`
- `step`
- `episode`
- `eval/mean_reward`
- `eval/best_mean_reward`

Report 中的 training curve 應使用:

```txt
x-axis: environment steps
y-axis: evaluation score 或 episodic return
```

若目前只有 training return，後續仍應補上 fixed-seed evaluation curve，讓 report 更符合 PDF 要求。

## 8. 常見問題

### `ModuleNotFoundError: No module named 'torch'`

代表目前 Python 環境沒有安裝 PyTorch。請啟動 `.venv` 並安裝 dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `python3.11: command not found`

代表系統沒有 Python 3.11。請先安裝 Python 3.11，或使用其他已安裝且支援 PyTorch 的 Python 版本。

### 訓練後沒有 snapshot

程式只會在 episode 結束且 reward 超過目前 best score 時保存模型。若 smoke test 太短，也應該會至少保存一次；若沒有，請檢查目前目錄是否可寫入。

### 評估分數很差

先確認:

- training 是否真的跑完。
- evaluation 是否載入正確 snapshot。
- action 在 eval 時是否使用 deterministic mean action。
- reward 是否隨 episode 有改善。

若仍然很差，再調整 learning rate 或 entropy weight。

### `_pickle.UnpicklingError: Weights only load failed`

這通常發生在 PyTorch 2.6+。PyTorch 2.6 將 `torch.load()` 的 `weights_only` 預設改成 `True`，但本程式保存的 snapshot 不是單純 state dict，還包含 `total_step`、`score`、`seed` 等 metadata。

請先更新到包含 checkpoint loading 修正的最新版程式，再重新執行 evaluation:

```bash
git pull
python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

只對自己訓練產生、可信任來源的 `.pt` 檔案使用這個載入流程。

## 9. Report 可使用的指令區塊

Report 中可以放以下 reproducibility commands:

```bash
pip install -r requirements.txt

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
  --no-wandb

python a2c_pendulum.py \
  --mode eval \
  --model-path LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt \
  --seed-start 0 \
  --seed-end 19 \
  --eval-episodes 20 \
  --no-wandb
```

## 10. 訓練完成後檢查清單

- [ ] `python -m py_compile a2c_pendulum.py` 通過。
- [ ] smoke test 可以跑完至少 2 episodes。
- [ ] 已產生 `LAB7_314553032_task1_a2c_pendulum_trainbest_untilpass.pt`。
- [ ] 已產生 `LAB7_314553032_task1_a2c_pendulum_evalbest_untilpass.pt`。
- [ ] 已完成 seeds `0` 到 `19` evaluation。
- [ ] evaluation mean reward 已記錄。
- [ ] evaluation screenshot 已保存供 report 使用。
- [ ] 若使用 W&B，training curve 已確認 x-axis 可對應 environment steps。
- [ ] demo video 有包含 Task 1 表現。
