# Lab 7 Policy-Based Reinforcement Learning Plan

## 作業需求整理

- 課程作業: NYCU CS 535518 Spring 2026 Deep Learning Lab 7
- 主題: Policy-Based Reinforcement Learning
- 截止時間: 2026/05/26 Tuesday 23:59
- 核心方法: A2C, PPO-Clip, GAE
- 實作框架: PyTorch + Gymnasium/OpenAI Gym environments
- 建議追蹤工具: Weights & Biases
- 主要環境:
  - Task 1: `Pendulum-v1`
  - Task 2: `Pendulum-v1`
  - Task 3: `Walker2d-v5`

## 三個任務

### Task 1: A2C on Pendulum

目標: 以 PyTorch 實作 A2C，訓練 `Pendulum-v1`。

需求:
- 以 `a2c.py` 為起點，本專案目前對應檔案是 `a2c_pendulum-2.py`。
- 使用簡單 fully connected policy network。
- Lab 7 全部任務都使用 Gaussian policy。
- 評估並繪製 total episodic reward vs environment steps。
- Pendulum 滿分附近約為 `-130`。
- 模型 snapshot 需在 20 個 evaluation episodes 平均分數達到 `>-150`。
- 若在 200k env steps 內達標，Task 1 performance 可拿滿。

### Task 2: PPO-Clip with GAE on Pendulum

目標: 將 A2C 擴充為 PPO-Clip，加入 clipped surrogate objective 與 GAE，並在 `Pendulum-v1` 評估。

需求:
- 在 A2C 實作基礎上新增 PPO clipped objective。
- 實作 Generalized Advantage Estimation。
- 評估並繪製 total episodic reward vs environment steps。
- 和 Task 1 A2C 比較 sample efficiency、learning speed、training stability。
- 模型 snapshot 需在 20 個 evaluation episodes 平均分數達到 `>-150`。
- 若在 200k env steps 內達標，Task 2 performance 可拿滿。

### Task 3: PPO-Clip on Walker2d

目標: 調整 PPO-Clip 與 hyperparameters，訓練 `Walker2d-v5`。

需求:
- 搜尋或參考 PPO paper 的 MuJoCo hyperparameter 設定。
- 和 A2C on Walker2d 比較 training performance。
- 分析 clipping parameter 與 entropy coefficient 對訓練的影響。
- 需提交 5 個固定步數模型 snapshot:
  - 1M steps
  - 1.5M steps
  - 2M steps
  - 2.5M steps
  - 3M steps
- 另提交一個 `best` snapshot，代表任一步數達到 Walker2d score 2500 的最佳模型。
- Task 3 grading 依達到 Walker2d score 2500 所需 env steps 計分。

## Package 與環境要求

PDF 建議版本:

```txt
Python >= 3.8, tested on 3.11
gymnasium == 1.1.1
torch >= 2.0, tested on 2.11
wandb >= 0.16, tested on 0.26
mujoco >= 3.0, tested on 3.8
tqdm
numpy
```

注意:
- `requirements.txt` 必須 pin 實際使用版本，方便 TA 重現。
- Pendulum template 預設 discount factor `gamma = 0.9` 是刻意設計，建議保留。
- Walker2d 必須使用 `gamma = 0.99`。
- Pendulum 約 100k steps 可在 10 分鐘內完成。
- Walker2d 可能需要 1M 到 3M steps，訓練時間可達數小時。

## Report 需求與評分重點

Report 最高 60%。

### Introduction 5%

- 高階介紹本作業。
- 簡述最重要發現。
- 說明 report 組織。

### Implementation 20%

每一項都要先解釋概念，再說明程式如何實作:
- A2C 如何取得 stochastic policy gradient。
- A2C 如何計算 TD error。
- PPO clipped objective 如何實作。
- GAE estimator 如何計算。
- 如何從環境收集 samples。
- 如何維持 exploration。
- 如何用 W&B 記錄 model performance 與 losses:
  - actor loss
  - critic loss
  - entropy

### Analysis and Discussions 25%

- 分別繪製 Task 1, Task 2, Task 3 的 evaluation score vs environment steps，x-axis 必須是 environment steps。
- 貼上 Task 1, Task 2, Task 3 各自 20 episodes、seed 0 到 19 的 reproducible evaluation screenshot。
- screenshot 內需包含 environment step。
- 比較 A2C 與 PPO 的 sample efficiency 與 training stability。
- 做 clipping parameter 與 entropy coefficient 的 empirical study。

### Bonus up to 10%

- 可加入其他 training strategy 的額外分析。

### Reproducibility

- Report 必須包含可重現 evaluation 的執行指令。
- 若 TA 無法重現結果，report 會扣 10 分。

## Performance、Demo Video、Snapshots

這部分最高 50%。

### Demo Video

- 長度: 5 到 6 分鐘。
- 語言: English，除非事先和 TA 討論特殊情況。
- 內容:
  - 約 2 分鐘說明 source code 與實作。
  - 約 3 分鐘 demo Task 1, Task 2, Task 3 模型表現。
- 若沒有有效 demo video，model snapshots 不會被評分。

### Evaluation Protocol

- 需提供可直接執行的 command 或 `.sh` script。
- 需提供 `requirements.txt`。
- 需提供每個 task 的 evaluation screenshot。
- 每缺一張 screenshot，從 report 扣 3 分。
- 需提交達成 report 數據的 exact model files。
- 若 TA 不能重現，每個 task 可能扣 5 分。
- Evaluation seeds: `0` 到 `19`。

## 繳交格式

ZIP 檔名:

```txt
LAB7_StudentID.zip
```

ZIP 結構:

```txt
LAB7_StudentID.zip
|-- LAB7_StudentID_Code/
|   |-- ppo_walker.py
|   |-- ppo_pendulum.py
|   |-- a2c_pendulum.py
|   |-- requirements.txt
|   |-- any other .py files
|   |-- any other .sh files
|-- LAB7_StudentID.pdf
|-- LAB7_StudentID.mp4
|-- LAB7_StudentID_task1_a2c_pendulum.pt
|-- LAB7_StudentID_task2_ppo_pendulum.pt
|-- LAB7_StudentID_task3_ppo_1m.pt
|-- LAB7_StudentID_task3_ppo_1p5m.pt
|-- LAB7_StudentID_task3_ppo_2m.pt
|-- LAB7_StudentID_task3_ppo_2p5m.pt
|-- LAB7_StudentID_task3_ppo_3m.pt
|-- LAB7_StudentID_task3_best.pt
```

注意:
- Code folder 只放 source code。
- Report、video、model snapshots 必須放在 ZIP root。
- 檔名或資料夾結構錯誤會扣 5 分。
- 2026/05/26 23:59 後不接受遲交，成績為 0。

## 實作計畫

### Phase 0: 專案整理與環境確認

- 將目前檔案整理成最終命名:
  - `a2c_pendulum-2.py` -> `a2c_pendulum.py`
  - `ppo_pendulum-3.py` -> `ppo_pendulum.py`
  - `ppo_walker-3.py` -> `ppo_walker.py`
- 建立 `requirements.txt` 並記錄實際版本。
- 確認可建立以下環境:
  - `Pendulum-v1`
  - `Walker2d-v5`
- 確認 MuJoCo rendering、evaluation、model loading 都可執行。

### Phase 1: Task 1 A2C

- 檢查 actor 是否輸出 Gaussian policy 的 mean 與 std 或 log_std。
- 確認 action sampling、log probability、entropy 計算正確。
- 實作或確認 critic value prediction。
- 使用 TD error:

```txt
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

- Actor update 使用 `log_prob(action) * delta_t`。
- Critic update 使用 TD error squared。
- 記錄:
  - episodic reward
  - evaluation reward
  - actor loss
  - critic loss
  - entropy
  - environment steps
- 儲存 best snapshot:

```txt
LAB7_StudentID_task1_a2c_pendulum.pt
```

- 執行 seed 0 到 19 的 20 episodes evaluation。

### Phase 2: Task 2 PPO Pendulum

- 實作 rollout buffer，保存:
  - states
  - actions
  - rewards
  - dones
  - values
  - old log probabilities
- 實作 GAE:

```txt
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
adv_t = delta_t + gamma * lambda * (1 - done_t) * adv_{t+1}
return_t = adv_t + V(s_t)
```

- 實作 PPO clipped objective:

```txt
ratio = exp(new_log_prob - old_log_prob)
loss_actor = -mean(min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv))
```

- 加入 critic loss 與 entropy bonus。
- 多 epoch、多 mini-batch 更新同一批 rollout。
- 和 A2C 使用相同 x-axis: environment steps。
- 儲存 best snapshot:

```txt
LAB7_StudentID_task2_ppo_pendulum.pt
```

- 執行 seed 0 到 19 的 20 episodes evaluation。

### Phase 3: Task 3 PPO Walker2d

- 將 PPO Pendulum 擴充到 `Walker2d-v5`。
- 保留 `gamma = 0.99`。
- 初始 hyperparameters 可從 PPO paper 或 common MuJoCo PPO 設定開始:
  - rollout length: 2048 或 4096
  - epochs: 10
  - mini-batch size: 64 到 256
  - clip epsilon: 0.1 到 0.3
  - GAE lambda: 0.95
  - entropy coefficient: 0 到 0.01
  - value coefficient: 0.5
  - learning rate: 3e-4 或 1e-4
  - max grad norm: 0.5
- 儲存固定步數 snapshots:

```txt
LAB7_StudentID_task3_ppo_1m.pt
LAB7_StudentID_task3_ppo_1p5m.pt
LAB7_StudentID_task3_ppo_2m.pt
LAB7_StudentID_task3_ppo_2p5m.pt
LAB7_StudentID_task3_ppo_3m.pt
LAB7_StudentID_task3_best.pt
```

- 執行 seed 0 到 19 的 20 episodes evaluation。

### Phase 4: Empirical Study

- Clipping parameter study:
  - `eps = 0.1`
  - `eps = 0.2`
  - `eps = 0.3`
- Entropy coefficient study:
  - `entropy_coef = 0`
  - `entropy_coef = 0.001`
  - `entropy_coef = 0.01`
- 每組至少記錄:
  - evaluation reward curve
  - actor loss
  - critic loss
  - entropy
  - final 20 episodes mean reward
- Report 中比較:
  - learning speed
  - stability
  - final performance
  - exploration effect

### Phase 5: Evaluation Scripts

- 建立或確認三個 evaluation command:

```bash
python a2c_pendulum.py --mode eval --model_path LAB7_StudentID_task1_a2c_pendulum.pt --seed_start 0 --seed_end 19
python ppo_pendulum.py --mode eval --model_path LAB7_StudentID_task2_ppo_pendulum.pt --seed_start 0 --seed_end 19
python ppo_walker.py --mode eval --model_path LAB7_StudentID_task3_best.pt --seed_start 0 --seed_end 19
```

- 評估輸出需包含:
  - model path
  - environment name
  - training environment step
  - each seed reward
  - mean reward over 20 episodes

### Phase 6: Report 與影片

- Report 架構:
  - Introduction
  - Method and Implementation
  - Experiment Setup
  - Results
  - A2C vs PPO Discussion
  - Hyperparameter Empirical Study
  - Reproducibility Commands
- 插入所有訓練曲線與 evaluation screenshots。
- Demo video 腳本:
  - 0:00-2:00 code walkthrough
  - 2:00-3:00 Task 1 model demo
  - 3:00-4:00 Task 2 model demo
  - 4:00-5:30 Task 3 model demo
  - 5:30-6:00 key results summary

## 最終檢查清單

- [ ] `requirements.txt` 已 pin 實際版本。
- [ ] `a2c_pendulum.py` 可 train/eval。
- [ ] `ppo_pendulum.py` 可 train/eval。
- [ ] `ppo_walker.py` 可 train/eval。
- [ ] Task 1 20 seeds evaluation screenshot 已放入 report。
- [ ] Task 2 20 seeds evaluation screenshot 已放入 report。
- [ ] Task 3 20 seeds evaluation screenshot 已放入 report。
- [ ] 三個 task 的 evaluation commands 已寫入 report。
- [ ] W&B 或本地圖表包含 reward vs environment steps。
- [ ] Report 轉成 `LAB7_StudentID.pdf`。
- [ ] Demo video 5 到 6 分鐘，且為英文。
- [ ] 所有 snapshots 檔名符合規定。
- [ ] ZIP root 與 Code folder 結構符合 PDF 要求。
- [ ] 最後用全新目錄解壓 ZIP，跑一次 evaluation commands 確認可重現。
