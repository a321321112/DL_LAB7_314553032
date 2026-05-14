# Lab 7 原理導讀: Policy-Based Reinforcement Learning

這份文件的目的，是幫助你在開始寫 Lab 7 程式前，先理解這份作業真正要你學會的觀念。請不要把這份作業看成「調參讓 reward 變高」而已。Lab 7 的核心，是理解 policy-based reinforcement learning 如何直接學習一個可以輸出連續動作的 policy，並透過 value function、advantage estimation、PPO clipping 與 GAE 讓訓練變得更穩定。

## 1. 這份作業想訓練你的能力

Lab 7 的三個任務有一個清楚的學習路徑:

1. 先用 A2C 學會 actor-critic 架構。
2. 再用 PPO-Clip 和 GAE 改善 A2C 的不穩定問題。
3. 最後把 PPO 套到更困難的 MuJoCo robot locomotion 任務。

你需要理解的是:

- policy 如何在 continuous action space 中輸出動作。
- actor 和 critic 各自負責什麼。
- advantage 為什麼比單純 reward 更適合用來更新 policy。
- PPO 為什麼要限制新舊 policy 差距。
- GAE 如何在 bias 和 variance 之間取得折衷。
- 為什麼同一個演算法在 Pendulum 和 Walker2d 上需要不同超參數。

## 2. Reinforcement Learning 的基本問題

在 reinforcement learning 中，agent 會和 environment 互動:

```txt
state s_t -> agent chooses action a_t -> environment returns reward r_t and next state s_{t+1}
```

agent 的目標不是只讓當下 reward 最大，而是讓整段 episode 的 discounted return 最大:

```txt
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

其中 `gamma` 是 discount factor。`gamma` 越接近 1，agent 越重視長期 reward；`gamma` 越小，agent 越重視短期 reward。

在這份作業中:

- Pendulum episode 較短，文件建議使用 `gamma = 0.9`。
- Walker2d 是長期 locomotion 任務，應使用 `gamma = 0.99`。

這不是任意設定，而是和任務時間尺度有關。

## 3. 為什麼使用 Policy-Based Method

有些 RL 方法會先學一個 action-value function，例如 `Q(s, a)`，再選出 value 最大的 action。這種方法在 discrete action space 中很自然，但在 continuous action space 中會變得困難，因為 action 有無限多種可能。

Lab 7 的三個任務都是 continuous control:

- Pendulum 的 action 是連續 torque。
- Walker2d 的 action 是多個 joint motor controls。

因此我們直接學一個 policy:

```txt
pi_theta(a | s)
```

意思是: 在 state `s` 下，由參數 `theta` 控制的 policy 產生 action `a` 的機率分布。

Policy-based method 的精神是:

```txt
如果某個 action 讓結果變好，就提高它未來被選到的機率。
如果某個 action 讓結果變差，就降低它未來被選到的機率。
```

這就是 policy gradient 的核心。

## 4. Gaussian Policy: 連續動作怎麼被取樣

在 continuous action task 中，policy 不能輸出「第幾個 action」，而是要輸出一個連續分布。Lab 7 假設所有任務都使用 Gaussian policy。

常見做法是讓 actor network 輸出:

```txt
mean = mu_theta(s)
std = sigma_theta(s)
```

然後從 Gaussian distribution 取樣:

```txt
a_t ~ Normal(mu_theta(s_t), sigma_theta(s_t))
```

這裡有兩個重要觀念:

- `mean` 代表 policy 目前認為比較好的動作中心。
- `std` 代表 exploration 的程度。

如果 `std` 太小，agent 太早變得保守，可能卡在不好的策略；如果 `std` 太大，agent 會一直亂試，學習也會不穩。

在實作中，通常會學 `log_std`，因為標準差必須是正數，而 `log_std` 比直接學 `std` 更穩定。

## 5. Policy Gradient 的直覺

Policy gradient 的目標是最大化 expected return:

```txt
J(theta) = E[return]
```

實際更新時，我們會使用以下形式:

```txt
gradient ~= log_prob(action) * learning_signal
```

在 A2C 中，這個 `learning_signal` 通常是 advantage。

請注意 `log_prob(action)` 的角色。因為 action 是從 stochastic policy sample 出來的，我們不能直接對「抽樣結果」做普通微分，但可以對「這個 action 被 policy 選到的 log probability」做微分。

直覺上:

- 如果 advantage 是正的，代表這個 action 比預期好，增加它的 log probability。
- 如果 advantage 是負的，代表這個 action 比預期差，降低它的 log probability。

## 6. Actor-Critic 架構

Actor-critic 將問題拆成兩個網路:

```txt
Actor:  pi_theta(a | s), 負責選 action
Critic: V_w(s),          負責估計 state value
```

Actor 要回答:

```txt
在目前 state 下，我應該怎麼行動？
```

Critic 要回答:

```txt
目前 state 本身大概有多好？
```

為什麼需要 critic？因為單純用 episode return 更新 policy，variance 通常很大。Critic 提供 baseline，讓 actor 不只看 reward 大小，而是看「這個 action 是否比原本預期更好」。

## 7. Advantage 的意義

Advantage 表示某個 action 相對於目前 state 平均期待表現的好壞:

```txt
A(s, a) = Q(s, a) - V(s)
```

如果 `A(s, a) > 0`:

```txt
這個 action 比這個 state 下的平均選擇更好。
```

如果 `A(s, a) < 0`:

```txt
這個 action 比平均選擇更差。
```

Lab 7 的 A2C 使用 TD error 作為 advantage 的估計:

```txt
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

這個式子可以理解成:

```txt
實際看到的一步 reward 加上下一個 state 的估計價值，是否比原本對 s_t 的估計更好？
```

所以 TD error 同時可以用來:

- 更新 critic，讓 value prediction 更準。
- 更新 actor，讓好的 action 更常被選到。

## 8. A2C 的原理

A2C 是 Advantage Actor-Critic。它的基本流程是:

1. Actor 根據目前 state sample 一個 action。
2. Environment 執行 action，回傳 reward 和 next state。
3. Critic 計算 `V(s_t)` 和 `V(s_{t+1})`。
4. 用 TD error 估計 advantage。
5. 用 TD error 更新 critic。
6. 用 `log_prob(action) * advantage` 更新 actor。

Critic loss 通常是:

```txt
critic_loss = (r_t + gamma * V(s_{t+1}) - V(s_t))^2
```

Actor loss 在實作中通常寫成要 minimize 的形式:

```txt
actor_loss = -log_prob(action) * advantage
```

負號的原因是 PyTorch optimizer 預設是 minimize loss，但 policy gradient 原本是 maximize objective。

### A2C 的限制

A2C 的每次 policy update 都直接改變 policy。如果 learning rate 太大，或 advantage estimate 太 noisy，policy 可能一步更新太多，導致訓練曲線震盪甚至崩潰。

這就是為什麼 Task 2 要引入 PPO。

## 9. PPO 的核心問題: 如何限制 policy 更新幅度

PPO 的核心想法是:

```txt
我們希望 policy 變好，但不要一次改太多。
```

在 on-policy RL 中，資料是由舊 policy 收集的。如果更新後的新 policy 和舊 policy 差太多，這批資料就不再能可靠代表新 policy 的行為。

PPO 用 probability ratio 衡量新舊 policy 的差異:

```txt
ratio = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
```

也常寫成:

```txt
ratio = exp(new_log_prob - old_log_prob)
```

ratio 的意義:

- `ratio = 1`: 新舊 policy 對這個 action 的機率一樣。
- `ratio > 1`: 新 policy 更傾向選這個 action。
- `ratio < 1`: 新 policy 較不傾向選這個 action。

## 10. PPO-Clip 的原理

PPO-Clip 的 objective 是:

```txt
min(
  ratio * advantage,
  clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
)
```

這個式子的直覺是:

```txt
如果 policy update 已經讓 objective 改善太多，就不要再繼續鼓勵它往同方向大幅更新。
```

`epsilon` 是 clipping parameter，常見值是 `0.1` 到 `0.3`，例如 `0.2`。

舉例:

- 如果 `epsilon = 0.2`，ratio 通常被限制在 `[0.8, 1.2]`。
- 這代表新 policy 不應該比舊 policy 對同一個 action 的機率大幅改變太多。

PPO 不是完全禁止 policy 改變，而是限制「有利更新」過度放大，避免 policy collapse。

## 11. 為什麼 PPO 可以多次更新同一批資料

A2C 通常每收集一小段資料就更新一次。PPO 則會:

1. 用 old policy 收集一批 rollout。
2. 固定 old log probabilities。
3. 對這批資料做多個 epoch 的 mini-batch updates。

因為 PPO 有 clipping 保護，所以可以比較安全地重複使用同一批 on-policy data。這提升了 sample efficiency。

但要注意，PPO 仍然是 on-policy method。資料不能被無限制重複使用，否則新 policy 和收集資料的 old policy 差距仍會過大。

## 12. GAE 的原理

GAE 是 Generalized Advantage Estimation。它處理的是 advantage estimate 的 bias-variance trade-off。

最簡單的一步 TD advantage 是:

```txt
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
```

這個估計 variance 低，但 bias 可能較高，因為它很依賴 critic 的估計。

另一種做法是看完整 future return。這樣 bias 較低，但 variance 很高。

GAE 用 `lambda` 在兩者之間折衷:

```txt
A_t = delta_t + gamma * lambda * delta_{t+1}
    + (gamma * lambda)^2 * delta_{t+2}
    + ...
```

實作時通常反向遞推:

```txt
adv_t = delta_t + gamma * lambda * (1 - done_t) * adv_{t+1}
```

`lambda` 的意義:

- `lambda` 越小，越接近 one-step TD，variance 較低但 bias 較高。
- `lambda` 越大，越接近 Monte Carlo return，bias 較低但 variance 較高。

常用值是:

```txt
lambda = 0.95
```

這也是 PPO 在 MuJoCo 任務中常見的設定。

## 13. Entropy Bonus 和 Exploration

A2C 和 PPO 都是 on-policy method，但仍然需要 exploration。對 Gaussian policy 來說，exploration 主要來自 action sampling 的隨機性。

Entropy 衡量 policy 的不確定性。Entropy 越高，policy 越願意探索；entropy 越低，policy 越接近 deterministic。

PPO objective 中常加入 entropy bonus:

```txt
total_objective = policy_objective - value_loss + entropy_coef * entropy
```

在 minimize loss 的程式中常寫成:

```txt
loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
```

`entropy_coef` 太小:

```txt
agent 可能太早收斂到不好的策略。
```

`entropy_coef` 太大:

```txt
agent 可能一直太隨機，無法穩定提升 performance。
```

這就是 Task 3 要你做 entropy coefficient empirical study 的原因。

## 14. Clipping Parameter 的影響

PPO 的 clipping parameter `epsilon` 控制 policy update 的保守程度。

如果 `epsilon` 太小:

```txt
policy 每次只能改一點點，訓練可能很穩，但 learning speed 慢。
```

如果 `epsilon` 太大:

```txt
policy 可以改很多，可能學得快，但也更容易不穩定。
```

Task 3 要求分析 clipping parameter，是希望你看到:

- learning speed 和 stability 之間的 trade-off。
- 在簡單任務有效的 epsilon，不一定適合複雜任務。
- reward curve 的震盪程度也是重要觀察，不只看最高分。

## 15. 為什麼 Pendulum 和 Walker2d 難度差很多

Pendulum 是低維度 classic control task:

- state 維度低。
- action 維度低。
- episode 長度短。
- reward structure 較直接。

Walker2d 是 MuJoCo locomotion task:

- state 維度高。
- action 維度高。
- dynamics 複雜。
- agent 需要學會平衡、前進、避免摔倒。
- reward curve 初期可能非常不穩。

因此，Pendulum 能很快驗證演算法是否正確；Walker2d 則測試你的 PPO implementation、hyperparameters 和 training stability。

## 16. 如何判斷你的實作是否合理

不要只看最後 reward。你應該同時觀察:

- evaluation reward 是否隨 environment steps 上升。
- actor loss 是否出現極端爆炸。
- critic loss 是否長期不下降或過度震盪。
- entropy 是否太快降到接近 0。
- action distribution 是否過早變得 deterministic。
- PPO ratio 或 approximate KL 是否異常大。
- Pendulum 是否能在合理步數內達到 `>-150`。
- Walker2d 是否能逐漸從站不穩進展到穩定前進。

如果 reward 長期沒有進步，常見原因包括:

- action scaling 錯誤。
- Gaussian log probability 計算錯誤。
- 忘記保存 old log probabilities。
- advantage 沒有 detach，導致梯度流到不該更新的地方。
- GAE done mask 處理錯誤。
- value target 和 critic output shape 不一致。
- reward curve x-axis 不是 environment steps。
- evaluation 時仍然用隨機 sampling 而不是 deterministic mean action。

## 17. Report 應該怎麼寫才像理解原理

Report 不是只貼圖。你應該把每個方法和程式連起來說明。

例如說 A2C 時，不要只寫:

```txt
We implemented A2C.
```

而應該寫清楚:

```txt
The actor outputs a Gaussian policy. During training, we sample actions from this distribution and compute the log probability of the sampled action. The critic estimates V(s). We compute the TD error r + gamma V(s') - V(s), use its squared value as the critic loss, and use it as the advantage estimate for the actor update.
```

說 PPO 時，也不要只寫:

```txt
We used PPO clipping.
```

而要說清楚:

```txt
We store the old log probabilities during rollout collection. During policy updates, we recompute the new log probabilities and form the ratio exp(new_log_prob - old_log_prob). The clipped surrogate objective limits how much the new policy can increase or decrease the probability of sampled actions, which improves stability compared with A2C.
```

說 GAE 時，要能解釋:

```txt
GAE computes a discounted sum of TD errors. It reduces variance compared with Monte Carlo returns while reducing bias compared with one-step TD estimates.
```

## 18. 這份作業最重要的觀念總結

如果你要用一句話理解 Task 1:

```txt
A2C 用 critic 提供 advantage，讓 actor 知道哪些 sampled actions 應該被鼓勵或抑制。
```

如果你要用一句話理解 Task 2:

```txt
PPO 在 A2C 的基礎上限制 policy update 幅度，並用 GAE 取得更穩定的 advantage estimate。
```

如果你要用一句話理解 Task 3:

```txt
Walker2d 測試 PPO 在高維連續控制上的 sample efficiency、training stability 與 hyperparameter sensitivity。
```

最後請記住，這份作業真正要評量的不是你是否能跑出一條漂亮曲線，而是你是否能說明:

- 為什麼這個 loss 是這樣設計的。
- 每個 tensor 在演算法中代表什麼。
- 為什麼 PPO 比 A2C 穩定。
- 為什麼 GAE 能改善 advantage estimation。
- 為什麼 clipping parameter 和 entropy coefficient 會影響訓練結果。
