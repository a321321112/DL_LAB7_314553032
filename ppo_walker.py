#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Initialize one linear layer uniformly."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Actor(nn.Module):
    """Gaussian policy for Walker2d continuous actions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_log_std: float = -0.5,
        min_log_std: float = -2.0,
        max_log_std: float = 0.5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = init_layer_uniform(nn.Linear(128, out_dim))
        self.log_std = nn.Parameter(torch.full((out_dim,), float(init_log_std)))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    def set_log_std_bounds(self, min_log_std: float, max_log_std: float) -> None:
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = 2.0 * torch.tanh(self.mu(x))
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist


class Critic(nn.Module):
    """State-value network."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = init_layer_uniform(nn.Linear(128, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.value(x)


def compute_gae(
    next_value: torch.Tensor,
    rewards: List[torch.Tensor],
    masks: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets."""
    values_with_bootstrap = values + [next_value]
    gae = torch.zeros_like(next_value)
    returns = []

    with torch.no_grad():
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values_with_bootstrap[step + 1] * masks[step] - values_with_bootstrap[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values_with_bootstrap[step])

    returns_tensor = torch.cat(returns).detach()
    values_tensor = torch.cat(values).detach()
    advantages = returns_tensor - values_tensor
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns_tensor, advantages.detach()


def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Yield shuffled mini-batches for PPO updates."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        indices = torch.randperm(batch_size, device=states.device)
        for start in range(0, batch_size, mini_batch_size):
            batch_ids = indices[start : start + mini_batch_size]
            if batch_ids.numel() == 0:
                continue
            yield (
                states[batch_ids],
                actions[batch_ids],
                old_log_probs[batch_ids],
                returns[batch_ids],
                advantages[batch_ids],
            )


class PPOAgent:
    """PPO-Clip agent for Walker2d-v5."""

    def __init__(self, env: gym.Env, args):
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.max_grad_norm = args.max_grad_norm
        self.init_log_std = args.init_log_std
        self.min_log_std = args.min_log_std
        self.max_log_std = args.max_log_std
        self.target_eval_mean = args.target_eval_mean
        self.use_wandb = bool(getattr(args, "use_wandb", False)) and wandb is not None

        self.save_dir = Path(args.save_dir)
        model_path = Path(args.model_path)
        eval_model_path = Path(args.eval_model_path)
        self.model_path = model_path if model_path.is_absolute() else self.save_dir / model_path
        self.eval_model_path = eval_model_path if eval_model_path.is_absolute() else self.save_dir / eval_model_path
        self.eval_interval = int(args.eval_interval)
        self.eval_seed_start = int(args.eval_seed_start)
        self.eval_seed_end = int(args.eval_seed_end)
        self.train_eval_enabled = not args.disable_train_eval and self.eval_interval > 0

        if self.min_log_std > self.max_log_std:
            raise ValueError("--min-log-std must be less than or equal to --max-log-std")
        if not self.min_log_std <= self.init_log_std <= self.max_log_std:
            raise ValueError("--init-log-std must be within [--min-log-std, --max-log-std]")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(
            self.obs_dim,
            self.action_dim,
            init_log_std=self.init_log_std,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
        ).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        self.total_step = 0
        self.loaded_training_step = 0
        self.best_score = -float("inf")
        self.eval_best_score = -float("inf")
        self.is_test = False

    def _env_id(self) -> str:
        spec = getattr(getattr(self.env, "unwrapped", self.env), "spec", None)
        return getattr(spec, "id", "Walker2d-v5")

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).view(-1)

    def _deterministic_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = self._state_tensor(state)
        with torch.no_grad():
            _, dist = self.actor(state_tensor)
            action = dist.mean
        return action.clamp(-2.0, 2.0).cpu().numpy()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = self._state_tensor(state)
        action, dist = self.actor(state_tensor)
        selected_action = dist.mean if self.is_test else action
        selected_action = selected_action.clamp(-2.0, 2.0)

        if not self.is_test:
            value = self.critic(state_tensor).view(1)
            log_prob = dist.log_prob(selected_action).sum(dim=-1).view(1)
            self.states.append(state_tensor.detach())
            self.actions.append(selected_action.detach())
            self.values.append(value.detach())
            self.log_probs.append(log_prob.detach())

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.rewards.append(torch.tensor([float(reward)], dtype=torch.float32, device=self.device))
            self.masks.append(torch.tensor([0.0 if done else 1.0], dtype=torch.float32, device=self.device))

        return next_state, float(reward), done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float, float, float, float]:
        next_state_tensor = self._state_tensor(next_state)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).view(1)

        returns, advantages = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.cat(self.log_probs)

        actor_losses, critic_losses, entropies = [], [], []
        approx_kls, clip_fractions = [], []

        for state, action, old_log_prob, return_, adv in ppo_iter(
            self.update_epoch,
            self.batch_size,
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
        ):
            _, dist = self.actor(state)
            new_log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            ratio = (new_log_prob - old_log_prob).exp()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy

            value = self.critic(state).squeeze(-1)
            critic_loss = F.mse_loss(value, return_)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            with torch.no_grad():
                self.actor.log_std.clamp_(self.actor.min_log_std, self.actor.max_log_std)

            with torch.no_grad():
                approx_kl = (old_log_prob - new_log_prob).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.item())
            approx_kls.append(approx_kl.item())
            clip_fractions.append(clip_fraction.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        return (
            float(np.mean(actor_losses)),
            float(np.mean(critic_losses)),
            float(np.mean(entropies)),
            float(np.mean(approx_kls)),
            float(np.mean(clip_fractions)),
        )

    def save_checkpoint(
        self,
        path: Path,
        score: float,
        selection_metric: str,
        eval_seed_start: Optional[int] = None,
        eval_seed_end: Optional[int] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "total_step": int(self.total_step),
            "score": float(score),
            "seed": int(self.seed),
            "selection_metric": selection_metric,
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "epsilon": float(self.epsilon),
            "entropy_weight": float(self.entropy_weight),
            "rollout_len": int(self.rollout_len),
            "update_epoch": int(self.update_epoch),
            "batch_size": int(self.batch_size),
            "init_log_std": float(self.init_log_std),
            "min_log_std": float(self.actor.min_log_std),
            "max_log_std": float(self.actor.max_log_std),
        }
        if eval_seed_start is not None and eval_seed_end is not None:
            checkpoint["eval_mean_reward"] = float(score)
            checkpoint["eval_seed_start"] = int(eval_seed_start)
            checkpoint["eval_seed_end"] = int(eval_seed_end)
        torch.save(checkpoint, path)

    def save_model(self, score: float) -> None:
        self.save_checkpoint(self.model_path, score, "episode_return")

    def save_eval_model(self, mean_reward: float) -> None:
        self.save_checkpoint(
            self.eval_model_path,
            mean_reward,
            "eval_mean_reward",
            self.eval_seed_start,
            self.eval_seed_end,
        )

    def load_model(self, model_path: str) -> None:
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "min_log_std" in checkpoint and "max_log_std" in checkpoint:
            self.actor.set_log_std_bounds(checkpoint["min_log_std"], checkpoint["max_log_std"])
        self.loaded_training_step = int(checkpoint.get("total_step", 0))
        self.best_score = float(checkpoint.get("score", -float("inf")))

    def evaluate_policy(
        self,
        env: gym.Env,
        seed_start: int,
        seed_end: int,
        verbose: bool = True,
    ) -> Tuple[float, List[float]]:
        was_training = self.actor.training
        self.actor.eval()
        rewards: List[float] = []

        for seed in range(seed_start, seed_end + 1):
            state, _ = env.reset(seed=seed)
            done = False
            score = 0.0
            while not done:
                action = self._deterministic_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                score += float(reward)
            rewards.append(score)
            if verbose:
                print(f"seed={seed} reward={score:.3f}")

        if was_training:
            self.actor.train()
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        return mean_reward, rewards

    def train(self) -> None:
        self.is_test = False
        eval_env = gym.make(self._env_id()) if self.train_eval_enabled else None
        next_eval_step = self.eval_interval
        stop_training = False

        state, _ = self.env.reset(seed=self.seed)
        score = 0.0
        episode_count = 0

        for _ in tqdm(range(1, self.num_episodes + 1)):
            for _ in range(self.rollout_len):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                self.total_step += 1
                score += reward
                state = next_state

                while eval_env is not None and self.total_step >= next_eval_step:
                    mean_reward, _ = self.evaluate_policy(
                        eval_env,
                        self.eval_seed_start,
                        self.eval_seed_end,
                        verbose=False,
                    )
                    if mean_reward > self.eval_best_score:
                        self.eval_best_score = mean_reward
                        self.save_eval_model(mean_reward)
                    print(
                        f"Eval at step {self.total_step}: "
                        f"mean reward = {mean_reward:.3f} | "
                        f"best eval mean = {self.eval_best_score:.3f}"
                    )
                    if self.use_wandb:
                        wandb.log({
                            "step": self.total_step,
                            "eval/mean_reward": mean_reward,
                            "eval/best_mean_reward": self.eval_best_score,
                        })
                    if self.target_eval_mean is not None and self.eval_best_score > self.target_eval_mean:
                        print(
                            f"Target eval mean reached: {self.eval_best_score:.3f} "
                            f"> {self.target_eval_mean:.3f}. Stopping training."
                        )
                        stop_training = True
                    next_eval_step += self.eval_interval
                    if stop_training:
                        break

                if done:
                    episode_count += 1
                    if score > self.best_score:
                        self.best_score = score
                        self.save_model(score)
                    print(f"Episode {episode_count}: Total Reward = {score:.3f} | Steps = {self.total_step}")
                    if self.use_wandb:
                        wandb.log({
                            "step": self.total_step,
                            "episode": episode_count,
                            "return": score,
                            "best_return": self.best_score,
                        })
                    state, _ = self.env.reset()
                    score = 0.0

                if stop_training:
                    break

            if self.states:
                actor_loss, critic_loss, entropy, approx_kl, clip_fraction = self.update_model(state)
                if self.use_wandb:
                    wandb.log({
                        "step": self.total_step,
                        "actor loss": actor_loss,
                        "critic loss": critic_loss,
                        "entropy": entropy,
                        "approx_kl": approx_kl,
                        "clip_fraction": clip_fraction,
                        "action/log_std": self.actor.log_std.detach().mean().item(),
                    })

            if stop_training:
                break

        if eval_env is not None:
            eval_env.close()

    def test(self, video_folder: str) -> None:
        self.is_test = True
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0.0
        while not done:
            action = self.select_action(state)
            state, reward, done = self.step(action)
            score += reward
        print("score: ", score)
        self.env.close()
        self.env = tmp_env

    def evaluate(self, seed_start: int, seed_end: int) -> Tuple[float, List[float]]:
        self.is_test = True
        mean_reward, rewards = self.evaluate_policy(self.env, seed_start, seed_end, verbose=True)
        print(f"model_path={self.model_path}")
        print(f"training_environment_step={self.loaded_training_step}")
        print(f"mean_reward={mean_reward:.3f}")
        return mean_reward, rewards


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2048)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--min-log-std", type=float, default=-2.0)
    parser.add_argument("--max-log-std", type=float, default=0.5)
    parser.add_argument("--model-path", type=str, default="LAB7_314553032_task3_train_latest.pt")
    parser.add_argument("--eval-model-path", type=str, default="LAB7_314553032_task3_best.pt")
    parser.add_argument("--eval-interval", type=int, default=50000)
    parser.add_argument("--eval-seed-start", type=int, default=0)
    parser.add_argument("--eval-seed-end", type=int, default=19)
    parser.add_argument("--target-eval-mean", type=float, default=None)
    parser.add_argument("--disable-train-eval", action="store_true")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=19)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-folder", type=str, default="videos/ppo_walker")
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    return args


if __name__ == "__main__":
    args = parse_args()

    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    random.seed(args.seed)
    np.random.seed(args.seed)
    seed_torch(args.seed)

    if args.use_wandb and wandb is not None:
        wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    elif args.use_wandb and wandb is None:
        print("wandb is not installed; continuing without W&B logging.")
        args.use_wandb = False

    agent = PPOAgent(env, args)
    if args.mode == "train":
        agent.train()
    else:
        agent.load_model(str(agent.model_path))
        if args.render_video:
            agent.test(args.video_folder)
        eval_seed_end = args.seed_start + args.eval_episodes - 1 if args.eval_episodes else args.seed_end
        agent.evaluate(args.seed_start, min(args.seed_end, eval_seed_end))
