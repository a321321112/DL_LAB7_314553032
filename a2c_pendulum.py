#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_log_std: float = 0.0,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mu = nn.Linear(128, out_dim)
        self.log_std = nn.Parameter(torch.full((out_dim,), float(init_log_std)))
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)

        initialize_uniformly(self.mu)

    def set_log_std_bounds(self, min_log_std: float, max_log_std: float) -> None:
        """Update action log standard deviation clamp bounds."""
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = 2.0 * torch.tanh(self.mu(x))
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

        initialize_uniformly(self.value)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.n_step = int(getattr(args, "n_step", 5))
        self.reward_scale = float(getattr(args, "reward_scale", 10.0))
        self.normalize_advantage = not bool(getattr(args, "disable_advantage_norm", False))
        self.init_log_std = float(getattr(args, "init_log_std", 0.0))
        self.min_log_std = float(getattr(args, "min_log_std", -5.0))
        self.max_log_std = float(getattr(args, "max_log_std", 2.0))
        self.use_wandb = bool(getattr(args, "use_wandb", False)) and wandb is not None
        self.save_dir = Path(getattr(args, "save_dir", "."))
        model_path = Path(getattr(args, "model_path", "LAB7_314553032_task1_a2c_pendulum.pt"))
        self.model_path = model_path if model_path.is_absolute() else self.save_dir / model_path
        eval_model_path = Path(getattr(args, "eval_model_path", "LAB7_314553032_task1_a2c_pendulum_evalbest.pt"))
        self.eval_model_path = eval_model_path if eval_model_path.is_absolute() else self.save_dir / eval_model_path
        self.eval_interval = int(getattr(args, "eval_interval", 20000))
        self.eval_seed_start = int(getattr(args, "eval_seed_start", 0))
        self.eval_seed_end = int(getattr(args, "eval_seed_end", 19))
        self.train_eval_enabled = not bool(getattr(args, "disable_train_eval", False)) and self.eval_interval > 0
        self.best_score = -float("inf")
        self.eval_best_score = -float("inf")
        self.loaded_training_step = 0
        if self.n_step < 1:
            raise ValueError("--n-step must be at least 1")
        if self.reward_scale <= 0:
            raise ValueError("--reward-scale must be greater than 0")
        if self.min_log_std > self.max_log_std:
            raise ValueError("--min-log-std must be less than or equal to --max-log-std")
        if not self.min_log_std <= self.init_log_std <= self.max_log_std:
            raise ValueError("--init-log-std must be within [--min-log-std, --max-log-std]")
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(
            obs_dim,
            action_dim,
            init_log_std=self.init_log_std,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
        ).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, entropy, next_state, reward, done)
        self.transition: list = list()
        self.rollout: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def _env_id(self) -> str:
        """Return the Gymnasium id used to build a separate eval environment."""
        spec = getattr(getattr(self.env, "unwrapped", self.env), "spec", None)
        return getattr(spec, "id", "Pendulum-v1")

    def _deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """Select the Gaussian mean action without touching training transition state."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            _, dist = self.actor(state_tensor)
            action = dist.mean
        return action.clamp(-2.0, 2.0).cpu().numpy()

    def save_checkpoint(
        self,
        path: Path,
        score: float,
        selection_metric: str,
        eval_seed_start: Optional[int] = None,
        eval_seed_end: Optional[int] = None,
    ) -> None:
        """Save actor/critic weights and selection metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "total_step": int(self.total_step),
            "score": float(score),
            "seed": int(self.seed),
            "selection_metric": selection_metric,
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
        """Save the current actor and critic snapshot."""
        self.save_checkpoint(self.model_path, score, "episode_return")

    def save_eval_model(self, mean_reward: float) -> None:
        """Save the current snapshot selected by periodic evaluation mean."""
        self.save_checkpoint(
            self.eval_model_path,
            mean_reward,
            "eval_mean_reward",
            eval_seed_start=self.eval_seed_start,
            eval_seed_end=self.eval_seed_end,
        )

    def load_model(self, model_path: str) -> None:
        """Load actor and critic weights from a saved snapshot."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "min_log_std" in checkpoint and "max_log_std" in checkpoint:
            self.actor.set_log_std_bounds(checkpoint["min_log_std"], checkpoint["max_log_std"])
            self.min_log_std = float(checkpoint["min_log_std"])
            self.max_log_std = float(checkpoint["max_log_std"])
        self.loaded_training_step = int(checkpoint.get("total_step", 0))
        self.best_score = float(checkpoint.get("score", -float("inf")))

    def evaluate_policy(
        self,
        env: gym.Env,
        seed_start: int,
        seed_end: int,
        verbose: bool = True,
    ) -> Tuple[float, List[float]]:
        """Evaluate deterministic mean actions on a provided environment."""
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            self.transition = [state, log_prob, entropy]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update actor and critic with n-step returns from the rollout buffer."""
        states = torch.stack([transition[0] for transition in self.rollout])
        log_probs = torch.stack([transition[1] for transition in self.rollout])
        entropies = torch.stack([transition[2] for transition in self.rollout])
        rewards = [float(transition[4]) / self.reward_scale for transition in self.rollout]
        dones = [bool(transition[5]) for transition in self.rollout]
        last_next_state = torch.FloatTensor(self.rollout[-1][3]).to(self.device)
        last_done = dones[-1]

        with torch.no_grad():
            running_return = torch.tensor(0.0, device=self.device)
            if not last_done:
                running_return = self.critic(last_next_state).squeeze(-1)

            returns = []
            for reward, done in zip(reversed(rewards), reversed(dones)):
                if done:
                    running_return = torch.tensor(0.0, device=self.device)
                running_return = torch.tensor(reward, dtype=torch.float32, device=self.device) + self.gamma * running_return
                returns.insert(0, running_return)
            returns = torch.stack(returns)

        values = self.critic(states).squeeze(-1)
        advantages = returns - values
        actor_advantages = advantages.detach()
        if self.normalize_advantage and actor_advantages.numel() > 1:
            actor_advantages = (actor_advantages - actor_advantages.mean()) / (
                actor_advantages.std(unbiased=False) + 1e-8
            )

        value_loss = F.mse_loss(values, returns)
        policy_loss = -(log_probs * actor_advantages).mean() - self.entropy_weight * entropies.mean()

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        with torch.no_grad():
            self.actor.log_std.clamp_(self.actor.min_log_std, self.actor.max_log_std)

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        eval_env = gym.make(self._env_id()) if self.train_eval_enabled else None
        next_eval_step = self.eval_interval
        
        state, _ = self.env.reset(seed=self.seed)
        for ep in tqdm(range(1, self.num_episodes + 1)):
            actor_losses, critic_losses, scores = [], [], []
            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False
            while not done:
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                self.rollout.append(self.transition)
                actor_loss, critic_loss = None, None
                if len(self.rollout) >= self.n_step or done:
                    actor_loss, critic_loss = self.update_model()
                    self.rollout = []
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                state = next_state
                score += reward
                self.total_step += 1

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
                    next_eval_step += self.eval_interval

                # W&B logging
                if self.use_wandb and actor_loss is not None and critic_loss is not None:
                    wandb.log({
                        "step": self.total_step,
                        "actor loss": actor_loss,
                        "critic loss": critic_loss,
                        "n_step": self.n_step,
                        "reward_scale": self.reward_scale,
                        "action/log_std": self.actor.log_std.detach().mean().item(),
                        "action/clamped_log_std": self.actor.log_std.detach().clamp(
                            self.actor.min_log_std,
                            self.actor.max_log_std,
                        ).mean().item(),
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    if score > self.best_score:
                        self.best_score = score
                        self.save_model(score)
                    print(f"Episode {ep}: Total Reward = {score} | Steps = {self.total_step}")
                    # W&B logging
                    if self.use_wandb:
                        wandb.log({
                            "episode": ep,
                            "return": score,
                            "best_return": self.best_score,
                        })
        if eval_env is not None:
            eval_env.close()

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

    def evaluate(self, seed_start: int, seed_end: int) -> Tuple[float, List[float]]:
        """Evaluate the deterministic policy over an inclusive seed range."""
        self.is_test = True
        mean_reward, rewards = self.evaluate_policy(self.env, seed_start, seed_end, verbose=True)

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        print(f"model_path={self.model_path}")
        print(f"training_environment_step={self.loaded_training_step}")
        print(f"mean_reward={mean_reward:.3f}")
        return mean_reward, rewards

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--disable-advantage-norm", action="store_true")
    parser.add_argument("--init-log-std", type=float, default=0.0)
    parser.add_argument("--min-log-std", type=float, default=-5.0)
    parser.add_argument("--max-log-std", type=float, default=2.0)
    parser.add_argument("--model-path", type=str, default="LAB7_314553032_task1_a2c_pendulum.pt")
    parser.add_argument("--eval-model-path", type=str, default="LAB7_314553032_task1_a2c_pendulum_evalbest.pt")
    parser.add_argument("--eval-interval", type=int, default=20000)
    parser.add_argument("--eval-seed-start", type=int, default=0)
    parser.add_argument("--eval-seed-end", type=int, default=19)
    parser.add_argument("--disable-train-eval", action="store_true")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=19)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-folder", type=str, default="videos/a2c_pendulum")
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    random.seed(args.seed)
    np.random.seed(args.seed)
    seed_torch(args.seed)
    if args.use_wandb and wandb is not None:
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    elif args.use_wandb and wandb is None:
        print("wandb is not installed; continuing without W&B logging.")
        args.use_wandb = False
    
    agent = A2CAgent(env, args)
    if args.mode == "train":
        agent.train()
    else:
        agent.load_model(str(agent.model_path))
        if args.render_video:
            agent.test(args.video_folder)
        eval_seed_end = args.seed_start + args.eval_episodes - 1 if args.eval_episodes else args.seed_end
        agent.evaluate(args.seed_start, min(args.seed_end, eval_seed_end))
