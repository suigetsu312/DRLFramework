# Agent/PPO.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from Agent.Registry import register_agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from gymnasium import spaces

from Agent.Agent import AgentBase, Transition
from Policy.Builder import build_model_from_config  # 建 policy head 的模型（含 backbone）
from Policy.Categorical import CategoricalHead
from Policy.Gaussian import GaussianHead


# --------- 小工具 ---------
def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    dev_str = str(cfg["DRL"]["Params"].get("device", "auto")).lower()
    if dev_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev_str.startswith("cuda"):
        if torch.cuda.is_available(): return torch.device(dev_str)
        print("[PPO] CUDA not available, fallback to CPU")
        return torch.device("cpu")
    return torch.device("cpu")

def _is_discrete(space) -> bool:
    return isinstance(space, spaces.Discrete)

def _action_dim(space) -> int:
    return (space.n if isinstance(space, spaces.Discrete)
            else int(space.shape[0]))


# --------- Rollout Buffer ---------
class RolloutBuffer:
    def __init__(self, capacity: int, obs_shape, action_space, device: torch.device):
        self.capacity = int(capacity)
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.next_obs = []  # for bootstrap convenience
        self.device = device
        self.discrete = isinstance(action_space, spaces.Discrete)

    def add(self, t: Transition, value: float, log_prob: float):
        self.obs.append(t.state)
        self.actions.append(t.action)
        self.rewards.append(float(t.reward))
        self.dones.append(bool(t.done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.next_obs.append(t.next_state)

    def __len__(self): return len(self.rewards)

    def clear(self):
        self.obs.clear(); self.actions.clear(); self.rewards.clear(); self.dones.clear()
        self.values.clear(); self.log_probs.clear(); self.next_obs.clear()

    def to_tensors(self):
        obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        if self.discrete:
            actions = torch.as_tensor(self.actions, dtype=torch.long, device=self.device)
        else:
            actions = torch.as_tensor(self.actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(self.values, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(self.log_probs, dtype=torch.float32, device=self.device)
        return obs, actions, rewards, dones, values, old_logp


# --------- Value Head（簡單線性；可抽到 Policy/value.py）---------
class ValueHead(nn.Module):
    def __init__(self, in_features: int, bias: bool = True):
        super().__init__()
        self.v = nn.Linear(in_features, 1, bias=bias)
    def forward(self, feat):
        return self.v(feat).squeeze(-1)  # [B]


# --------- Actor-Critic 包裝：共用 backbone，分別掛 policy / value head ---------
class ActorCritic(nn.Module):
    def __init__(self, backbone: nn.Sequential, policy_head: nn.Module, value_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head

    def features(self, obs: torch.Tensor):
        return self.backbone(obs)

    # for convenience
    def policy_dist(self, obs: torch.Tensor):
        feat = self.features(obs)
        if isinstance(self.policy_head, CategoricalHead):
            logits = self.policy_head(feat)
            return D.Categorical(logits=logits)
        elif isinstance(self.policy_head, GaussianHead):
            mu, std = self.policy_head(feat)
            return D.Normal(mu, std)
        else:
            raise TypeError("Unknown policy head type")

    def value(self, obs: torch.Tensor):
        feat = self.features(obs)
        return self.value_head(feat)


# --------- PPO Agent ---------
@register_agent("PPO")
class PPOAgent(AgentBase):
    def __init__(self, cfg: Dict[str, Any], obs_space, action_space):
        super().__init__()
        self.cfg = cfg
        P = cfg["DRL"]["Params"]
        self.device = _device_from_cfg(cfg)

        # === 建 policy backbone+head ===
        # 用你現有 Builder 的輸出：Sequential(backbone, head)
        # 我們把它拆開：假設最後一層是 head，其前面的 Sequential[:-1] 是 backbone
        # （你的 Builder 寫法正是這樣）
        policy_model = build_model_from_config(cfg["Model"], obs_space, action_space).to(self.device)
        assert isinstance(policy_model, nn.Sequential) and len(policy_model) == 2, \
            "Builder expected to return nn.Sequential(backbone, head)"
        backbone: nn.Sequential = policy_model[0]
        policy_head: nn.Module = policy_model[1]

        # === 建 value head（共享同一個 backbone）===
        # 直接用 backbone 最後輸出維度 → ValueHead(in_features=…)
        # 推斷 in_features：建一個假的張量跑一次
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_space.shape, dtype=torch.float32).to(self.device)
            feat_dim = backbone(dummy).shape[-1]
        value_head = ValueHead(feat_dim).to(self.device)

        self.ac = ActorCritic(backbone, policy_head, value_head).to(self.device)
        self.discrete = _is_discrete(action_space)
        self.action_dim = _action_dim(action_space)

        # === Optimizer ===
        opt_cfg = cfg["DRL"]["Optimizer"]
        lr = float(opt_cfg["params"].get("lr", 3e-4))
        wd = float(opt_cfg["params"].get("weight_decay", 0.0))
        eps = float(opt_cfg["params"].get("eps", 1e-5))
        self.opt = getattr(optim, opt_cfg["type"])(self.ac.parameters(), lr=lr, weight_decay=wd, eps=eps)

        # === 超參數 ===
        self.gamma = float(P["gamma"])
        self.lam = float(P.get("gae_lambda", 0.95))
        self.clip = float(P.get("clip_range", 0.2))
        self.vf_clip = float(cfg["DRL"]["Loss"]["params"].get("vf_clip_range", 0.2)) if cfg["DRL"]["Loss"]["params"] else 0.2
        self.update_epochs = int(P.get("update_epochs", 4))
        self.minibatch_size = int(P.get("minibatch_size", 64))
        self.rollout_steps = int(P.get("rollout_steps", 2048))
        self.vf_coef = float(P.get("vf_coef", 0.5))
        self.ent_coef = float(P.get("ent_coef", 0.0))
        self.max_grad_norm = float(P.get("max_grad_norm", 0.5))
        self.target_kl = P.get("target_kl", None)
        self.normalize_adv = bool(P.get("normalize_adv", True))

        # === buffer ===
        self.buf = RolloutBuffer(self.rollout_steps, obs_space.shape, action_space, self.device)
        self._steps_since_update = 0

    # -------- AgentBase API --------
    def act(self, state, step: int, deterministic: bool = False) -> Tuple[Any, Dict[str, Any]]:
        self.ac.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.ac.policy_dist(s)
            if self.discrete:
                action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
                logp = dist.log_prob(action)
                value = self.ac.value(s)
                a = int(action.item())
                meta = {"mode": "greedy" if deterministic else "sample",
                        "logp": float(logp.item()), "value": float(value.item())}
                return a, meta
            else:
                a_t = dist.mean if deterministic else dist.rsample()
                logp = dist.log_prob(a_t).sum(-1)
                value = self.ac.value(s)
                a = a_t.squeeze(0).detach().cpu().numpy()
                meta = {"mode": "mean" if deterministic else "rsample",
                        "logp": float(logp.item()), "value": float(value.item())}
                return a, meta

    def record(self, t: Transition) -> None:
        # 重新計算 value / log_prob（避免 Trainer 不傳 meta）
        with torch.no_grad():
            s = torch.as_tensor(t.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.ac.policy_dist(s)
            if self.discrete:
                a = torch.as_tensor([t.action], device=self.device)
                logp = dist.log_prob(a).squeeze(0).item()
            else:
                a = torch.as_tensor(t.action, dtype=torch.float32, device=self.device).unsqueeze(0)
                logp = dist.log_prob(a).sum(-1).item()
            v = self.ac.value(s).item()
        self.buf.add(t, v, logp)
        self._steps_since_update += 1

    def learn_if_ready(self, step: int) -> Dict[str, float]:
        if self._steps_since_update < self.rollout_steps:
            return {}

        # 需要 bootstrap 的 next value（用最後一個 next_obs）
        with torch.no_grad():
            last_next_obs = torch.as_tensor(self.buf.next_obs[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.ac.value(last_next_obs).item()

        logs = self._update(last_value)
        self.buf.clear()
        self._steps_since_update = 0
        return logs

    def save(self, path: str, step: Optional[int] = None, episode: Optional[int] = None) -> None:
        os.makedirs(path, exist_ok=True)
        tag = f"step{step}" if step is not None else ("ep" + str(episode) if episode is not None else "latest")
        torch.save({"ac": self.ac.state_dict(),
                    "opt": self.opt.state_dict()}, os.path.join(path, f"ckpt_{tag}.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(ckpt["ac"])
        if "opt" in ckpt: self.opt.load_state_dict(ckpt["opt"])

    def train_mode(self) -> None: self.ac.train()
    def eval_mode(self)  -> None: self.ac.eval()

    # -------- PPO internals --------
    def _compute_gae(self, rewards, dones, values, last_value) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            next_value = (values[t+1] if t < T-1 else last_value)
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def _update(self, last_value: float) -> Dict[str, float]:
        obs, actions, rewards, dones, values, old_logp = self.buf.to_tensors()

        # GAE
        adv, ret = self._compute_gae(rewards, dones, values, last_value)
        if self.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 多 epoch 隨機 minibatch
        N = obs.size(0)
        inds = torch.arange(N, device=self.device)
        policy_losses = []; value_losses = []; entropies = []; kls = []

        for _ in range(self.update_epochs):
            perm = inds[torch.randperm(N)]
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)
                mb = perm[start:end]
                mb_obs, mb_actions = obs[mb], actions[mb]
                mb_old_logp, mb_adv, mb_ret, mb_values = old_logp[mb], adv[mb], ret[mb], values[mb]

                dist = self.ac.policy_dist(mb_obs)
                if self.discrete:
                    new_logp = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                else:
                    new_logp = dist.log_prob(mb_actions).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_logp - mb_old_logp)  # pi(a|s)/pi_old(a|s)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss（可選 clip）
                new_values = self.ac.value(mb_obs)
                if self.vf_clip is not None:
                    v_clipped = mb_values + torch.clamp(new_values - mb_values, -self.vf_clip, self.vf_clip)
                    v_loss1 = (new_values - mb_ret).pow(2)
                    v_loss2 = (v_clipped  - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - mb_ret).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().clamp_min(0.0)
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                kls.append(approx_kl.item())

            # early stop by KL
            if self.target_kl is not None:
                if sum(kls[-(N//self.minibatch_size+1):]) / max(1, (N//self.minibatch_size+1)) > float(self.target_kl):
                    break

        logs = {
            "ppo/policy_loss": float(sum(policy_losses)/len(policy_losses)),
            "ppo/value_loss": float(sum(value_losses)/len(value_losses)),
            "ppo/entropy": float(sum(entropies)/len(entropies)),
            "ppo/kl": float(sum(kls)/len(kls)),
            "ppo/num_steps": float(N),
        }
        return logs
