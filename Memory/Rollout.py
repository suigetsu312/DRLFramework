# Memory/rollout.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable
import torch
from dataclasses import dataclass

@dataclass
class Step:
    obs: Any
    action: Any
    reward: float
    done: bool
    value: float
    logp: float
    next_obs: Any

class RolloutBuffer:
    """
    通用 on-policy buffer（PPO/A2C 可用）
    - 先 add() 收集 transition（含 value/logp）
    - finalize(last_value, gamma, lam) 產生 advantages/returns
    - iter_minibatches(bs, shuffle=True) 取 batch
    """
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.steps: List[Step] = []
        self.adv = None
        self.ret = None

    def __len__(self): return len(self.steps)
    def clear(self):
        self.steps.clear()
        self.adv = self.ret = None

    def add(self, obs, action, reward, done, value, logp, next_obs):
        if len(self) >= self.capacity:
            raise RuntimeError(f"RolloutBuffer full: capacity={self.capacity}")
        self.steps.append(Step(obs, action, float(reward), bool(done), float(value), float(logp), next_obs))

    def _stack(self, key: str):
        xs = [getattr(s, key) for s in self.steps]
        if key in ("action",):
            # 可能是 scalar/int 或向量
            if isinstance(xs[0], (list, tuple)):
                return torch.as_tensor(xs, dtype=torch.float32, device=self.device)
            if isinstance(xs[0], (int,)):
                return torch.as_tensor(xs, dtype=torch.long, device=self.device)
            return torch.as_tensor(xs, dtype=torch.float32, device=self.device)
        elif key in ("done",):
            return torch.as_tensor(xs, dtype=torch.float32, device=self.device)
        else:
            return torch.as_tensor(xs, dtype=torch.float32, device=self.device)

    def tensors(self) -> Dict[str, torch.Tensor]:
        return dict(
            obs=self._stack("obs"),
            action=self._stack("action"),
            reward=self._stack("reward"),
            done=self._stack("done"),
            value=self._stack("value"),
            logp=self._stack("logp"),
            next_obs=self._stack("next_obs"),
        )

    def finalize(self, gamma: float, lam: float, bootstrap_value: float):
        T = len(self)
        data = self.tensors()
        r, d, v = data["reward"], data["done"], data["value"]
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - d[t]
            nv = v[t+1] if t < T-1 else torch.tensor(bootstrap_value, device=self.device)
            delta = r[t] + gamma * nv * mask - v[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        ret = adv + v
        self.adv, self.ret = adv, ret
        return adv, ret

    def normalized_adv(self, eps: float = 1e-8):
        assert self.adv is not None
        return (self.adv - self.adv.mean()) / (self.adv.std() + eps)

    def iter_minibatches(self, batch_size: int, shuffle: bool = True) -> Iterable[Dict[str, torch.Tensor]]:
        data = self.tensors()
        if self.adv is None or self.ret is None:
            raise RuntimeError("Call finalize() before iterating batches.")
        data["adv"] = self.adv
        data["ret"] = self.ret

        N = len(self)
        idx = torch.randperm(N, device=self.device) if shuffle else torch.arange(N, device=self.device)
        for s in range(0, N, batch_size):
            mb = idx[s:s+batch_size]
            yield {k: v[mb] for k, v in data.items()}
