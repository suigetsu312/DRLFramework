# Memory/replay.py
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, size: int, obs_shape, action_dtype=np.int64):
        self.size = int(size); self.ptr = 0; self.full = False
        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.act = np.zeros((size,), dtype=action_dtype)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i] = o; self.act[i] = a; self.rew[i] = r
        self.next_obs[i] = no; self.done[i] = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0: self.full = True

    def __len__(self): return self.size if self.full else self.ptr

    def sample(self, batch_size: int, device="cpu"):
        hi = len(self)
        idx = np.random.randint(0, hi, size=int(batch_size))
        return dict(
            obs=torch.as_tensor(self.obs[idx], device=device),
            act=torch.as_tensor(self.act[idx], device=device),
            rew=torch.as_tensor(self.rew[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            done=torch.as_tensor(self.done[idx], device=device),
        )
