# Agent/DQN.py
import os, random
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from Agent.Agent import AgentBase, Transition
from Policy.Builder import build_model_from_config
from Memory.Replay import ReplayBuffer
from Utils.Scheduler import EpsilonGreedy
from Agent.Registry import register_agent

@register_agent("DQN")
class DQNAgent(AgentBase):
    def __init__(self, cfg: Dict[str, Any], obs_space, action_space):
        super().__init__()
        self.cfg = cfg
        P = cfg["DRL"]["Params"]
        self.device = torch.device(P.get("device", "cpu"))

        # model
        self.q = build_model_from_config(cfg["Model"], obs_space, action_space).to(self.device)
        self.q_tgt = build_model_from_config(cfg["Model"], obs_space, action_space).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict()); self.q_tgt.eval()
        self.num_actions = int(action_space.n)

        # optim/loss
        lr = float(P["learning_rate"])
        opt_cfg = cfg["DRL"]["Optimizer"]
        opt_lr = opt_cfg["params"].get("lr") or lr
        self.opt = getattr(optim, opt_cfg["type"])(self.q.parameters(),
                                                   lr=opt_lr,
                                                   weight_decay=opt_cfg["params"].get("weight_decay", 0.0),
                                                   amsgrad=opt_cfg["params"].get("amsgrad", False))
        loss_type = str(cfg["DRL"]["Loss"]["type"]).lower()
        if loss_type == "huber":
            self.crit = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.crit = nn.MSELoss(reduction="none")
        else:
            raise ValueError("Loss must be Huber or MSE")

        # scheduler（可選）
        self.sched = None
        sch_type = cfg["DRL"]["Scheduler"]["type"]
        if sch_type:
            sp = cfg["DRL"]["Scheduler"]["params"]
            if sch_type == "StepLR":
                self.sched = optim.lr_scheduler.StepLR(self.opt, **sp)
            elif sch_type == "CosineAnnealing":
                self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, **sp)
            else:
                raise ValueError(f"Unknown scheduler {sch_type}")

        # buffer & exploration
        self.buffer = ReplayBuffer(P["buffer_size"], tuple(obs_space.shape))
        eps_cfg = cfg["DRL"]["Exploration"]["epsilon"]
        self.eps = EpsilonGreedy(start=eps_cfg["start"], end=eps_cfg["end"],
                                 decay=eps_cfg["decay"], min_eps=eps_cfg["min"],
                                 warmup_steps=eps_cfg.get("warmup_steps", 0))

        # hparams
        self.gamma = float(P["gamma"])
        self.batch_size = int(P["batch_size"])
        self.learning_starts = int(P["learning_starts"])
        self.train_freq = int(P["train_freq"])
        self.grad_accumulate = int(P.get("grad_accumulate", 1))
        self.max_grad_norm = float(P["max_grad_norm"])
        self.double = bool(P.get("double_dqn", True))
        self.tau = P.get("tau", None)
        self.target_update_freq = int(P["target_update_freq"])
        self._updates = 0

    # === AgentBase ===
    def act(self, state, step: int, deterministic: bool = False):
        eps = 0.0 if deterministic else self.eps.value(step)
        if not deterministic and random.random() < eps:
            a = random.randrange(self.num_actions)
            return a, {"mode": "explore", "epsilon": eps}
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(s)
            a = int(q.argmax(dim=-1).item())
            return a, {"mode": "greedy", "epsilon": eps, "q_max": float(q.max().item())}

    def record(self, t: Transition) -> None:
        self.buffer.add(t.state, t.action, t.reward, t.next_state, float(t.done))

    def learn_if_ready(self, step: int) -> Dict[str, float]:
        if len(self.buffer) < max(self.learning_starts, self.batch_size): return {}
        if step % self.train_freq != 0: return {}

        last_logs = {}
        for _ in range(self.grad_accumulate):
            batch = self.buffer.sample(self.batch_size, device=self.device)
            loss, td_err = self._compute_loss(batch)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
            self.opt.step()
            self._updates += 1
            last_logs = {
                "loss/q": float(loss.item()),
                "td_err/mean": float(td_err.abs().mean().item()),
                "buffer/size": float(len(self.buffer)),
                "lr": float(self.opt.param_groups[0]["lr"]),
            }

        if self.sched: self.sched.step()

        # target sync
        if self.tau is not None:
            self._soft_update(float(self.tau))
        elif self._updates % self.target_update_freq == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())
            last_logs["target_synced"] = 1.0

        return last_logs

    def save(self, path: str, step: Optional[int] = None, episode: Optional[int] = None) -> None:
        os.makedirs(path, exist_ok=True)
        tag = f"step{step}" if step is not None else ("ep" + str(episode) if episode is not None else "latest")
        torch.save({"q": self.q.state_dict(),
                    "opt": self.opt.state_dict(),
                    "updates": self._updates}, os.path.join(path, f"ckpt_{tag}.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q"]); self.q_tgt.load_state_dict(ckpt["q"])
        if "opt" in ckpt: self.opt.load_state_dict(ckpt["opt"])
        self._updates = ckpt.get("updates", 0)

    def train_mode(self) -> None: self.q.train(); self.q_tgt.eval()
    def eval_mode(self)  -> None: self.q.eval();  self.q_tgt.eval()

    # === internals ===
    def _compute_loss(self, B: Dict[str, torch.Tensor]):
        obs, act, rew, next_obs, done = B["obs"], B["act"].long(), B["rew"], B["next_obs"], B["done"]
        q_sa = self.q(obs).gather(1, act.view(-1,1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                na = self.q(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.q_tgt(next_obs).gather(1, na).squeeze(1)
            else:
                next_q = self.q_tgt(next_obs).max(dim=1)[0]
            target = rew + (1.0 - done) * self.gamma * next_q

        td = q_sa - target
        # SmoothL1Loss/MSE 都已設 reduction='none'
        loss = self.crit(q_sa, target).mean()
        return loss, td

    def _soft_update(self, tau: float):
        with torch.no_grad():
            for tp, p in zip(self.q_tgt.parameters(), self.q.parameters()):
                tp.data.lerp_(p.data, tau)
